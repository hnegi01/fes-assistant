"""
LangSmith trace-tree reporting via the LangSmith SDK.

Replaces LiteLLM's bundled langsmith callback, which drops custom metadata keys
and cannot build a run tree (every call landed as an isolated root run). Here we
own the tree: one root ``agent_turn`` run (run_type=chain) per user turn, with
every LLM call and every MCP tool execution as child runs. ``thread_id``
metadata on all runs groups a session's turns in LangSmith's Threads view.

Data boundaries (enforced here, in code — LangSmith is an EXTERNAL cloud
service, a separate trust boundary from the answering LLM provider):

- ``tool`` children NEVER carry result payloads — tool_id, scrubbed args,
  ok/error, row count, duration only.
- ``llm`` children carry message/response CONTENT only when
  ``FES_LANGSMITH_LOG_CONTENT=true``. Default is shape-only (roles + sizes +
  token counts): decide/verify prompts embed tool result data via the
  transcript, so content is closed by default even when summarization is on.
- The root's final reply is likewise content-gated (a summ-on reply can quote
  result data); the user's own prompt is always included (their words).

Every call in here is best-effort: tracing must never break or slow a turn, so
failures are swallowed and logged at debug level. Enabled by
``LANGSMITH_TRACING=true`` (same master switch as before).
"""

from __future__ import annotations

import contextvars
import os
from typing import Any, Dict, List, Optional

from ._config import _scrub_secrets, logger

try:
    from langsmith.run_trees import RunTree
except ImportError:  # pragma: no cover — langsmith is in requirements
    RunTree = None  # type: ignore[assignment]

# The current turn's root run. Set by runtime at turn start; child runs attach
# from anywhere in the call stack (including fan-out branch tasks, which copy
# the parent context).
_CURRENT_ROOT: contextvars.ContextVar[Optional["RunTree"]] = contextvars.ContextVar("fes_langsmith_root", default=None)


def _enabled() -> bool:
    return RunTree is not None and os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"


def _log_content() -> bool:
    """Whether full prompt/response content may leave for LangSmith.
    Independent of the summarization flag — different destination, own switch."""
    return os.getenv("FES_LANGSMITH_LOG_CONTENT", "false").strip().lower() == "true"


def _message_shapes(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Privacy-safe view of a message list: roles and sizes, no content."""
    return [{"role": str(m.get("role", "?")), "chars": len(str(m.get("content") or ""))} for m in messages or []]


def start_turn_trace(user_text: str, session_id: str, mode: str) -> None:
    """Create + post the root ``agent_turn`` run and bind it to the context."""
    if not _enabled():
        return
    try:
        root = RunTree(
            name="agent_turn",
            run_type="chain",
            inputs={"prompt": (user_text or "")[:2000]},
            project_name=os.getenv("LANGSMITH_PROJECT", "default"),
            extra={"metadata": {"thread_id": session_id or "unknown", "mode": mode}},
        )
        root.post()
        _CURRENT_ROOT.set(root)
    except Exception as exc:  # noqa: BLE001 — observability never breaks a turn
        logger.debug("LangSmith start_turn_trace failed: %s", exc)


def end_turn_trace(reply: Optional[str] = None, outcome: str = "ok") -> None:
    """End + patch the root run. Idempotent: the first call clears the context."""
    root = _CURRENT_ROOT.get()
    if root is None:
        return
    _CURRENT_ROOT.set(None)
    try:
        outputs: Dict[str, Any] = {"outcome": outcome}
        if reply is not None:
            if _log_content():
                outputs["reply"] = reply[:4000]
            else:
                outputs["reply_chars"] = len(reply)
        root.end(outputs=outputs)
        root.patch()
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith end_turn_trace failed: %s", exc)


def log_llm_child(
    name: str,
    messages: List[Dict[str, Any]],
    response: Optional[Dict[str, Any]],
    duration_ms: int,
    n_tools: int = 0,
    error: Optional[str] = None,
) -> None:
    """One LLM call as a child ``llm`` run. Content only when opted in."""
    root = _CURRENT_ROOT.get()
    if root is None:
        return
    try:
        show = _log_content()
        inputs: Dict[str, Any] = (
            {"messages": messages} if show else {"messages": _message_shapes(messages), "n_tools": n_tools}
        )
        outputs: Dict[str, Any] = {}
        usage = (response or {}).get("usage") or {}
        choices = (response or {}).get("choices") or []
        msg = (choices[0].get("message") or {}) if choices and isinstance(choices[0], dict) else {}
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            outputs["tool_selected"] = [str(((tc.get("function") or {}).get("name")) or "?") for tc in tool_calls]
        if show:
            outputs["content"] = (msg.get("content") or "")[:4000]
        else:
            outputs["content_chars"] = len(msg.get("content") or "")
        outputs["usage_metadata"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        child = root.create_child(
            name=name or "llm-call",
            run_type="llm",
            inputs=inputs,
            extra={"metadata": dict((root.extra or {}).get("metadata") or {})},
        )
        child.end(outputs=outputs, error=error)
        # Duration: RunTree stamps start/end itself; ours is close enough that we
        # keep only the measured value as metadata for exact cross-checks.
        child.extra.setdefault("metadata", {})["measured_ms"] = duration_ms
        child.post()
        child.patch()
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith log_llm_child failed: %s", exc)


def log_tool_child(
    tool_id: str,
    args: Dict[str, Any],
    ok: Optional[bool],
    count: Optional[int],
    duration_ms: int,
    error: Optional[str] = None,
) -> None:
    """One MCP tool execution as a child ``tool`` run — metadata ONLY, always.
    Result payloads never leave for LangSmith regardless of any flag."""
    root = _CURRENT_ROOT.get()
    if root is None:
        return
    try:
        child = root.create_child(
            name=tool_id or "tool",
            run_type="tool",
            inputs={"tool_id": tool_id, "args": _scrub_secrets(args or {})},
            extra={"metadata": dict((root.extra or {}).get("metadata") or {})},
        )
        outputs: Dict[str, Any] = {"ok": ok, "duration_ms": duration_ms}
        if count is not None:
            outputs["count"] = count
        child.end(outputs=outputs, error=error)
        child.post()
        child.patch()
    except Exception as exc:  # noqa: BLE001
        logger.debug("LangSmith log_tool_child failed: %s", exc)
