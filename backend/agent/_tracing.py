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
- ``llm`` children always show their messages — system prompts, user text,
  plan text carry no Sisense data. Only the DATA-BEARING parts are gated by
  ``FES_LANGSMITH_LOG_CONTENT`` (default false): ``role:"tool"`` messages (tool
  results embedded in decide/verify/finalize prompts for dependent tasks) and
  prior assistant replies in history (they quote data), which are replaced by
  ``[... hidden · N chars]`` placeholders. Outputs of answer-producing calls
  (decide/finalize) are likewise length-only, since a final answer quotes data.
- The root's final reply is content-gated for the same reason; the user's own
  prompt is always included (their words).

Every call in here is best-effort: tracing must never break or slow a turn, so
failures are swallowed and logged at debug level. Enabled by
``LANGSMITH_TRACING=true`` (same master switch as before).
"""

from __future__ import annotations

import contextvars
import os
from typing import Any, Dict, List, Optional

from ._config import LLM_CONFIG, LLM_PROVIDER, _scrub_secrets, logger

try:
    from langsmith.run_trees import RunTree
except ImportError:  # pragma: no cover — langsmith is in requirements
    RunTree = None  # type: ignore[assignment]

# The current turn's root run. Set by runtime at turn start; child runs attach
# from anywhere in the call stack (including fan-out branch tasks, which copy
# the parent context).
_CURRENT_ROOT: contextvars.ContextVar[Optional["RunTree"]] = contextvars.ContextVar("fes_langsmith_root", default=None)

# Result-DERIVED texts this turn (decide's CONTINUE ops, replan output): they
# embed values from tool results (adaptive value-passing — "the group
# 'Everyone'"), so with content off they must be redacted wherever they appear
# in later prompts. Registered by the loop, checked by _sanitized_messages.
_TAINTED: contextvars.ContextVar[tuple] = contextvars.ContextVar("fes_langsmith_tainted", default=())


def mark_tainted(text: Optional[str]) -> None:
    """Register a result-derived text fragment for redaction in traced prompts."""
    t = (text or "").strip()
    if len(t) >= 8:  # ignore trivial fragments that would over-redact
        _TAINTED.set(_TAINTED.get() + (t,))


def _enabled() -> bool:
    return RunTree is not None and os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"


def _log_content() -> bool:
    """Whether full prompt/response content may leave for LangSmith.
    Independent of the summarization flag — different destination, own switch."""
    return os.getenv("FES_LANGSMITH_LOG_CONTENT", "false").strip().lower() == "true"


# Plan text stashed in the transcript is safe (orchestrator prose, no data);
# any OTHER assistant text in a prompt is a prior reply, which quotes data.
_SAFE_ASSISTANT_PREFIXES = ("PLAN:", "REVISED PLAN")

# Calls whose OUTPUT derives from reading result data — final answers quote it
# (decide/finalize), and verify/replan reason over the data-bearing transcript —
# so their output content is gated like the root reply.
_DATA_OUTPUT_LABELS = frozenset({"decide", "finalize", "verify", "replan"})

_MSG_CAP = 8000  # per-message char cap so giant catalogs don't bloat ingestion


def _sanitized_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Message list with ONLY the data-bearing parts redacted.

    System prompts, user text, and plan text carry no Sisense data → shown.
    Redacted (length-only placeholders): role:"tool" messages (embedded tool
    results — the dependent-task case) and prior assistant replies from history
    (they quote data). Assistant tool_call stubs and PLAN text stay visible."""
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        role = str(m.get("role", "?"))
        content = str(m.get("content") or "")
        if role == "tool":
            out.append(
                {
                    "role": "tool",
                    "name": m.get("name"),
                    "content": f"[tool result hidden · {len(content)} chars · FES_LANGSMITH_LOG_CONTENT=false]",
                }
            )
        elif content and any(t in content for t in _TAINTED.get()):
            # Derived step text (adaptive value-passing): decide/replan lifted a
            # value out of a result ("the group 'Everyone'") into this text.
            out.append({"role": role, "content": f"[result-derived text hidden · {len(content)} chars]"})
        elif (
            role == "assistant"
            and content
            and not content.startswith(_SAFE_ASSISTANT_PREFIXES)
            and not m.get("tool_calls")
        ):
            out.append({"role": "assistant", "content": f"[prior reply hidden · {len(content)} chars]"})
        else:
            kept: Dict[str, Any] = {"role": role, "content": content[:_MSG_CAP]}
            if m.get("tool_calls"):
                kept["tool_calls"] = m.get("tool_calls")
            out.append(kept)
    return out


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
        _TAINTED.set(())
    except Exception as exc:  # noqa: BLE001 — observability never breaks a turn
        logger.debug("LangSmith start_turn_trace failed: %s", exc)


def end_turn_trace(
    reply: Optional[str] = None,
    outcome: str = "ok",
    steps: Optional[int] = None,
    tools: Optional[List[str]] = None,
) -> None:
    """End + patch the root run. Idempotent: the first call clears the context.
    steps/tools are metadata (never data) → shown regardless of the flag, so the
    collapsed row is scannable even in private mode."""
    root = _CURRENT_ROOT.get()
    if root is None:
        return
    _CURRENT_ROOT.set(None)
    try:
        outputs: Dict[str, Any] = {"outcome": outcome}
        if steps is not None:
            outputs["steps"] = steps
        if tools:
            outputs["tools"] = tools
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
        inputs: Dict[str, Any] = {
            "messages": messages if show else _sanitized_messages(messages),
            "n_tools": n_tools,
        }
        outputs: Dict[str, Any] = {}
        usage = (response or {}).get("usage") or {}
        choices = (response or {}).get("choices") or []
        msg = (choices[0].get("message") or {}) if choices and isinstance(choices[0], dict) else {}
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            outputs["tool_selected"] = [str(((tc.get("function") or {}).get("name")) or "?") for tc in tool_calls]
        content = msg.get("content") or ""
        if show or (name or "") not in _DATA_OUTPUT_LABELS:
            outputs["content"] = content[:4000]
        else:
            # decide/finalize output IS the answer → quotes result data → gated.
            outputs["content_chars"] = len(content)
        outputs["usage_metadata"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        # ls_provider / ls_model_name are LangSmith's convention for automatic
        # cost calculation (token usage x its model price table). Prefer the
        # response's exact model version (e.g. gpt-4o-2024-11-20) over config.
        child_md = dict((root.extra or {}).get("metadata") or {})
        model_name = str((response or {}).get("model") or LLM_CONFIG.model or "")
        if "/" in model_name:  # litellm prefix form, e.g. "azure/gpt-4o"
            model_name = model_name.split("/", 1)[1]
        child_md["ls_provider"] = LLM_PROVIDER
        child_md["ls_model_name"] = model_name
        child = root.create_child(
            name=name or "llm-call",
            run_type="llm",
            inputs=inputs,
            extra={"metadata": child_md},
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
