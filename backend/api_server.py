"""
FastAPI HTTP API for the FES Assistant backend.

Responsibilities
----------------
- Expose health, tools, and /agent/turn endpoints (for Streamlit UI or any client).
- Load the shared tool registry via backend.agent.llm_agent and select tools per mode:
  - chat: non-migration tools
  - migration: migration tools only
- Enforce summarization policy at the backend layer.
- Bridge each UI request into backend.runtime.run_turn_once.
- Support both JSON and SSE responses on /agent/turn based on the Accept header.

Notes
-----
- The UI sends the full conversation history in `messages`.
- Tool execution + LLM orchestration happen inside call_llm_with_tools (via runtime).
- For SSE: progress events are pushed via runtime's progress callback mechanism.
"""

from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Optional .env loading (local/dev only; safe for Docker/prod)
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parents[1]

if load_dotenv is not None:
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

import asyncio
import contextlib
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from backend.agent import llm_agent
from backend.agent._config import pop_turn_usage
from backend.runtime import cancel_active_turn, run_turn_once

# -----------------------------------------------------------------------------
# Logging (dedicated file for backend API)
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("backend.api")


def _setup_logger() -> None:
    """
    Configure a rotating file logger for the API layer.

    Notes
    -----
    This function is idempotent and avoids adding duplicate handlers.
    """
    log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logger.setLevel(log_level)
    logger.propagate = False

    if any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
        logger.info(
            "backend.api logger already configured at level %s (env %s)",
            log_level_name,
            LOG_LEVEL_ENV_VAR,
        )
        return

    fh = TimedRotatingFileHandler(
        LOG_DIR / "backend_api.log",
        when="midnight",  # daily file; 7 dated backups = 7 days kept, older deleted
        backupCount=7,
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(
        "backend.api logger initialized at level %s (env %s)",
        log_level_name,
        LOG_LEVEL_ENV_VAR,
    )


_setup_logger()


# -----------------------------------------------------------------------------
# Summarization policy (backend enforcement)
# -----------------------------------------------------------------------------
ALLOW_SUMMARIZATION_TOGGLE_ENV_VAR = "FES_ALLOW_SUMMARIZATION_TOGGLE"
ALLOW_SUMMARIZATION_TOGGLE = os.getenv(ALLOW_SUMMARIZATION_TOGGLE_ENV_VAR, "true").strip().lower() == "true"

logger.info(
    "Summarization toggle allowed (backend): %s (env %s)",
    ALLOW_SUMMARIZATION_TOGGLE,
    ALLOW_SUMMARIZATION_TOGGLE_ENV_VAR,
)


# -----------------------------------------------------------------------------
# Helpers: tool selection per mode
# -----------------------------------------------------------------------------
def _select_tools_for_mode(mode: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load tools from the shared registry and return those relevant for the given mode.

    Parameters
    ----------
    mode
        Logical mode: "chat" or "migration" (defaults to "chat" if None/unknown).

    Returns
    -------
    list of dict
        OpenAI-style tool definitions.

    Notes
    -----
    - mode == "migration": return only tools where registry meta module == "migration"
    - otherwise: return tools where registry meta module != "migration"

    If filtering yields an empty list (e.g., registry missing metadata), fall back
    to returning all tools to keep the agent usable.
    """
    all_tools = llm_agent.load_tools_for_llm()
    registry = getattr(llm_agent, "TOOL_REGISTRY", {}) or {}

    tools_by_name: Dict[str, Dict[str, Any]] = {t["function"]["name"]: t for t in all_tools}
    mode_normalized = (mode or "chat").strip().lower()

    if mode_normalized == "migration":
        allowed_names = [tool_id for tool_id, meta in registry.items() if meta.get("module") == "migration"]
    else:
        allowed_names = [tool_id for tool_id, meta in registry.items() if meta.get("module") != "migration"]

    selected = [tools_by_name[name] for name in allowed_names if name in tools_by_name]

    if not selected:
        logger.warning(
            "No tools selected for mode=%s (registry entries=%d, total tools=%d). Falling back to all tools.",
            mode_normalized,
            len(registry),
            len(all_tools),
        )
        selected = all_tools

    logger.info(
        "Selected %d tools for mode=%s (registry entries=%d, total tools=%d)",
        len(selected),
        mode_normalized,
        len(registry),
        len(all_tools),
    )
    return selected


# -----------------------------------------------------------------------------
# SSE helpers
# -----------------------------------------------------------------------------
def _sse_pack(data: Dict[str, Any], event: str = "message") -> str:
    """
    Format one SSE event frame.

    Parameters
    ----------
    data
        JSON-serializable payload.
    event
        SSE event name.

    Returns
    -------
    str
        SSE frame:
          event: <event>
          data: <json>
    """
    try:
        payload = json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        payload = json.dumps({"ok": False, "error": "Failed to JSON encode SSE payload."})
    return f"event: {event}\ndata: {payload}\n\n"


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class AgentTurnRequest(BaseModel):
    """
    Payload sent from the UI (Streamlit or any client) to /agent/turn.
    """

    messages: List[Dict[str, Any]]
    user_input: str = ""

    # Chat mode: single Sisense deployment config
    tenant_config: Optional[Dict[str, Any]] = None

    # Migration mode: source/target deployment config
    migration_config: Optional[Dict[str, Any]] = None

    # Approved mutating tool calls, keyed (tool_id, canonical-JSON args) —
    # must match llm_agent._approval_key exactly (sort_keys=True JSON)
    approved_keys: Optional[List[Tuple[str, str]]] = None

    # Long-lived session identifier from the UI
    session_id: str

    # Per-turn summarization flag requested by the client
    allow_summarization: Optional[bool] = None

    # Logical mode: "chat" or "migration"
    mode: Optional[str] = "chat"


class AgentTurnResponse(BaseModel):
    """
    Response for a single agent turn.

    Attributes
    ----------
    reply
        Natural-language assistant reply.
    tool_result
        The latest raw tool payload captured by llm_agent, if any.
    step_results
        Every tool result of the turn in order ([{step, tool_id, result}]), so the
        UI can show the whole chain — not just the final tool_result.
    trace_id
        The turn's trace id — the same id that groups this turn's rows in
        llm_calls.csv / llm_traces.csv. The UI attaches it to user feedback
        (thumbs up/down) so a vote joins back to the exact calls it judges.
    """

    reply: str
    tool_result: Optional[Dict[str, Any]] = None
    step_results: Optional[List[Dict[str, Any]]] = None
    trace_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None  # {tokens_in, tokens_out, cost} — cost None = pricing unknown
    # Screen-only lines the UI renders under the reply, NEVER stored in a
    # message's content (so they can't re-enter LLM prompts via history) —
    # e.g. clarification option names, shown in every summarization mode.
    display_hints: Optional[List[str]] = None


class CancelRequest(BaseModel):
    session_id: str


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="FES Assistant Backend API",
    version="2.0.0",  # keep in sync with pyproject.toml [project] version
    description="HTTP API for running FES agent turns with MCP tools.",
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Simple health endpoint to check if the backend is up.
    """
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    """
    Return OpenAI-style tools plus a JSON-safe tool registry.

    This is used by the UI for display and metadata (counts per mode, etc.).
    """
    tools = llm_agent.load_tools_for_llm()
    registry_raw = getattr(llm_agent, "TOOL_REGISTRY", {}) or {}

    # Keep only JSON-safe fields that UI needs.
    registry_public: Dict[str, Dict[str, Any]] = {}
    for tool_id, meta in registry_raw.items():
        # example[0] only — the curated one, already shown to users in
        # clarifications and approval dialogs. The UI's capability view reuses
        # it as "you could ask …"; the uncurated siblings never leave the
        # registry (see test_tool_examples.py for the bar example[0] meets).
        examples = meta.get("examples") or []
        first_query = (examples[0] or {}).get("user_query") if examples else None
        registry_public[tool_id] = {
            "id": meta.get("id", tool_id),
            "module": meta.get("module"),
            "sub_module": meta.get("sub_module"),
            "mutates": bool(meta.get("mutates", False)),
            "description": meta.get("description"),
            "example": first_query,
        }

    return {"tools": tools, "registry": registry_public}


@app.post("/agent/cancel")
async def cancel_agent_turn(req: CancelRequest) -> Dict[str, Any]:
    """Cancel any active agent turn for the given session."""
    await cancel_active_turn(req.session_id)
    logger.info("Cancel requested for session %s", req.session_id)
    return {"cancelled": True, "session_id": req.session_id}


@app.post("/agent/turn")
async def agent_turn(request: Request, payload: AgentTurnRequest):
    """
    Main entrypoint: run one agent turn.

    Behavior
    --------
    - If client requests SSE via Accept: text/event-stream, respond as a stream.
    - Otherwise, respond as JSON (backward compatible).
    """
    accept = (request.headers.get("accept") or "").lower()
    wants_sse = "text/event-stream" in accept

    selected_tools = _select_tools_for_mode(payload.mode)

    # Client requested flag
    user_flag = payload.allow_summarization if payload.allow_summarization is not None else False

    # Backend policy enforcement: if toggle is disabled, force False
    effective_allow = bool(user_flag) if ALLOW_SUMMARIZATION_TOGGLE else False

    approved_set: Set[Tuple[str, str]] = set(payload.approved_keys or [])

    logger.info(
        "Received /agent/turn: messages=%d mode=%s tools_for_mode=%d "
        "tenant=%s migration=%s approvals=%d session_id=%s "
        "allow_summarization=%s effective_allow_summarization=%s wants_sse=%s",
        len(payload.messages),
        payload.mode,
        len(selected_tools),
        bool(payload.tenant_config),
        bool(payload.migration_config),
        len(approved_set),
        payload.session_id,
        bool(user_flag),
        bool(effective_allow),
        wants_sse,
    )

    # -------------------------------------------------------------------------
    # JSON path (default)
    # -------------------------------------------------------------------------
    if not wants_sse:
        try:
            turn = await run_turn_once(
                session_id=payload.session_id,
                messages=payload.messages,
                user_input=payload.user_input,
                tools=selected_tools,
                tenant_config=payload.tenant_config,
                approved_keys=approved_set,
                migration_config=payload.migration_config,
                allow_summarization=effective_allow,
            )

            # Everything comes from the turn's own snapshot — never from
            # llm_agent module globals, which another session's turn may have
            # overwritten by the time this coroutine resumes.
            reply = turn["reply"]
            tool_result = turn["tool_result"]
            step_results = turn["step_results"]
            trace_id = turn["trace_id"]
            usage = pop_turn_usage(trace_id or "")

            logger.info("Agent turn completed successfully (JSON).")
            logger.debug("Reply (truncated): %s", reply[:500] if isinstance(reply, str) else repr(reply))

            return AgentTurnResponse(
                reply=reply,
                tool_result=tool_result,
                step_results=step_results,
                trace_id=trace_id,
                usage=usage,
                display_hints=turn.get("display_hints") or None,
            )

        except Exception as exc:
            logger.exception("Error while handling /agent/turn (JSON): %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # -------------------------------------------------------------------------
    # SSE path
    # -------------------------------------------------------------------------
    q: asyncio.Queue[Optional[Tuple[str, Dict[str, Any]]]] = asyncio.Queue()

    async def _progress_cb(event: Dict[str, Any]) -> None:
        """
        Receives progress events from runtime.publish_progress() and forwards them to SSE.
        """
        try:
            await q.put(("progress", event))
        except Exception:
            logger.debug("Failed to enqueue progress event; ignored.", exc_info=True)

    async def _run_turn_and_publish() -> None:
        """
        Worker task that runs the agent turn and publishes status/progress/result into the queue.
        """
        try:
            await q.put(("status", {"phase": "started"}))

            turn = await run_turn_once(
                session_id=payload.session_id,
                messages=payload.messages,
                user_input=payload.user_input,
                tools=selected_tools,
                tenant_config=payload.tenant_config,
                approved_keys=approved_set,
                migration_config=payload.migration_config,
                allow_summarization=effective_allow,
                progress_cb=_progress_cb,
            )

            # Same rule as the JSON path: read the turn snapshot, not globals.
            reply = turn["reply"]
            tool_result = turn["tool_result"]
            step_results = turn["step_results"]
            trace_id = turn["trace_id"]
            usage = pop_turn_usage(trace_id or "")

            await q.put(
                (
                    "result",
                    {
                        "reply": reply,
                        "tool_result": tool_result,
                        "step_results": step_results,
                        "trace_id": trace_id,
                        "usage": usage,
                        "display_hints": turn.get("display_hints") or None,
                    },
                )
            )
            await q.put(("status", {"phase": "completed"}))

        except asyncio.CancelledError:
            # Turn was cancelled (likely due to client disconnect). Emit a final status if you want.
            logger.info("Turn cancelled (session_id=%s).", payload.session_id)
            with contextlib.suppress(Exception):
                await q.put(("status", {"phase": "cancelled"}))
            raise

        except Exception as exc:
            logger.exception("Error while handling /agent/turn (SSE): %s", exc)
            await q.put(("error", {"ok": False, "error": str(exc), "error_type": type(exc).__name__}))

        finally:
            await q.put(None)

    async def _event_generator() -> AsyncIterator[str]:
        """
        SSE generator that drains queue items and emits SSE frames, with keepalives.
        Also detects client disconnect and cancels the running turn.
        """
        turn_task = asyncio.create_task(_run_turn_and_publish())
        keepalive_seconds = 10.0

        try:
            while True:
                # Explicit disconnect check (refresh/close tab)
                if await request.is_disconnected():
                    logger.info("SSE client disconnected; cancelling turn (session_id=%s).", payload.session_id)

                    # REQUIRED: notify runtime to cancel the active MCP session (which calls MCP /mcp/cancel).
                    with contextlib.suppress(Exception):
                        from backend import runtime as runtime_mod

                        logger.info("Calling runtime.cancel_active_turn session_id=%s", payload.session_id)
                        await runtime_mod.cancel_active_turn(payload.session_id)
                        logger.info("runtime.cancel_active_turn returned session_id=%s", payload.session_id)

                    if not turn_task.done():
                        turn_task.cancel()

                    break

                try:
                    item = await asyncio.wait_for(q.get(), timeout=keepalive_seconds)
                except asyncio.TimeoutError:
                    # Keepalive to reduce risk of proxy/client idle timeouts.
                    yield _sse_pack({"keepalive": True}, event="keepalive")
                    continue

                if item is None:
                    break

                event_name, data = item
                yield _sse_pack(data, event=event_name)

        except asyncio.CancelledError:
            # If Starlette cancels the generator itself, also cancel the running turn.
            logger.info("SSE generator cancelled; cancelling turn (session_id=%s).", payload.session_id)

            with contextlib.suppress(Exception):
                from backend import runtime as runtime_mod

                logger.info("Calling runtime.cancel_active_turn session_id=%s", payload.session_id)
                await runtime_mod.cancel_active_turn(payload.session_id)
                logger.info("runtime.cancel_active_turn returned session_id=%s", payload.session_id)

            if not turn_task.done():
                turn_task.cancel()
            raise

        finally:
            if not turn_task.done():
                turn_task.cancel()
                with contextlib.suppress(Exception):
                    await turn_task

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
