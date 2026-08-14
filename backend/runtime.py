"""
Runtime layer for the FES Assistant.

Responsibilities
----------------
- Maintain a pool of long-lived McpClient instances keyed by session_id.
- For each agent turn, look up or create the McpClient for the given session_id
  and current config (tenant_config or migration_config).
- Replace clients when:
  - the session was idle too long, or
  - the config changed (tenant/migration).
- Delegate LLM + tools orchestration to backend.agent.llm_agent.call_llm_with_tools.
- Support per-turn progress publishing via a ContextVar-scoped async callback.
- Track the active turn per session_id and support cancellation (disconnect/refresh).

Notes
-----
- `messages` is the full UI conversation history (user + assistant).
- LLM/tool orchestration is handled by call_llm_with_tools.
- The MCP client is intentionally kept alive across turns for efficiency.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from backend.agent import llm_agent
from backend.agent._tracing import end_turn_trace, start_turn_trace
from backend.agent.llm_agent import call_llm_with_tools
from backend.agent.mcp_client import McpClient

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("backend.runtime")


def _setup_logger() -> None:
    """
    Configure a rotating file logger for the runtime layer.

    Notes
    -----
    - This function is idempotent and avoids duplicating handlers.
    """
    log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logger.setLevel(log_level)
    logger.propagate = False

    if any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
        logger.info(
            "backend.runtime logger already configured at level %s (env %s)",
            log_level_name,
            LOG_LEVEL_ENV_VAR,
        )
        return

    fh = TimedRotatingFileHandler(
        LOG_DIR / "backend_runtime.log",
        when="midnight",  # daily file; 7 dated backups = 7 days kept, older deleted
        backupCount=7,
        encoding="utf-8",
    )
    fh.setLevel(log_level)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(
        "backend.runtime logger initialized at level %s (env %s)",
        log_level_name,
        LOG_LEVEL_ENV_VAR,
    )


_setup_logger()

# -----------------------------------------------------------------------------
# Progress callback context (per agent turn)
# -----------------------------------------------------------------------------
ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]

_CURRENT_PROGRESS_CB: contextvars.ContextVar[Optional[ProgressCallback]] = contextvars.ContextVar(
    "fes_current_progress_cb",
    default=None,
)


async def publish_progress(event: Dict[str, Any]) -> None:
    """
    Best-effort publish of a progress event for the current agent turn.

    Parameters
    ----------
    event
        Progress event payload. Should be JSON-serializable.

    Notes
    -----
    This is ContextVar-scoped so:
    - concurrent sessions do not mix events, and
    - only the active request (for example, an SSE response) receives progress updates.
    """
    cb = _CURRENT_PROGRESS_CB.get()
    if cb is None:
        return

    try:
        await cb(event)
    except Exception as exc:
        # Never let progress publishing break the agent turn.
        logger.debug("Progress callback raised and was ignored: %s", exc)


@contextlib.asynccontextmanager
async def _progress_context(progress_cb: Optional[ProgressCallback]) -> AsyncIterator[None]:
    """
    Context manager that binds a per-turn progress callback for publish_progress().
    """
    token = _CURRENT_PROGRESS_CB.set(progress_cb)
    try:
        yield
    finally:
        _CURRENT_PROGRESS_CB.reset(token)


# -----------------------------------------------------------------------------
# Active turn tracking + cancellation (per UI session)
# -----------------------------------------------------------------------------
# session_id -> asyncio.Task[str]
_ACTIVE_TURNS: Dict[str, asyncio.Task] = {}
_ACTIVE_TURNS_LOCK = asyncio.Lock()


async def _best_effort_cancel_mcp(session_id: str) -> None:
    """
    Best-effort notify the MCP server to cancel any active streaming tool call
    for this UI session, using the cached McpClient (if present).

    Notes
    -----
    - Uses asyncio.shield() around the actual HTTP cancel so it still completes
      even if the caller is being cancelled (for example, SSE disconnect).
    """
    client: Optional[McpClient] = None

    async with SESSION_POOL_LOCK:
        entry = SESSION_POOL.get(session_id)
        if entry is not None:
            client = entry.mcp_client

    if client is None:
        logger.info("No McpClient found in SESSION_POOL for session_id=%s; skipping MCP cancel.", session_id)
        return

    cancel_fn = getattr(client, "cancel_session", None)
    if cancel_fn is None or not callable(cancel_fn):
        logger.warning(
            "McpClient for session_id=%s does not expose cancel_session(); skipping MCP cancel.",
            session_id,
        )
        return

    logger.info("Calling McpClient.cancel_session session_id=%s", session_id)

    try:
        # Critical: prevent the cancel HTTP request from being cancelled by the caller.
        await asyncio.shield(cancel_fn())
        logger.info("McpClient.cancel_session returned session_id=%s", session_id)
    except asyncio.CancelledError:
        # If our caller is cancelled, we still want to have attempted the shielded cancel.
        # Do not re-raise; this is best-effort.
        logger.info("Runtime cancellation occurred while requesting MCP cancel (session_id=%s).", session_id)
    except Exception:
        logger.exception("McpClient.cancel_session failed session_id=%s", session_id)


async def cancel_active_turn(session_id: str) -> None:
    """
    Cancel the currently active agent turn for a session (best-effort).

    Notes
    -----
    - Intended for disconnect/refresh cancellation from the API layer.
    - Performs two actions:
        1) Triggers MCP server-side cancel via the dedicated MCP cancel endpoint.
        2) Cancels the local asyncio.Task for the active turn.

    Critical behavior:
    - The MCP cancel is shielded so it can still reach the server even if
      the caller (SSE generator) is already being cancelled.
    """
    task: Optional[asyncio.Task] = None

    async with _ACTIVE_TURNS_LOCK:
        task = _ACTIVE_TURNS.get(session_id)

    # Always attempt MCP cancel first (even if the local task is already done).
    try:
        await _best_effort_cancel_mcp(session_id)
    except asyncio.CancelledError:
        # Best-effort: do not let caller cancellation stop the server-side cancel attempt.
        logger.info("cancel_active_turn cancelled while attempting MCP cancel (session_id=%s).", session_id)
    except Exception:
        logger.exception("Unexpected error attempting MCP cancel (session_id=%s).", session_id)

    if task is None or task.done():
        return

    logger.info("Cancelling active turn task for session %s.", session_id)
    task.cancel()

    # Best-effort wait to let cancellation propagate and resources close.
    try:
        await task
    except asyncio.CancelledError:
        # Do not re-raise; this is a cancellation helper.
        return
    except Exception:
        return


# -----------------------------------------------------------------------------
# MCP client pool (per UI session)
# -----------------------------------------------------------------------------
SESSION_IDLE_TIMEOUT = timedelta(hours=9)


@dataclass
class SessionEntry:
    """
    Cached MCP client entry for a UI session.

    Attributes
    ----------
    mcp_client
        Connected McpClient instance.
    tenant_config
        Chat-mode Sisense connection config (domain/token/ssl or similar).
    migration_config
        Migration-mode source/target config.
    last_used
        Timestamp of last successful reuse.
    """

    mcp_client: McpClient
    tenant_config: Optional[Dict[str, Any]]
    migration_config: Optional[Dict[str, Any]]
    last_used: datetime
    # Step 7: carried-over clarification state when a turn paused for a missing
    # required arg. {tool_id, missing_fields, filled_args, attempts} or None.
    pending_clarification: Optional[Dict[str, Any]] = None
    # Step 8: carried-over agentic-loop state when a turn paused mid-loop for a
    # mutation approval. {transcript, steps_executed, tool_id, arguments} or None.
    # On the approval turn the loop resumes from the paused step (Option A).
    pending_loop: Optional[Dict[str, Any]] = None


# session_id -> SessionEntry
SESSION_POOL: Dict[str, SessionEntry] = {}

# Global lock to protect SESSION_POOL structure (not the network calls)
SESSION_POOL_LOCK = asyncio.Lock()


def _now_utc() -> datetime:
    """
    Return a UTC timestamp.

    Notes
    -----
    Uses naive UTC timestamps consistently across this module.
    """
    return datetime.utcnow()


def _config_changed(
    current_tenant: Optional[Dict[str, Any]],
    current_migration: Optional[Dict[str, Any]],
    previous_entry: SessionEntry,
) -> bool:
    """
    Determine whether the session configuration changed since last turn.
    """
    return (current_tenant != previous_entry.tenant_config) or (current_migration != previous_entry.migration_config)


async def _close_client_safely(session_id: str, client: McpClient, reason: str) -> None:
    """
    Close an McpClient, swallowing errors but logging them.

    Parameters
    ----------
    session_id
        UI session id associated with the client.
    client
        McpClient to close.
    reason
        Reason for closure (for logs).
    """
    try:
        await client.close()
        logger.info("Closed McpClient for session %s (%s).", session_id, reason)
    except Exception:
        logger.exception("Error while closing McpClient for session %s (%s).", session_id, reason)


async def _get_or_create_mcp_client(
    session_id: str,
    tenant_config: Optional[Dict[str, Any]],
    migration_config: Optional[Dict[str, Any]],
) -> McpClient:
    """
    Return a connected McpClient for the given session_id.

    Parameters
    ----------
    session_id
        Logical UI session (for example, one Streamlit tab).
    tenant_config
        Chat-mode Sisense connection config.
    migration_config
        Migration-mode source/target config.

    Returns
    -------
    McpClient
        A connected client suitable for tool invocation.

    Notes
    -----
    - This keeps clients alive across turns for efficiency.
    - We do not hold the global lock while doing network operations (connect/close),
      to avoid blocking other sessions.
    """
    now = _now_utc()

    # Step 1: Decide whether we can reuse, or need to replace.
    old_client_to_close: Optional[McpClient] = None
    close_reason: Optional[str] = None

    async with SESSION_POOL_LOCK:
        entry = SESSION_POOL.get(session_id)

        if entry is not None:
            idle = now - entry.last_used

            if idle > SESSION_IDLE_TIMEOUT:
                old_client_to_close = entry.mcp_client
                close_reason = f"idle_timeout (idle={idle}, timeout={SESSION_IDLE_TIMEOUT})"
                SESSION_POOL.pop(session_id, None)
                entry = None
            elif _config_changed(tenant_config, migration_config, entry):
                old_client_to_close = entry.mcp_client
                close_reason = "config_changed"
                SESSION_POOL.pop(session_id, None)
                entry = None
            else:
                entry.last_used = now
                logger.debug("Reusing McpClient for session %s", session_id)
                return entry.mcp_client

    # Step 2: Close old client outside the lock (if needed).
    if old_client_to_close is not None and close_reason is not None:
        await _close_client_safely(session_id, old_client_to_close, close_reason)

    # Step 3: Create and connect a new client outside the lock.
    logger.info("Creating new McpClient for session %s", session_id)
    new_client = McpClient(tenant_config=tenant_config, migration_config=migration_config)
    await new_client.connect()
    logger.info("McpClient connected for session %s", session_id)

    # Step 4: Store it, but handle a possible race where another request created one first.
    async with SESSION_POOL_LOCK:
        existing = SESSION_POOL.get(session_id)
        if existing is not None:
            # Another request won the race; use the existing client and close ours.
            logger.info("Detected concurrent client creation for session %s. Using existing client.", session_id)

            # Release lock before closing (network call).
            chosen = existing.mcp_client
        else:
            SESSION_POOL[session_id] = SessionEntry(
                mcp_client=new_client,
                tenant_config=tenant_config,
                migration_config=migration_config,
                last_used=now,
            )
            return new_client

    # If we reach here, we need to close new_client and return chosen.
    await _close_client_safely(session_id, new_client, "race_lost")
    return chosen


# -----------------------------------------------------------------------------
# Core "one turn" runtime
# -----------------------------------------------------------------------------
async def _run_turn_once(
    session_id: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tenant_config: Optional[Dict[str, Any]],
    migration_config: Optional[Dict[str, Any]],
    approved_keys: Set[Tuple[str, str]],
    allow_summarization: Optional[bool],
    progress_cb: Optional[ProgressCallback],
) -> str:
    """
    Execute a single agent turn.

    Parameters
    ----------
    session_id
        Logical UI session.
    messages
        Full conversation history (UI-managed).
    tools
        Tool definitions available for this turn.
    tenant_config
        Chat mode Sisense connection.
    migration_config
        Migration mode source/target config.
    approved_keys
        Approved mutating tool calls for this turn.
    allow_summarization
        Per-turn override for tool-result summarization behavior in the LLM layer.
    progress_cb
        Optional async callback used to emit progress events during the turn.

    Returns
    -------
    str
        Final assistant reply.
    """
    logger.info("=== run_turn_once START (session_id=%s) ===", session_id)
    logger.debug(
        "Inputs: messages=%d, tools=%d, tenant_config=%s, migration_config=%s, approvals=%d, allow_summarization=%s",
        len(messages),
        len(tools),
        bool(tenant_config),
        bool(migration_config),
        len(approved_keys),
        allow_summarization,
    )

    mcp_client = await _get_or_create_mcp_client(
        session_id=session_id,
        tenant_config=tenant_config,
        migration_config=migration_config,
    )

    # Step 7: restore any clarification state saved on the previous turn so the
    # LLM layer can skip routing and resume the pinned tool.
    entry = SESSION_POOL.get(session_id)
    prior_clarification = entry.pending_clarification if entry else None
    if prior_clarification:
        logger.info(
            "Resuming clarification for session %s: tool=%s attempt=%s",
            session_id,
            prior_clarification.get("tool_id"),
            prior_clarification.get("attempts"),
        )

    # Step 8: restore paused loop state so an approval turn resumes mid-loop
    # instead of re-planning from scratch.
    prior_loop = entry.pending_loop if entry else None
    if prior_loop:
        logger.info(
            "Resuming paused agent loop for session %s: step=%s gated_tool=%s",
            session_id,
            prior_loop.get("steps_executed"),
            prior_loop.get("tool_id"),
        )

    # LangSmith: one root `agent_turn` run per turn; every LLM call and tool
    # execution attaches as a child via ContextVar. thread_id = the UI session,
    # so a whole conversation groups in the Threads view. Best-effort only.
    _last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    start_turn_trace(str(_last_user), session_id, "migration" if migration_config else "chat")
    try:
        async with _progress_context(progress_cb):
            reply = await call_llm_with_tools(
                messages=messages,
                tools=tools,
                mcp_client=mcp_client,
                approved_mutations=approved_keys,
                allow_summarization=allow_summarization,
                pending_clarification=prior_clarification,
                pending_loop=prior_loop,
            )

        # Persist or clear pause state for the next turn. The LLM layer sets these
        # globals only when it paused (to ask / for approval); any other outcome
        # (executed, gave up, topic change) leaves them None → cleared.
        if entry is not None:
            entry.pending_clarification = getattr(llm_agent, "LAST_PENDING_CLARIFICATION", None)
            entry.pending_loop = getattr(llm_agent, "LAST_PENDING_LOOP", None)

        logger.info("call_llm_with_tools completed successfully for session %s.", session_id)
        logger.debug("Agent reply (truncated): %s", reply[:500] if isinstance(reply, str) else repr(reply))
        _steps = getattr(llm_agent, "LAST_STEP_RESULTS", None) or []
        end_turn_trace(
            reply=reply if isinstance(reply, str) else None,
            outcome="ok",
            steps=len(_steps),
            tools=[str(st.get("tool_id") or "?") for st in _steps],
        )
        return reply

    except asyncio.CancelledError:
        logger.info("Agent turn cancelled (session_id=%s).", session_id)
        end_turn_trace(outcome="cancelled")
        raise

    except Exception as exc:
        logger.exception("Error during agent turn (session_id=%s): %s", session_id, exc)
        end_turn_trace(outcome=f"error: {str(exc)[:200]}")
        raise

    finally:
        # Idempotent — a no-op when one of the paths above already closed it.
        end_turn_trace(outcome="unknown")
        logger.info("=== run_turn_once END (session_id=%s) ===", session_id)


async def run_turn_once(
    session_id: str,
    messages: List[Dict[str, Any]],
    user_input: str,
    tools: List[Dict[str, Any]],
    tenant_config: Optional[Dict[str, Any]] = None,
    approved_keys: Optional[Set[Tuple[str, str]]] = None,
    migration_config: Optional[Dict[str, Any]] = None,
    allow_summarization: Optional[bool] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """
    Public API: run one agent turn.

    This is the entrypoint the API layer calls for each user turn.

    Parameters
    ----------
    session_id
        Logical UI session, for example one Streamlit tab.
    messages
        Full conversation history (UI-managed; includes prior user/assistant turns).
    user_input
        Latest user message. This is kept for API symmetry and future use.
        The runtime currently uses the full `messages` history instead.
    tools
        OpenAI-style tool definitions available for this turn.
    tenant_config
        Chat mode Sisense connection config.
    approved_keys
        Approved mutating tool calls for this turn.
    migration_config
        Migration mode source/target config.
    allow_summarization
        Per-turn override for whether tool results can be sent to the LLM.
        If None, the LLM layer falls back to its global configuration.
    progress_cb
        Optional async callback used to emit progress events during the turn
        (only relevant when the API responds via SSE).

    Returns
    -------
    str
        Final assistant reply for the turn.
    """
    _ = user_input  # reserved for future use

    if approved_keys is None:
        approved_keys = set()

    turn_task: asyncio.Task = asyncio.create_task(
        _run_turn_once(
            session_id=session_id,
            messages=messages,
            tools=tools,
            tenant_config=tenant_config,
            migration_config=migration_config,
            approved_keys=approved_keys,
            allow_summarization=allow_summarization,
            progress_cb=progress_cb,
        )
    )

    prev_task: Optional[asyncio.Task] = None
    async with _ACTIVE_TURNS_LOCK:
        prev_task = _ACTIVE_TURNS.get(session_id)
        _ACTIVE_TURNS[session_id] = turn_task

    # Best-effort: cancel any previous turn still running for this same session_id.
    if prev_task is not None and prev_task is not turn_task and not prev_task.done():
        logger.info("Cancelling previous active turn for session %s.", session_id)

        # Critical: shield MCP cancel so it reaches the server even if this turn is under cancellation pressure.
        try:
            await _best_effort_cancel_mcp(session_id)
        except asyncio.CancelledError:
            logger.info("run_turn_once cancelled while attempting MCP cancel (session_id=%s).", session_id)
        except Exception:
            logger.exception("Unexpected error attempting MCP cancel (session_id=%s).", session_id)

        prev_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await prev_task

    try:
        return await turn_task
    finally:
        async with _ACTIVE_TURNS_LOCK:
            current = _ACTIVE_TURNS.get(session_id)
            if current is turn_task:
                _ACTIVE_TURNS.pop(session_id, None)
