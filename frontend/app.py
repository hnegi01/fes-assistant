# app.py
#
# Streamlit UI for the FES Assistant.
# - Connects to the backend FastAPI (/agent/turn) for each turn.
# - Manages per-tab session_id, Sisense tenant configs, and mutation approvals.
# - Uses MCP + PySisense tools exposed by the backend/MCP server.
#
# Notes:
# - Keep the ROOT_DIR + .env loading BEFORE other imports. Some imports read env vars at import time.

import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Bootstrap: sys.path + env loading FIRST (before other imports)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

# -----------------------------------------------------------------------------
# Standard imports (safe after env loading)
# -----------------------------------------------------------------------------
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
log_level = getattr(logging, log_level_name, logging.INFO)

LOG_DIR = ROOT_DIR / "logs"
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except FileExistsError:
    pass

logger = logging.getLogger("app")
logger.setLevel(log_level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,  # keep 5 old files
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.info("App logger initialized at level %s (env var %s)", log_level_name, LOG_LEVEL_ENV_VAR)

# -----------------------------------------------------------------------------
# Summarization policy (UI permission)
# -----------------------------------------------------------------------------
ALLOW_SUMMARIZATION_TOGGLE_ENV_VAR = "FES_ALLOW_SUMMARIZATION_TOGGLE"
raw_toggle = os.getenv(ALLOW_SUMMARIZATION_TOGGLE_ENV_VAR, "true")
ALLOW_SUMMARIZATION_TOGGLE = raw_toggle.lower() == "true"

logger.debug(
    "Summarization toggle allowed: %s (env %s=%r)",
    ALLOW_SUMMARIZATION_TOGGLE,
    ALLOW_SUMMARIZATION_TOGGLE_ENV_VAR,
    raw_toggle,
)

# -----------------------------------------------------------------------------
# Backend API URL
# -----------------------------------------------------------------------------
BACKEND_URL = os.getenv("FES_BACKEND_URL", "http://localhost:8001").rstrip("/")
logger.debug("Using BACKEND_URL=%s", BACKEND_URL)

# -----------------------------------------------------------------------------
# UI session idle timeout (hours)
# -----------------------------------------------------------------------------
UI_IDLE_TIMEOUT_HOURS = float(os.getenv("FES_UI_IDLE_TIMEOUT_HOURS", "9"))


def check_ui_session_timeout() -> None:
    """
    Enforce a simple idle timeout for the Streamlit session.

    If the last activity was more than UI_IDLE_TIMEOUT_HOURS ago,
    clear session_state so the user has to reconnect.
    """
    now = datetime.utcnow()
    last_key = "last_activity_utc"
    last_raw = st.session_state.get(last_key)
    expired = False

    if last_raw:
        last_dt = None
        try:
            if isinstance(last_raw, str):
                last_dt = datetime.fromisoformat(last_raw)
            elif isinstance(last_raw, datetime):
                last_dt = last_raw
        except Exception:
            last_dt = None

        if last_dt and (now - last_dt) > timedelta(hours=UI_IDLE_TIMEOUT_HOURS):
            expired = True

    if expired:
        logger.info(
            "UI session idle for more than %s hours; resetting Streamlit session_state.",
            UI_IDLE_TIMEOUT_HOURS,
        )
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["session_expired"] = True

    st.session_state[last_key] = now.isoformat()


# -----------------------------------------------------------------------------
# SSE parsing helper
# -----------------------------------------------------------------------------
def _iter_sse_events(resp: requests.Response) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Parse a text/event-stream response into (event_name, data_dict).

    Expected format:
      event: <name>
      data: <json>
      <blank line>
    """
    event_name: str = "message"
    data_lines: List[str] = []

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue

        line = raw_line.rstrip("\r")

        # End of one SSE frame
        if line == "":
            if data_lines:
                data_str = "\n".join(data_lines)
                data_lines = []
                try:
                    obj = json.loads(data_str)
                    if isinstance(obj, dict):
                        yield event_name, obj
                    else:
                        yield event_name, {"value": obj}
                except Exception:
                    yield event_name, {"ok": False, "error": "Failed to parse SSE JSON payload."}
            event_name = "message"
            continue

        # Comments / keep-alives
        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event_name = line[len("event:") :].strip() or "message"
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue

    # Flush if stream ends without trailing blank line
    if data_lines:
        data_str = "\n".join(data_lines)
        try:
            obj = json.loads(data_str)
            if isinstance(obj, dict):
                yield event_name, obj
            else:
                yield event_name, {"value": obj}
        except Exception:
            yield event_name, {"ok": False, "error": "Failed to parse SSE JSON payload (final chunk)."}


# -----------------------------------------------------------------------------
# Progress UX helpers
# -----------------------------------------------------------------------------
_LAST_RUN_LOG_STATE_KEY = "_fes_last_run_log"

# Migration turn threading state keys
_MIG_TURN_IN_PROGRESS_KEY = "_mig_turn_in_progress"
_MIG_TURN_CTX_KEY = "_mig_turn_ctx"
_MIG_PENDING_TURN_META_KEY = "_mig_pending_turn_meta"


def _write_run_log(run_log: Any, _run_log_out: Optional[Dict[str, Any]] = None) -> None:
    """Write run_log to the optional out-dict and, when on the main Streamlit thread, to session state."""
    if _run_log_out is not None:
        _run_log_out["run_log"] = run_log
    try:
        st.session_state[_LAST_RUN_LOG_STATE_KEY] = run_log
    except Exception:
        pass  # Called from background thread — session state not accessible


def _render_agent_progress(
    placeholder: Any,
    completed_steps: List[Dict[str, Any]],
    status_line: Optional[str],
) -> None:
    """Render the agentic-loop progress block: collapsed checklist of completed
    steps + a live status line for the current phase."""
    if placeholder is None:
        return
    lines: List[str] = []
    for s in completed_steps:
        mark = "✅" if s.get("ok") else "⚠️"
        lines.append(f"- {mark} Step {s.get('step')}: `{s.get('tool_id', '?')}`")
    md = ""
    if lines:
        md += "\n".join(lines) + "\n\n"
    if status_line:
        md += f"*{status_line}*"
    if md:
        placeholder.markdown(md)


def _agent_progress_status_line(data: Dict[str, Any]) -> Optional[str]:
    """Map an agent_progress event to a human status line (None for 'completed')."""
    phase = data.get("phase")
    step = data.get("step")
    max_steps = data.get("max_steps")
    tool_id = data.get("tool_id")
    if phase == "deciding":
        return "🤔 Checking progress against your request…"
    if phase == "verifying":
        return "🔎 Double-checking the result covers your whole request…"
    if phase == "planning":
        return f"🧭 Planning step {step}/{max_steps}…"
    if phase == "executing":
        return f"⏳ Step {step}/{max_steps}: running `{tool_id}`…"
    return None


def _cancel_backend_turn(sid: str) -> None:
    """POST /agent/cancel for the given session id; errors are logged and swallowed."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/agent/cancel",
            json={"session_id": sid},
            timeout=10,
        )
        r.raise_for_status()
        logger.info("Cancel request sent for session %s", sid)
    except Exception as exc:
        logger.warning("Cancel request failed for session %s: %s", sid, exc)


def _extract_progress_payload(data: Any) -> Any:
    """
    Best-effort extraction of the 'useful' payload for progress rendering.

    Example MCP notification envelope:
      {"source":"mcp","type":"notification","method":"notifications/message","params":{"level":"info","data":{...}}}

    We want params.data when present; otherwise return original.
    """
    if not isinstance(data, dict):
        return data

    params = data.get("params")
    if isinstance(params, dict):
        inner = params.get("data")
        if isinstance(inner, dict):
            return inner

    inner = data.get("data")
    if isinstance(inner, dict):
        return inner

    return data


def _format_progress_line(payload: Any) -> str:
    """
    Render one progress payload as a single human-readable line.

    We keep it generic: prefer 'message', then add a few optional hints if present.
    """
    if not isinstance(payload, dict):
        return str(payload)

    msg = payload.get("message")
    if not isinstance(msg, str) or not msg.strip():
        msg = None

    step = payload.get("step")
    if not isinstance(step, str) or not step.strip():
        step = None

    parts: List[str] = []
    if step:
        parts.append(f"[{step}]")
    if msg:
        parts.append(msg)
    else:
        t = payload.get("type")
        if isinstance(t, str) and t.strip():
            parts.append(t)
        else:
            parts.append("update")

    hints: List[str] = []
    for k in [
        "batch_number",
        "batches_total",
        "processed_so_far",
        "total_count",
        "succeeded_total",
        "failed_total",
        "skipped_total",
        "pages_fetched",
    ]:
        v = payload.get(k)
        if isinstance(v, (int, float, str)) and str(v) != "":
            hints.append(f"{k}={v}")

    if hints:
        parts.append(f"({', '.join(hints)})")

    return " ".join(parts).strip()


def render_run_log(run_log: Optional[Dict[str, Any]]) -> None:
    """
    Render a run log in a collapsed expander.

    run_log shape:
      {"started_at": "...", "events": [{"event": "...", "payload": {...}}, ...]}
    """
    if not run_log or not isinstance(run_log, dict):
        return

    events = run_log.get("events") or []
    if not isinstance(events, list) or not events:
        return

    started_at = run_log.get("started_at")
    header = "Run log"
    if isinstance(started_at, str) and started_at.strip():
        header = f"Run log ({started_at})"

    with st.expander(header, expanded=False):
        max_lines = 200
        tail = events[-max_lines:]
        lines: List[str] = []
        for item in tail:
            if isinstance(item, dict):
                payload = item.get("payload")
                lines.append(_format_progress_line(payload))
            else:
                lines.append(str(item))

        if len(events) > max_lines:
            st.caption(f"Showing last {max_lines} updates out of {len(events)}.")

        st.markdown("\n".join([f"- {ln}" for ln in lines]))


def _launch_migration_turn(
    meta: Dict[str, Any],
    messages: List[Dict[str, Any]],
    user_input: str,
    tenant_config: Optional[Dict],
    approved_keys: Optional[Any],
    migration_config: Optional[Dict],
    session_id: str,
    allow_summarization: bool,
    mode: str,
) -> None:
    """
    Start call_backend_turn in a background thread for migration mode.

    Writes results into st.session_state[_MIG_TURN_CTX_KEY] so the
    polling block can pick them up on subsequent Streamlit reruns.
    meta keys used by _process_mig_turn_result:
      clear_pending (bool) — whether to clear MIG_PENDING_KEY on completion.
    """
    ctx: Dict[str, Any] = {
        "done": False,
        "reply": None,
        "tool_result": None,
        "error_str": None,
        "progress_lines": [],
        "run_log": None,
    }
    st.session_state[_MIG_TURN_CTX_KEY] = ctx
    st.session_state[_MIG_PENDING_TURN_META_KEY] = meta
    st.session_state[_MIG_TURN_IN_PROGRESS_KEY] = True

    def _progress_cb(line: str) -> None:
        ctx["progress_lines"].append(line)

    call_kwargs: Dict[str, Any] = {
        "messages": messages,
        "user_input": user_input,
        "tenant_config": tenant_config,
        "approved_keys": approved_keys,
        "migration_config": migration_config,
        "session_id": session_id,
        "allow_summarization": allow_summarization,
        "mode": mode,
        "progress_placeholder": None,
        "progress_callback": _progress_cb,
        "_run_log_out": ctx,
    }

    def _run() -> None:
        try:
            reply, tool_result = call_backend_turn(**call_kwargs)
            ctx["reply"] = reply
            ctx["tool_result"] = tool_result
        except Exception as exc:
            logger.exception("Background migration turn failed: %s", exc)
            ctx["error_str"] = str(exc)
        finally:
            ctx["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info(
        "Launched migration thread for session %s (kind=%s)",
        session_id,
        meta.get("kind", "unknown"),
    )


def call_backend_turn(
    messages,
    user_input,
    tenant_config=None,
    approved_keys=None,
    migration_config=None,
    session_id=None,
    allow_summarization=None,
    mode=None,
    progress_placeholder: Optional[Any] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    _run_log_out: Optional[Dict[str, Any]] = None,
):
    """
    Thin HTTP client for the backend /agent/turn API.

    Strategy (both modes request SSE):
    - Migration mode: render SDK progress lines + store run_log.
    - Chat mode: render agentic-loop step progress (agent_progress events) so
      multi-step turns show live "planning / running tool X" status instead of
      a silent spinner.
    """
    payload = {
        "messages": messages,
        "user_input": user_input,
        "tenant_config": tenant_config,
        "migration_config": migration_config,
        "approved_keys": list(approved_keys) if approved_keys else [],
        "session_id": session_id,
        "allow_summarization": allow_summarization,
        "mode": mode,
    }

    logger.info("Calling backend /agent/turn (mode=%s, session_id=%s)", mode, session_id)

    is_migration = mode == BACKEND_MODE_MIGRATION

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
    }

    # Timeouts: keep connect timeout reasonable; allow long reads for migration.
    timeout = (30, 1800)

    resp = requests.post(
        f"{BACKEND_URL}/agent/turn",
        json=payload,
        headers=headers,
        timeout=timeout,
        stream=True,
    )
    resp.raise_for_status()

    # SSE expected (but backend might still return JSON).
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if "text/event-stream" in ctype:
        final_reply: Optional[str] = None
        final_tool_result: Optional[Dict[str, Any]] = None

        run_log: Dict[str, Any] = {
            "started_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "events": [],
        }

        progress_lines: List[str] = []
        agent_steps: List[Dict[str, Any]] = []

        for event, data in _iter_sse_events(resp):
            if event == "keepalive":
                continue

            cleaned_payload = _extract_progress_payload(data)
            run_log["events"].append({"event": event, "payload": cleaned_payload})

            # Agentic-loop step progress (Step 8) — dedicated checklist rendering.
            if event == "progress" and data.get("type") == "agent_progress":
                if data.get("phase") == "completed":
                    agent_steps.append({"step": data.get("step"), "tool_id": data.get("tool_id"), "ok": data.get("ok")})
                    _render_agent_progress(progress_placeholder, agent_steps, None)
                else:
                    _render_agent_progress(progress_placeholder, agent_steps, _agent_progress_status_line(data))
                if progress_callback is not None:
                    line = _agent_progress_status_line(data)
                    if line:
                        progress_callback(line)
                continue

            if event == "status":
                phase = data.get("phase")
                if isinstance(phase, str) and phase.strip():
                    new_line = f"Status: {phase}"
                    progress_lines.append(new_line)
                    if progress_callback is not None:
                        progress_callback(new_line)
                continue

            if event == "progress":
                msg = data.get("message") or data.get("detail")
                if isinstance(msg, str) and msg.strip():
                    new_line = msg.strip()
                else:
                    new_line = _format_progress_line(cleaned_payload)
                progress_lines.append(new_line)
                if progress_callback is not None:
                    progress_callback(new_line)

            elif event == "result":
                final_reply = data.get("reply", "")
                final_tool_result = data.get("tool_result")

            elif event == "error":
                err = data.get("error") or "Unknown error"
                _write_run_log(run_log, _run_log_out)
                raise RuntimeError(err)

            else:
                new_line = _format_progress_line(cleaned_payload)
                progress_lines.append(new_line)
                if progress_callback is not None:
                    progress_callback(new_line)

            # Migration-style rolling log; chat placeholders render agent_progress only.
            if is_migration and progress_placeholder is not None and progress_lines:
                tail = progress_lines[-20:]
                progress_placeholder.markdown("**Progress**\n\n" + "\n".join([f"- {x}" for x in tail]))

        _write_run_log(run_log, _run_log_out)

        if final_reply is None and final_tool_result is None:
            raise RuntimeError("SSE stream ended without a final result.")

        return final_reply or "", final_tool_result

    # Fallback: backend returned JSON even though we asked for SSE
    _write_run_log(None, _run_log_out)
    data = resp.json()
    reply = data.get("reply", "")
    tool_result = data.get("tool_result")
    return reply, tool_result


# -----------------------------------------------------------------------------
# Tool result rendering
# -----------------------------------------------------------------------------
def _approval_key(tool_id: str, args: Dict[str, Any]) -> Tuple[str, str]:
    return tool_id, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def fetch_tools_from_backend():
    """
    Fetch OpenAI-style tools and registry metadata from the backend.
    """
    url = f"{BACKEND_URL}/tools"
    logger.debug("Fetching tools from backend: %s", url)

    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        logger.exception("Request to /tools failed: %s", e)
        st.error(
            "Could not reach the backend /tools endpoint. Check that the backend is running and BACKEND_URL is correct."
        )
        st.stop()

    if not resp.ok:
        logger.error("Backend /tools returned %s: %s", resp.status_code, resp.text[:500])
        st.error(f"Backend /tools failed with status {resp.status_code}. See backend logs for details.")
        st.stop()

    try:
        data = resp.json()
    except ValueError as e:
        logger.exception("Failed to decode JSON from /tools: %s", e)
        st.error("Backend /tools did not return valid JSON. See backend logs.")
        st.stop()

    tools = data.get("tools") or []
    registry = data.get("registry") or {}

    if not isinstance(tools, list):
        logger.error("Unexpected tools payload type from /tools: %r", type(tools))
        st.error("Backend /tools returned tools in an unexpected format.")
        st.stop()

    if not isinstance(registry, dict):
        logger.error("Unexpected registry payload type from /tools: %r", type(registry))
        st.error("Backend /tools returned registry in an unexpected format.")
        st.stop()

    logger.debug("Loaded %d tools and %d registry entries from backend", len(tools), len(registry))
    return tools, registry


def render_tool_result(tr: dict):
    if not tr or not isinstance(tr, dict):
        return

    tool_name = tr.get("tool_id", "")
    if tool_name:
        st.caption(f"Tool called: `{tool_name}`")

    if tr.get("ok", True):
        data = tr.get("result")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)

            # Make mixed-type columns safe for Arrow / Streamlit
            for col in df.columns:
                try:
                    if df[col].map(type).nunique() > 1:
                        df[col] = df[col].astype(str)
                except Exception:
                    df[col] = df[col].astype(str)

            st.markdown("**Result**")
            # Change suggested: use_container_width is the stable option across Streamlit versions.
            st.dataframe(df, width="stretch")
        else:
            st.markdown("**Result (JSON)**")
            st.code(json.dumps(data, indent=2), language="json")
    else:
        if not tr.get("pending_confirmation"):
            st.markdown("**Tool error**")
            st.code(json.dumps(tr, indent=2), language="json")


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FES Assistant", page_icon="images/sisense.png")

check_ui_session_timeout()

st.markdown(
    """
    <style>
    button[data-testid="stBaseButton-headerNoPadding"] {
        display: none !important;
    }
    [data-testid="stSidebar"] {
        min-width: 360px;
        max-width: 360px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="font-size: 2.5rem; font-weight: 700; line-height: 1.15; margin: 0 0 0.25rem 0;">
        FES Assistant: Agentic Sisense Co-Pilot
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='font-size: 0.95rem; opacity: 0.85; margin-top: 0;'>Powered by the FES Agent (MCP + PySisense)</p>",
    unsafe_allow_html=True,
)

if st.session_state.get("session_expired"):
    st.info(
        "Your session was idle for a long time, so it was reset. Please reconnect your Sisense deployment to continue."
    )
    del st.session_state["session_expired"]


# -----------------------------------------------------------------------------
# Per-session id (one per browser tab)
# -----------------------------------------------------------------------------
SESSION_ID_KEY = "fes_session_id"
if SESSION_ID_KEY not in st.session_state:
    st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())
    logger.info("Initialized new UI session: %s", st.session_state[SESSION_ID_KEY])

session_id = st.session_state[SESSION_ID_KEY]


# -----------------------------------------------------------------------------
# Global Privacy & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="font-weight: 700; font-size: 1.1rem; margin-top: 10px;">
            Data sharing &amp; agent capability
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "allow_summarization" not in st.session_state:
        st.session_state["allow_summarization"] = False

    _summ_help = (
        "Controls whether tool **results (your Sisense data)** are sent to the LLM. "
        "This is not only a privacy switch — it sets **how capable the assistant is**.\n\n"
        "**On** — the assistant can:\n"
        "- write natural-language answers, not just raw tables\n"
        "- chain steps that depend on each other (e.g. find a user's role, then "
        "list everyone with that role)\n"
        "- independently double-check that it actually finished your whole request\n\n"
        "…but result data leaves your Sisense instance and goes to the LLM "
        "provider — enable only if you trust it with that data.\n\n"
        "**Off** — data stays private: the LLM only ever sees *which* operations "
        "ran and whether they succeeded, never the data itself. The assistant still "
        "handles independent multi-step requests, but it **can't** pass data between "
        "steps, **can't** verify the goal was met, and returns **raw results** "
        "instead of a written summary."
    )

    if ALLOW_SUMMARIZATION_TOGGLE:
        st.checkbox(
            "Send results to the LLM — fuller answers & smarter steps",
            key="allow_summarization",
            help=_summ_help,
        )
        if st.session_state["allow_summarization"]:
            st.caption(
                "🟢 **Full capability** — adaptive multi-step, goal double-check, "
                "written answers. Result data is sent to the LLM."
            )
        else:
            st.caption(
                "🔒 **Private** — data stays in Sisense. Independent multi-step still "
                "works; no data-chaining between steps, no goal check, raw results only."
            )
    else:
        st.session_state["allow_summarization"] = False
        st.checkbox(
            "Send results to the LLM (disabled by admin)",
            key="allow_summarization",
            disabled=True,
            help=_summ_help + "\n\n_Disabled in the server configuration._",
        )
        st.caption(
            "🔒 Disabled by the administrator — result data is never sent to the LLM. "
            "The assistant runs in private mode: independent multi-step only, no "
            "data-chaining, goal check, or written summaries."
        )


# -----------------------------------------------------------------------------
# Mode selection: Chat vs Migration
# -----------------------------------------------------------------------------
MODE_CHAT = "Chat with deployment"
MODE_MIGRATION = "Migrate between deployments"

BACKEND_MODE_CHAT = "chat"
BACKEND_MODE_MIGRATION = "migration"

mode = st.radio(
    "Mode",
    [MODE_CHAT, MODE_MIGRATION],
    horizontal=True,
    label_visibility="collapsed",
)

logger.debug("Current mode: %s", mode)


# -----------------------------------------------------------------------------
# Load tools once (for display/metadata)
# -----------------------------------------------------------------------------
if "tools" not in st.session_state or "tool_registry" not in st.session_state:
    tools, registry = fetch_tools_from_backend()
    st.session_state.tools = tools
    st.session_state.tool_registry = registry

    logger.debug(
        "Loaded TOOL_REGISTRY with %d tools: %s",
        len(registry),
        list(registry.keys()),
    )

    logger.debug(
        "Tools fetched from backend (for display/metadata): %d tools",
        len(st.session_state.tools),
    )

if "chat_tools" not in st.session_state or "migration_tools" not in st.session_state:
    registry = st.session_state.tool_registry
    all_tools = st.session_state.tools
    tools_by_name = {t["function"]["name"]: t for t in all_tools}

    chat_tool_names: List[str] = []
    migration_tool_names: List[str] = []

    for tid, row in registry.items():
        module = row.get("module")
        if module == "migration":
            migration_tool_names.append(tid)
        else:
            chat_tool_names.append(tid)

    st.session_state.chat_tools = [tools_by_name[name] for name in chat_tool_names if name in tools_by_name]
    st.session_state.migration_tools = [tools_by_name[name] for name in migration_tool_names if name in tools_by_name]

    logger.debug(
        "Per-mode tools (for display): chat_tools=%d, migration_tools=%d",
        len(st.session_state.chat_tools),
        len(st.session_state.migration_tools),
    )

all_tools = st.session_state.tools
chat_tools = st.session_state.chat_tools
migration_tools = st.session_state.migration_tools


# =============================================================================
# MODE 1: CHAT WITH DEPLOYMENT
# - Chat mode does NOT show Progress/Run Log (no SDK progress events in chat tools).
# - Still supports approvals for mutating tools.
# - Also fixes the "hide previous user request" behavior (applies hide index when rendering).
# =============================================================================
if mode == MODE_CHAT:
    CHAT_TENANT_KEY = "chat_tenant_config"
    CHAT_MESSAGES_KEY = "chat_messages"
    CHAT_LAST_USER_IDX_KEY = "chat_last_user_idx"
    CHAT_HIDE_USER_IDX_KEY = "chat_hide_user_idx"
    CHAT_PENDING_KEY = "chat_pending_confirmation"
    CHAT_APPROVED_KEY = "chat_approved_mutations"

    if CHAT_TENANT_KEY not in st.session_state:
        st.session_state[CHAT_TENANT_KEY] = None

    def render_chat_tenant_form():
        st.subheader("Connect your Sisense deployment")

        with st.form("chat_tenant_form"):
            domain = st.text_input("Sisense domain", placeholder="https://your-domain.sisense.com")
            token = st.text_input("API token", type="password")
            ssl = st.checkbox("Verify SSL", value=True)
            submitted = st.form_submit_button("Connect")

        if submitted:
            if not domain or not token:
                st.error("Domain and token are required.")
                return

            st.session_state[CHAT_TENANT_KEY] = {
                "domain": domain.strip(),
                "token": token.strip(),
                "ssl": ssl,
            }
            logger.info("[CHAT] Tenant configured for domain=%s, ssl=%s", domain.strip(), ssl)
            st.success("Connected. You can now chat with your Sisense deployment.")
            st.rerun()

    if st.session_state[CHAT_TENANT_KEY] is None:
        with st.sidebar:
            st.subheader("Status:")
            st.write(f"Chat tools available to LLM: **{len(chat_tools)}**")
            st.markdown("**Mode:** Chat with deployment")
            st.markdown("---")
            st.caption(
                "Connect your Sisense deployment to start chatting. "
                "Switch to 'Migrate between deployments' mode to migrate assets between environments."
            )
        render_chat_tenant_form()
        st.stop()

    chat_tenant_config = st.session_state[CHAT_TENANT_KEY]

    if CHAT_MESSAGES_KEY not in st.session_state:
        st.session_state[CHAT_MESSAGES_KEY] = [
            {"role": "assistant", "content": "Hi! Ask me about your Sisense deployment."},
        ]
        logger.debug("[CHAT] Chat history initialized with greeting only (system prompt handled in backend).")

    if CHAT_LAST_USER_IDX_KEY not in st.session_state:
        st.session_state[CHAT_LAST_USER_IDX_KEY] = None
    if CHAT_HIDE_USER_IDX_KEY not in st.session_state:
        st.session_state[CHAT_HIDE_USER_IDX_KEY] = None
    if CHAT_PENDING_KEY not in st.session_state:
        st.session_state[CHAT_PENDING_KEY] = None
    if CHAT_APPROVED_KEY not in st.session_state:
        st.session_state[CHAT_APPROVED_KEY] = set()

    with st.sidebar:
        st.subheader("Status:")
        st.write(f"Chat tools available to LLM: **{len(chat_tools)}**")
        st.markdown("**Mode:** Chat with deployment")

        st.markdown("**Connected tenant**")
        st.write(f"Domain: `{chat_tenant_config.get('domain', '')}`")
        st.write(f"SSL verification: `{chat_tenant_config.get('ssl', True)}`")

        if st.button("Disconnect tenant"):
            logger.info("[CHAT] Disconnecting tenant.")
            st.session_state[CHAT_TENANT_KEY] = None
            for key in [
                CHAT_MESSAGES_KEY,
                CHAT_LAST_USER_IDX_KEY,
                CHAT_HIDE_USER_IDX_KEY,
                CHAT_PENDING_KEY,
                CHAT_APPROVED_KEY,
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        with st.expander("Examples", expanded=False):
            st.markdown(
                """
- Show me all users
- List all dashboards
- Show all data models
- Show all tables and columns in 'ecommerce_db' datamodel
- Add a table called "top_customers" in datamodel "ecommerce_db"
- Create an elasticube called "nyctaxi_ec" using connection "pysense_databricks",
  database "samples", schema "nyctaxi". Add tables trips and vendors.
"""
            )

        st.markdown("---")
        st.caption(
            "Agentic assistant for Sisense, powered by an LLM and MCP. It plans, "
            "runs PySisense tools one step at a time, and verifies its work against "
            "your request — chaining multiple steps when needed."
        )

    # Render chat history (with hide support for approved mutation reruns)
    for i, msg in enumerate(st.session_state[CHAT_MESSAGES_KEY]):
        if msg.get("role") not in ("user", "assistant"):
            continue

        # Apply the hide index in chat mode
        if st.session_state[CHAT_HIDE_USER_IDX_KEY] is not None and i == st.session_state[CHAT_HIDE_USER_IDX_KEY]:
            continue

        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                tr = msg.get("tool_result")
                if tr:
                    render_tool_result(tr)

                # Chat mode: do NOT render run log (no progress emitted for these tools)

            st.markdown(msg.get("content", ""))

    # Clear the one-shot hide flag after rendering once
    if st.session_state[CHAT_HIDE_USER_IDX_KEY] is not None:
        st.session_state[CHAT_HIDE_USER_IDX_KEY] = None

    # Pending mutation approval UX (Chat)
    pending = st.session_state[CHAT_PENDING_KEY]
    if pending and isinstance(pending, dict):
        st.info(
            pending.get("reason")
            or "This action requires approval before it can make changes to your Sisense deployment."
        )
        with st.expander("View operation details", expanded=True):
            st.markdown("**Tool:** `{}`".format(pending.get("tool_id", "")))
            st.code(json.dumps(pending.get("arguments", {}), indent=2), language="json")

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Approve", type="primary"):
                key = _approval_key(pending["tool_id"], pending.get("arguments", {}))
                st.session_state[CHAT_APPROVED_KEY].add(key)

                _agent_ph = st.empty()
                with st.spinner("Running approved action..."):
                    try:
                        reply, tr = call_backend_turn(
                            messages=st.session_state[CHAT_MESSAGES_KEY],
                            user_input="",
                            tenant_config=chat_tenant_config,
                            approved_keys=st.session_state[CHAT_APPROVED_KEY],
                            migration_config=None,
                            session_id=session_id,
                            allow_summarization=st.session_state["allow_summarization"],
                            mode=BACKEND_MODE_CHAT,
                            progress_placeholder=_agent_ph,
                        )
                    except Exception as e:
                        logger.exception("Agent run after approval failed: %s", e)
                        st.error("The approved action failed.")
                        st.exception(e)
                        st.session_state[CHAT_PENDING_KEY] = None
                        st.rerun()
                _agent_ph.empty()

                # Ensure run log is not shown/stored for chat
                st.session_state[_LAST_RUN_LOG_STATE_KEY] = None

                st.session_state[CHAT_MESSAGES_KEY].append(
                    {"role": "assistant", "content": reply, "tool_result": tr, "run_log": None}
                )

                # Hide the previous user request on next render
                st.session_state[CHAT_HIDE_USER_IDX_KEY] = st.session_state[CHAT_LAST_USER_IDX_KEY]
                st.session_state[CHAT_PENDING_KEY] = None
                st.rerun()

        with cols[1]:
            if st.button("Cancel"):
                st.session_state[CHAT_PENDING_KEY] = None
                st.session_state[CHAT_MESSAGES_KEY].append({"role": "assistant", "content": "Action cancelled."})
                st.rerun()

    # Chat input (Chat mode)
    user_input = (st.chat_input("Ask something about Sisense...") or "").strip() or None

    if user_input:
        logger.debug("[CHAT] User question: %s", user_input)

        st.session_state[CHAT_LAST_USER_IDX_KEY] = len(st.session_state[CHAT_MESSAGES_KEY])
        st.session_state[CHAT_MESSAGES_KEY].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Live agentic-loop progress (step checklist + current phase).
            _agent_ph = st.empty()
            _call_failed = False
            with st.spinner("Thinking..."):
                try:
                    reply, tr = call_backend_turn(
                        messages=st.session_state[CHAT_MESSAGES_KEY],
                        user_input=user_input,
                        tenant_config=chat_tenant_config,
                        approved_keys=None,
                        migration_config=None,
                        session_id=session_id,
                        allow_summarization=st.session_state["allow_summarization"],
                        mode=BACKEND_MODE_CHAT,
                        progress_placeholder=_agent_ph,
                    )
                except Exception as e:
                    _call_failed = True
                    logger.exception("LLM+tools call failed: %s", e)
                    st.error("Sorry, something went wrong while calling the agent.")
                    st.exception(e)
                    reply = f"Error: {e}"
                    tr = None
            _agent_ph.empty()

            # Ensure run log is not shown/stored for chat
            st.session_state[_LAST_RUN_LOG_STATE_KEY] = None

            if isinstance(tr, dict) and tr.get("pending_confirmation"):
                st.session_state[CHAT_PENDING_KEY] = tr["pending_confirmation"]

                st.info(
                    tr["pending_confirmation"].get("reason")
                    or "This action requires approval before it can make changes to your Sisense deployment."
                )
                with st.expander("View operation details", expanded=True):
                    pc = tr["pending_confirmation"]
                    st.markdown("**Tool:** `{}`".format(pc.get("tool_id", "")))
                    st.code(json.dumps(pc.get("arguments", {}), indent=2), language="json")

                cols = st.columns([1, 1])
                with cols[0]:
                    if st.button("Approve", type="primary"):
                        key = _approval_key(pc["tool_id"], pc.get("arguments", {}))
                        st.session_state[CHAT_APPROVED_KEY].add(key)

                        _agent_ph2 = st.empty()
                        with st.spinner("Running approved action..."):
                            try:
                                reply2, tr2 = call_backend_turn(
                                    messages=st.session_state[CHAT_MESSAGES_KEY],
                                    user_input=user_input,
                                    tenant_config=chat_tenant_config,
                                    approved_keys=st.session_state[CHAT_APPROVED_KEY],
                                    migration_config=None,
                                    session_id=session_id,
                                    allow_summarization=st.session_state["allow_summarization"],
                                    mode=BACKEND_MODE_CHAT,
                                    progress_placeholder=_agent_ph2,
                                )
                            except Exception as e:
                                logger.exception("Agent run after approval failed: %s", e)
                                st.error("The approved action failed.")
                                st.exception(e)
                                st.session_state[CHAT_PENDING_KEY] = None
                                st.rerun()
                        _agent_ph2.empty()

                        st.session_state[_LAST_RUN_LOG_STATE_KEY] = None

                        st.session_state[CHAT_MESSAGES_KEY].append(
                            {"role": "assistant", "content": reply2, "tool_result": tr2, "run_log": None}
                        )

                        st.session_state[CHAT_HIDE_USER_IDX_KEY] = st.session_state[CHAT_LAST_USER_IDX_KEY]
                        st.session_state[CHAT_PENDING_KEY] = None
                        st.rerun()

                with cols[1]:
                    if st.button("Cancel"):
                        st.session_state[CHAT_PENDING_KEY] = None
                        st.session_state[CHAT_MESSAGES_KEY].append(
                            {"role": "assistant", "content": "Action cancelled."}
                        )
                        st.rerun()
            else:
                st.session_state[CHAT_MESSAGES_KEY].append(
                    {"role": "assistant", "content": reply, "tool_result": tr, "run_log": None}
                )
                if not _call_failed:
                    st.rerun()


# =============================================================================
# MODE 2: MIGRATE BETWEEN DEPLOYMENTS
# =============================================================================
if mode == MODE_MIGRATION:
    MIG_SRC_KEY = "migration_source_config"
    MIG_TGT_KEY = "migration_target_config"
    MIG_MESSAGES_KEY = "migration_messages"
    MIG_LAST_USER_IDX_KEY = "migration_last_user_idx"
    MIG_HIDE_USER_IDX_KEY = "migration_hide_user_idx"
    MIG_PENDING_KEY = "migration_pending_confirmation"
    MIG_APPROVED_KEY = "migration_approved_mutations"

    if MIG_SRC_KEY not in st.session_state:
        st.session_state[MIG_SRC_KEY] = None
    if MIG_TGT_KEY not in st.session_state:
        st.session_state[MIG_TGT_KEY] = None

    st.subheader("Connect source and target Sisense environments")

    cols = st.columns(2)

    with cols[0]:
        st.markdown("**Source environment**")
        src_cfg = st.session_state[MIG_SRC_KEY] or {}
        with st.form("source_form"):
            src_domain = st.text_input(
                "Source domain", value=src_cfg.get("domain", ""), placeholder="https://source.sisense.com"
            )
            src_token = st.text_input("Source API token", type="password", value=src_cfg.get("token", ""))
            src_ssl = st.checkbox("Verify SSL (source)", value=src_cfg.get("ssl", True))
            src_submitted = st.form_submit_button("Connect source")

        if src_submitted:
            if not src_domain or not src_token:
                st.error("Source domain and token are required.")
            else:
                st.session_state[MIG_SRC_KEY] = {
                    "domain": src_domain.strip(),
                    "token": src_token.strip(),
                    "ssl": src_ssl,
                }
                logger.info("[MIGRATION] Source configured for domain=%s ssl=%s", src_domain.strip(), src_ssl)
                st.success("Source environment connected.")
                st.rerun()

        if st.session_state[MIG_SRC_KEY] is not None:
            if st.button("Disconnect source"):
                logger.info("[MIGRATION] Disconnecting source.")
                st.session_state[MIG_SRC_KEY] = None
                st.rerun()

    with cols[1]:
        st.markdown("**Target environment**")
        tgt_cfg = st.session_state[MIG_TGT_KEY] or {}
        with st.form("target_form"):
            tgt_domain = st.text_input(
                "Target domain", value=tgt_cfg.get("domain", ""), placeholder="https://target.sisense.com"
            )
            tgt_token = st.text_input("Target API token", type="password", value=tgt_cfg.get("token", ""))
            tgt_ssl = st.checkbox("Verify SSL (target)", value=tgt_cfg.get("ssl", True))
            tgt_submitted = st.form_submit_button("Connect target")

        if tgt_submitted:
            if not tgt_domain or not tgt_token:
                st.error("Target domain and token are required.")
            else:
                st.session_state[MIG_TGT_KEY] = {
                    "domain": tgt_domain.strip(),
                    "token": tgt_token.strip(),
                    "ssl": tgt_ssl,
                }
                logger.info("[MIGRATION] Target configured for domain=%s ssl=%s", tgt_domain.strip(), tgt_ssl)
                st.success("Target environment connected.")
                st.rerun()

        if st.session_state[MIG_TGT_KEY] is not None:
            if st.button("Disconnect target"):
                logger.info("[MIGRATION] Disconnecting target.")
                st.session_state[MIG_TGT_KEY] = None
                st.rerun()

    with st.sidebar:
        st.subheader("Status:")
        st.write(f"Migration tools available to LLM: **{len(migration_tools)}**")
        st.markdown("**Mode:** Migrate between deployments")

        src_cfg = st.session_state[MIG_SRC_KEY]
        tgt_cfg = st.session_state[MIG_TGT_KEY]

        st.markdown("**Source**")
        if src_cfg:
            st.write(f"Domain: `{src_cfg.get('domain', '')}`")
            st.write(f"SSL verification: `{src_cfg.get('ssl', True)}`")
        else:
            st.write("_Not connected_")

        st.markdown("**Target**")
        if tgt_cfg:
            st.write(f"Domain: `{tgt_cfg.get('domain', '')}`")
            st.write(f"SSL verification: `{tgt_cfg.get('ssl', True)}`")
        else:
            st.write("_Not connected_")

        if src_cfg and tgt_cfg:
            st.markdown(
                """
**Examples:**
- Migrate these groups from source to target: `group_a`, `group_b`
- Migrate dashboards "Sales Overview" and "Customer KPIs"
- Migrate datamodel "ecommerce_db" from source to target
"""
            )
        else:
            st.markdown("_Connect both source and target environments to see migration examples._")
        st.markdown("---")
        st.caption("Migration mode uses source and target connections to migrate assets between environments.")

    if not st.session_state[MIG_SRC_KEY] or not st.session_state[MIG_TGT_KEY]:
        st.info("Connect both source and target environments to start using the Migration assistant.")
        st.stop()

    src_cfg = st.session_state[MIG_SRC_KEY]
    tgt_cfg = st.session_state[MIG_TGT_KEY]
    migration_config = {"source": src_cfg, "target": tgt_cfg}

    if MIG_MESSAGES_KEY not in st.session_state:
        st.session_state[MIG_MESSAGES_KEY] = [
            {
                "role": "assistant",
                "content": (
                    "You are connected to a **source** and a **target** Sisense "
                    "deployment. Describe what you want to migrate between them."
                ),
            },
        ]
        logger.debug("[MIGRATION] Chat history initialized with greeting only (system prompt handled in backend).")

    if MIG_LAST_USER_IDX_KEY not in st.session_state:
        st.session_state[MIG_LAST_USER_IDX_KEY] = None
    if MIG_HIDE_USER_IDX_KEY not in st.session_state:
        st.session_state[MIG_HIDE_USER_IDX_KEY] = None
    if MIG_PENDING_KEY not in st.session_state:
        st.session_state[MIG_PENDING_KEY] = None
    if MIG_APPROVED_KEY not in st.session_state:
        st.session_state[MIG_APPROVED_KEY] = set()
    if _MIG_TURN_IN_PROGRESS_KEY not in st.session_state:
        st.session_state[_MIG_TURN_IN_PROGRESS_KEY] = False
    if _MIG_TURN_CTX_KEY not in st.session_state:
        st.session_state[_MIG_TURN_CTX_KEY] = {}
    if _MIG_PENDING_TURN_META_KEY not in st.session_state:
        st.session_state[_MIG_PENDING_TURN_META_KEY] = {}

    def _process_mig_turn_result() -> None:
        """
        Called from the polling block once the background thread is done.
        Reads the completed ctx and meta, updates session state, does NOT call st.rerun().
        """
        ctx = st.session_state.get(_MIG_TURN_CTX_KEY, {})
        meta = st.session_state.get(_MIG_PENDING_TURN_META_KEY, {})
        st.session_state[_MIG_TURN_IN_PROGRESS_KEY] = False

        run_log = ctx.get("run_log") or st.session_state.get(_LAST_RUN_LOG_STATE_KEY)
        try:
            st.session_state[_LAST_RUN_LOG_STATE_KEY] = run_log
        except Exception:
            pass

        if ctx.get("error_str"):
            st.session_state[MIG_MESSAGES_KEY].append({"role": "assistant", "content": f"Error: {ctx['error_str']}"})
            st.session_state[MIG_PENDING_KEY] = None
            return

        reply = ctx.get("reply") or ""
        tr = ctx.get("tool_result")

        if isinstance(tr, dict) and tr.get("pending_confirmation"):
            st.session_state[MIG_PENDING_KEY] = tr["pending_confirmation"]
        else:
            if meta.get("clear_pending"):
                st.session_state[MIG_PENDING_KEY] = None
            st.session_state[MIG_MESSAGES_KEY].append(
                {"role": "assistant", "content": reply, "tool_result": tr, "run_log": run_log}
            )

    for i, msg in enumerate(st.session_state[MIG_MESSAGES_KEY]):
        if msg.get("role") not in ("user", "assistant"):
            continue
        if st.session_state[MIG_HIDE_USER_IDX_KEY] is not None and i == st.session_state[MIG_HIDE_USER_IDX_KEY]:
            continue
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                tr = msg.get("tool_result")
                if tr:
                    render_tool_result(tr)
                render_run_log(msg.get("run_log"))
            st.markdown(msg.get("content", ""))

    if st.session_state[MIG_HIDE_USER_IDX_KEY] is not None:
        st.session_state[MIG_HIDE_USER_IDX_KEY] = None

    # -------------------------------------------------------------------------
    # Turn-in-progress polling block
    # Shown while a background migration thread is running.
    # Calls st.rerun() to poll until done; blocks the rest of the UI meanwhile.
    # -------------------------------------------------------------------------
    if st.session_state.get(_MIG_TURN_IN_PROGRESS_KEY):
        ctx = st.session_state.get(_MIG_TURN_CTX_KEY, {})

        _col_status, _col_stop = st.columns([5, 1])
        with _col_stop:
            if st.button("⏹ Stop", key="mig_stop_btn"):
                _cancel_backend_turn(session_id)
                st.session_state[_MIG_TURN_IN_PROGRESS_KEY] = False
                st.session_state[MIG_MESSAGES_KEY].append(
                    {"role": "assistant", "content": "Migration stopped by user."}
                )
                st.rerun()
        with _col_status:
            prog = ctx.get("progress_lines", [])
            if prog:
                tail = prog[-20:]
                st.markdown("**Progress**\n\n" + "\n".join([f"- {x}" for x in tail]))
            else:
                st.markdown("*Planning migration...*")

        if ctx.get("done"):
            _process_mig_turn_result()
        else:
            time.sleep(0.4)
        st.rerun()

    pending_mig = st.session_state[MIG_PENDING_KEY]
    if pending_mig and isinstance(pending_mig, dict):
        st.info(
            pending_mig.get("reason")
            or "This migration action requires approval before it can make changes to your Sisense deployments."
        )
        with st.expander("View operation details", expanded=True):
            st.markdown("**Tool:** `{}`".format(pending_mig.get("tool_id", "")))
            st.code(json.dumps(pending_mig.get("arguments", {}), indent=2), language="json")

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Approve migration", type="primary"):
                key = _approval_key(pending_mig["tool_id"], pending_mig.get("arguments", {}))
                st.session_state[MIG_APPROVED_KEY].add(key)
                _launch_migration_turn(
                    meta={"kind": "approval", "clear_pending": True},
                    messages=st.session_state[MIG_MESSAGES_KEY],
                    user_input="",
                    tenant_config=None,
                    approved_keys=st.session_state[MIG_APPROVED_KEY],
                    migration_config=migration_config,
                    session_id=session_id,
                    allow_summarization=st.session_state["allow_summarization"],
                    mode=BACKEND_MODE_MIGRATION,
                )
                st.rerun()

        with cols[1]:
            if st.button("Cancel migration"):
                st.session_state[MIG_PENDING_KEY] = None
                st.session_state[MIG_MESSAGES_KEY].append(
                    {"role": "assistant", "content": "Migration action cancelled."}
                )
                st.rerun()

    mig_input = st.chat_input("Describe what you want to migrate...")

    if mig_input and not st.session_state.get(_MIG_TURN_IN_PROGRESS_KEY):
        logger.debug("[MIGRATION] User request: %s", mig_input)
        st.session_state[MIG_LAST_USER_IDX_KEY] = len(st.session_state[MIG_MESSAGES_KEY])
        st.session_state[MIG_MESSAGES_KEY].append({"role": "user", "content": mig_input})
        _launch_migration_turn(
            meta={"kind": "input", "clear_pending": False},
            messages=st.session_state[MIG_MESSAGES_KEY],
            user_input=mig_input,
            tenant_config=None,
            approved_keys=None,
            migration_config=migration_config,
            session_id=session_id,
            allow_summarization=st.session_state["allow_summarization"],
            mode=BACKEND_MODE_MIGRATION,
        )
        st.rerun()
