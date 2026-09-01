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
import itertools
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from streamlit.components.v1 import html as components_html

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

if not any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
    fh = TimedRotatingFileHandler(
        LOG_DIR / "app.log",
        when="midnight",  # daily file; 7 dated backups = 7 days kept, older deleted
        backupCount=7,
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

# Synthetic tool_id the backend uses to key a WHOLE-PLAN migration approval —
# see PLAN_TOOL_ID in backend/agent/migration_flow.py. Duplicated rather than
# imported: the UI is a separate process and imports nothing from the backend.
# It is not a real tool and never appears in the registry.
MIGRATION_PLAN_TOOL_ID = "migration.plan"


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
    plan_text: Optional[str] = None,
) -> None:
    """Render the agentic-loop progress block: the current plan (when the
    strategist made one), a collapsed checklist of completed steps, and a live
    status line for the current phase."""
    if placeholder is None:
        return
    md = ""
    if plan_text:
        plan_lines = "\n".join(f"> {ln}" for ln in plan_text.splitlines())
        md += f"**📋 Plan**\n{plan_lines}\n\n"
    lines: List[str] = []
    for s in completed_steps:
        mark = "✅" if s.get("ok") else "⚠️"
        lines.append(f"- {mark} Step {s.get('step')}: `{s.get('tool_id', '?')}`")
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
    tool_id = data.get("tool_id")
    if phase == "deciding":
        return "🤔 Checking progress against your request…"
    if phase == "replanning":
        return "🧠 That approach didn't work — rethinking the plan…"
    if phase == "replanned":
        return "📋 Plan revised — continuing…"
    if phase == "verifying":
        return "🔎 Double-checking the result covers your whole request…"
    if phase == "planning":
        return f"🧭 Planning step {step}…"
    if phase == "executing":
        return f"⏳ Step {step}: running `{tool_id}`…"
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


# Per-item narration — one line per exported/imported asset — drowns the
# batch-level story in the live progress view (875 of ~1,060 events in one
# real run, 2026-08-14). Hidden from the LIVE view only: the run-log expander
# and the server logs keep every event for troubleshooting.
_LIVE_HIDDEN_STEPS = {"import_datamodels", "export_datamodels"}


def _is_live_progress(payload: Any) -> bool:
    """Should this progress event appear in the live progress view?"""
    return not (isinstance(payload, dict) and payload.get("step") in _LIVE_HIDDEN_STEPS)


def _push_progress_line(lines: List[str], line: str) -> None:
    """Append a live-progress line, or REPLACE the previous one from the same phase.

    The live view is a progress indicator, not a record — the run log keeps
    every event. So a phase should occupy one line that updates, rather than
    scrolling a fixed 20-line window with its own history.

    It matters because the SDK emits twice per page while paginating
    (`"Fetching dashboards page…"` at the top of the loop and `"Fetched
    dashboards page."` at the bottom), and the first of each pair reports the
    counter BEFORE the increment — so the pair renders as the same sentence
    with the same number, twice. Migrating 501 dashboards produced 22 fetch
    lines and 51 batch lines; a reader watching it just wants "which phase, how
    far".

    Phase = the `[step]` prefix `_format_progress_line` puts on. Lines without
    one (chat's agent_progress milestones) always append, since each is a
    distinct event rather than a running counter.
    """
    prefix = line[: line.index("]") + 1] if line.startswith("[") and "]" in line else None
    if prefix and lines and lines[-1].startswith(prefix):
        lines[-1] = line
        return
    lines.append(line)


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

    # Human phrasing, not key=value: "batch 3/10, processed 120 of 500" reads;
    # "(batch_number=3, batches_total=10, processed_so_far=120)" leaks API keys
    # into the run log the user reads.
    hints: List[str] = []

    def _val(k: str) -> Optional[str]:
        v = payload.get(k)
        return str(v) if isinstance(v, (int, float, str)) and str(v) != "" else None

    batch, batches = _val("batch_number"), _val("batches_total")
    if batch and batches:
        hints.append(f"batch {batch}/{batches}")
    elif batch:
        hints.append(f"batch {batch}")

    done, total = _val("processed_so_far"), _val("total_count")
    if done and total:
        hints.append(f"processed {done} of {total}")
    elif done:
        hints.append(f"processed {done}")
    elif total:
        hints.append(f"{total} total")

    for k, label in [
        ("succeeded_total", "succeeded"),
        ("failed_total", "failed"),
        ("skipped_total", "skipped"),
        ("pages_fetched", "pages fetched"),
    ]:
        v = _val(k)
        if v:
            hints.append(f"{v} {label}")

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
        # Agentic-loop state, filled by the SSE reader on the worker thread and
        # rendered by the polling block on the main thread (see call_backend_turn).
        "agent_plan": None,
        "agent_steps": [],
        "agent_status": None,
    }
    st.session_state[_MIG_TURN_CTX_KEY] = ctx
    st.session_state[_MIG_PENDING_TURN_META_KEY] = meta
    st.session_state[_MIG_TURN_IN_PROGRESS_KEY] = True

    def _progress_cb(line: str) -> None:
        # Same phase-collapsing as the inline placeholder. There are TWO lists:
        # the SSE reader keeps a local one for the inline view, and this one
        # crosses the worker-thread boundary to the migration panel. Collapsing
        # only the first left the migration feed — the long-running case the
        # collapsing is FOR — still showing every event.
        _push_progress_line(ctx["progress_lines"], line)

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
            reply, tool_result, step_results, trace_id, usage, _display_hints = call_backend_turn(**call_kwargs)
            ctx["reply"] = reply
            ctx["tool_result"] = tool_result
            ctx["step_results"] = step_results
            ctx["trace_id"] = trace_id
            ctx["usage"] = usage
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
        final_step_results: Optional[List[Dict[str, Any]]] = None
        final_trace_id: Optional[str] = None
        final_usage: Optional[Dict[str, Any]] = None
        final_display_hints: Optional[List[str]] = None

        run_log: Dict[str, Any] = {
            "started_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "events": [],
        }

        progress_lines: List[str] = []
        agent_steps: List[Dict[str, Any]] = []
        agent_plan: Optional[str] = None

        for event, data in _iter_sse_events(resp):
            if event == "keepalive":
                continue

            cleaned_payload = _extract_progress_payload(data)
            # The run log is the SDK's per-asset activity trail ("[fetch_users]
            # Fetching users… (3 of 10)") plus errors — nothing else. Internal
            # frames (status, result, the agent_progress checklist) used to be
            # recorded too, and on turns with no SDK activity the expander
            # showed only their type names as bullets: "update / agent_progress
            # / update". With this filter such turns record nothing and the
            # expander does not appear at all.
            if event == "error" or (event == "progress" and data.get("type") != "agent_progress"):
                run_log["events"].append({"event": event, "payload": cleaned_payload})

            # Agentic-loop step progress (Step 8) — dedicated checklist rendering.
            if event == "progress" and data.get("type") == "agent_progress":
                if data.get("plan"):
                    agent_plan = data["plan"]
                if data.get("phase") == "completed":
                    agent_steps.append({"step": data.get("step"), "tool_id": data.get("tool_id"), "ok": data.get("ok")})
                    _render_agent_progress(progress_placeholder, agent_steps, None, agent_plan)
                else:
                    _render_agent_progress(
                        progress_placeholder, agent_steps, _agent_progress_status_line(data), agent_plan
                    )
                # Migration runs this loop on a background thread, where touching
                # a Streamlit placeholder is illegal — hand the same structured
                # state to the polling block instead, which renders on the main
                # thread. Plain dict writes only; no st.* calls off-thread.
                if _run_log_out is not None:
                    _run_log_out["agent_plan"] = agent_plan
                    _run_log_out["agent_steps"] = list(agent_steps)
                    _run_log_out["agent_status"] = _agent_progress_status_line(data)
                elif progress_callback is not None:
                    # No structured channel — fall back to a flat status line.
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
                # Run log (above) already recorded the event; the live view
                # shows milestones only. The step name lives in the EXTRACTED
                # payload — `data` is the MCP notification envelope, which has
                # no `step` key (the filter's first version tested `data` and
                # therefore never fired; caught live 2026-08-14).
                if not _is_live_progress(cleaned_payload):
                    continue
                msg = data.get("message") or data.get("detail")
                if isinstance(msg, str) and msg.strip():
                    new_line = msg.strip()
                else:
                    new_line = _format_progress_line(cleaned_payload)
                _push_progress_line(progress_lines, new_line)
                if progress_callback is not None:
                    progress_callback(new_line)

            elif event == "result":
                final_reply = data.get("reply", "")
                final_tool_result = data.get("tool_result")
                final_step_results = data.get("step_results")
                final_trace_id = data.get("trace_id")
                final_usage = data.get("usage")
                final_display_hints = data.get("display_hints")

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

        return (
            final_reply or "",
            final_tool_result,
            final_step_results,
            final_trace_id,
            final_usage,
            final_display_hints,
        )

    # Fallback: backend returned JSON even though we asked for SSE
    _write_run_log(None, _run_log_out)
    data = resp.json()
    reply = data.get("reply", "")
    tool_result = data.get("tool_result")
    step_results = data.get("step_results")
    return reply, tool_result, step_results, data.get("trace_id"), data.get("usage"), data.get("display_hints")


# -----------------------------------------------------------------------------
# Tool result rendering
# -----------------------------------------------------------------------------
def _normalize_domain(raw: str) -> str:
    """'mycompany.sisense.com/' -> 'https://mycompany.sisense.com'. A bare
    domain defaults to https (the token must not travel in cleartext by
    accident); an EXPLICIT http:// is honored — some internal deployments
    genuinely run it. The SDK accepts every form; this is for safety and a
    consistent sidebar display."""
    d = (raw or "").strip().rstrip("/")
    if d and "://" not in d:
        d = f"https://{d}"
    return d


AUTH_TOKEN = "API token"
AUTH_PASSWORD = "Username & password"


def _sdk_base_url(domain: str, verify_ssl: bool) -> str:
    """The URL the SDK will actually talk to, asked of the SDK rather than guessed.

    Sisense's non-SSL listener is on a different port (30845 on Linux), and the
    SDK rebuilds the base URL from `domain` + `is_ssl` rather than taking the
    domain verbatim — with ssl off it appends that port itself, and it does so
    even when the user typed a port of their own. So logging in against the
    domain as typed can authenticate somewhere the agent will never call.

    Rather than hardcode 30845 here — SDK knowledge that would silently drift
    the day they change it — construct the client the same way the backend will
    and read the URL it derived. If the SDK is unavailable (someone running the
    UI standalone), fall back to the domain as given; the request either works
    or reports honestly.
    """
    try:
        from pysisense import SisenseClient  # local: keeps UI startup independent of the SDK

        # The token is a placeholder: this client is only ever asked what URL it
        # derived and never issues a request. It cannot be empty — the SDK
        # rejects a domain without one.
        client = SisenseClient(config_file=None, domain=domain, token="url-derivation-only", is_ssl=verify_ssl)
        return str(client.base_url)
    except Exception:  # noqa: BLE001 — any SDK problem falls back, never blocks sign-in
        return domain


def _login_for_token(domain: str, username: str, password: str, verify_ssl: bool) -> str:
    """Exchange a Sisense username/password for an API token.

    THE PASSWORD STOPS HERE. Streamlit runs server-side, so this call goes
    straight from the UI process to Sisense; the backend, the MCP server, the
    session pool and every log line downstream still only ever see the token,
    exactly as when the user pastes one. Routing the password through those
    three hops instead would turn each of them into a place a reusable
    credential can leak, and buy nothing — the token is what every one of them
    actually needs.

    Nothing here is stored: the caller keeps the returned token and drops the
    password, and the form widget holding it stops being rendered on the next
    frame.
    """
    base = _sdk_base_url(domain, verify_ssl)
    try:
        resp = requests.post(
            f"{base}/api/v1/authentication/login",
            data={"username": username, "password": password},
            timeout=30,
            verify=verify_ssl,
        )
    except requests.exceptions.SSLError:
        raise RuntimeError(
            "TLS verification failed. If this deployment uses a self-signed certificate, untick 'Verify SSL'."
        ) from None
    except requests.exceptions.RequestException as exc:
        # Name the URL actually attempted, not the domain as typed — they differ
        # whenever the SDK rewrote it (non-SSL port), and the difference is
        # usually the reason it failed.
        raise RuntimeError(f"Could not reach {base}: {exc}") from None

    # Sisense answers a bad password with 401 and a JSON body; relay ITS words
    # rather than inventing a reason ("wrong password" may actually be a locked
    # account, an SSO-only tenant, or a disabled user).
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not resp.ok or not body.get("success", resp.ok):
        # Sisense nests it: {"error": {"code": 5001, "message": "Invalid
        # credentials.", ...}}. Reach the sentence rather than showing the
        # user a dict — but still fall back to the raw body when the shape is
        # one we do not recognise, so nothing is ever swallowed.
        err = body.get("error")
        detail = (
            (err.get("message") if isinstance(err, dict) else None)
            or body.get("message")
            or (err if isinstance(err, str) else None)
            or (resp.text or "").strip()[:200]
        )
        raise RuntimeError(f"Sign-in failed (HTTP {resp.status_code}){f': {detail}' if detail else ''}")

    token = body.get("access_token") or body.get("token")
    if not token:
        raise RuntimeError("Sign-in succeeded but Sisense returned no access token.")
    return str(token)


def _auth_mode_radio(key: str) -> str:
    """Rendered OUTSIDE the form on purpose: a radio inside a form does not
    rerun until submit, so the fields below it could not follow the choice."""
    return st.radio("Sign in with", [AUTH_TOKEN, AUTH_PASSWORD], key=key, horizontal=True)


def _credential_inputs(mode: str, label_prefix: str = "", token_value: str = "") -> Dict[str, str]:
    """The credential fields for the chosen mode. Call INSIDE a form."""
    p = f"{label_prefix} " if label_prefix else ""
    if mode == AUTH_TOKEN:
        return {"token": st.text_input(f"{p}API token".strip(), type="password", value=token_value)}
    return {
        "username": st.text_input(f"{p}Username".strip()),
        "password": st.text_input(f"{p}Password".strip(), type="password"),
    }


def _resolve_token(
    domain: str, mode: str, creds: Dict[str, str], verify_ssl: bool
) -> Tuple[Optional[str], Optional[str]]:
    """(token, error). Password mode exchanges via Sisense; token mode passes through."""
    if mode == AUTH_TOKEN:
        token = (creds.get("token") or "").strip()
        return (token, None) if token else (None, "API token is required.")

    username = (creds.get("username") or "").strip()
    password = creds.get("password") or ""
    if not username or not password:
        return None, "Username and password are required."
    try:
        return _login_for_token(domain, username, password, verify_ssl), None
    except RuntimeError as exc:
        return None, str(exc)


def _approval_key(tool_id: str, args: Dict[str, Any]) -> Tuple[str, str]:
    return tool_id, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_tools_cached(url: str):
    """One HTTP fetch of /tools, cached so reruns (and browser refreshes) don't
    re-hit the backend. Raises on any problem; the caller handles retry/errors.
    Only successful returns are cached — a raise is retried next call."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    tools = data.get("tools") or []
    registry = data.get("registry") or {}
    if not isinstance(tools, list) or not isinstance(registry, dict):
        raise ValueError("Unexpected /tools payload shape")
    return tools, registry


def fetch_tools_from_backend():
    """
    Fetch OpenAI-style tools and registry metadata from the backend.

    Cached + retried: on a browser refresh Streamlit reruns this and the
    reconnecting websocket can make a single request race and fail. Retrying a
    couple of times (and caching the success) stops a transient blip from
    flashing a scary error on every reload.
    """
    url = f"{BACKEND_URL}/tools"
    last_err = None
    for attempt in range(3):
        try:
            return _fetch_tools_cached(url)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("Fetch /tools attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.4 * (attempt + 1))

    logger.exception("Request to /tools failed after retries: %s", last_err)
    st.error(
        "Could not reach the backend /tools endpoint after a few tries. "
        "Check that the backend is running and BACKEND_URL is correct."
    )
    st.stop()


# Unique keys for the download buttons: render_tool_result runs inside the
# history loops, and two identical payloads in one conversation would collide
# on Streamlit's auto-generated IDs. app.py is the main script, re-executed
# top-to-bottom every rerun, so a fresh positional counter is stable per frame.
_DL_SEQ = itertools.count()

# Starting height (px) of the scrollable conversation area — deliberately
# TALLER than most windows: the auto-height JS reliably SHRINKS the box to
# fit the viewer's window (growing gets reverted by Streamlit re-renders),
# so starting tall and shrinking covers every monitor. The container itself
# exists so the chat input can be NESTED (inline) instead of Streamlit's
# pinned sticky bar (the ghost-bar fix).
CHAT_BOX_HEIGHT = 1600


# -----------------------------------------------------------------------------
# "What can I ask?" — the capability browser
# -----------------------------------------------------------------------------
# Built from the LIVE registry the backend serves, never a hand-written list, so
# it cannot drift from what the assistant can actually do: enable a tool and it
# appears here, delist one and it disappears. Rendered with Streamlit's own
# widgets rather than embedded HTML so it inherits the app's theme.
_CAPABILITY_AREAS = {
    "access_management": "Users, groups & access",
    "dashboard": "Dashboards & widgets",
    "datamodel": "Data models & connections",
    "migration": "Moving between environments",
    "wellcheck": "Health checks",
    "folder": "Folders",
    "metadata": "Measures & metadata",
    "queries": "Queries",
    "report_manager": "Report Manager",
    "custom_code": "Notebooks & custom code",
    "blox": "BloX widgets",
    "plugins": "Plugins",
    "encryption": "Encryption",
}


def _render_capabilities(registry: Dict[str, Any], mode_label: str) -> None:
    """Searchable list of everything the assistant can do, grouped by area."""
    rows = [
        {
            "id": tid,
            "name": tid.split(".", 1)[-1],
            "area": (meta.get("module") or "other"),
            "write": bool(meta.get("mutates")),
            "desc": (meta.get("description") or "").strip().splitlines()[0] if meta.get("description") else "",
            "example": meta.get("example") or "",
        }
        for tid, meta in (registry or {}).items()
    ]
    # Mode decides the surface, exactly as it does for the agent: migration
    # tools are unusable in chat and vice versa, so listing them here would
    # promise something this session cannot do.
    want_migration = mode_label == "migration"
    rows = [r for r in rows if (r["area"] == "migration") == want_migration]
    writes = sum(1 for r in rows if r["write"])

    st.caption(
        f"**{len(rows)}** operations available in this mode · **{writes}** of them change "
        "something and always ask for your approval first · everything runs with *your* "
        "Sisense token, so it can only do what you can do."
    )
    term = (
        st.text_input(
            "Search",
            key=f"cap_q_{mode_label}",
            placeholder="Search — try “user”, “dashboard”, “unused”",
            label_visibility="collapsed",
        )
        .strip()
        .lower()
    )
    kind = st.radio(
        "Show",
        ["Everything", "Read only", "Changes things"],
        horizontal=True,
        key=f"cap_kind_{mode_label}",
        label_visibility="collapsed",
    )

    shown = 0
    for area in [a for a in _CAPABILITY_AREAS if any(r["area"] == a for r in rows)]:
        group = [r for r in rows if r["area"] == area]
        if kind == "Read only":
            group = [r for r in group if not r["write"]]
        elif kind == "Changes things":
            group = [r for r in group if r["write"]]
        if term:
            group = [
                r
                for r in group
                if term in r["name"].lower() or term in r["desc"].lower() or term in r["example"].lower()
            ]
        if not group:
            continue
        shown += len(group)
        with st.expander(f"{_CAPABILITY_AREAS.get(area, area)}  ·  {len(group)}", expanded=bool(term)):
            for r in sorted(group, key=lambda x: (x["write"], x["name"])):
                tag = "🔸 changes things" if r["write"] else "🔹 read only"
                st.markdown(
                    f"**{r['desc'] or r['name']}**  \n<span style='opacity:.7;font-size:.85em'>{tag}</span>",
                    unsafe_allow_html=True,
                )
                if r["example"]:
                    st.caption(f"Try: *{r['example']}*")
    if shown == 0:
        st.info("No operation matches that. Try a different word.")


@st.dialog("What can I ask?", width="large")
def _capabilities_dialog(registry: Dict[str, Any], mode_label: str) -> None:
    _render_capabilities(registry, mode_label)


def render_tool_result(tr: dict):
    if not tr or not isinstance(tr, dict):
        return

    tool_name = tr.get("tool_id", "")
    if tool_name:
        st.caption(f"Tool called: `{tool_name}`")

    _fname = (tool_name or "result").replace(".", "_")

    if tr.get("ok", True):
        data = tr.get("result")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)

            # Make columns safe for Arrow / Streamlit. Mixed-type columns AND
            # columns of nested containers: a column where EVERY cell is a dict
            # is uniformly typed (the old nunique guard passed it), but nested
            # dicts/lists of varying inner shape crash pyarrow's conversion —
            # a native segfault, not an exception, so no try/except downstream
            # can save the process (fes-ui exit=139, live 2026-08-27, EC2).
            # Nested values render as compact JSON strings instead.
            def _cell_safe(v):
                if isinstance(v, (dict, list, tuple, set)):
                    try:
                        return json.dumps(v, ensure_ascii=False, default=str)
                    except Exception:
                        return str(v)
                return v

            for col in df.columns:
                try:
                    has_nested = df[col].map(lambda v: isinstance(v, (dict, list, tuple, set, bytes))).any()
                    if has_nested or df[col].map(type).nunique() > 1:
                        df[col] = df[col].map(_cell_safe).astype(str)
                except Exception:
                    df[col] = df[col].astype(str)

            st.markdown("**Result**")
            # Interactive dataframe restored (canvas was EXONERATED for the
            # ghost-bar bug — it reproduced with plain HTML tables too; the
            # structural fix is the nested inline chat input, see the
            # conversation-box notes). Its grid also hands trackpad scrolling
            # back to the page properly, unlike a plain overflow div.
            st.dataframe(df, width="stretch")
            # Export without copy-paste, in whichever format the user wants.
            _c1, _c2, _c3, _ = st.columns([1, 1, 1, 4])
            _c1.download_button(
                "CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{_fname}.csv",
                mime="text/csv",
                key=f"dl_{next(_DL_SEQ)}",
            )
            _c2.download_button(
                "JSON",
                json.dumps(data, indent=2, ensure_ascii=False),
                file_name=f"{_fname}.json",
                mime="application/json",
                key=f"dl_{next(_DL_SEQ)}",
            )
            _c3.download_button(
                "TXT",
                df.to_string(index=False),
                file_name=f"{_fname}.txt",
                mime="text/plain",
                key=f"dl_{next(_DL_SEQ)}",
            )
        else:
            st.markdown("**Result (JSON)**")
            st.code(json.dumps(data, indent=2), language="json")
            if data is not None:
                _payload = json.dumps(data, indent=2, ensure_ascii=False)
                _c1, _c2, _ = st.columns([1, 1, 5])
                _c1.download_button(
                    "JSON",
                    _payload,
                    file_name=f"{_fname}.json",
                    mime="application/json",
                    key=f"dl_{next(_DL_SEQ)}",
                )
                _c2.download_button(
                    "TXT",
                    _payload if not isinstance(data, str) else data,
                    file_name=f"{_fname}.txt",
                    mime="text/plain",
                    key=f"dl_{next(_DL_SEQ)}",
                )
    else:
        if not tr.get("pending_confirmation"):
            st.markdown("**Tool error**")
            st.code(json.dumps(tr, indent=2), language="json")


def _render_result_expander(label: str, res: Any, expanded_when_ok: bool = False) -> None:
    """Every raw result lives in an expander. A SINGLE step's result opens by
    default (the expander is for collapsing, not hiding — user feedback,
    2026-08-17); multi-step results stay collapsed (scrolling past several
    payloads to reach the answer buries the answer, 2026-08-14). A FAILED
    result always opens: the raw payload is the tool's own account of what
    went wrong — not worth hiding behind a click."""
    ok = res.get("ok", True) if isinstance(res, dict) else True
    rows = res.get("result") if isinstance(res, dict) else None
    n = f" · {len(rows)} rows" if isinstance(rows, list) else ""
    mark = "" if ok else " ⚠️"
    with st.expander(f"{label}{n}{mark}", expanded=(not ok) or expanded_when_ok):
        render_tool_result(res)


def render_results(step_results, fallback_tr=None):
    """Show every step's raw result, labeled by tool, so a multi-step answer is
    legible instead of surfacing only the last table. Falls back to the single
    tool_result for older messages / single-step turns."""
    steps = [s for s in (step_results or []) if isinstance(s, dict)]
    if len(steps) > 1:
        st.caption(f"This answer used {len(steps)} steps — raw output of each:")
        for s in steps:
            _render_result_expander(f"Step {s.get('step', '?')} · `{s.get('tool_id', '?')}`", s.get("result") or {})
    elif len(steps) == 1:
        s = steps[0]
        _render_result_expander(f"Result · `{s.get('tool_id', '?')}`", s.get("result") or {}, expanded_when_ok=True)
    elif fallback_tr:
        _render_result_expander("Result", fallback_tr, expanded_when_ok=True)


# -----------------------------------------------------------------------------
# User feedback (thumbs up/down per answer) → logs/feedback.csv
# Joined to llm_calls.csv / llm_traces.csv by trace_id, a vote becomes a
# LABELED row for cross-model accuracy comparison ("wrong tool ran" etc.).
# Always on — it is an explicit user action, tiny volume, and the whole point
# is collecting the labels.
# -----------------------------------------------------------------------------
_FEEDBACK_CSV_PATH = LOG_DIR / "feedback.csv"
_FEEDBACK_COLUMNS = ["timestamp", "session_id", "mode", "trace_id", "verdict", "comment", "question", "tools"]


def _turn_tools(msg: Dict[str, Any]) -> str:
    """Tool ids this answer's turn executed, ';'-joined (for the feedback row)."""
    ids: List[str] = []
    for s in msg.get("step_results") or []:
        if isinstance(s, dict) and s.get("tool_id"):
            ids.append(str(s["tool_id"]))
    if not ids and isinstance(msg.get("tool_result"), dict) and msg["tool_result"].get("tool_id"):
        ids.append(str(msg["tool_result"]["tool_id"]))
    return ";".join(ids)


def _record_feedback(mode_label: str, msg: Dict[str, Any], question: str, verdict: str, comment: str = "") -> None:
    """Append one feedback event. Swallows errors — feedback must never break the UI."""
    import csv as _csv

    try:
        write_header = not _FEEDBACK_CSV_PATH.exists()
        with _FEEDBACK_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=_FEEDBACK_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": st.session_state.get("fes_session_id", ""),
                    "mode": mode_label,
                    "trace_id": msg.get("trace_id") or "",
                    "verdict": verdict,
                    "comment": (comment or "")[:500],
                    "question": (question or "")[:300],
                    "tools": _turn_tools(msg),
                }
            )
        logger.info("Feedback recorded: %s (trace_id=%s)", verdict, msg.get("trace_id"))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to record feedback; ignored.")


def _question_before(messages: List[Dict[str, Any]], idx: int) -> str:
    """The user message this answer responds to (nearest user turn above it)."""
    for m in reversed(messages[:idx]):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _usage_caption(u: Optional[Dict[str, Any]]) -> Optional[str]:
    """'X tokens · ~$Y' line for a turn; None when nothing was recorded.
    cost=None means the model isn't in the pricing map — say so explicitly
    (a silent absence or a $0.00 would both read as 'broken')."""
    if not isinstance(u, dict):
        return None
    total = int(u.get("tokens_in") or 0) + int(u.get("tokens_out") or 0)
    if total <= 0:
        return None
    cost = u.get("cost")
    if cost is None:
        return f"⚡ {total:,} tokens · cost n/a for this model"
    return f"⚡ {total:,} tokens" + (f" · ~${float(cost):.4f}" if float(cost) > 0 else "")


def render_feedback_controls(mode_label: str, messages: List[Dict[str, Any]], idx: int, msg: Dict[str, Any]) -> None:
    """Thumbs up/down under an assistant answer, plus an optional what-went-wrong
    note on a thumbs-down. Each event appends a feedback.csv row; the vote is
    remembered on the message dict so a rerun doesn't double-write."""
    fb = st.feedback("thumbs", key=f"fb_{mode_label}_{idx}")
    if fb is not None and msg.get("feedback") != fb:
        msg["feedback"] = fb
        verdict = "up" if fb == 1 else "down"
        _record_feedback(mode_label, msg, _question_before(messages, idx), verdict)
        st.toast("Thanks — feedback recorded.", icon="👍" if fb == 1 else "👎")
    if msg.get("feedback") == 0:
        # One comment per thumbs-down: once recorded, the input is gone —
        # otherwise every further Enter in the lingering box records again.
        if msg.get("feedback_comment"):
            st.caption(f"📝 Feedback noted: {msg['feedback_comment']}")
        else:
            note = st.text_input(
                "What went wrong? (optional — e.g. wrong tool, wrong data, unclear answer)",
                key=f"fbc_{mode_label}_{idx}",
            )
            if note:
                msg["feedback_comment"] = note
                _record_feedback(mode_label, msg, _question_before(messages, idx), "down-comment", comment=note)
                st.toast("Noted — thank you.", icon="📝")
                st.rerun()


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
# Wide layout: the default centered column is ~730px, which wastes most of a
# wide monitor. The CSS below re-caps content at a readable width — tables and
# run logs get the room, chat text lines stay a comfortable length.
# (Bisected and exonerated for the sticky-bar paint bug, 2026-08-17.)
st.set_page_config(page_title="FES Assistant", page_icon="frontend/assets/sisense.png", layout="wide")

check_ui_session_timeout()

# Tab bar linking this app to the Fieldnotes tool sharing the same domain
# (nginx routes "/" here, "/fieldnotes/" to the other). Plain <a> tags, not
# st.page_link: Fieldnotes is a separate Flask app, not a page in this one.
#
# Lives inside Streamlit's own header bar, not in the page content below it.
# st.markdown can only add content in the normal document flow, and that
# header is a fixed overlay drawn on top of the page rather than pushing it
# down, so anything placed "below" it in markup either gets covered or needs
# a margin guess to clear it. A components.html script reaches past its own
# iframe into window.parent.document and appends straight into
# [data-testid="stHeader"] instead — checked that this actually survives a
# real Streamlit rerun (not just the initial paint) before relying on it.
components_html(
    """
    <script>
    (function () {
        const doc = window.parent.document;
        if (doc.getElementById("app-tabs-nav")) return;  // already injected, reruns call this again

        const header = doc.querySelector('[data-testid="stHeader"]');
        if (!header) return;

        const style = doc.createElement("style");
        style.textContent = `
            #app-tabs-nav {
                position: absolute;
                left: 1rem;
                top: 50%;
                transform: translateY(-50%);
                /* Streamlit's own header content wrapper is a plain static
                   sibling, but it still won the paint/click order without
                   this — verified empirically (elementFromPoint returned the
                   wrapper, not this nav) rather than trusted on stacking
                   rules alone. */
                z-index: 1000000;
                display: inline-flex;
                padding: 0.25rem;
                gap: 0.2rem;
                background: rgba(128, 128, 128, 0.08);
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 10px;
            }
            #app-tabs-nav a, #app-tabs-nav a * {
                text-decoration: none !important;
            }
            .app-tab {
                display: inline-flex;
                align-items: baseline;
                gap: 0.4rem;
                padding: 0.4rem 0.9rem;
                border-radius: 7px;
                font-size: 0.88rem;
                font-weight: 600;
                color: inherit;
                opacity: 0.6;
            }
            .app-tab.is-active {
                opacity: 1;
                background: rgba(128, 128, 128, 0.22);
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
            }
            .app-tab-sub {
                font-size: 0.7rem;
                font-weight: 500;
                opacity: 0.75;
            }
        `;
        doc.head.appendChild(style);

        const nav = doc.createElement("nav");
        nav.id = "app-tabs-nav";
        nav.innerHTML = `
            <a class="app-tab is-active" href="/">FES Assistant <span class="app-tab-sub">MCP</span></a>
            <a class="app-tab" href="/fieldnotes/">Fieldnotes <span class="app-tab-sub">FES Tickets</span></a>
        `;
        header.appendChild(nav);
    })();
    </script>
    """,
    height=0,
)


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
    /* Wide layout, but not wall-to-wall: cap the main column at a width where
       tables breathe and text lines stay readable, centered in the leftover.
       scrollbar-gutter keeps content from shifting sideways when a dropdown
       toggle adds/removes the scrollbar. Streamlit's default ~6rem top
       padding is cut so header + conversation + input fit the window without
       page-level scrolling. */
    [data-testid="stMainBlockContainer"] {
        max-width: 1250px;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    [data-testid="stAppScrollToBottomContainer"] {
        scrollbar-gutter: stable;
    }
    /* Help ("?") tooltips: Streamlit's default is white-on-black, which blends
       into dark content on the main pane. Light card + dark text instead —
       the tooltip renders in a BaseWeb portal, so target the portal body and
       force the color on descendants (the markdown inside carries its own). */
    div[data-baseweb="tooltip"], [data-testid="stTooltipContent"] {
        background-color: #ffffff !important;
        color: #31333f !important;
        border: 1px solid #d5dae5 !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18) !important;
    }
    [data-testid="stTooltipContent"] p,
    [data-testid="stTooltipContent"] li,
    [data-testid="stTooltipContent"] em,
    [data-testid="stTooltipContent"] code {
        color: #31333f !important;
    }
    /* Conversation boxes: no browser scroll anchoring. Chrome's automatic
       scroll adjustment on content growth makes an expander toggle feel like
       a double render (settle, then adjust — seen live 2026-08-18). */
    .st-key-chat_box, .st-key-chat_box [data-testid="stVerticalBlock"],
    .st-key-mig_box, .st-key-mig_box [data-testid="stVerticalBlock"] {
        overflow-anchor: none;
        /* Streamlit animates its scroll adjustment when box content grows
           (expander opens) — a visible bottom-top-bottom bounce. Instant
           scrolling turns the dance into an imperceptible snap. */
        scroll-behavior: auto !important;
    }
    /* The conversation "box" must EXIST (its nested chat input renders
       inline — the ghost-bar fix) but should not LOOK or FEEL like a box:
       no border, and stretched to fill the viewport so the layout reads as
       a normal full-page chat with the composer at the bottom. The keyed
       stVerticalBlock itself carries Streamlit's height and border. */
    /* Conversation area: no box look. Height comes from the
       st.container(height=...) parameter (sidebar slider) — CSS overrides
       of Streamlit's height wrapper proved unreliable (cascade race with
       emotion styles; do not retry calc(100vh) here). Tune the env knob per
       monitor instead. */
    div.stVerticalBlock.st-key-chat_box, div.stVerticalBlock.st-key-mig_box {
        border: none !important;
        /* No visible inner scrollbar: with wheel-forwarding making the whole
           page scroll the conversation, a scrollbar at the box's right edge
           (mid-screen) is the one tell that a box exists. Scrolling still
           works everywhere; tables keep their own scrollbars. */
        scrollbar-width: none;
    }
    div.stVerticalBlock.st-key-chat_box::-webkit-scrollbar,
    div.stVerticalBlock.st-key-mig_box::-webkit-scrollbar {
        display: none;
    }
    /* Expanders snap instead of animating: inside the height-container,
       Streamlit re-measures content on toggle and can replay the collapse
       transition ("it collapses twice"). No animation, nothing to replay. */
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpanderDetails"] {
        transition: none !important;
        animation: none !important;
    }
    /* NO pinned header. Tried (sticky via a :has() selector on the
       stLayoutWrapper around the keyed container) and REVERTED 2026-08-17:
       Chrome intermittently left a stale paint of the chat-input bar
       mid-screen after result-expander toggles — visible but unclickable;
       toggling the expander repainted it away. Chrome's :has() style
       invalidation under heavy DOM churn was the prime suspect, and no
       :has-free selector reaches that wrapper. If pinning returns, use a
       non-CSS mechanism. */
    /* Chat-input bar: STOCK STREAMLIT STYLING ONLY — do not add transform /
       will-change / z-index here. A translateZ(0) "mitigation" (2026-08-17)
       put the sticky bar on its own compositor layer, which broke Chrome's
       sticky compensation during trackpad (compositor-thread) scrolling:
       the bar's pixels scrolled away with the content and floated mid-page
       (painted position == layout position minus scrollTop, verified against
       a live window screenshot). Programmatic scrolls — and thus headless
       tests — go through the main thread and never reproduced it. */
    /* Title row: the h1 and the capability button sit side by side, centred
       against each other. Streamlit stacks a container's children vertically,
       so the inner vertical block is flipped to a row here. */
    .st-key-fes_title_row [data-testid="stVerticalBlock"] {
        flex-direction: row;
        align-items: center;
        gap: 18px;
        width: auto;
    }
    /* The button sits inside a nudge/calm wrapper (see the animation note
       below), so BOTH the wrapper and the button block must opt out of the
       row's stretch — otherwise the flex child is the wrapper and the button
       is pushed wide inside it. */
    .st-key-fes_title_row .st-key-cap_slot_nudge,
    .st-key-fes_title_row .st-key-cap_slot_calm,
    .st-key-fes_title_row .st-key-cap_btn_top { width: auto; flex: 0 0 auto; }
    .st-key-fes_title_row .st-key-cap_btn_top button { width: auto; white-space: nowrap; }
    /* The capability button is the only affordance a first-time user has for
       "what do I type?", and as a stock secondary button it read as a label
       sitting next to the wordmark rather than something to press. Given an
       accent of its own it becomes the one coloured thing on an otherwise
       neutral header, which is what makes it findable.
       Colour, border and shadow only — NO transform (see the chat-input note
       above; this app has been bitten by compositor-layer side effects). */
    .st-key-cap_btn_top button {
        border-radius: 999px;
        padding: 0.35rem 1.05rem;
        font-weight: 600;
        background: #eaf3fb;
        color: #14507d;
        border: 1px solid #b9d6ec;
        box-shadow: none;
        transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }
    .st-key-cap_btn_top button:hover {
        background: #d8e9f7;
        border-color: #8dbfe2;
        color: #0e3c60;
        box-shadow: 0 1px 6px rgba(20, 80, 125, 0.18);
    }
    .st-key-cap_btn_top button:focus-visible {
        outline: 2px solid #14507d;
        outline-offset: 2px;
    }
    @media (prefers-color-scheme: dark) {
        .st-key-cap_btn_top button {
            background: rgba(86, 156, 214, 0.16);
            color: #a8d3f0;
            border-color: rgba(86, 156, 214, 0.38);
        }
        .st-key-cap_btn_top button:hover {
            background: rgba(86, 156, 214, 0.26);
            border-color: rgba(86, 156, 214, 0.6);
            color: #cbe6fa;
            box-shadow: 0 1px 8px rgba(0, 0, 0, 0.35);
        }
        .st-key-cap_btn_top button:focus-visible { outline-color: #a8d3f0; }
    }
    /* A few slow rings, then still. Only while the user has not asked anything
       yet — the wrapper key changes to cap_slot_calm on their first message, so
       this never fires again and cannot pulse on every rerun. */
    @keyframes fesCapNudge {
        0%   { box-shadow: 0 0 0 0 rgba(20, 80, 125, 0.35); }
        70%  { box-shadow: 0 0 0 9px rgba(20, 80, 125, 0); }
        100% { box-shadow: 0 0 0 0 rgba(20, 80, 125, 0); }
    }
    .st-key-cap_slot_nudge .st-key-cap_btn_top button {
        animation: fesCapNudge 2.4s ease-out 3;
    }
    @media (prefers-reduced-motion: reduce) {
        .st-key-cap_slot_nudge .st-key-cap_btn_top button { animation: none; }
    }
    .st-key-app_header h1 {
        font-size: 2.75rem;
        padding-bottom: 0;
        margin-bottom: 0;
    }
    .fes-subtitle {
        font-size: 1rem;
        opacity: 0.85;
        margin: 0.6rem 0 0.9rem 0;
    }
    /* "Mode" label hugs its radio pills, not the subtitle above it */
    .st-key-app_header [data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
        margin-bottom: 0;
        padding-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# NO pinned chat input (final resolution of the 2026-08-17/18 ghost-bar
# saga). Root-level st.chat_input makes Streamlit pin the bar in a sticky
# stBottom strip inside its scroll-to-bottom container — and Chrome
# intermittently paints that sticky bar at a stale scroll offset after
# repeated expander toggles (Safari clean; reproduced with FULLY STOCK
# Streamlit in scratchpad repro_v3.py; upstream: streamlit issue #8480).
# Every mitigation attempted (scroll guards, compositor layer promotion,
# repaint nudges) failed or aggravated it. The durable fix is structural:
# messages render inside a fixed-height scrollable container and the chat
# input is NESTED (documented Streamlit behavior: nested chat inputs render
# INLINE) — so the pinned bar, the sticky strip, and the page-level
# auto-scroll container cease to exist. Nothing left to mis-composite.

# ONE scoped JS guard on the conversation box (this is NOT the banned sticky-
# bar patching — that element is gone): Streamlit auto-scrolls height-
# containers with chat messages whenever their content changes, so an
# expander toggle yanks the box down-then-up ("someone is scrolling").
# Programmatic scrolls are allowed for a 2s grace window after each rerun —
# preserving the GOOD auto-scroll to a fresh answer — and blocked after,
# which is when expander toggles happen (they don't rerun). Installed per
# component-iframe with NO once-per-session flag: reruns destroy the iframe
# and its timers, so the newest iframe must always re-install (hard lesson,
# 2026-08-17).
components_html(
    """
    <script>
    (() => {
        const w = window.parent;
        // This zero-height iframe still occupies a flex slot and collects the
        // page's element-gap spacing — hide our own container so we add no
        // blank band to the layout (hidden iframes keep running their timers
        // in Chrome; the interval below merely throttles, which is fine).
        try { w.frameElement.parentElement.style.display = "none"; } catch (e) {}
        const findBox = () => w.document.querySelector(".st-key-chat_box, .st-key-mig_box");

        // NOTE (2026-08-18): NO scroll guard and NO follow logic here — on the
        // box architecture, Streamlit's native height-container behavior is
        // already correct (sticks to the bottom only when at the bottom;
        // never yanks a scrolled-away view — verified headless both ways).
        // A guard inherited from the old pinned-input architecture used to
        // block that native behavior, and three generations of hand-rolled
        // "follow" logic tried to rebuild it and lost timing races. Do not
        // reintroduce either. The only jobs left for JS are the two things
        // native cannot do:

        // 1. AUTO-HEIGHT: size the conversation area to the viewer's window
        //    (inline styles win where CSS calc() lost the cascade). The gap
        //    below the box is MEASURED, so an approval dialog appearing below
        //    shrinks the box and the input stays on screen.
        const autoHeight = () => {
            const box = findBox();
            if (!box) return;
            const inp = w.document.querySelector('[data-testid="stChatInput"]');
            if (!inp) return;
            const slack = Math.round((w.innerHeight - 12) - inp.getBoundingClientRect().bottom);
            if (Math.abs(slack) <= 4) return;
            const cur = box.getBoundingClientRect().height;
            const h = Math.min(Math.max(320, Math.round(cur + slack)), w.innerHeight - 180);
            if (Math.abs(cur - h) > 4) {
                // Was the view at the bottom BEFORE our resize? Then keep it
                // there after: native stickiness covers content growth, not a
                // viewport we shrink ourselves (e.g. when an approval dialog
                // appears below and takes room from the box).
                const atBottom = (box.scrollHeight - cur - box.scrollTop) < 80;
                for (const el of [box, box.parentElement]) {
                    el.style.setProperty("height", h + "px", "important");
                    el.style.setProperty("max-height", h + "px", "important");
                }
                if (atBottom) box.scrollTop = box.scrollHeight;
            }
        };
        setInterval(autoHeight, 400);
        autoHeight();
        if (w.__fesResizeHandler) w.removeEventListener("resize", w.__fesResizeHandler);
        w.__fesResizeHandler = autoHeight;
        w.addEventListener("resize", w.__fesResizeHandler);

        // 2. SCROLL-ANYWHERE: wheel motion from OUTSIDE the box (header, side
        //    margins, input area) forwards into the box — but stands down
        //    whenever the PAGE itself can scroll (tall dialog, small window),
        //    else one gesture double-scrolls box and page. Sidebar excluded.
        if (w.__fesWheelHandler) {
            w.document.removeEventListener("wheel", w.__fesWheelHandler);
        }
        w.__fesWheelHandler = (e) => {
            const box = findBox();
            if (!box || box.contains(e.target)) return;
            if (e.target.closest && e.target.closest('[data-testid="stSidebar"]')) return;
            const main = w.document.querySelector('[data-testid="stMain"]') || w.document.scrollingElement;
            if (main && main.scrollHeight > main.clientHeight + 8) return;
            box.scrollTop += e.deltaY;
        };
        w.document.addEventListener("wheel", w.__fesWheelHandler, {passive: true});
    })();
    </script>
    """,
    height=0,
)

# Header: title + subtitle + the mode radio (added into this container
# further down). Not pinned — see the CSS note above for why.
_header = st.container(key="app_header")
with _header:
    # Title + capability button as one FLEX ROW (see the CSS for
    # .st-key-fes_title_row). st.columns was tried first and is wrong here:
    # its widths are proportional, so the button drifted further from the
    # wordmark the wider the window got. A flex row keeps it hugging the title
    # at any viewport, and align-items centres it against the 2.75rem h1
    # instead of floating above it.
    _title_row = st.container(key="fes_title_row")
    with _title_row:
        st.title("FES Assistant")
        # Filled after the registry loads. An OWNED st.empty() slot so a rerun
        # replaces the button in place rather than leaving a ghost widget.
        _capability_slot = st.empty()
    st.markdown(
        '<p class="fes-subtitle">Explore, manage and migrate your Sisense environment — just ask. '
        "Scoped to your API token's permissions, and every change asks before it runs.</p>",
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

    # Widen the ? help tooltip so multi-line explanations don't scroll in a
    # narrow one-line-wide box. The tooltip renders at document level, so this
    # style is global (not scoped to the sidebar).
    st.markdown(
        """
        <style>
        [data-testid="stTooltipContent"] {
            max-width: 460px !important;
            max-height: 70vh !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "allow_summarization" not in st.session_state:
        st.session_state["allow_summarization"] = False

    _summ_help = (
        "Controls whether tool **results (your Sisense data)** are sent to the LLM. "
        "This is not only a privacy switch — it sets **how capable the assistant is**.\n\n"
        "**On** — the assistant can:\n"
        "- answer in natural language, not just raw data\n"
        "- chain steps that depend on each other (e.g. find a user's role, then "
        "list everyone with that role)\n"
        "- independently double-check that it actually finished your whole request\n\n"
        "…but result data leaves your Sisense instance and goes to the LLM "
        "provider — enable only if you trust it with that data.\n\n"
        "**Off** — data stays private: the LLM only ever sees *which* operations "
        "ran and whether they succeeded, never the data itself. The assistant still "
        "handles independent multi-step requests, but it **can't** pass data between "
        "steps, **can't** verify the goal was met, and returns **raw data** "
        "instead of a natural-language answer."
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
                "🔒 **Private** — results are fetched and shown to you here, but "
                "never sent to the LLM. Independent multi-step still works; no "
                "data-chaining between steps, no goal check, raw results only."
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

    # (Chat-area height is fully automatic — sized to the viewer's window by
    # the auto-height JS below; no user knob needed.)


# -----------------------------------------------------------------------------
# Mode selection: Chat vs Migration
# -----------------------------------------------------------------------------
MODE_CHAT = "Chat with deployment"
MODE_MIGRATION = "Migrate between deployments"

BACKEND_MODE_CHAT = "chat"
BACKEND_MODE_MIGRATION = "migration"

with _header:
    mode = st.radio(
        "Mode",
        [MODE_CHAT, MODE_MIGRATION],
        horizontal=True,
        key="mode_radio",
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

with _capability_slot.container():
    # Nudge only until the user has actually asked something. Keyed on a real
    # user turn, not on "messages is empty": the greeting is seeded into the
    # list up front, so an emptiness check would go quiet before anyone had
    # typed anything. The key drives which CSS rule matches — a rerun with the
    # same key restarts nothing, so sending a message simply stops the animation
    # instead of re-triggering it on every frame.
    _msgs = st.session_state.get("migration_messages" if mode == MODE_MIGRATION else "chat_messages") or []
    _asked_something = any(isinstance(m, dict) and m.get("role") == "user" for m in _msgs)
    with st.container(key="cap_slot_calm" if _asked_something else "cap_slot_nudge"):
        if st.button("✨ What can I ask?", key="cap_btn_top", width="stretch"):
            _capabilities_dialog(
                st.session_state.tool_registry,
                BACKEND_MODE_MIGRATION if mode == MODE_MIGRATION else BACKEND_MODE_CHAT,
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
    CHAT_PENDING_KEY = "chat_pending_confirmation"
    CHAT_APPROVED_KEY = "chat_approved_mutations"

    if CHAT_TENANT_KEY not in st.session_state:
        st.session_state[CHAT_TENANT_KEY] = None

    def render_chat_tenant_form():
        st.subheader("Connect your Sisense deployment")

        mode = _auth_mode_radio("chat_auth_mode")

        with st.form("chat_tenant_form"):
            domain = st.text_input("Sisense domain", placeholder="https://your-domain.sisense.com")
            creds = _credential_inputs(mode)
            ssl = st.checkbox("Verify SSL", value=True)
            submitted = st.form_submit_button("Connect")

        if mode == AUTH_PASSWORD:
            st.caption(
                "Your password is exchanged for an API token and then discarded — "
                "it is never stored or sent anywhere else. Deployments using SSO "
                "should use an API token instead."
            )

        if submitted:
            if not domain:
                st.error("Domain is required.")
                return
            token, err = _resolve_token(_normalize_domain(domain), mode, creds, ssl)
            if err:
                st.error(err)
                return

            st.session_state[CHAT_TENANT_KEY] = {
                "domain": _normalize_domain(domain),
                "token": token,
                "ssl": ssl,
            }
            logger.info("[CHAT] Tenant configured for domain=%s, ssl=%s", domain.strip(), ssl)
            # st.toast survives the rerun; st.success here would be erased by it
            st.toast("Connected. You can now chat with your Sisense deployment.", icon="✅")
            st.rerun()

    if st.session_state[CHAT_TENANT_KEY] is None:
        with st.sidebar:
            st.subheader("Connection")
            st.write(f"Available tools: **{len(chat_tools)}**")
            st.caption(
                "Everything in this deployment — users, groups, dashboards, data "
                "models, health checks. To move assets between environments, "
                "switch to Migration mode."
            )
            st.markdown("**Mode:** Chat with deployment")
            st.markdown("---")
            st.caption("Connect your Sisense deployment to start.")
        render_chat_tenant_form()
        st.stop()

    chat_tenant_config = st.session_state[CHAT_TENANT_KEY]

    if CHAT_MESSAGES_KEY not in st.session_state:
        st.session_state[CHAT_MESSAGES_KEY] = [
            {"role": "assistant", "content": "Hi! Ask me about your Sisense deployment."},
        ]
        logger.debug("[CHAT] Chat history initialized with greeting only (system prompt handled in backend).")

    if CHAT_PENDING_KEY not in st.session_state:
        st.session_state[CHAT_PENDING_KEY] = None
    if CHAT_APPROVED_KEY not in st.session_state:
        st.session_state[CHAT_APPROVED_KEY] = set()

    with st.sidebar:
        st.subheader("Connection")
        st.write(f"Available tools: **{len(chat_tools)}**")
        st.caption(
            "Everything in this deployment — users, groups, dashboards, data "
            "models, health checks. To move assets between environments, "
            "switch to Migration mode."
        )
        st.markdown("**Mode:** Chat with deployment")

        st.markdown("**Connected tenant**")
        st.write(f"Domain: `{chat_tenant_config.get('domain', '')}`")
        st.write(f"SSL verification: `{chat_tenant_config.get('ssl', True)}`")

        # Two-click disconnect: it also deletes the chat transcript, which is
        # not what someone expects from a "disconnect" — confirm first.
        if st.button("Disconnect tenant", key="chat_disconnect"):
            st.session_state["_chat_disconnect_confirm"] = True
        if st.session_state.get("_chat_disconnect_confirm"):
            st.warning("Disconnecting also clears this chat's history.")
            _dc1, _dc2 = st.columns(2)
            if _dc1.button("Disconnect", type="primary", key="chat_disconnect_yes"):
                logger.info("[CHAT] Disconnecting tenant.")
                st.session_state[CHAT_TENANT_KEY] = None
                for key in [
                    CHAT_MESSAGES_KEY,
                    CHAT_PENDING_KEY,
                    CHAT_APPROVED_KEY,
                    "_chat_disconnect_confirm",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            if _dc2.button("Keep", key="chat_disconnect_no"):
                st.session_state["_chat_disconnect_confirm"] = False
                st.rerun()

        with st.expander("Examples", expanded=False):
            st.markdown(
                """
**Look things up**
- Show me all users
- Which dashboards use the 'ecommerce_db' datamodel?

**Ask for several things at once**
- List all datamodels, all user groups, and all folders

**Let it chain steps** *(needs "Send results to the LLM" on)*
- Which groups does jane@acme.com belong to, and who else is in them?
- Get john@acme.com's role, then list everyone with that same role

**Audit & health-check**
- Find unused columns in the 'ecommerce_db' datamodel
- Check all datamodels for many-to-many relationships

**Make changes** *(always asks for your approval first)*
- Create a user analyst@acme.com with role Viewer
- Add a table called "top_customers" in datamodel "ecommerce_db"
"""
            )

        st.markdown("---")
        st.caption(
            "Describe what you need in your own words — the assistant works out "
            "the steps, runs them against your Sisense environment, and checks "
            "the result actually answers your question. Anything that changes "
            "data always asks for your approval first."
        )

    # Render chat history inside a fixed-height scrollable box. The box is
    # load-bearing, not cosmetic: with the conversation in its own scroll
    # container and the chat input NESTED below (see _chat_input_holder),
    # Streamlit renders the input INLINE instead of as the pinned sticky
    # stBottom bar — the element Chrome kept mis-painting (ghost-bar saga,
    # 2026-08-17/18). Dialogs and the input stay visible without scrolling.
    _chat_box = st.container(height=CHAT_BOX_HEIGHT, key="chat_box")
    with _chat_box:
        for _i, msg in enumerate(st.session_state[CHAT_MESSAGES_KEY]):
            if msg.get("role") not in ("user", "assistant"):
                continue

            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    tr = msg.get("tool_result")
                    sr = msg.get("step_results")
                    if sr or tr:
                        render_results(sr, fallback_tr=tr)

                    # Chat mode: do NOT render run log (no progress emitted for these tools)

                st.markdown(msg.get("content", ""))

                # Screen-only hints (e.g. clarification option names): rendered
                # under the reply, NEVER merged into `content` — content is the
                # only thing the backend's LLM path reads from history, so this
                # is how live values stay visible in every summarization mode
                # without ever reaching the model unless the user types them.
                if msg["role"] == "assistant" and msg.get("display_hints"):
                    for _hint in msg["display_hints"]:
                        st.info(_hint)

                # Usage line + thumbs on real turns (a trace_id means a backend turn ran)
                if msg["role"] == "assistant" and msg.get("trace_id"):
                    _uc = _usage_caption(msg.get("usage"))
                    if _uc:
                        st.caption(_uc)
                    render_feedback_controls("chat", st.session_state[CHAT_MESSAGES_KEY], _i, msg)

    # Pending mutation approval UX (Chat) — the ONLY dialog renderer, rendered
    # INSIDE the conversation box right after the last message: a confirmation
    # is part of the conversation, so it sits glued under the prompt that
    # caused it (rendering it between box and input left a dead band on
    # sparse histories — seen live 2026-08-18) and the box's native
    # stick-to-bottom lands the view on it. Exactly one Approve/Cancel widget
    # pair per frame (an inline second copy used to collide on widget IDs).
    pending = st.session_state[CHAT_PENDING_KEY]
    if pending and isinstance(pending, dict):
        with _chat_box:
            st.warning(
                pending.get("reason")
                or "This action requires approval before it can make changes to your Sisense deployment."
            )
            with st.expander("View operation details", expanded=True):
                st.markdown("**Tool:** `{}`".format(pending.get("tool_id", "")))
                st.code(json.dumps(pending.get("arguments", {}), indent=2), language="json")

            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("Approve", type="primary", key="chat_approve"):
                    key = _approval_key(pending["tool_id"], pending.get("arguments", {}))
                    # Replace, never accumulate: this turn carries exactly the one
                    # approval the user just gave. The backend consumes it on use, so
                    # the same operation asked for again gates again.
                    st.session_state[CHAT_APPROVED_KEY] = {key}

                    _agent_ph = st.empty()
                    _approve_failed = False
                    with st.spinner("Running approved action..."):
                        try:
                            reply, tr, sr, tid, usage, hints = call_backend_turn(
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
                            # Put the failure in history BEFORE rerunning — st.error
                            # followed by st.rerun() is a frame the user never sees.
                            _approve_failed = True
                            reply, tr, sr, tid, usage, hints = (
                                f"The approved action failed: {e}",
                                None,
                                None,
                                None,
                                None,
                                None,
                            )
                    _agent_ph.empty()

                    # Ensure run log is not shown/stored for chat
                    st.session_state[_LAST_RUN_LOG_STATE_KEY] = None

                    st.session_state[CHAT_MESSAGES_KEY].append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "tool_result": tr,
                            "step_results": sr,
                            "run_log": None,
                            "trace_id": tid,
                            "usage": usage,
                            "display_hints": hints,
                        }
                    )

                    st.session_state[CHAT_PENDING_KEY] = None
                    st.rerun()

            with cols[1]:
                if st.button("Cancel", key="chat_cancel"):
                    st.session_state[CHAT_PENDING_KEY] = None
                    st.session_state[CHAT_MESSAGES_KEY].append({"role": "assistant", "content": "Action cancelled."})
                    st.rerun()

    # Chat input (Chat mode) — NESTED in a container so Streamlit renders it
    # INLINE below the message box, not as the pinned sticky bar (see the
    # ghost-bar note above the message box).
    with st.container():
        # A pending approval is a MODAL decision: freeze the input until the
        # user clicks Approve or Cancel. Without this, an accidental Enter
        # silently abandoned the gated action (the topic-change rule below
        # still backstops any stale state).
        _gate_open = bool(st.session_state.get(CHAT_PENDING_KEY))
        user_input = (
            st.chat_input(
                "Approve or cancel the pending action above first…" if _gate_open else "Ask something about Sisense...",
                disabled=_gate_open,
            )
            or ""
        ).strip() or None

    if user_input:
        logger.debug("[CHAT] User question: %s", user_input)

        # Typing instead of clicking answers the pending dialog with a topic
        # change: drop the stale gate so it cannot reappear under the new
        # answer and execute an operation the user has moved past.
        st.session_state[CHAT_PENDING_KEY] = None
        st.session_state[CHAT_APPROVED_KEY] = set()

        st.session_state[CHAT_MESSAGES_KEY].append({"role": "user", "content": user_input})

        with _chat_box, st.chat_message("user"):
            st.markdown(user_input)

        with _chat_box, st.chat_message("assistant"):
            # Live agentic-loop progress (step checklist + current phase).
            _agent_ph = st.empty()
            _call_failed = False
            with st.spinner("Thinking..."):
                try:
                    reply, tr, sr, tid, usage, hints = call_backend_turn(
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
                    reply = f"Error: {e}"
                    tr = None
                    sr = None  # else the PREVIOUS message's results re-render under this error
                    tid = None
                    usage = None
                    hints = None
            _agent_ph.empty()

            # Ensure run log is not shown/stored for chat
            st.session_state[_LAST_RUN_LOG_STATE_KEY] = None

            if isinstance(tr, dict) and tr.get("pending_confirmation"):
                # Store the gate and rerun into the persisted dialog — the one
                # and only Approve/Cancel renderer (an inline copy here used to
                # duplicate widget IDs). The reason is NOT also appended to
                # history: the dialog shows it while pending, and the same
                # sentence in a chat bubble AND the warning box read as a
                # rendering glitch (seen live 2026-08-18). The durable record
                # is the resolution message (result / "Action cancelled.").
                st.session_state[CHAT_PENDING_KEY] = tr["pending_confirmation"]
                st.rerun()
            else:
                st.session_state[CHAT_MESSAGES_KEY].append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "tool_result": tr,
                        "step_results": sr,
                        "run_log": None,
                        "trace_id": tid,
                        "usage": usage,
                        # Screen-only: rendered under the reply, never part of
                        # `content`, so it can't re-enter LLM prompts via history.
                        "display_hints": hints,
                    }
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
    MIG_PENDING_KEY = "migration_pending_confirmation"
    MIG_APPROVED_KEY = "migration_approved_mutations"

    if MIG_SRC_KEY not in st.session_state:
        st.session_state[MIG_SRC_KEY] = None
    if MIG_TGT_KEY not in st.session_state:
        st.session_state[MIG_TGT_KEY] = None

    def _mig_disconnect(which_key: str) -> None:
        """Drop one environment AND the conversation state: a stale approval
        dialog re-rendered against NEW credentials would migrate into the
        wrong environment (chat mode's disconnect does the same). The turn
        threading keys must go too — a surviving _mig_turn_in_progress=True
        keeps the chat input disabled with no turn left to finish it."""
        st.session_state[which_key] = None
        for key in [
            MIG_MESSAGES_KEY,
            MIG_PENDING_KEY,
            MIG_APPROVED_KEY,
            _MIG_TURN_IN_PROGRESS_KEY,
            _MIG_TURN_CTX_KEY,
            _MIG_PENDING_TURN_META_KEY,
            _LAST_RUN_LOG_STATE_KEY,
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Connect forms show only while a side is missing — once both are
    # connected they disappear (chat-mode parity) so the conversation gets
    # the vertical space; disconnect lives in the sidebar.
    if not (st.session_state[MIG_SRC_KEY] and st.session_state[MIG_TGT_KEY]):
        st.subheader("Connect source and target Sisense environments")

        cols = st.columns(2)

        with cols[0]:
            st.markdown("**Source environment**")
            src_cfg = st.session_state[MIG_SRC_KEY] or {}
            src_mode = _auth_mode_radio("mig_src_auth_mode")
            with st.form("source_form"):
                src_domain = st.text_input(
                    "Source domain", value=src_cfg.get("domain", ""), placeholder="https://source.sisense.com"
                )
                src_creds = _credential_inputs(src_mode, "Source", src_cfg.get("token", ""))
                src_ssl = st.checkbox("Verify SSL (source)", value=src_cfg.get("ssl", True))
                src_submitted = st.form_submit_button("Connect source")

            if src_submitted and not src_domain:
                st.error("Source domain is required.")
            elif src_submitted:
                src_token, src_err = _resolve_token(_normalize_domain(src_domain), src_mode, src_creds, src_ssl)
                if src_err:
                    st.error(src_err)
                else:
                    st.session_state[MIG_SRC_KEY] = {
                        "domain": _normalize_domain(src_domain),
                        "token": src_token,
                        "ssl": src_ssl,
                    }
                    logger.info("[MIGRATION] Source configured for domain=%s ssl=%s", src_domain.strip(), src_ssl)
                    st.toast("Source environment connected.", icon="✅")
                    st.rerun()

            if st.session_state[MIG_SRC_KEY] is not None:
                if st.button("Disconnect source", key="mig_disconnect_src_form"):
                    logger.info("[MIGRATION] Disconnecting source.")
                    _mig_disconnect(MIG_SRC_KEY)

        with cols[1]:
            st.markdown("**Target environment**")
            tgt_cfg = st.session_state[MIG_TGT_KEY] or {}
            tgt_mode = _auth_mode_radio("mig_tgt_auth_mode")
            with st.form("target_form"):
                tgt_domain = st.text_input(
                    "Target domain", value=tgt_cfg.get("domain", ""), placeholder="https://target.sisense.com"
                )
                tgt_creds = _credential_inputs(tgt_mode, "Target", tgt_cfg.get("token", ""))
                tgt_ssl = st.checkbox("Verify SSL (target)", value=tgt_cfg.get("ssl", True))
                tgt_submitted = st.form_submit_button("Connect target")

            if tgt_submitted and not tgt_domain:
                st.error("Target domain is required.")
            elif tgt_submitted:
                tgt_token, tgt_err = _resolve_token(_normalize_domain(tgt_domain), tgt_mode, tgt_creds, tgt_ssl)
                if tgt_err:
                    st.error(tgt_err)
                else:
                    st.session_state[MIG_TGT_KEY] = {
                        "domain": _normalize_domain(tgt_domain),
                        "token": tgt_token,
                        "ssl": tgt_ssl,
                    }
                    logger.info("[MIGRATION] Target configured for domain=%s ssl=%s", tgt_domain.strip(), tgt_ssl)
                    st.toast("Target environment connected.", icon="✅")
                    st.rerun()

            if st.session_state[MIG_TGT_KEY] is not None:
                if st.button("Disconnect target", key="mig_disconnect_tgt_form"):
                    logger.info("[MIGRATION] Disconnecting target.")
                    _mig_disconnect(MIG_TGT_KEY)

    with st.sidebar:
        st.subheader("Connection")
        st.write(f"Available tools: **{len(migration_tools)}**")
        st.markdown("**Mode:** Migrate between deployments")

        src_cfg = st.session_state[MIG_SRC_KEY]
        tgt_cfg = st.session_state[MIG_TGT_KEY]

        st.markdown("**Source**")
        if src_cfg:
            st.write(f"Domain: `{src_cfg.get('domain', '')}`")
            st.write(f"SSL verification: `{src_cfg.get('ssl', True)}`")
            if st.button("Disconnect source", key="mig_disconnect_src"):
                logger.info("[MIGRATION] Disconnecting source.")
                _mig_disconnect(MIG_SRC_KEY)
        else:
            st.write("_Not connected_")

        st.markdown("**Target**")
        if tgt_cfg:
            st.write(f"Domain: `{tgt_cfg.get('domain', '')}`")
            st.write(f"SSL verification: `{tgt_cfg.get('ssl', True)}`")
            if st.button("Disconnect target", key="mig_disconnect_tgt"):
                logger.info("[MIGRATION] Disconnecting target.")
                _mig_disconnect(MIG_TGT_KEY)
        else:
            st.write("_Not connected_")

        if src_cfg and tgt_cfg:
            with st.expander("Examples", expanded=False):
                st.markdown(
                    """
*Move specific assets*
- Migrate the "Sales Team" group and the user jane@acme.com
- Migrate dashboards "Sales Overview" and "Customer KPIs"

*Move in bulk — planned in the right order automatically*
- Migrate all users, all groups and all datamodels
- Migrate all dashboards, overwriting existing ones
"""
                )
        else:
            st.markdown("_Connect both source and target environments to see migration examples._")
        st.markdown("---")
        st.caption(
            "Describe what to move from source to target — the assistant plans "
            "every step in dependency order (groups before users, datamodels "
            "before dashboards) and shows you the full plan for approval before "
            "anything runs. A failed step stops the run instead of cascading."
        )

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
        sr = ctx.get("step_results")

        if isinstance(tr, dict) and tr.get("pending_confirmation"):
            # The dialog is the record while pending — the reason is NOT also
            # appended to history (the same sentence in a bubble AND the
            # warning box reads as a rendering glitch, seen live 2026-08-18).
            # The durable record is the resolution message.
            st.session_state[MIG_PENDING_KEY] = tr["pending_confirmation"]
        else:
            if meta.get("clear_pending"):
                st.session_state[MIG_PENDING_KEY] = None
            st.session_state[MIG_MESSAGES_KEY].append(
                {
                    "role": "assistant",
                    "content": reply,
                    "tool_result": tr,
                    "step_results": sr,
                    "run_log": run_log,
                    "trace_id": ctx.get("trace_id"),
                    "usage": ctx.get("usage"),
                }
            )

    # Fixed-height scrollable conversation box — same load-bearing role as in
    # chat mode: with the input NESTED below it renders inline, and the pinned
    # sticky bar Chrome kept mis-painting never exists.
    _mig_box = st.container(height=CHAT_BOX_HEIGHT, key="mig_box")
    with _mig_box:
        for _i, msg in enumerate(st.session_state[MIG_MESSAGES_KEY]):
            if msg.get("role") not in ("user", "assistant"):
                continue
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    tr = msg.get("tool_result")
                    sr = msg.get("step_results")
                    if sr or tr:
                        render_results(sr, fallback_tr=tr)
                    render_run_log(msg.get("run_log"))
                st.markdown(msg.get("content", ""))

                # Screen-only hints (e.g. clarification option names): rendered
                # under the reply, NEVER merged into `content` — content is the
                # only thing the backend's LLM path reads from history, so this
                # is how live values stay visible in every summarization mode
                # without ever reaching the model unless the user types them.
                if msg["role"] == "assistant" and msg.get("display_hints"):
                    for _hint in msg["display_hints"]:
                        st.info(_hint)

                # Usage line + thumbs on real turns (a trace_id means a backend turn ran)
                if msg["role"] == "assistant" and msg.get("trace_id"):
                    _uc = _usage_caption(msg.get("usage"))
                    if _uc:
                        st.caption(_uc)
                    render_feedback_controls("migration", st.session_state[MIG_MESSAGES_KEY], _i, msg)

    # One position-stable slot for the approval dialog, INSIDE the conversation
    # box right after the last message (same placement as chat mode — a
    # confirmation is part of the conversation, and the box's native
    # stick-to-bottom lands the view on it). The polling branch clears it
    # EXPLICITLY every frame: script runs that end in st.rerun() never reach
    # Streamlit's stale-element pruning, so a dialog drawn by the Approve
    # click-run otherwise lingers as ghost widgets — visible but inert —
    # below the live progress (seen live 2026-08-14, twice).
    with _mig_box:
        mig_dialog_slot = st.empty()

    # -------------------------------------------------------------------------
    # Turn-in-progress polling block
    # Shown while a background migration thread is running.
    # Calls st.rerun() to poll until done; blocks the rest of the UI meanwhile.
    # -------------------------------------------------------------------------
    if st.session_state.get(_MIG_TURN_IN_PROGRESS_KEY):
        mig_dialog_slot.empty()
        ctx = st.session_state.get(_MIG_TURN_CTX_KEY, {})

        # Inside the conversation box: the live plan/progress is conversation
        # content — outside the box it rendered below the window edge until
        # the approval dialog replaced it (seen live 2026-08-18).
        with _mig_box:
            _col_status, _col_stop = st.columns([5, 1])
            with _col_stop:
                if st.button("⏹ Stop", key="mig_stop_btn"):
                    _cancel_backend_turn(session_id)
                    st.session_state[_MIG_TURN_IN_PROGRESS_KEY] = False
                    # The approval that launched this run is spent — leaving it in
                    # state re-renders the dialog under the "stopped" message, and
                    # re-approving would run the whole migration AGAIN (seen live
                    # 2026-08-14). A stopped run ends the exchange; ask again to redo.
                    st.session_state[MIG_PENDING_KEY] = None
                    st.session_state[MIG_APPROVED_KEY] = set()
                    st.session_state[MIG_MESSAGES_KEY].append(
                        {"role": "assistant", "content": "Migration stopped by user."}
                    )
                    st.rerun()
            with _col_status:
                # The plan checklist, same as chat mode — a migration is exactly
                # where seeing the ordered steps up front matters most. Rendered
                # here on the main thread from state the worker collected.
                _render_agent_progress(
                    st.empty(),
                    ctx.get("agent_steps") or [],
                    ctx.get("agent_status"),
                    ctx.get("agent_plan"),
                )
                # Then the SDK's own per-asset progress ("migrated 12 of 40...").
                # Agent phase lines are not in here — they go to the checklist above.
                prog = ctx.get("progress_lines") or []
                if prog:
                    tail = prog[-20:]
                    st.markdown("**Progress**\n\n" + "\n".join([f"- {x}" for x in tail]))
                elif not ctx.get("agent_plan") and not ctx.get("agent_status"):
                    st.markdown("*Planning migration...*")

        if ctx.get("done"):
            _process_mig_turn_result()
        else:
            time.sleep(0.4)
        st.rerun()

    pending_mig = st.session_state[MIG_PENDING_KEY]
    # Never render the approval dialog while a migration turn is in flight:
    # Approve was already clicked, so showing live buttons invites a second
    # click from a user who thinks the run is stuck (seen live 2026-08-14 —
    # only Streamlit's rerun timing prevented a duplicate approval turn).
    if pending_mig and isinstance(pending_mig, dict) and not st.session_state.get(_MIG_TURN_IN_PROGRESS_KEY):
        with mig_dialog_slot.container():
            # st.info, not st.warning: this carries the full numbered PLAN
            # document — a wall of amber is unreadable (user call, 2026-08-18).
            # Chat's dialog keeps the warning: its reason is one line.
            st.info(
                pending_mig.get("reason")
                or "This migration action requires approval before it can make changes to your Sisense deployments."
            )
            # A whole-plan approval already lists every step and its arguments in the
            # message above, in plain English. Repeating it as JSON under a synthetic
            # tool name (`migration.plan` is not a real tool) adds nothing but noise
            # in front of a destructive action. Single-tool approvals still get it.
            if pending_mig.get("tool_id") != MIGRATION_PLAN_TOOL_ID:
                with st.expander("View operation details", expanded=True):
                    st.markdown("**Tool:** `{}`".format(pending_mig.get("tool_id", "")))
                    st.code(json.dumps(pending_mig.get("arguments", {}), indent=2), language="json")

            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("Approve", type="primary", key="mig_approve"):
                    key = _approval_key(pending_mig["tool_id"], pending_mig.get("arguments", {}))
                    # Single-use, as in chat mode — and it matters more here: a
                    # silently-repeated migration writes to the target twice.
                    st.session_state[MIG_APPROVED_KEY] = {key}
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
                if st.button("Cancel", key="mig_cancel"):
                    st.session_state[MIG_PENDING_KEY] = None
                    st.session_state[MIG_MESSAGES_KEY].append({"role": "assistant", "content": "Action cancelled."})
                    st.rerun()

    # Nested → inline input (see the conversation-box note above)
    with st.container():
        # Same modal freeze as chat mode: decide on the pending migration
        # plan before asking for anything else.
        _mig_gate_open = bool(st.session_state.get(MIG_PENDING_KEY)) and not st.session_state.get(
            _MIG_TURN_IN_PROGRESS_KEY
        )
        mig_input = st.chat_input(
            "Approve or cancel the pending migration above first…"
            if _mig_gate_open
            else "Describe what you want to migrate...",
            disabled=_mig_gate_open,
        )

    if mig_input and not st.session_state.get(_MIG_TURN_IN_PROGRESS_KEY):
        logger.debug("[MIGRATION] User request: %s", mig_input)
        # Typing instead of clicking answers the pending dialog with a topic
        # change: drop the stale gate now, so it cannot re-render under the new
        # turn's answer and fire an operation the user has moved past.
        st.session_state[MIG_PENDING_KEY] = None
        st.session_state[MIG_APPROVED_KEY] = set()
        st.session_state[MIG_MESSAGES_KEY].append({"role": "user", "content": mig_input})
        _launch_migration_turn(
            meta={"kind": "input", "clear_pending": True},
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
