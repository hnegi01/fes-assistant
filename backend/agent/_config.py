"""
backend/agent/_config.py

LLM provider configuration, logging, env helpers, and observability utilities.

What lives here:
  - LiteLLM + LangSmith wiring (_configure_langsmith_tracing)
  - Logging setup (logger, audit_logger, LOG_DIR, LLM_TRACES_PATH)
  - Env helpers (_require_env, _env_bool)
  - AWS Secrets Manager fallback for Azure OpenAI credentials
  - Mutation and summarization control flags
  - _LlmConfig dataclass and _build_llm_config() factory
  - LLM_CONFIG singleton (built at import)
  - Observability helpers: _scrub_secrets, _log_json
  - _LLM_TRACE_COLUMNS, _write_llm_trace (CSV tracing)
"""

from __future__ import annotations

import contextvars
import csv
import datetime
import json
import logging
import os
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm

# Drop unsupported params silently so the same call works across providers.
litellm.drop_params = True


def _configure_langsmith_tracing() -> None:
    """Prepare the process for LangSmith tracing when LANGSMITH_TRACING=true.

    Reporting itself is done by backend/agent/_tracing.py via the LangSmith SDK
    (a proper run tree: root agent_turn + llm/tool children). LiteLLM's bundled
    "langsmith" callback is deliberately NOT registered: it drops custom
    metadata keys and posts every call as an isolated root run, so it can
    neither group a turn's calls nor power the Threads view.
    """
    litellm.success_callback = []
    litellm.failure_callback = []
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true":
        # macOS venv Python often lacks system CA certs; point to certifi's bundle.
        try:
            import certifi

            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass


_configure_langsmith_tracing()

# Optional: boto3 for AWS Secrets Manager (Azure OpenAI credentials fallback)
try:
    import boto3
    from botocore.exceptions import ClientError

    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

_log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LLM_TRACES_PATH = LOG_DIR / "llm_traces.csv"
# Per-CALL log: one row per LLM call (route/plan/decide/verify/...), all rows of
# one turn sharing the same trace_id. Join to llm_traces.csv (per-turn) on trace_id.
LLM_CALLS_PATH = LOG_DIR / "llm_calls.csv"
TOOL_CALLS_PATH = LOG_DIR / "tool_calls.csv"


def _csv_observability_enabled() -> bool:
    """Local CSV observability (llm_traces / llm_calls / tool_calls) is opt-in:
    FES_CSV_OBSERVABILITY=true. Off by default — same closed-by-default stance
    as the LangSmith content flag, each destination with its own switch. The
    mutations audit log is NOT gated by this: audit is a requirement, not
    observability."""
    return os.getenv("FES_CSV_OBSERVABILITY", "false").strip().lower() == "true"


logger = logging.getLogger("backend.agent.llm_agent")
logger.setLevel(_log_level)
logger.propagate = False

if not any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
    _fh = TimedRotatingFileHandler(
        LOG_DIR / "llm_agent.log",
        when="midnight",  # daily file; 7 dated backups = 7 days kept, older deleted
        backupCount=7,
        encoding="utf-8",
    )
    _fh.setLevel(_log_level)
    _fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(_fh)

logger.info("llm_agent logger initialized at level %s (env %s)", _log_level_name, LOG_LEVEL_ENV_VAR)


def _make_module_logger(name: str, filename: str) -> logging.Logger:
    """Create a RotatingFileHandler logger for a sub-module (routing, registry, etc.)."""
    mod_logger = logging.getLogger(name)
    mod_logger.setLevel(_log_level)
    mod_logger.propagate = False
    if not any(isinstance(h, TimedRotatingFileHandler) for h in mod_logger.handlers):
        _fh = TimedRotatingFileHandler(
            LOG_DIR / filename,
            when="midnight",  # daily file; 7 dated backups = 7 days kept, older deleted
            backupCount=7,
            encoding="utf-8",
        )
        _fh.setLevel(_log_level)
        _fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        mod_logger.addHandler(_fh)
    mod_logger.info("%s logger initialized at level %s", name, _log_level_name)
    return mod_logger


# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------
def _require_env(name: str) -> str:
    """Read a required environment variable, raising RuntimeError if missing or empty."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# -----------------------------------------------------------------------------
# AWS Secrets Manager fallback (Azure OpenAI only)
# -----------------------------------------------------------------------------
AWS_SECRET_ID_ENV_VAR = "FES_AZURE_OPENAI_SECRET_ID"
AWS_REGION_ENV_VAR = "AWS_REGION"


def _get_azure_openai_secrets_from_aws() -> Dict[str, str]:
    """
    Fetch Azure OpenAI credentials from AWS Secrets Manager.

    Requires env: FES_AZURE_OPENAI_SECRET_ID (secret name/id), AWS_REGION.
    Expects the secret to be a JSON string with keys such as:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and optionally AZURE_OPENAI_DEPLOYMENT.
    """
    if not _BOTO3_AVAILABLE:
        raise RuntimeError("AWS Secrets Manager fallback requires boto3. Install with: pip install boto3")

    secret_id = os.getenv(AWS_SECRET_ID_ENV_VAR)
    region = os.getenv(AWS_REGION_ENV_VAR)

    if not (secret_id and region):
        raise RuntimeError(
            f"To use AWS Secrets Manager for Azure OpenAI credentials, "
            f"set {AWS_SECRET_ID_ENV_VAR} and {AWS_REGION_ENV_VAR}"
        )

    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_id)
    except ClientError as e:
        logger.exception("Failed to get secret %s from AWS Secrets Manager: %s", secret_id, e)
        raise RuntimeError(f"Failed to get Azure OpenAI secret from AWS Secrets Manager: {e}") from e

    secret_str = response.get("SecretString")
    if not secret_str:
        raise RuntimeError(f"Secret {secret_id} has no SecretString (binary secrets not supported)")

    try:
        data = json.loads(secret_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Secret {secret_id} is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError(f"Secret {secret_id} must be a JSON object")

    endpoint = (data.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    api_key = (data.get("AZURE_OPENAI_API_KEY") or "").strip()

    if not endpoint or not api_key:
        raise RuntimeError(f"Secret {secret_id} must contain AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")

    logger.info(
        "Loaded Azure OpenAI credentials from AWS Secrets Manager (secret_id=%s, region=%s)",
        secret_id,
        region,
    )

    return {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_DEPLOYMENT": (data.get("AZURE_OPENAI_DEPLOYMENT") or "").strip(),
    }


# -----------------------------------------------------------------------------
# Mutation + summarization controls
# -----------------------------------------------------------------------------
# If False, mutating tools are not included in the tool list sent to the LLM.
ALLOW_MUTATING_TOOLS: bool = True

# If True, mutating tool calls require explicit UI approval (two-phase).
REQUIRE_MUTATION_CONFIRM: bool = True

# Global hard cap: if False, tool results are never sent to the LLM for summarization.
ALLOW_SUMMARIZATION: bool = _env_bool("ALLOW_SUMMARIZATION", default=True)
logger.info("ALLOW_SUMMARIZATION=%s", ALLOW_SUMMARIZATION)

# Separate audit logger for mutations
audit_logger = logging.getLogger("backend.agent.llm_agent.mutations")
audit_logger.setLevel(_log_level)
audit_logger.propagate = False
if not any(isinstance(h, logging.FileHandler) for h in audit_logger.handlers):
    _audit_fh = logging.FileHandler(LOG_DIR / "mutations.log", encoding="utf-8")
    _audit_fh.setLevel(_log_level)
    _audit_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    audit_logger.addHandler(_audit_fh)


# -----------------------------------------------------------------------------
# LLM provider config
# -----------------------------------------------------------------------------
MAX_LLM_HTTP_RETRIES: int = int(os.getenv("LLM_HTTP_MAX_RETRIES", "3"))
LLM_HTTP_RETRY_BASE_DELAY: float = float(os.getenv("LLM_HTTP_RETRY_BASE_DELAY", "0.5"))
LLM_PLANNING_HISTORY_TURNS: int = int(os.getenv("LLM_PLANNING_HISTORY_TURNS", "5"))
CLARIFY_MAX_ATTEMPTS: int = int(os.getenv("FES_CLARIFY_MAX_ATTEMPTS", "2"))
# Step 8: hard ceiling on tool-executing iterations per agent turn. Reaching it
# returns partial progress with an "incomplete" note — never a silent stop.
MAX_AGENT_STEPS: int = int(os.getenv("FES_MAX_AGENT_STEPS", "8"))


def _cfg_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


# Independent goal checker (verify #3): before accepting a "done" answer, a
# separate adversarial LLM call re-checks the whole prompt against the results
# to catch "declared victory too early". Only the goal-completion verify is
# checked — per-step verify (schema + ok flag) is deterministic code and needs no
# second opinion. Set false to skip the extra call.
VERIFY_GOAL: bool = _cfg_flag("FES_VERIFY_GOAL", "true")
# How many times the checker may override a "done" and push the loop to continue.
# Bounds the extra cost; the step cap still applies on top.
VERIFY_MAX_RECHECKS: int = int(os.getenv("FES_VERIFY_MAX_RECHECKS", "1"))
# Plan→replan: how many times per turn the planner may REVISE the plan after
# a step's outcome shows the approach can't work (failed / not found / wrong
# kind of data). 0 disables replanning. The step cap still applies on top.
MAX_REPLANS: int = int(os.getenv("FES_MAX_REPLANS", "1"))
# Parallel fan-out: how many INDEPENDENT plan steps may execute concurrently
# (each with its own route→plan→execute pipeline; results join into the shared
# transcript). 1 disables fan-out. Concurrency downstream is bounded by the MCP
# server's PYSISENSE_MAX_CONCURRENT_READ_TOOLS semaphore.
MAX_PARALLEL_STEPS: int = int(os.getenv("FES_MAX_PARALLEL_STEPS", "3"))
# Migration turns plan every step in ONE call, sort them into dependency order in
# code, and execute in sequence (backend/agent/migration_flow.py). No migration
# tool consumes a value another produces, so the chat loop's per-step "what
# next?" calls buy nothing there. Set false to route migration through the
# reactive loop instead — a kill switch for a path that writes to live targets,
# not a mode anyone should need.
MIGRATION_SINGLE_SHOT: bool = _cfg_flag("FES_MIGRATION_SINGLE_SHOT", "true")
# Optional second pair of eyes on a migration plan: one extra LLM call asking a
# fresh reader "which asset kinds did the user request that this plan omits?",
# with one re-plan if it names any. OFF by default (2026-08-10): the root causes
# of dropped calls were fixed (dedicated planning prompt, no history), and every
# plan is shown to a human as a numbered step list before anything runs — you
# asked for three kinds and see two steps, you cancel. Turn on for unattended
# or API-driven use, where nobody is reading the dialog.
MIGRATION_COMPLETENESS_CHECK: bool = _cfg_flag("FES_MIGRATION_COMPLETENESS_CHECK", "false")


def _env_int_clamped(name: str, default: int, lo: int, hi: int) -> int:
    """Read an int env var, clamping to [lo, hi]; falls back to default if unparseable."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(lo, min(hi, int(raw.strip())))
    except ValueError:
        logger.warning("%s=%r is not an integer — using %d", name, raw, default)
        return default


# Few-shot examples on the tool-SELECTION call: how many of each tool's curated
# `user_query → arguments` examples (from the registry) to append to its
# description. 0 = none, which reproduces the pre-flag prompt byte for byte.
# Only example[0] has been curated to never invent a value the query didn't
# supply, so 1 is the vetted setting; 2-3 pull in less-vetted siblings.
# Examples go ONLY to the tool-selection call — the planner writes prose steps
# and never emits arguments, so they would be pure cost there.
TOOL_EXAMPLES_COUNT: int = _env_int_clamped("FES_TOOL_EXAMPLES", 0, 0, 3)

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "databricks").strip().lower()
logger.info("Using LLM_PROVIDER=%s", LLM_PROVIDER)


@dataclass(frozen=True)
class _LlmConfig:
    model: str  # LiteLLM model string: "azure/gpt-4o", "databricks/ep", "huggingface/..."
    api_key: str
    api_base: Optional[str]  # Provider base URL; None = LiteLLM default
    api_version: Optional[str]  # Azure legacy API version only
    timeout_seconds: float


def _build_llm_config() -> _LlmConfig:
    """
    Build LiteLLM-compatible configuration from environment variables.

    LiteLLM routes by model string prefix:
      "openai/<name>"       → OpenAI-compatible endpoint (Azure v1 style)
      "azure/<name>"        → Azure OpenAI (legacy deployment URL)
      "databricks/<name>"   → Databricks Model Serving
      "huggingface/<name>"  → HuggingFace Inference API
    """
    timeout_seconds = float(os.getenv("LLM_HTTP_TIMEOUT", "60"))

    if LLM_PROVIDER == "azure":
        az_style = os.getenv("AZURE_OPENAI_API_STYLE", "v1").strip().lower()
        az_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        az_deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
        az_api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()

        # If endpoint or api_key empty, try AWS Secrets Manager
        if not az_endpoint or not az_api_key:
            secrets = _get_azure_openai_secrets_from_aws()
            if not az_endpoint:
                az_endpoint = secrets["AZURE_OPENAI_ENDPOINT"].rstrip("/")
            if not az_api_key:
                az_api_key = secrets["AZURE_OPENAI_API_KEY"]
            if not az_deployment and secrets.get("AZURE_OPENAI_DEPLOYMENT"):
                az_deployment = secrets["AZURE_OPENAI_DEPLOYMENT"]

        if not az_endpoint or not az_api_key:
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
                "(set in .env or in AWS Secrets Manager via FES_AZURE_OPENAI_SECRET_ID and AWS_REGION)."
            )
        if not az_deployment:
            raise RuntimeError("Azure OpenAI requires AZURE_OPENAI_DEPLOYMENT (set in .env or in the AWS secret).")

        az_api_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-11-20").strip()

        if az_style == "v1":
            # Non-standard Azure route: {endpoint}/openai/v1/chat/completions
            return _LlmConfig(
                model=f"openai/{az_deployment}",
                api_key=az_api_key,
                api_base=f"{az_endpoint}/openai/v1",
                api_version=None,
                timeout_seconds=timeout_seconds,
            )
        else:
            # Standard Azure deployment URL
            return _LlmConfig(
                model=f"azure/{az_deployment}",
                api_key=az_api_key,
                api_base=az_endpoint,
                api_version=az_api_ver,
                timeout_seconds=timeout_seconds,
            )

    if LLM_PROVIDER == "databricks":
        host = _require_env("DATABRICKS_HOST").rstrip("/")
        token = _require_env("DATABRICKS_TOKEN")
        endpoint = _require_env("LLM_ENDPOINT")
        return _LlmConfig(
            model=f"databricks/{endpoint}",
            api_key=token,
            api_base=host,
            api_version=None,
            timeout_seconds=timeout_seconds,
        )

    if LLM_PROVIDER == "huggingface":
        hf_api_key = _require_env("HUGGINGFACE_API_KEY")
        hf_model = _require_env("HUGGINGFACE_MODEL")
        return _LlmConfig(
            model=f"huggingface/{hf_model}",
            api_key=hf_api_key,
            api_base=None,
            api_version=None,
            timeout_seconds=timeout_seconds,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r}. Valid: azure, databricks, huggingface")


LLM_CONFIG = _build_llm_config()


# -----------------------------------------------------------------------------
# Observability helpers (credential-safe logging + CSV tracing)
# -----------------------------------------------------------------------------
def _scrub_secrets(obj: Any) -> Any:
    """Recursively redact credential fields from dicts/lists before logging."""
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            key_l = str(k).lower()
            if key_l in (
                "token",
                "api_key",
                "api-key",
                "apikey",
                "authorization",
                "auth",
                "password",
                "passwd",
                "secret",
                "access_token",
                "refresh_token",
                "id_token",
                "bearer_token",
            ):
                cleaned[k] = "***REDACTED***"
            else:
                cleaned[k] = _scrub_secrets(v)
        return cleaned
    if isinstance(obj, list):
        return [_scrub_secrets(x) for x in obj]
    return obj


def _log_json(title: str, obj: Any) -> None:
    """Log the FULL JSON representation of obj, secrets scrubbed.

    Full by decision (2026-08-10): payloads used to be cut at 2,000 chars, which
    made incidents undiagnosable from disk — the whole reason a live turn had to
    be re-captured in-process to be read. Disk is bounded by time instead: every
    log file rotates daily and keeps 7 days. Scrubbing stays unconditional —
    completeness is for payloads, never credentials."""
    try:
        text = json.dumps(_scrub_secrets(obj), indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = repr(obj)
    logger.debug("%s:\n%s", title, text)


_LLM_TRACE_COLUMNS: List[str] = [
    "timestamp",
    "trace_id",
    "mode",
    "user_message",
    "model",
    "provider",
    "tools_available",
    "routing_module",
    "routing_latency_ms",
    "tool_selected",
    "outcome",
    "planning_tokens_in",
    "planning_tokens_out",
    "planning_latency_ms",
    "summary_tokens_in",
    "summary_tokens_out",
    "summary_latency_ms",
    "summarization_used",
    "agent_steps",  # tool-executing steps this turn
    "goal_rechecks",  # times the goal checker (verify) pushed one more step
]


def _csv_needs_header(path: Path, columns: List[str]) -> bool:
    """Return True if a header row should be written. If the file exists but its
    header no longer matches `columns` (we added/removed a column), rotate the old
    file aside to `<name>.old` so a fresh, correctly-headed file starts — instead
    of appending rows with more values than the header names."""
    if not path.exists():
        return True
    try:
        with path.open("r", encoding="utf-8") as f:
            existing = f.readline().rstrip("\r\n")
    except Exception:  # noqa: BLE001
        return False
    if existing == ",".join(columns):
        return False
    try:
        path.replace(path.with_name(path.name + ".old"))  # keep history, start fresh
    except Exception:  # noqa: BLE001
        return False
    return True


def _write_llm_trace(trace: Dict[str, Any]) -> None:
    """Append one row to llm_traces.csv. Swallows all errors — never breaks a turn."""
    if not _csv_observability_enabled():
        return
    try:
        write_header = _csv_needs_header(LLM_TRACES_PATH, _LLM_TRACE_COLUMNS)
        with LLM_TRACES_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_LLM_TRACE_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({col: trace.get(col, "") for col in _LLM_TRACE_COLUMNS})
    except Exception as exc:
        logger.warning("Failed to write LLM trace: %s", exc)


# -----------------------------------------------------------------------------
# Per-CALL tracing — one row per LLM call, grouped by turn via trace_id.
# The current turn's id + user message are stashed in a ContextVar at turn start
# so call_llm_raw can stamp every call it makes without threading them through.
# -----------------------------------------------------------------------------
_CURRENT_TURN: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar("fes_current_turn", default={})

_LLM_CALL_COLUMNS: List[str] = [
    "timestamp",
    "trace_id",  # same for every call in one turn — group by this
    "user_message",  # repeated per row, so a turn's rows are eyeball-able
    "call_type",  # route / plan / decide / verify / decompose / ...
    "model",
    "provider",
    "messages",  # count of messages sent
    "tools",  # count of tools offered
    "latency_ms",
    "tokens_in",
    "tokens_out",
    "ok",  # True / False
    "error",
]


def set_current_turn(trace_id: str, user_message: str) -> contextvars.Token:
    """Mark the turn each subsequent LLM call belongs to (for per-call tracing)."""
    return _CURRENT_TURN.set({"trace_id": trace_id or "", "user_message": (user_message or "")[:300]})


def reset_current_turn(token: contextvars.Token) -> None:
    try:
        _CURRENT_TURN.reset(token)
    except Exception:  # noqa: BLE001
        pass


def write_llm_call(
    *,
    call_type: str,
    n_messages: int,
    n_tools: int,
    latency_ms: int,
    tokens_in: int = 0,
    tokens_out: int = 0,
    ok: bool = True,
    error: str = "",
) -> None:
    """Append one row per LLM call. Reads turn id + user message from the
    ContextVar. Swallows all errors — tracing must never break a turn."""
    if not _csv_observability_enabled():
        return
    try:
        turn = _CURRENT_TURN.get()
        row = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "trace_id": turn.get("trace_id", ""),
            "user_message": turn.get("user_message", ""),
            "call_type": call_type or "unknown",
            "model": LLM_CONFIG.model,
            "provider": LLM_PROVIDER,
            "messages": n_messages,
            "tools": n_tools,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "ok": ok,
            "error": (error or "")[:300],
        }
        write_header = _csv_needs_header(LLM_CALLS_PATH, _LLM_CALL_COLUMNS)
        with LLM_CALLS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_LLM_CALL_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write LLM call trace: %s", exc)


_TOOL_CALL_COLUMNS: List[str] = [
    "timestamp",
    "trace_id",  # same for every call in one turn — group by this
    "user_message",
    "tool_id",
    "ok",  # True / False
    "count",  # rows in the result, when list-shaped
    "latency_ms",
    "mutates",
    "error",
]


def write_tool_call(
    *,
    tool_id: str,
    ok: Optional[bool],
    count: Optional[int],
    latency_ms: int,
    mutates: bool = False,
    error: str = "",
) -> None:
    """Append one row per MCP tool execution (tool_calls.csv) — the tool-side
    twin of write_llm_call. Metadata only: never result payloads."""
    if not _csv_observability_enabled():
        return
    try:
        turn = _CURRENT_TURN.get()
        row = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "trace_id": turn.get("trace_id", ""),
            "user_message": turn.get("user_message", ""),
            "tool_id": tool_id or "unknown",
            "ok": ok,
            "count": count if count is not None else "",
            "latency_ms": latency_ms,
            "mutates": mutates,
            "error": (error or "")[:300],
        }
        write_header = _csv_needs_header(TOOL_CALLS_PATH, _TOOL_CALL_COLUMNS)
        with TOOL_CALLS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_TOOL_CALL_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write tool call trace: %s", exc)
