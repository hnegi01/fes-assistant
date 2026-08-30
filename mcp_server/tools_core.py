# mcp_server/tools_core.py
#
# Core helpers for the PySisense MCP tool server.
#
# This module is responsible for:
# - Loading and normalizing the tool registry (JSON)
# - Building Sisense SDK clients from inline connection args (domain/token/ssl)
# - Routing tool_id + arguments to the correct SDK method
# - Returning a normalized payload for callers (HTTP, MCP, etc.)
# - Streaming progress events for a small set of migration tools that support an `emit` callback

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import os
import threading
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

import urllib3

# Many Sisense environments use self-signed certs; SSL verification is controlled via tool args.
# Suppress urllib3 warnings so logs remain readable for users who intentionally disable SSL verify.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------------------------
# Cancellation (per session_id)
# -----------------------------------------------------------------------------
_CANCEL_FLAGS: dict[str, threading.Event] = {}
_CANCEL_FLAGS_LOCK = threading.Lock()


def get_cancel_flag(session_id: str) -> threading.Event:
    """
    Get or create a per-session cancellation flag.

    Notes
    -----
    - Uses threading.Event so it is safe to check from worker threads (SDK code).
    """
    with _CANCEL_FLAGS_LOCK:
        flag = _CANCEL_FLAGS.get(session_id)
        if flag is None:
            flag = threading.Event()
            _CANCEL_FLAGS[session_id] = flag
        return flag


def request_cancel(session_id: str) -> None:
    """
    Signal cancellation for the given session_id.
    """
    logger.info("Cancellation requested for session_id=%s", session_id)
    get_cancel_flag(session_id).set()


def is_cancel_requested(session_id: str) -> bool:
    """
    Return True if cancel has been requested for session_id.
    """
    with _CANCEL_FLAGS_LOCK:
        flag = _CANCEL_FLAGS.get(session_id)
        return bool(flag and flag.is_set())


def release_cancel_flag(session_id: str) -> None:
    """
    Remove the stored cancel flag for the given session_id.

    Notes
    -----
    Prevents _CANCEL_FLAGS from growing forever if many sessions are created.
    """
    with _CANCEL_FLAGS_LOCK:
        _CANCEL_FLAGS.pop(session_id, None)


# -----------------------------------------------------------------------------
# Emit support detection
# -----------------------------------------------------------------------------
def _callable_supports_emit(fn: Callable[..., Any]) -> bool:
    """
    Determine whether a callable can accept an `emit` parameter.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    if "emit" in sig.parameters:
        return True

    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return False


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL_ENV_VAR = "FES_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("mcp_server")


def _setup_logger() -> None:
    """
    Configure the rotating file logger for this module.

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
            "mcp_server logger already configured at level %s (env %s)",
            log_level_name,
            LOG_LEVEL_ENV_VAR,
        )
        return

    fh = TimedRotatingFileHandler(
        LOG_DIR / "tools_core.log",
        when="midnight",  # daily file; 7 dated backups = 7 days kept, older deleted
        backupCount=7,
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(
        "mcp_server core logger initialized at level %s (env %s)",
        log_level_name,
        LOG_LEVEL_ENV_VAR,
    )


_setup_logger()


def _log_json(label: str, obj: Any) -> None:
    """Debug-log the FULL JSON, secrets scrubbed HERE — this helper used to dump
    its input raw and rely on every caller remembering to scrub first. With full
    payloads that is one forgotten call away from a token on disk, so scrubbing
    is unconditional. Disk is bounded by the 7-day rotation."""
    try:
        text = json.dumps(_scrub_secrets(obj), indent=2, default=str)
    except Exception:
        text = str(obj)
    logger.debug("%s:\n%s", label, text)


def _scrub_secrets(obj: Any) -> Any:
    """
    Redact likely secret fields from nested dict/list structures before logging.
    """
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            key_l = str(k).lower()
            if (
                "token" in key_l
                or key_l in ("api-key", "apikey")
                or "authorization" in key_l
                or key_l in ("auth", "password", "passwd", "secret")
            ):
                cleaned[k] = "***REDACTED***"
            else:
                cleaned[k] = _scrub_secrets(v)
        return cleaned

    if isinstance(obj, list):
        return [_scrub_secrets(x) for x in obj]

    return obj


# -----------------------------------------------------------------------------
# SDK debug toggle
# -----------------------------------------------------------------------------
SDK_DEBUG_ENV_VAR = "PYSISENSE_SDK_DEBUG"
_raw_sdk_debug = os.getenv(SDK_DEBUG_ENV_VAR)

_log_level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)

if _raw_sdk_debug is None:
    SDK_DEBUG = _log_level == logging.DEBUG
else:
    SDK_DEBUG = _raw_sdk_debug.strip().lower() == "true"

logger.info(
    "PYSISENSE SDK debug=%s (env %s=%r, LOG_LEVEL=%s)",
    SDK_DEBUG,
    SDK_DEBUG_ENV_VAR,
    _raw_sdk_debug,
    _log_level_name,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
REGISTRY_JSON = os.environ.get("PYSISENSE_REGISTRY_PATH", "config/tools.registry.with_examples.json")

ALLOW_MUTATIONS_ENV_VAR = "PYSISENSE_ALLOW_MUTATIONS"
ALLOW_MUTATIONS_DEFAULT = "true"


# Curated tool surface. The backend reads the same file (see
# backend/agent/_registry.py::allowed_tool_ids) but this process enforces it
# independently: the MCP server is the dispatch boundary, so a delisted tool must
# be unreachable even from a client that is not our backend.
ALLOWLIST_PATH = os.environ.get("FES_TOOL_ALLOWLIST", "config/allowed_tools.txt")


def _env_flag(name: str, default: str = "false") -> bool:
    """
    Parse a boolean-ish environment variable.
    """
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


ALLOW_MUTATIONS = _env_flag(ALLOW_MUTATIONS_ENV_VAR, ALLOW_MUTATIONS_DEFAULT)

logger.info("Config:")
logger.info("  REGISTRY_JSON      = %s", REGISTRY_JSON)
logger.info("  ALLOW_MUTATIONS    = %s (env %s)", ALLOW_MUTATIONS, ALLOW_MUTATIONS_ENV_VAR)


# Audit logger for mutating operations (separate file)
audit_logger = logging.getLogger("mcp_server.mutations")
audit_logger.setLevel(_log_level)
audit_logger.propagate = False

if not any(isinstance(h, logging.FileHandler) for h in audit_logger.handlers):
    audit_fh = logging.FileHandler(LOG_DIR / "server_mutations.log", encoding="utf-8")
    audit_fh.setLevel(_log_level)
    audit_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    audit_fh.setFormatter(audit_fmt)
    audit_logger.addHandler(audit_fh)


# -----------------------------------------------------------------------------
# Concurrency caps + thread offload helper (single-worker friendly)
# -----------------------------------------------------------------------------
MAX_CONCURRENT_MIGRATIONS_ENV_VAR = "PYSISENSE_MAX_CONCURRENT_MIGRATIONS"
MAX_CONCURRENT_READ_TOOLS_ENV_VAR = "PYSISENSE_MAX_CONCURRENT_READ_TOOLS"


def _env_int(name: str, default: int) -> int:
    """
    Parse an integer env var with a safe fallback.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        val = int(raw.strip())
        if val > 0:
            return val
        return default
    except Exception:
        return default


MAX_CONCURRENT_MIGRATIONS = _env_int(MAX_CONCURRENT_MIGRATIONS_ENV_VAR, default=1)
MAX_CONCURRENT_READ_TOOLS = _env_int(MAX_CONCURRENT_READ_TOOLS_ENV_VAR, default=5)

_MIGRATION_SEM = asyncio.Semaphore(MAX_CONCURRENT_MIGRATIONS)
_READ_SEM = asyncio.Semaphore(MAX_CONCURRENT_READ_TOOLS)

logger.info(
    "Concurrency caps: MAX_CONCURRENT_MIGRATIONS=%d (env %s), MAX_CONCURRENT_READ_TOOLS=%d (env %s)",
    MAX_CONCURRENT_MIGRATIONS,
    MAX_CONCURRENT_MIGRATIONS_ENV_VAR,
    MAX_CONCURRENT_READ_TOOLS,
    MAX_CONCURRENT_READ_TOOLS_ENV_VAR,
)


# -----------------------------------------------------------------------------
# Streaming: tools that support emit
# -----------------------------------------------------------------------------
STREAMING_TOOL_IDS = {
    "migration.migrate_all_groups",
    "migration.migrate_all_users",
    "migration.migrate_dashboards",
    "migration.migrate_all_dashboards",
    "migration.migrate_datamodels",
    "migration.migrate_all_datamodels",
}

_STREAM_SENTINEL = object()


# -----------------------------------------------------------------------------
# SDK imports / init
# -----------------------------------------------------------------------------
try:
    from pysisense import *  # noqa: F401, F403 — brings in all facade classes from __all__
    from pysisense import SisenseClient  # explicit re-import for static analysis
except Exception as exc:
    logger.exception("Failed to import pysisense SDK")
    raise RuntimeError(f"Failed to import pysisense SDK: {exc}") from exc

logger.info("pysisense SDK imported successfully. Clients will be created from inline connection at runtime.")

# Maps registry module key → facade class, derived from the SDK itself.
#
# pysisense >= 1.1 exports FACADES — the explicit tuple of tool-bearing classes
# — so a new SDK package dispatches without editing this file. The previous
# hardcoded map silently stranded new packages: report_manager shipped in
# 1.1.0, passed the allowlist, and would have failed here with "unsupported
# module" (found live 2026-08-29). Exposure is still gated entirely by
# config/allowed_tools.txt; this only decides whether an ALREADY-allowed tool
# can be dispatched.
#
# Excluded deliberately:
#   SisenseClient — infrastructure, not a facade
#   migration     — different constructor (source_client/target_client), handled
#                   on its own path below
#   mergetool     — duplicates the migration surface; kept off the agent
_EXCLUDED_FACADES = {"SisenseClient", "Migration", "MergeTool"}


def _discover_module_classes() -> Dict[str, type]:
    import inspect as _inspect

    import pysisense as _pysisense

    facades = getattr(_pysisense, "FACADES", None)
    candidates = (
        list(facades) if facades else [getattr(_pysisense, n, None) for n in getattr(_pysisense, "__all__", [])]
    )
    out: Dict[str, type] = {}
    for obj in candidates:
        if not _inspect.isclass(obj) or obj.__name__ in _EXCLUDED_FACADES:
            continue
        mod = (getattr(obj, "__module__", "") or "").split(".")
        key = mod[1] if len(mod) >= 2 and mod[0] == "pysisense" else obj.__name__.lower()
        out[key] = obj
    return out


_MODULE_CLASSES: Dict[str, type] = _discover_module_classes()

SUPPORTED_MODULES = sorted([*_MODULE_CLASSES.keys(), "migration"])


# -----------------------------------------------------------------------------
# Registry load / normalize
# -----------------------------------------------------------------------------
def _normalize_parameters_schema(raw: Any) -> Dict[str, Any]:
    """
    Normalize tool parameters schema.
    """
    if raw is None:
        return {"type": "object", "properties": {}, "required": []}

    if isinstance(raw, dict):
        schema = copy.deepcopy(raw)
    else:
        schema = {"type": "object", "properties": {}, "required": []}

    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])

    if not isinstance(schema["properties"], dict):
        schema["properties"] = {}
    if not isinstance(schema["required"], list):
        schema["required"] = []

    return schema


def _load_registry(path: str) -> List[Dict[str, Any]]:
    """
    Load the registry JSON file.
    """
    logger.info("Loading tool registry from %s", path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError as exc:
        logger.exception("Registry file not found")
        raise RuntimeError(f"Registry file not found: {path}. Generate it before starting the server.") from exc
    except Exception as exc:
        logger.exception("Failed to load registry JSON")
        raise RuntimeError(f"Failed to load registry JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError(f"Registry JSON must be a list. Got: {type(payload).__name__}")

    return payload


def _load_allowlist(path: str) -> Optional[Set[str]]:
    """
    Read the curated tool allowlist: one tool_id per line, '#' starts a comment.

    Returns None when the file is absent — meaning no allowlist is in force and
    every registry tool is exposed. A missing file must not silently empty the
    tool surface, so "absent" is allow-all, not deny-all.
    """
    p = Path(path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    if not p.exists():
        logger.warning(
            "Tool allowlist not found at %s — ALL registry tools are exposed. "
            "Create it with: python scripts/04_generate_tool_allowlist.py --init",
            p,
        )
        return None
    try:
        ids = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                ids.add(entry)
        logger.info("Loaded tool allowlist: %d tool(s) permitted (path=%s)", len(ids), p)
        return ids
    except Exception:
        logger.exception("Failed to read tool allowlist %s — allowing all tools", p)
        return None


REGISTRY = _load_registry(REGISTRY_JSON)
ALLOWED_TOOL_IDS = _load_allowlist(ALLOWLIST_PATH)

TOOLS_BY_ID: Dict[str, Dict[str, Any]] = {}
_skipped_missing = 0
_skipped_allowlist = 0

for row in REGISTRY:
    tool_id = row.get("tool_id")
    module = row.get("module")
    method = row.get("method")

    if not tool_id or not module or not method:
        _skipped_missing += 1
        continue

    if ALLOWED_TOOL_IDS is not None and tool_id not in ALLOWED_TOOL_IDS:
        _skipped_allowlist += 1
        continue

    normalized = dict(row)
    normalized["mutates"] = bool(normalized.get("mutates", False))

    params = normalized.get("parameters")
    if params is None and "parameters_json" in normalized:
        try:
            params = json.loads(normalized["parameters_json"])
        except Exception:
            params = None

    normalized["parameters"] = _normalize_parameters_schema(params)
    TOOLS_BY_ID[tool_id] = normalized

logger.info("Registry summary:")
logger.info("  Total rows in JSON      : %d", len(REGISTRY))
logger.info("  Loaded into TOOLS_BY_ID : %d", len(TOOLS_BY_ID))
logger.info("  Skipped (missing fields): %d", _skipped_missing)
logger.info("  Skipped (allowlist)     : %d", _skipped_allowlist)


# -----------------------------------------------------------------------------
# Small helpers for readability
# -----------------------------------------------------------------------------
def _one_liner(text: Optional[str]) -> str:
    """
    Return only the first line of a description for compact listings.
    """
    if not text:
        return "No description."
    return text.strip().splitlines()[0]


def _bucket_for_tool(tool_id: str) -> str:
    """
    Classify a tool as 'migration' vs 'read' for concurrency throttling.
    """
    meta = TOOLS_BY_ID.get(tool_id)
    module = meta.get("module") if isinstance(meta, dict) else None
    if module == "migration":
        return "migration"
    return "read"


def _coerce_json_strings(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce JSON-looking strings (e.g., '{"a":1}', '[1,2]') into Python objects.
    """
    coerced: Dict[str, Any] = {}
    for k, v in (arguments or {}).items():
        if isinstance(v, str):
            vs = v.strip()
            if (vs.startswith("{") and vs.endswith("}")) or (vs.startswith("[") and vs.endswith("]")):
                try:
                    coerced[k] = json.loads(vs)
                    continue
                except Exception:
                    pass
        coerced[k] = v
    return coerced


# -----------------------------------------------------------------------------
# MULTITENANT: per-request tenant helpers
# -----------------------------------------------------------------------------
def _extract_tenant_from_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pop and return tenant fields from arguments for single-tenant tools.
    """
    domain = arguments.pop("domain", None)
    token = arguments.pop("token", None)
    ssl = arguments.pop("ssl", None)

    if ssl is None:
        ssl = True

    # Credentials come from the caller (the backend injects the UI's sidebar
    # config) — NEVER from env. An env fallback existed for credential-less
    # clients (Claude Desktop); removed 2026-08-14: with the assistant as the
    # only client, a fallback just turns "creds missing" into a silent write
    # against whatever the env last pointed at.
    if domain and token:
        logger.info("Using tenant connection: domain=%s ssl=%s", domain, ssl)
        return {"domain": domain, "token": token, "ssl": ssl}

    logger.error("Missing tenant domain/token. Pass domain/token in tool args.")
    raise RuntimeError("Tenant domain and token are required for SisenseClient.from_connection.")


def _extract_migration_tenants_from_arguments(arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Pop and return source/target tenant fields from arguments for migration tools.
    """
    _MISSING = object()

    src_domain = arguments.pop("source_domain", None)
    src_token = arguments.pop("source_token", None)
    src_ssl_raw = arguments.pop("source_ssl", _MISSING)

    tgt_domain = arguments.pop("target_domain", None)
    tgt_token = arguments.pop("target_token", None)
    tgt_ssl_raw = arguments.pop("target_ssl", _MISSING)

    # Respect explicit caller values. Source/target credentials come from the
    # caller only — the env fallback (PYSISENSE_USE_DEFAULT_MIGRATION_TENANTS)
    # was removed 2026-08-14: a silent default TARGET is where migration
    # writes land when config is missing, which is exactly the wrong failure mode.
    src_ssl = True if src_ssl_raw is _MISSING else bool(src_ssl_raw)
    tgt_ssl = True if tgt_ssl_raw is _MISSING else bool(tgt_ssl_raw)

    src = {"domain": src_domain, "token": src_token, "ssl": src_ssl}
    tgt = {"domain": tgt_domain, "token": tgt_token, "ssl": tgt_ssl}

    if not (src["domain"] and src["token"]):
        raise RuntimeError("Migration tools require source_domain and source_token to be provided.")
    if not (tgt["domain"] and tgt["token"]):
        raise RuntimeError("Migration tools require target_domain and target_token to be provided.")

    logger.info(
        "Using migration connections: source_domain=%s source_ssl=%s | target_domain=%s target_ssl=%s",
        src["domain"],
        src["ssl"],
        tgt["domain"],
        tgt["ssl"],
    )
    return src, tgt


def _build_sisense_client(tenant: Dict[str, Any]) -> "SisenseClient":
    """
    Build a SisenseClient from tenant dict.
    """
    if not tenant.get("domain") or not tenant.get("token"):
        raise RuntimeError("Tenant domain and token are required for SisenseClient.")
    return SisenseClient.from_connection(
        domain=tenant["domain"],
        token=tenant["token"],
        is_ssl=tenant.get("ssl", True),
        debug=SDK_DEBUG,
    )


def _get_module_instance(module: str, tenant: Dict[str, Any]) -> Any:
    """
    Return an SDK module instance for the requested module.
    """
    klass = _MODULE_CLASSES.get(module)
    if klass is None:
        raise LookupError(f"Module '{module}' not recognized.")
    client = _build_sisense_client(tenant)
    return klass(api_client=client)


# -----------------------------------------------------------------------------
# Core dispatcher
# -----------------------------------------------------------------------------
def _resolve_sdk_callable(
    tool_id: str, arguments: Dict[str, Any]
) -> Tuple[Callable[..., Any], Dict[str, Any], Dict[str, Any]]:
    """
    Resolve tool_id to an SDK callable and prepared arguments.
    """
    meta = TOOLS_BY_ID.get(tool_id)
    if not meta:
        raise ValueError(f"Unknown tool_id: {tool_id}")

    if meta.get("mutates") and not ALLOW_MUTATIONS:
        raise PermissionError(f"Tool '{tool_id}' is mutating and ALLOW_MUTATIONS is false.")

    module = meta["module"]
    method = meta["method"]

    if module == "migration":
        src_tenant, tgt_tenant = _extract_migration_tenants_from_arguments(arguments)
        src_client = _build_sisense_client(src_tenant)
        tgt_client = _build_sisense_client(tgt_tenant)
        instance = Migration(source_client=src_client, target_client=tgt_client, debug=SDK_DEBUG)  # noqa: F405
    else:
        tenant = _extract_tenant_from_arguments(arguments)
        instance = _get_module_instance(module, tenant)

    if not hasattr(instance, method):
        raise LookupError(f"Method '{module}.{method}' not found on SDK instance.")

    schema = meta.get("parameters", {})
    required = schema.get("required", [])
    missing = [k for k in required if k not in arguments or arguments.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Missing required arguments: {missing}")

    coerced = _coerce_json_strings(arguments)
    coerced.pop("emit", None)  # safety: never forward client-supplied emit

    func = getattr(instance, method)
    return func, meta, coerced


# Keys an SDK failure envelope may carry alongside "error". Deliberately a
# CLOSED set rather than a bare `"error" in candidate`: a legitimate data row
# can carry an incidental error field (a per-item build status, a wellcheck
# finding), and reading one of those as a failed call would turn a good result
# into a false failure — the opposite of the bug this guards.
_ERROR_ENVELOPE_KEYS = {"error", "failed_references", "error_type", "status_code"}


def _sdk_error_payload(tool_id: str, result: Any) -> Optional[Dict[str, Any]]:
    """
    SDK methods that fail return {"error": "..."} instead of raising — sometimes
    wrapped in a single-item list ([{"error": "..."}]) by list-returning methods,
    and sometimes with extra detail beside the message
    ({"error": ..., "failed_references": [...]}). Detect all of these and
    normalise to ok=False so callers don't have to inspect the result themselves.

    The match was once `list(keys) == ["error"]` — an exact SOLE-key test, which
    silently stopped recognising failures the moment the SDK enriched the
    envelope. get_unused_columns_bulk (pysisense, 2026-08-29) is the case that
    found it: asked about a data model that does not exist, it returns
    {"error": "None of the given data model references could be processed …",
    "failed_references": [...]} — two keys, so the old test returned None and the
    call was reported as `ok: true`.

    That is worst with summarization OFF, which is the production default: the
    payload never reaches the LLM, so it sees `ok: true` with no count and
    reports a successful lookup of a data model that isn't there. The whole
    point of the envelope is to announce the failure; matching it loosely enough
    to actually catch it is what makes `ok` trustworthy downstream.
    """
    candidate = result
    if isinstance(result, list) and len(result) == 1:
        candidate = result[0]
    if isinstance(candidate, dict) and candidate.get("error") and set(candidate) <= _ERROR_ENVELOPE_KEYS:
        return {"tool_id": tool_id, "ok": False, "error": candidate["error"], "error_type": "SDKError"}
    return None


def _add_unused_columns_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add summary stats for access.get_unused_columns result payloads.
    """
    result = payload.get("result")
    if not isinstance(result, list):
        return payload

    total_columns = len(result)
    used_count = 0
    unused_count = 0

    for row in result:
        if not isinstance(row, dict):
            continue
        if row.get("used") is True:
            used_count += 1
            continue
        if row.get("used") is False:
            unused_count += 1
            continue

    payload["total_columns"] = total_columns
    payload["used_count"] = used_count
    payload["unused_count"] = unused_count
    return payload


def invoke_tool(
    tool_id: str,
    arguments: Optional[Dict[str, Any]] = None,
    emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Invoke a tool synchronously and return a normalized payload.

    When `emit` is given and the SDK method supports it, it is injected as the
    progress callback. (This absorbed invoke_tool_with_emit, 2026-08-17 — the
    two bodies were ~60 duplicated lines apart from the injection block.)
    """
    try:
        safe_args = copy.deepcopy(arguments or {})
    except Exception:
        safe_args = dict(arguments or {})

    logger.info("Dispatching tool call: tool_id=%s emit=%s", tool_id, emit is not None)
    _log_json("Incoming arguments (scrubbed)", _scrub_secrets(safe_args))

    try:
        func, meta, coerced = _resolve_sdk_callable(tool_id, safe_args)

        if emit is not None and _callable_supports_emit(func):
            coerced["emit"] = emit
        elif emit is not None and tool_id in STREAMING_TOOL_IDS:
            with contextlib.suppress(Exception):
                emit(
                    {
                        "type": "warning",
                        "step": "emit",
                        "message": f"Tool '{tool_id}' does not accept emit; running without SDK progress callbacks.",
                    }
                )

        if meta.get("mutates"):
            audit_logger.info(
                "EXECUTING mutation tool=%s args=%s",
                tool_id,
                json.dumps(_scrub_secrets(coerced), default=str),
            )

        result = func(**coerced)
        _log_json("SDK method result (full, scrubbed)", result)

        err = _sdk_error_payload(tool_id, result)
        if err:
            return err

        payload: Dict[str, Any] = {"tool_id": tool_id, "ok": True, "result": result}

        # The one sanctioned per-tool special case: this tool's answer IS a
        # count, and the LLM-bound payload truncates long lists, so the counts
        # must be precomputed or the model cannot answer. Evaluated for
        # relocation 2026-08-17 and kept: moving it changes the UI-visible
        # payload for no structural gain now that there is a single call site.
        if tool_id == "access_management.get_unused_columns":
            return _add_unused_columns_summary(payload)

        return payload

    except asyncio.CancelledError:
        return {"tool_id": tool_id, "ok": False, "error": "Cancelled", "error_type": "CancelledError"}

    except TypeError as te:
        try:
            expected = str(inspect.signature(func))  # type: ignore[name-defined]
        except Exception:
            expected = None

        msg = f"Argument error: {te}"
        if expected:
            msg += f" | expected signature: {expected}"

        return {"tool_id": tool_id, "ok": False, "error": msg, "error_type": "TypeError"}

    except Exception as exc:
        return {"tool_id": tool_id, "ok": False, "error": str(exc), "error_type": type(exc).__name__}


async def invoke_tool_async(
    tool_id: str,
    arguments: Optional[Dict[str, Any]] = None,
    emit_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    *,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Invoke a tool on a worker thread and return a normalized payload.

    session_id
        If provided, allows best-effort cancellation before starting execution.
    """
    if session_id and is_cancel_requested(session_id):
        return {"tool_id": tool_id, "ok": False, "error": "Cancelled", "error_type": "CancelledError"}

    bucket = _bucket_for_tool(tool_id)
    sem = _MIGRATION_SEM if bucket == "migration" else _READ_SEM

    try:
        safe_args = copy.deepcopy(arguments or {})
    except Exception:
        safe_args = dict(arguments or {})

    safe_args.pop("emit", None)

    async with sem:
        if emit_cb is not None and tool_id in STREAMING_TOOL_IDS:
            return await asyncio.to_thread(invoke_tool, tool_id, safe_args, emit_cb)

        return await asyncio.to_thread(invoke_tool, tool_id, safe_args)


async def invoke_tool_stream_async(
    tool_id: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream tool execution events.

    Notes
    -----
    Cancellation is best-effort:
    - If the underlying SDK/migration code calls emit periodically, cancellation
      will stop quickly because emit will raise once cancel is requested.
    - If the SDK code does not call emit often (or is stuck in a single long
      blocking request), cancellation may be delayed.
    """
    bucket = _bucket_for_tool(tool_id)
    sem = _MIGRATION_SEM if bucket == "migration" else _READ_SEM

    try:
        safe_args = copy.deepcopy(arguments or {})
    except Exception:
        safe_args = dict(arguments or {})

    safe_args.pop("emit", None)

    cancel_flag: Optional["threading.Event"] = None
    if session_id:
        cancel_flag = get_cancel_flag(session_id)

    async with sem:
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Any]" = asyncio.Queue()

        def _emit(event: Any) -> None:
            """
            Thread-safe emit callback injected into the SDK method.

            Cancellation behavior
            ---------------------
            If cancel is requested, raise CancelledError. This typically stops
            the migration loop at the next emit call.
            """
            if cancel_flag is not None and cancel_flag.is_set():
                raise asyncio.CancelledError("Cancellation requested.")

            try:
                if isinstance(event, dict):
                    payload = dict(event)
                else:
                    payload = {"type": "progress", "message": str(event)}

                payload.setdefault("tool_id", tool_id)
                loop.call_soon_threadsafe(q.put_nowait, payload)
            except Exception:
                return

        def _finish(final_payload: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(q.put_nowait, {"type": "final", "tool_id": tool_id, "payload": final_payload})
            loop.call_soon_threadsafe(q.put_nowait, _STREAM_SENTINEL)

        async def _runner() -> None:
            try:
                if tool_id in STREAMING_TOOL_IDS:
                    final_payload = await asyncio.to_thread(invoke_tool, tool_id, safe_args, _emit)
                else:
                    final_payload = await asyncio.to_thread(invoke_tool, tool_id, safe_args)
                _finish(final_payload)

            except asyncio.CancelledError:
                _finish({"tool_id": tool_id, "ok": False, "error": "Cancelled", "error_type": "CancelledError"})

            except Exception as exc:
                _finish({"tool_id": tool_id, "ok": False, "error": str(exc), "error_type": type(exc).__name__})

        task = asyncio.create_task(_runner())

        try:
            while True:
                item = await q.get()
                if item is _STREAM_SENTINEL:
                    break
                if isinstance(item, dict):
                    yield item
        finally:
            with contextlib.suppress(Exception):
                await task
            if session_id:
                release_cancel_flag(session_id)


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------
def health_summary() -> Dict[str, Any]:
    """
    Return a small health summary for the server.
    """
    return {
        "ok": True,
        "modules": sorted(SUPPORTED_MODULES),
        "tools": len(TOOLS_BY_ID),
        "mutations_allowed": ALLOW_MUTATIONS,
        "registry_path": str(Path(REGISTRY_JSON).resolve()),
    }


def list_tools(module: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List tools from the loaded registry.
    """
    out: List[Dict[str, Any]] = []
    for tool_id, row in TOOLS_BY_ID.items():
        if module and row.get("module") != module:
            continue
        if tag and tag not in (row.get("tags") or []):
            continue

        out.append(
            {
                "tool_id": tool_id,
                "module": row.get("module"),
                "class": row.get("class"),
                "method": row.get("method"),
                "description": _one_liner(row.get("description")),
                "parameters": row.get("parameters"),
                "mutates": bool(row.get("mutates", False)),
                "tags": row.get("tags"),
                "examples": row.get("examples"),
                "sdk_version": row.get("sdk_version"),
                "updated_at": row.get("updated_at"),
            }
        )

    return out
