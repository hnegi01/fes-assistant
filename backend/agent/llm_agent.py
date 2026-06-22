"""
backend/agent/llm_agent.py

High-level LLM + tools orchestration for the FES Assistant.

What this module does
---------------------
1) Loads the tool registry JSON and exposes OpenAI-style tool definitions to the LLM.
2) Calls the LLM in "planning" mode to select the best tool + JSON arguments.
3) Executes selected tools via the MCP server (McpClient).
4) Optionally calls the LLM again to summarize tool results ("summarization mode").
5) Enforces mutation approvals (two-phase execution) and supports a privacy mode
   where tool results are NOT sent to the LLM for summarization.

Key Concepts
------------
- TOOL_REGISTRY: tool_id -> metadata (module, mutates, description, schema, etc.)
- LAST_TOOL_RESULT: last executed tool payload (for the API layer/UI to display)
- Mutations: tools marked "mutates" can be gated behind UI approval.
- Summarization: can be globally disabled and/or disabled per turn by the API layer.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema
import litellm

from .mcp_client import McpClient

# Drop unsupported params silently so the same call works across providers.
litellm.drop_params = True


def _configure_langsmith_tracing() -> None:
    """Wire LiteLLM → LangSmith when LANGSMITH_TRACING=true.

    Extracted into a function so tests can call it directly with monkeypatched
    env vars without reloading the module.
    """
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true":
        # macOS venv Python often lacks system CA certs; point to certifi's bundle.
        try:
            import certifi

            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass

        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]
    else:
        litellm.success_callback = []
        litellm.failure_callback = []


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

logger = logging.getLogger("backend.agent.llm_agent")
logger.setLevel(_log_level)
logger.propagate = False

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    _fh = RotatingFileHandler(
        LOG_DIR / "llm_agent.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    _fh.setLevel(_log_level)
    _fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(_fh)

logger.info("llm_agent logger initialized at level %s (env %s)", _log_level_name, LOG_LEVEL_ENV_VAR)


# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------
def _require_env(name: str) -> str:
    """
    Read a required environment variable.

    Parameters
    ----------
    name:
        Environment variable name.

    Returns
    -------
    str
        Value of the env var.

    Raises
    ------
    RuntimeError
        If the env var is missing or empty.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


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

    Returns
    -------
    dict
        At least "AZURE_OPENAI_ENDPOINT" and "AZURE_OPENAI_API_KEY";
        may include "AZURE_OPENAI_DEPLOYMENT".

    Raises
    ------
    RuntimeError
        If boto3 is missing, required env vars are missing, or the secret cannot be read.
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


def _env_bool(name: str, default: bool) -> bool:
    """
    Read a boolean environment variable.

    Parameters
    ----------
    name:
        Environment variable name.
    default:
        Value returned if env var is missing or blank.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


# -----------------------------------------------------------------------------
# Mutation + summarization controls (agent-level)
# -----------------------------------------------------------------------------
# If False, mutating tools are not included in the tool list sent to the LLM.
ALLOW_MUTATING_TOOLS: bool = True

# If True, mutating tool calls require explicit UI approval (two-phase).
REQUIRE_MUTATION_CONFIRM: bool = True

# Global hard cap: if False, tool results are never sent to the LLM for summarization.
# The API layer can further restrict summarization per turn.
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
# LLM provider config (azure | databricks)
# -----------------------------------------------------------------------------
MAX_LLM_HTTP_RETRIES: int = int(os.getenv("LLM_HTTP_MAX_RETRIES", "3"))
LLM_HTTP_RETRY_BASE_DELAY: float = float(os.getenv("LLM_HTTP_RETRY_BASE_DELAY", "0.5"))

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

    Returns
    -------
    _LlmConfig
        Fully resolved configuration.
    """
    timeout_seconds = float(os.getenv("LLM_HTTP_TIMEOUT", "60"))

    if LLM_PROVIDER == "azure":
        az_style = os.getenv("AZURE_OPENAI_API_STYLE", "v1").strip().lower()  # v1 | legacy
        az_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        az_deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
        az_api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()

        # If endpoint or api_key empty, try AWS Secrets Manager (requires AWS_REGION + FES_AZURE_OPENAI_SECRET_ID)
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
            # Non-standard Azure route: {endpoint}/openai/v1/chat/completions (requires "model" field).
            # Treat as a generic OpenAI-compatible endpoint — LiteLLM appends /chat/completions.
            return _LlmConfig(
                model=f"openai/{az_deployment}",
                api_key=az_api_key,
                api_base=f"{az_endpoint}/openai/v1",
                api_version=None,
                timeout_seconds=timeout_seconds,
            )
        else:
            # Standard Azure deployment URL.
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
        # LiteLLM constructs the full invocations URL from api_base + model suffix.
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
# Registry + globals
# -----------------------------------------------------------------------------
_registry_env = os.getenv("PYSISENSE_REGISTRY_PATH")
REGISTRY_PATH = Path(_registry_env) if _registry_env else (ROOT_DIR / "config" / "tools.registry.with_examples.json")

# Public globals used by the API layer
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
LAST_TOOL_RESULT: Optional[Dict[str, Any]] = None

# Internal cache so we don't re-read JSON on every request
_registry_cache_mtime: Optional[float] = None
_registry_cache_rows: List[Dict[str, Any]] = []


def _scrub_secrets(obj: Any) -> Any:
    """Recursively redact credential fields from dicts/lists before logging."""
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            key_l = str(k).lower()
            if (
                "token" in key_l
                or "api_key" in key_l
                or key_l in ("api-key", "apikey", "authorization", "auth", "password", "passwd", "secret")
            ):
                cleaned[k] = "***REDACTED***"
            else:
                cleaned[k] = _scrub_secrets(v)
        return cleaned
    if isinstance(obj, list):
        return [_scrub_secrets(x) for x in obj]
    return obj


_LLM_TRACE_COLUMNS = [
    "timestamp",
    "trace_id",
    "mode",
    "user_message",
    "model",
    "provider",
    "tools_available",
    "tool_selected",
    "outcome",
    "planning_tokens_in",
    "planning_tokens_out",
    "planning_latency_ms",
    "summary_tokens_in",
    "summary_tokens_out",
    "summary_latency_ms",
    "summarization_used",
]


def _write_llm_trace(trace: Dict[str, Any]) -> None:
    """Append one row to llm_traces.csv. Swallows all errors — never breaks a turn."""
    try:
        write_header = not LLM_TRACES_PATH.exists()
        with LLM_TRACES_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_LLM_TRACE_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({col: trace.get(col, "") for col in _LLM_TRACE_COLUMNS})
    except Exception as exc:
        logger.warning("Failed to write LLM trace: %s", exc)


def _log_json_truncated(title: str, obj: Any, max_len: int = 2000) -> None:
    """
    Log a JSON representation of obj, truncated for readability.

    Parameters
    ----------
    title:
        Label for the log entry.
    obj:
        Object to serialize and log.
    max_len:
        Maximum number of characters to log.
    """
    try:
        text = json.dumps(_scrub_secrets(obj), indent=2, ensure_ascii=False, default=str)
    except Exception:
        text = repr(obj)
    if len(text) > max_len:
        text = text[:max_len] + "... [truncated]"
    logger.debug("%s:\n%s", title, text)


def _load_registry_rows() -> List[Dict[str, Any]]:
    """
    Load tool registry JSON from disk with a simple mtime cache.

    Returns
    -------
    list[dict]
        Raw registry rows. Returns [] if the file is missing or invalid.
    """
    global _registry_cache_mtime, _registry_cache_rows

    if not REGISTRY_PATH.exists():
        logger.warning("Tool registry not found at %s", REGISTRY_PATH.resolve())
        _registry_cache_mtime = None
        _registry_cache_rows = []
        return []

    try:
        mtime = REGISTRY_PATH.stat().st_mtime
    except Exception:
        mtime = None

    if mtime is not None and _registry_cache_mtime == mtime and _registry_cache_rows:
        return list(_registry_cache_rows)

    try:
        raw = REGISTRY_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            logger.error("Registry JSON is not a list (path=%s)", REGISTRY_PATH.resolve())
            _registry_cache_rows = []
            _registry_cache_mtime = mtime
            return []
        _registry_cache_rows = data
        _registry_cache_mtime = mtime
        logger.info("Loaded registry with %d entries (path=%s)", len(data), REGISTRY_PATH.resolve())
        return list(data)
    except Exception as exc:
        logger.exception("Failed to load registry JSON: %s", exc)
        _registry_cache_rows = []
        _registry_cache_mtime = mtime
        return []


def load_tools_for_llm() -> List[Dict[str, Any]]:
    """
    Load tools from the registry and convert them to OpenAI-style tool definitions.

    Returns
    -------
    list[dict]
        OpenAI-style tools list.
    """
    global TOOL_REGISTRY

    rows = _load_registry_rows()
    if not rows:
        TOOL_REGISTRY = {}
        logger.warning("Registry empty; no tools available to LLM.")
        return []

    registry_by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        tid = row.get("tool_id")
        if tid:
            registry_by_id[tid] = row

    TOOL_REGISTRY = registry_by_id
    logger.info("TOOL_REGISTRY populated with %d tools", len(TOOL_REGISTRY))

    tools: List[Dict[str, Any]] = []
    skipped_mutating: List[str] = []

    for tid, meta in registry_by_id.items():
        mutates = bool(meta.get("mutates", False))
        if mutates and not ALLOW_MUTATING_TOOLS:
            skipped_mutating.append(tid)
            continue

        params = meta.get("parameters") or {}
        desc = meta.get("description") or ""
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tid,
                    "description": desc,
                    "parameters": params,
                },
            }
        )

    if skipped_mutating:
        logger.info("Mutating tools hidden (ALLOW_MUTATING_TOOLS=False): %s", skipped_mutating)

    logger.info("Tools loaded from registry: %d", len(tools))
    return tools


# -----------------------------------------------------------------------------
# Local result description (used when summarization is disabled)
# -----------------------------------------------------------------------------
def _describe_tool_result(tool_name: str, result: Optional[Dict[str, Any]]) -> str:
    """
    Generate a human-readable description of a tool result without calling the LLM.
    Used when ALLOW_SUMMARIZATION is off so the user gets something useful instead
    of the generic "Summarization disabled by configuration" message.
    """
    if not result or not isinstance(result, dict):
        return f"`{tool_name}` ran. Result shown above."

    # MCP-level failure
    if not result.get("ok", True):
        return f"`{tool_name}` failed. Details shown above."

    data = result.get("result")

    # API-level error embedded in the result payload
    if isinstance(data, dict) and "error" in data:
        return f"`{tool_name}` returned an error — {data['error']}"

    # List result
    if isinstance(data, list):
        n = len(data)
        if n == 0:
            return f"`{tool_name}` returned no results."
        noun = "result" if n == 1 else "results"
        return f"Found {n} {noun} from `{tool_name}`. Results shown above."

    # Single object result
    if isinstance(data, dict):
        return f"Got a response from `{tool_name}`. Details shown above."

    # Primitive or None
    return f"`{tool_name}` completed. Result shown above."


# -----------------------------------------------------------------------------
# Generic payload shrinker for LLM summarization
# -----------------------------------------------------------------------------
MAX_LIST_ITEMS_FOR_LLM = 20
MAX_KEYS_PER_OBJECT_FOR_LLM = 10
MAX_DEPTH_FOR_LLM = 8
MAX_STRING_LENGTH_FOR_LLM = 300
MAX_TOTAL_LENGTH_FOR_LLM = 10_000
TRUNCATION_NOTE_KEY = "_truncated"


def _shrink_for_llm(
    value: Any,
    *,
    max_list_items: int = MAX_LIST_ITEMS_FOR_LLM,
    max_keys_per_object: int = MAX_KEYS_PER_OBJECT_FOR_LLM,
    max_depth: int = MAX_DEPTH_FOR_LLM,
    max_string_length: int = MAX_STRING_LENGTH_FOR_LLM,
    max_total_length: int = MAX_TOTAL_LENGTH_FOR_LLM,
) -> Any:
    """
    Shrink tool results before sending them to the LLM for summarization.

    This is a generic, shape-aware shrinker:
    - Caps list length
    - Caps dict key count
    - Caps nesting depth
    - Caps string length
    - Enforces a rough overall char budget

    Parameters
    ----------
    value:
        Any JSON-serializable structure returned by tools.
    max_list_items:
        Maximum list items to keep.
    max_keys_per_object:
        Maximum keys to keep per dict.
    max_depth:
        Maximum nesting depth to expand.
    max_string_length:
        Maximum length for strings.
    max_total_length:
        Rough total character budget.

    Returns
    -------
    Any
        Shrunk structure.
    """
    budget = {"remaining": max_total_length}

    def take(n: int) -> None:
        budget["remaining"] = max(0, budget["remaining"] - n)

    def inner(obj: Any, depth: int) -> Any:
        if budget["remaining"] <= 0:
            return "... [truncated due to max_total_length]"

        if isinstance(obj, str):
            s = obj
            if len(s) > max_string_length:
                s = s[:max_string_length] + "... [truncated]"
            take(len(s))
            return s

        if isinstance(obj, (int, float, bool)) or obj is None:
            take(len(str(obj)))
            return obj

        if isinstance(obj, list):
            out: List[Any] = []
            total = len(obj)
            for item in obj[:max_list_items]:
                if budget["remaining"] <= 0:
                    break
                out.append(inner(item, depth + 1))
            if total > max_list_items:
                note = f"... [{total - max_list_items} more items omitted for summarization]"
                take(len(note))
                out.append(note)
            take(2 + len(out))
            return out

        if isinstance(obj, dict):
            if depth >= max_depth:
                summary_text = f"Nested content limited for summarization (object with {len(obj)} keys)"
                take(len(summary_text))
                return {"_summary": summary_text}

            out_dict: Dict[str, Any] = {}
            items = list(obj.items())
            total_keys = len(items)

            for idx, (k, v) in enumerate(items):
                if idx >= max_keys_per_object or budget["remaining"] <= 0:
                    break
                ks = str(k)
                take(len(ks))
                out_dict[ks] = inner(v, depth + 1)

            if total_keys > max_keys_per_object:
                note = f"{total_keys - max_keys_per_object} additional fields omitted for summarization"
                out_dict["_truncated_keys"] = note
                take(len(note))

            take(2 + len(out_dict))
            return out_dict

        s = repr(obj)
        if len(s) > max_string_length:
            s = s[:max_string_length] + "... [truncated]"
        take(len(s))
        return s

    shrunk = inner(value, depth=0)

    if budget["remaining"] <= 0:
        if isinstance(shrunk, dict):
            shrunk.setdefault(
                TRUNCATION_NOTE_KEY,
                "Payload limited due to summarization size constraints; only partial content shown.",
            )
        else:
            shrunk = {
                TRUNCATION_NOTE_KEY: (
                    "Payload limited due to summarization size constraints; only partial content shown."
                ),
                "partial": shrunk,
            }

    return shrunk


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
PLANNING_SYSTEM_PROMPT = """
You are a planning assistant for a Sisense tool-calling agent.

Your ONLY job is to decide which function tool to call and with what JSON arguments.
You are given:
- A natural-language user request.
- A list of tools (functions) with names and JSON parameter schemas.

Global rules:
- Prefer calling a single tool that best matches the request.
- The arguments MUST match the tool's JSON Schema:
  - If type is "array", pass a JSON array (e.g. ["Sales","Marketing"]), NOT a comma-separated string.
  - If type is "boolean", use true or false, NOT "true" or "false".
  - If type is "integer", pass a number, NOT a quoted string.
  - If an enum is defined, the value MUST be one of the allowed enum values.
- Optional parameters can be omitted if the user did not imply them.
- If no tool is clearly appropriate, answer the user directly in natural language
  and DO NOT call any tool.
- Do NOT try to summarise results or explain anything beyond choosing a tool and args.

Strict rules for list parameters (e.g. group_name_list, user_name_list,
dashboard_names, dashboard_ids, datamodel_names, datamodel_ids, dependencies):
- Always pass these as JSON arrays.
- Only include items that the user has explicitly mentioned in their latest message.
- Treat the user’s message as the complete list. DO NOT add extra items.

Additional guidance for dependencies:
- If the user explicitly says "all dependencies" or similar, map that to:
  ["dataSecurity", "formulas", "hierarchies", "perspectives"].
- Otherwise, only include the dependency types the user mentions.
""".strip()

CHAT_PLANNING_CONTEXT_PROMPT = """
The user is working with a single Sisense deployment (chat mode).
When selecting tools, assume there is exactly one active deployment configured.
""".strip()

MIGRATION_PLANNING_CONTEXT_PROMPT = """
The user is working in migration mode with a configured source and target
Sisense deployment. Prefer tools that migrate users, groups, datamodels, and dashboards.
""".strip()

SUMMARY_SYSTEM_PROMPT_CHAT = """
You are a Sisense analytics assistant. Summarise tool results for the user.

Rules:
- Base your answer only on the tool results; do NOT invent objects.
- If many rows are returned, do NOT list everything. Provide counts and a few examples.
- If few rows are returned (roughly <= 20), it is usually OK to list them when helpful.
""".strip()

SUMMARY_SYSTEM_PROMPT_MIGRATION = """
You are a Sisense migration assistant. Summarise tool results for the user.

Rules:
- Base your answer only on the tool results; do NOT invent objects.
- Prefer counts and a high-level summary. Provide a few examples only if useful.
""".strip()


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _approval_key(tool_id: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """
    Stable key for UI approval matching.

    Parameters
    ----------
    tool_id:
        Tool identifier.
    args:
        Tool arguments.

    Returns
    -------
    tuple[str, str]
        (tool_id, normalized_args_json)
    """
    return tool_id, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def _infer_mode_from_tools(tools: List[Dict[str, Any]]) -> str:
    """
    Infer mode ("chat" or "migration") based on registry metadata.

    Parameters
    ----------
    tools:
        OpenAI-style tool definitions passed to the LLM.

    Returns
    -------
    str
        "migration" if any included tool is in the migration module, else "chat".
    """
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        meta = TOOL_REGISTRY.get(name) or {}
        if meta.get("module") == "migration":
            return "migration"
    return "chat"


def _extract_latest_user_message(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get the last user message from a full UI conversation history.

    Parameters
    ----------
    messages:
        Full UI conversation history (user + assistant turns).

    Returns
    -------
    dict
        Latest user message.

    Raises
    ------
    ValueError
        If no user message exists.
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            return m
    raise ValueError("No user message found for LLM planning call.")


def _safe_json_loads(text: Any, default: Any) -> Any:
    """
    Best-effort JSON parse helper.

    Parameters
    ----------
    text:
        Input text to parse (usually a JSON string).
    default:
        Value returned on parse failure.

    Returns
    -------
    Any
        Parsed JSON or default.
    """
    if not isinstance(text, str) or not text.strip():
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _pick_tool_calls_from_llm_response(data: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Extract assistant content and tool_calls from an OpenAI-style response.

    Parameters
    ----------
    data:
        Raw LLM response dict.

    Returns
    -------
    tuple
        (content_text, tool_calls_list)
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, []
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None, []
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    return content if isinstance(content, str) else None, tool_calls if isinstance(tool_calls, list) else []


# -----------------------------------------------------------------------------
# LLM call (raw)
# -----------------------------------------------------------------------------
async def call_llm_raw(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Make a single LLM call via LiteLLM and return the response as a plain dict.

    Uses tool_choice="required" when tools are provided so the planner always
    selects a tool rather than answering in free text. Retries are delegated to
    LiteLLM (num_retries). Providers: azure, databricks, huggingface.

    Parameters
    ----------
    messages:
        Conversation messages to send to the LLM.
    tools:
        OpenAI-style tool definitions. When provided, tool_choice is set to "required".
    trace_id:
        Optional UUID string. When provided, LiteLLM passes it to the LangSmith
        callback so all calls within the same agent turn are grouped into one trace.
        Contains no credentials or customer data.

    Returns
    -------
    dict
        OpenAI-compatible response dict (choices[0].message.tool_calls / content).

    Raises
    ------
    litellm.exceptions.APIError (and subclasses)
        Propagated after retries are exhausted.
    """
    kwargs: Dict[str, Any] = {
        "model": LLM_CONFIG.model,
        "messages": messages,
        "api_key": LLM_CONFIG.api_key,
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
        "timeout": LLM_CONFIG.timeout_seconds,
        "num_retries": MAX_LLM_HTTP_RETRIES,
    }

    if LLM_CONFIG.api_base:
        kwargs["api_base"] = LLM_CONFIG.api_base
    if LLM_CONFIG.api_version:
        kwargs["api_version"] = LLM_CONFIG.api_version

    if tools:
        kwargs["tools"] = tools
        # "required" forces the planner to always emit a tool_call.
        # litellm.drop_params=True silently drops this for providers that don't support it.
        kwargs["tool_choice"] = "required"

    if trace_id:
        # Groups the planning and summarization calls for one turn into a single
        # LangSmith trace. No credentials or customer data included.
        kwargs["metadata"] = {"trace_id": trace_id}

    logger.info("LLM call start: model=%s messages=%d tools=%d", LLM_CONFIG.model, len(messages), len(tools or []))
    _log_json_truncated("LLM request kwargs (truncated)", {k: v for k, v in kwargs.items() if k != "api_key"})

    response = await litellm.acompletion(**kwargs)
    data = response.model_dump()
    _log_json_truncated("LLM raw response (truncated)", data)
    return data


# -----------------------------------------------------------------------------
# Fallback: direct tool (only for when planning fails hard)
# -----------------------------------------------------------------------------
async def _fallback_direct_tool(user_text: str, mcp_client: McpClient) -> Tuple[str, Dict[str, Any]]:
    """
    Run a safe, simple tool based on keywords if planning fails.

    Parameters
    ----------
    user_text:
        Latest user request.
    mcp_client:
        Connected MCP client.

    Returns
    -------
    tuple[str, dict]
        (assistant_summary, tool_result_payload)
    """
    text = (user_text or "").lower()

    # Keep fallback tools read-only (no mutations).
    if "user" in text:
        tool_id = "access.get_users_all"
        args: Dict[str, Any] = {}
    elif "dashboard" in text:
        tool_id = "dashboard.get_dashboards_all"
        args = {}
    elif "data model" in text or "datamodel" in text or "data models" in text:
        tool_id = "datamodel.get_all_datamodel"
        args = {}
    else:
        result = {
            "ok": False,
            "error": (
                "The planning step could not select a tool, and no safe fallback match was possible. "
                "Please rephrase your request (for example, 'show all users' or 'list dashboards')."
            ),
            "error_type": "PlanningFailed",
        }
        return result["error"], result

    logger.info("Fallback: executing tool directly without planning: %s", tool_id)
    result = await mcp_client.invoke_tool(tool_id, args)

    data = result.get("result")
    if isinstance(data, list):
        summary = (
            "The planning step failed, so a keyword-based fallback was used.\n\n"
            f"Executed `{tool_id}` and retrieved **{len(data)}** records. The full result is available in the UI."
        )
    else:
        summary = (
            "The planning step failed, so a keyword-based fallback was used.\n\n"
            f"Executed `{tool_id}`. The result is not a simple table, so the raw payload is provided."
        )

    return summary, result


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
async def call_llm_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    mcp_client: McpClient,
    approved_mutations: Optional[Set[Tuple[str, str]]] = None,
    allow_summarization: Optional[bool] = None,
) -> str:
    """
    Run a single agent turn: planning -> tool execution -> optional summarization.

    Parameters
    ----------
    messages:
        Full UI conversation history (user + assistant turns).
    tools:
        OpenAI-style tool definitions (already filtered by the API layer per mode).
    mcp_client:
        Connected MCP client for calling tools.
    approved_mutations:
        Set of approval keys allowing a mutating tool to execute this turn.
        Each key is (tool_id, normalized_args_json).
    allow_summarization:
        Per-turn override from the API layer.
        Global env ALLOW_SUMMARIZATION is still a hard cap.

    Returns
    -------
    str
        Final assistant reply for the UI.
    """
    global LAST_TOOL_RESULT

    approved_mutations = approved_mutations or set()

    # Global is a hard cap; per-turn can only further restrict.
    if allow_summarization is None:
        allow_summarization_flag = ALLOW_SUMMARIZATION
    else:
        allow_summarization_flag = ALLOW_SUMMARIZATION and bool(allow_summarization)

    # Ensure LAST_TOOL_RESULT is reset for this turn.
    LAST_TOOL_RESULT = None

    latest_user_message = _extract_latest_user_message(messages)
    user_text = str(latest_user_message.get("content", ""))

    mode = _infer_mode_from_tools(tools)
    planning_context = MIGRATION_PLANNING_CONTEXT_PROMPT if mode == "migration" else CHAT_PLANNING_CONTEXT_PROMPT
    summary_system_prompt = SUMMARY_SYSTEM_PROMPT_MIGRATION if mode == "migration" else SUMMARY_SYSTEM_PROMPT_CHAT

    # Cap applied here (after mode filtering) so migration tools are never silently dropped.
    # Databricks serving endpoints reject payloads with too many tools.
    max_tools = int(os.getenv("LLM_MAX_TOOLS", "80"))
    if len(tools) > max_tools:
        logger.warning(
            "Truncating tool list from %d to %d (LLM_MAX_TOOLS). Increase LLM_MAX_TOOLS if tools are missing.",
            len(tools),
            max_tools,
        )
        tools = tools[:max_tools]

    # One UUID per agent turn — groups planning + summarization LLM calls into a
    # single LangSmith trace. Contains no credentials or customer data.
    turn_trace_id = str(uuid.uuid4())

    _trace: Dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "trace_id": turn_trace_id,
        "mode": mode,
        "user_message": user_text[:500],
        "model": LLM_CONFIG.model,
        "provider": LLM_PROVIDER,
        "tools_available": len(tools),
        "tool_selected": "",
        "outcome": "unknown",
        "planning_tokens_in": 0,
        "planning_tokens_out": 0,
        "planning_latency_ms": 0,
        "summary_tokens_in": 0,
        "summary_tokens_out": 0,
        "summary_latency_ms": 0,
        "summarization_used": False,
    }

    logger.info(
        "call_llm_with_tools start: mode=%s tools=%d approvals=%d allow_summarization=%s trace_id=%s",
        mode,
        len(tools),
        len(approved_mutations),
        allow_summarization_flag,
        turn_trace_id,
    )

    # ---------------------------------------------------------------------
    # 1) Planning call (with tools)
    # ---------------------------------------------------------------------
    planning_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        {"role": "system", "content": planning_context},
        latest_user_message,
    ]

    _plan_t0 = time.perf_counter()
    try:
        planning_data = await call_llm_raw(planning_messages, tools=tools, trace_id=turn_trace_id)
    except Exception as exc:
        _trace["planning_latency_ms"] = int((time.perf_counter() - _plan_t0) * 1000)
        _trace["outcome"] = "fallback"
        logger.warning("Planning LLM call failed (%s). Using fallback direct tool.", exc)
        summary, result = await _fallback_direct_tool(user_text, mcp_client)
        LAST_TOOL_RESULT = result
        _write_llm_trace(_trace)
        return summary

    _trace["planning_latency_ms"] = int((time.perf_counter() - _plan_t0) * 1000)
    _plan_usage = planning_data.get("usage") or {}
    _trace["planning_tokens_in"] = _plan_usage.get("prompt_tokens", 0)
    _trace["planning_tokens_out"] = _plan_usage.get("completion_tokens", 0)

    planning_content, tool_calls = _pick_tool_calls_from_llm_response(planning_data)

    if tool_calls:
        _trace["tool_selected"] = (tool_calls[0].get("function") or {}).get("name", "")

    # If no tool calls, return the direct text answer.
    if not tool_calls:
        _trace["outcome"] = "no_tool"
        _write_llm_trace(_trace)
        return planning_content or ""

    # ---------------------------------------------------------------------
    # 2) Execute tools (via MCP) with mutation approval gating
    # ---------------------------------------------------------------------
    tool_messages_for_llm: List[Dict[str, Any]] = []

    # We read the planning assistant message as the "message" to include in the followup.
    planning_assistant_message: Dict[str, Any] = {
        "role": "assistant",
        "content": planning_content or "",
        "tool_calls": tool_calls,
    }

    for tool_call in tool_calls:
        fn = tool_call.get("function") or {}
        tool_id = fn.get("name")
        args_str = fn.get("arguments", "{}")

        if not isinstance(tool_id, str) or not tool_id:
            logger.warning("Skipping tool call with missing name: %s", tool_call)
            continue

        args = _safe_json_loads(args_str, default={})
        if not isinstance(args, dict):
            args = {}

        meta = TOOL_REGISTRY.get(tool_id) or {}
        mutates = bool(meta.get("mutates", False))

        # Validate args against the tool's JSON schema so bad LLM outputs are visible in logs.
        tool_schema = meta.get("parameters")
        if tool_schema:
            try:
                jsonschema.validate(instance=args, schema=tool_schema)
            except jsonschema.ValidationError as _ve:
                logger.warning("Tool %s arg validation warning: %s", tool_id, _ve.message)

        logger.info("Tool selected: %s (mutates=%s)", tool_id, mutates)
        _log_json_truncated("Tool args (from planner)", args)

        if mutates and REQUIRE_MUTATION_CONFIRM:
            key = _approval_key(tool_id, args)
            if key not in approved_mutations:
                pending = {
                    "tool_id": tool_id,
                    "arguments": args,
                    "reason": "Tool is mutating and requires confirmation in the UI.",
                }
                LAST_TOOL_RESULT = {"ok": False, "pending_confirmation": pending}
                logger.info(
                    "Pending mutation approval tool=%s args=%s",
                    tool_id,
                    json.dumps(_scrub_secrets(args), ensure_ascii=False),
                )
                _trace["outcome"] = "pending_mutation"
                _write_llm_trace(_trace)
                return "This action requires confirmation in the UI before proceeding."

        if mutates:
            audit_logger.info(
                "EXECUTING mutation tool=%s args=%s",
                tool_id,
                json.dumps(_scrub_secrets(args), ensure_ascii=False),
            )

        result = await mcp_client.invoke_tool(tool_id, args)
        LAST_TOOL_RESULT = result

        shrunk = _shrink_for_llm(result)

        tool_messages_for_llm.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "name": tool_id,
                "content": json.dumps(shrunk, ensure_ascii=False),
            }
        )

    # If nothing executed (all tool calls were invalid), just return planning text.
    if not tool_messages_for_llm:
        _trace["outcome"] = "no_execution"
        _write_llm_trace(_trace)
        return planning_content or ""

    # ---------------------------------------------------------------------
    # 3) Summarize (optional) or return local-only message if disabled
    # ---------------------------------------------------------------------
    if not allow_summarization_flag:
        last_name = tool_messages_for_llm[-1].get("name", "unknown")
        _trace["outcome"] = "summarization_disabled"
        _write_llm_trace(_trace)
        return _describe_tool_result(last_name, LAST_TOOL_RESULT)

    followup_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": summary_system_prompt},
        latest_user_message,
        planning_assistant_message,
    ] + tool_messages_for_llm

    _trace["summarization_used"] = True
    _sum_t0 = time.perf_counter()
    try:
        followup_data = await call_llm_raw(followup_messages, tools=None, trace_id=turn_trace_id)
    except Exception as exc:
        _trace["summary_latency_ms"] = int((time.perf_counter() - _sum_t0) * 1000)
        _trace["outcome"] = "summarization_failed"
        _write_llm_trace(_trace)
        logger.warning("Summarization LLM call failed (%s). Returning basic status.", exc)
        last_name = tool_messages_for_llm[-1].get("name")
        return f"I ran `{last_name}`, but the summarization step failed, so I cannot provide a richer summary."

    _trace["summary_latency_ms"] = int((time.perf_counter() - _sum_t0) * 1000)
    _sum_usage = followup_data.get("usage") or {}
    _trace["summary_tokens_in"] = _sum_usage.get("prompt_tokens", 0)
    _trace["summary_tokens_out"] = _sum_usage.get("completion_tokens", 0)
    _trace["outcome"] = "ok"
    _write_llm_trace(_trace)

    final_content, _ = _pick_tool_calls_from_llm_response(followup_data)
    return final_content or ""
