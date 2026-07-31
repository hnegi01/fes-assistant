"""
backend/agent/_routing.py

Two-stage module routing, conversation history, and raw LLM call.

What lives here:
  - System prompts re-exported from _prompts.py (edit prompts there, not here)
  - _MODULE_DESCRIPTIONS — module name → one-liner for the routing prompt
  - _build_planning_history() — last-N-turns context extraction
  - _parse_module_from_response() — extract module name from LLM response
  - _route_to_module() — stage-1 routing LLM call
  - _pick_tool_calls_from_llm_response() — parse tool_calls from OpenAI response
  - _extract_latest_user_message() — find the latest user message in history
  - call_llm_raw() — single LiteLLM call with retry and tracing
  - _fallback_direct_tool() — keyword-based fallback when planning fails
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm

from ._config import (
    ALLOW_MUTATING_TOOLS,
    LLM_CONFIG,
    MAX_LLM_HTTP_RETRIES,
    ROOT_DIR,
    _log_json_truncated,
    _make_module_logger,
    write_llm_call,
)
from ._prompts import (
    ROUTING_SYSTEM_PROMPT,
)
from ._tracing import log_llm_child
from .mcp_client import McpClient

logger = _make_module_logger("backend.agent.llm_routing", "llm_routing.log")

REGISTRY_DIR: Path = ROOT_DIR / "config" / "registry"


# -----------------------------------------------------------------------------
# Conversation history
# -----------------------------------------------------------------------------
def _build_planning_history(
    messages: List[Dict[str, Any]],
    latest_user_message: Dict[str, Any],
    n_turns: int,
) -> List[Dict[str, Any]]:
    """
    Extract the last n_turns of conversation history for the planning call.

    Rules:
    - Skips the latest_user_message (it is appended separately by the caller).
    - Assistant messages are stripped to their text content only — tool result
      payloads are excluded so they don't bloat the planning prompt.
    - Empty assistant messages (e.g. pending-confirmation turns) are skipped.
    - Returns at most n_turns * 2 messages (n_turns user + n_turns assistant).
    """
    history: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        if msg is latest_user_message:
            continue
        if role == "assistant":
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            history.append({"role": "assistant", "content": content})
        else:
            history.append({"role": "user", "content": msg.get("content", "")})

    if n_turns <= 0:
        return []
    return history[-(n_turns * 2) :]


# -----------------------------------------------------------------------------
# Two-stage routing helpers
# -----------------------------------------------------------------------------
def _parse_module_from_response(content: str, modules: Dict[str, str]) -> Optional[str]:
    """
    Extract a module name from an LLM routing response.

    Tries exact match first, then substring match. Returns None if no
    known module name is found in the response.
    """
    if not content:
        return None
    content_lower = content.strip().lower()
    if content_lower in modules:
        return content_lower
    for name in modules:
        if name in content_lower:
            return name
    return None


async def _route_to_module(
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    modules: Dict[str, str],
    trace_id: Optional[str],
) -> Tuple[Optional[str], int]:
    """
    Stage 1 of two-stage routing: ask the LLM which module best fits the request.

    Returns (module_name, latency_ms). module_name is None on any failure so
    the caller can fall back to the full tool list.
    """
    module_list = "\n".join(f"- {name}: {desc}" for name, desc in sorted(modules.items()))
    routing_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ROUTING_SYSTEM_PROMPT.format(module_list=module_list)},
        *history,
        latest_user_message,
    ]
    t0 = time.perf_counter()
    try:
        data = await call_llm_raw(routing_messages, tools=None, trace_id=trace_id, label="route")
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("Routing LLM call failed (%s). Falling back to full tool list.", exc)
        return None, latency_ms
    latency_ms = int((time.perf_counter() - t0) * 1000)

    content, _ = _pick_tool_calls_from_llm_response(data)
    raw = (content or "").strip().lower()
    if raw == "none":
        logger.info("Router signalled no Sisense intent for this message.")
        return "__unclear__", latency_ms
    chosen = _parse_module_from_response(content or "", modules)
    if not chosen:
        logger.warning("Router returned unrecognised response %r. Falling back.", raw[:80])
    return chosen, latency_ms


# -----------------------------------------------------------------------------
# 3-level hierarchical navigation
# -----------------------------------------------------------------------------


def _load_registry_index() -> Dict[str, Any]:
    """Load config/registry/index.json — Level 1 package descriptions."""
    path = REGISTRY_DIR / "index.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load registry index: %s", exc)
        return {}


def _load_package_index(package: str) -> Dict[str, Any]:
    """Load config/registry/{package}/index.json — Level 2 mixin descriptions."""
    path = REGISTRY_DIR / package / "index.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load package index for %s: %s", package, exc)
        return {}


def planner_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    """Schema variant sent to the planning LLM: same properties, but no `required` list.

    Marking a field required pressures the model to fill it with *something* —
    placeholders ("user@example.com"), empty strings, or words lifted from the
    request ("email"). With no required pressure it naturally omits values the
    user didn't provide; server-side validation against the real schema then
    routes genuinely missing fields into the clarification loop.
    """
    schema = copy.deepcopy(params or {})
    schema.pop("required", None)
    return schema


def _load_mixin_tools(package: str, mixin: str) -> List[Dict[str, Any]]:
    """
    Load tools from config/registry/{package}/{mixin}.json and convert to OpenAI format.
    Filters out mutating tools when ALLOW_MUTATING_TOOLS is False.
    """
    path = REGISTRY_DIR / package / f"{mixin}.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load mixin tools %s/%s: %s", package, mixin, exc)
        return []

    tools = []
    for row in rows:
        if not ALLOW_MUTATING_TOOLS and row.get("mutates"):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": row["tool_id"],
                    "description": row.get("description", ""),
                    "parameters": planner_schema(row.get("parameters") or {"type": "object", "properties": {}}),
                },
            }
        )
    return tools


def _load_all_package_tools(package: str) -> List[Dict[str, Any]]:
    """
    Load all tools for a package by combining every mixin file.
    Used for migration mode (all ~9 tools in one shot, no navigation needed).
    """
    pkg_dir = REGISTRY_DIR / package
    if not pkg_dir.is_dir():
        logger.warning("Package directory not found: %s", pkg_dir)
        return []
    tools = []
    for mixin_file in sorted(pkg_dir.glob("*.json")):
        if mixin_file.name == "index.json":
            continue
        tools.extend(_load_mixin_tools(package, mixin_file.stem))
    return tools


async def _navigate_to_tools(
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    trace_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], str, str, int]:
    """
    3-level navigation: package → mixin → tools.

    Level 1: LLM picks a package from config/registry/index.json descriptions.
    Level 2: LLM picks a mixin from {package}/index.json (skipped if only 1 mixin).
    Level 3: tools loaded from {package}/{mixin}.json — returned for the planning call.

    Returns (tools, chosen_package, chosen_mixin, total_routing_ms).
    Returns ([], "", "", ms) on any failure so the caller can fall back.
    """
    total_ms = 0

    # Level 1 — pick package
    index = _load_registry_index()
    packages = index.get("packages", {})
    if not packages:
        logger.warning("Registry index empty — cannot navigate")
        return [], "", "", 0

    pkg_descs = {pkg: info.get("description", "") for pkg, info in packages.items()}
    chosen_pkg, ms1 = await _route_to_module(latest_user_message, history, pkg_descs, trace_id)
    total_ms += ms1

    if chosen_pkg == "__unclear__":
        return [], "__unclear__", "", total_ms

    if not chosen_pkg:
        logger.warning("Level 1 navigation: no package selected")
        return [], "", "", total_ms

    logger.info("Level 1: chose package=%s (%dms)", chosen_pkg, ms1)

    # Level 2 — pick mixin (skip if only 1)
    pkg_index = _load_package_index(chosen_pkg)
    modules = pkg_index.get("modules", {})

    if not modules:
        logger.warning("Package %s has no modules in index", chosen_pkg)
        return [], chosen_pkg, "", total_ms

    if len(modules) == 1:
        chosen_mixin = list(modules.keys())[0]
        logger.info("Level 2 skipped — single mixin in %s: %s", chosen_pkg, chosen_mixin)
    else:
        chosen_mixin, ms2 = await _route_to_module(latest_user_message, history, modules, trace_id)
        total_ms += ms2
        if chosen_mixin == "__unclear__":
            # Router saw no clear intent at mixin level — propagate the same
            # short-circuit as a Level 1 unclear instead of falling through to a
            # failed file load (0 tools → full-registry fallback → forced tool call).
            logger.info("Level 2 navigation: unclear intent in %s — short-circuiting", chosen_pkg)
            return [], "__unclear__", "", total_ms
        if not chosen_mixin:
            logger.warning("Level 2 navigation: no mixin selected in %s", chosen_pkg)
            return [], chosen_pkg, "", total_ms
        logger.info("Level 2: chose mixin=%s (%dms)", chosen_mixin, ms2)

    # Load Level 3 tools
    tools = _load_mixin_tools(chosen_pkg, chosen_mixin)
    logger.info(
        "Navigation complete: %s → %s → %d tools (total routing_ms=%d)",
        chosen_pkg,
        chosen_mixin,
        len(tools),
        total_ms,
    )
    return tools, chosen_pkg, chosen_mixin, total_ms


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------
def _pick_tool_calls_from_llm_response(
    data: Dict[str, Any],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Extract assistant content and tool_calls from an OpenAI-style response."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, []
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None, []
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    return (
        content if isinstance(content, str) else None,
        tool_calls if isinstance(tool_calls, list) else [],
    )


def _extract_latest_user_message(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get the last user message from a full UI conversation history."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m
    raise ValueError("No user message found for LLM planning call.")


# -----------------------------------------------------------------------------
# Raw LLM call
# -----------------------------------------------------------------------------
async def call_llm_raw(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    trace_id: Optional[str] = None,
    tool_choice: Optional[str] = None,
    label: str = "",
) -> Dict[str, Any]:
    """
    Make a single LLM call via LiteLLM and return the response as a plain dict.

    When tools are provided, tool_choice defaults to "required" so the tool-selection
    call always selects a tool rather than answering in free text. Pass tool_choice
    explicitly (e.g. "auto") to let it decline — used on the
    clarification-resume turn, where a decline signals the user changed topic.
    Providers: azure, databricks, huggingface.
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
        # "required" forces the tool-selection call to always emit a tool_call; "auto" lets it
        # decline (return plain text). litellm.drop_params=True silently drops this
        # for providers that don't support it.
        kwargs["tool_choice"] = tool_choice or "required"

    if trace_id or label:
        # LangSmith metadata: name the run by its kind (route / plan / decide /
        # verify / ...) and tag the turn id for filtering. We deliberately do NOT
        # set the reserved `trace_id` key: LiteLLM would keep our turn id as the
        # LangSmith trace_id while generating a different run_id with no parent,
        # producing a dotted_order whose first segment != trace_id — which
        # LangSmith rejects (HTTP 400). Omitting it lets LiteLLM default
        # trace_id = run_id, so each call is a valid standalone trace. Per-turn
        # grouping lives in llm_calls.csv (grouped by trace_id). No creds/data.
        kwargs["metadata"] = {
            "run_name": label or "llm-call",
            "call_type": label or "unknown",
            "turn_id": trace_id or "",
        }

    logger.info(
        "LLM call start: kind=%s model=%s messages=%d tools=%d",
        label or "unknown",
        LLM_CONFIG.model,
        len(messages),
        len(tools or []),
    )
    _log_json_truncated("LLM request kwargs (truncated)", {k: v for k, v in kwargs.items() if k != "api_key"})

    _t0 = time.perf_counter()
    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:
        _ms = int((time.perf_counter() - _t0) * 1000)
        write_llm_call(
            call_type=label,
            n_messages=len(messages),
            n_tools=len(tools or []),
            latency_ms=_ms,
            ok=False,
            error=str(exc),
        )
        log_llm_child(label, messages, None, _ms, n_tools=len(tools or []), error=str(exc))
        raise
    data = response.model_dump()
    _usage = data.get("usage") or {}
    _ms = int((time.perf_counter() - _t0) * 1000)
    write_llm_call(
        call_type=label,
        n_messages=len(messages),
        n_tools=len(tools or []),
        latency_ms=_ms,
        tokens_in=_usage.get("prompt_tokens", 0) or 0,
        tokens_out=_usage.get("completion_tokens", 0) or 0,
        ok=True,
    )
    log_llm_child(label, messages, data, _ms, n_tools=len(tools or []))
    _log_json_truncated("LLM raw response (truncated)", data)
    return data


# -----------------------------------------------------------------------------
# Fallback: direct tool (only for when planning fails hard)
# -----------------------------------------------------------------------------
async def _fallback_direct_tool(user_text: str, mcp_client: McpClient) -> Tuple[str, Dict[str, Any]]:
    """Run a safe, simple tool based on keywords if planning fails."""
    text = (user_text or "").lower()

    # Keep fallback tools read-only (no mutations).
    if "user" in text:
        tool_id = "access_management.get_users_all"
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
