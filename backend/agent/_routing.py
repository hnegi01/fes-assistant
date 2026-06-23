"""
backend/agent/_routing.py

LLM prompts, two-stage module routing, conversation history, and raw LLM call.

What lives here:
  - System prompts: planning, context, summarization, routing
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
)

logger = _make_module_logger("backend.agent.llm_routing", "llm_routing.log")
from .mcp_client import McpClient

REGISTRY_DIR: Path = ROOT_DIR / "config" / "registry"


# -----------------------------------------------------------------------------
# System prompts
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
- Treat the user's message as the complete list. DO NOT add extra items.

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

ROUTING_SYSTEM_PROMPT = """
You are a request router for a Sisense administration assistant.

Your ONLY job is to identify which module best matches the user's request.

Available modules:
{module_list}

Rules:
- Reply with ONLY the module name — a single word, nothing else.
- Pick the module whose tools are most likely to fulfil the request.
- If the request spans multiple modules, pick the primary one.
""".strip()


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
    return history[-(n_turns * 2):]


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
        data = await call_llm_raw(routing_messages, tools=None, trace_id=trace_id)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("Routing LLM call failed (%s). Falling back to full tool list.", exc)
        return None, latency_ms
    latency_ms = int((time.perf_counter() - t0) * 1000)

    content, _ = _pick_tool_calls_from_llm_response(data)
    chosen = _parse_module_from_response(content or "", modules)
    if not chosen:
        logger.warning("Router returned unrecognised response %r. Falling back.", (content or "").strip()[:80])
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
        tools.append({
            "type": "function",
            "function": {
                "name": row["tool_id"],
                "description": row.get("description", ""),
                "parameters": row.get("parameters", {"type": "object", "properties": {}}),
            },
        })
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
        if not chosen_mixin:
            logger.warning("Level 2 navigation: no mixin selected in %s", chosen_pkg)
            return [], chosen_pkg, "", total_ms
        logger.info("Level 2: chose mixin=%s (%dms)", chosen_mixin, ms2)

    # Load Level 3 tools
    tools = _load_mixin_tools(chosen_pkg, chosen_mixin)
    logger.info(
        "Navigation complete: %s → %s → %d tools (total routing_ms=%d)",
        chosen_pkg, chosen_mixin, len(tools), total_ms,
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
) -> Dict[str, Any]:
    """
    Make a single LLM call via LiteLLM and return the response as a plain dict.

    Uses tool_choice="required" when tools are provided so the planner always
    selects a tool rather than answering in free text. Providers: azure, databricks,
    huggingface.
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
async def _fallback_direct_tool(
    user_text: str, mcp_client: McpClient
) -> Tuple[str, Dict[str, Any]]:
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
