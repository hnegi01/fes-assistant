"""
backend/agent/llm_agent.py

Main orchestration for the FES Assistant agent turn.

What lives here:
  - TOOL_REGISTRY and LAST_TOOL_RESULT — mutable globals read by the API layer
    (must stay on this module; api_server.py reads them via getattr(llm_agent, ...))
  - load_tools_for_llm() — loads registry JSON and populates TOOL_REGISTRY
  - _get_module_tools() — groups tools by module (uses TOOL_REGISTRY directly)
  - _infer_mode_from_tools() — detects chat vs migration mode
  - _approval_key() — stable key for mutation approval matching
  - call_llm_with_tools() — the main plan → execute → summarize pipeline

What is imported from sub-modules and re-exported for backward compatibility:
  - llm_config: logging, env helpers, LLM provider config, observability
  - llm_tools: registry I/O, payload shrinkers, result description
  - llm_routing: prompts, routing, planning history, raw LLM call, fallback

Dependency order (no circular imports):
  llm_config ← llm_tools ← llm_routing ← llm_agent
"""

from __future__ import annotations

import datetime
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema

# --- _config ---
from ._config import (
    ALLOW_MUTATING_TOOLS,
    ALLOW_SUMMARIZATION,
    CLARIFY_MAX_ATTEMPTS,
    LLM_CONFIG,
    LLM_PLANNING_HISTORY_TURNS,
    LLM_PROVIDER,
    REQUIRE_MUTATION_CONFIRM,
    _log_json_truncated,
    _scrub_secrets,
    _write_llm_trace,
    audit_logger,
    logger,
)

# --- _prompts ---
from ._prompts import (
    CHAT_PLANNING_CONTEXT_PROMPT,
    CLARIFY_QUESTION_SYSTEM_PROMPT,
    MIGRATION_PLANNING_CONTEXT_PROMPT,
    MUTATION_EXPLAIN_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT_CHAT,
    SUMMARY_SYSTEM_PROMPT_MIGRATION,
)

# --- _registry ---
from ._registry import (
    _describe_tool_result,
    _load_registry_rows,
    _safe_json_loads,  # re-exported: test_smoke.py imports from llm_agent
    _shrink_for_llm,  # re-exported: test_smoke.py imports from llm_agent
)

# --- _routing ---
from ._routing import (
    _build_planning_history,  # re-exported: test_planning_history.py calls m._build_planning_history
    _extract_latest_user_message,
    _fallback_direct_tool,
    _load_all_package_tools,
    _navigate_to_tools,
    _pick_tool_calls_from_llm_response,
    call_llm_raw,
)
from ._routing import (
    planner_schema as _planner_schema,
)
from .mcp_client import McpClient

# -----------------------------------------------------------------------------
# Mutable globals — must live on this module so api_server.py reads the live value.
# Re-assigning after a "from .llm_agent import X" would create stale bindings.
# -----------------------------------------------------------------------------
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
LAST_TOOL_RESULT: Optional[Dict[str, Any]] = None

# Set when a turn pauses to ask the user for a missing required argument.
# runtime.py reads this after each turn to persist/clear per-session clarification
# state. Kept separate from LAST_TOOL_RESULT so the API layer does not surface it
# as a tool payload — the clarifying question is delivered as the plain reply.
LAST_PENDING_CLARIFICATION: Optional[Dict[str, Any]] = None

# CLARIFY_MAX_ATTEMPTS imported from _config (env: FES_CLARIFY_MAX_ATTEMPTS, default 2)


# -----------------------------------------------------------------------------
# Registry → OpenAI-style tool definitions
# -----------------------------------------------------------------------------
def load_tools_for_llm() -> List[Dict[str, Any]]:
    """Load tools from the registry and convert them to OpenAI-style tool definitions."""
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
                    "parameters": _planner_schema(params),
                },
            }
        )

    if skipped_mutating:
        logger.info("Mutating tools hidden (ALLOW_MUTATING_TOOLS=False): %s", skipped_mutating)

    logger.info("Tools loaded from registry: %d", len(tools))
    return tools


# -----------------------------------------------------------------------------
# Helpers that depend on TOOL_REGISTRY (must stay on this module)
# -----------------------------------------------------------------------------
def _get_module_tools(tools: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group tools by their registry module. Returns {module_name: [tools]}."""
    by_module: Dict[str, List[Dict[str, Any]]] = {}
    for tool in tools:
        fn = tool.get("function") or {}
        name = fn.get("name", "")
        meta = TOOL_REGISTRY.get(name) or {}
        module = meta.get("module") or "unknown"
        by_module.setdefault(module, []).append(tool)
    return by_module


def _infer_mode_from_tools(tools: List[Dict[str, Any]]) -> str:
    """Infer mode ("chat" or "migration") based on registry metadata."""
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        meta = TOOL_REGISTRY.get(name) or {}
        if meta.get("module") == "migration":
            return "migration"
    return "chat"


def _approval_key(tool_id: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """Stable key for UI approval matching."""
    return tool_id, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


_CREDENTIAL_FIELDS: frozenset = frozenset(
    {
        "domain",
        "token",
        "ssl",
        "source_domain",
        "source_token",
        "source_ssl",
        "target_domain",
        "target_token",
        "target_ssl",
    }
)


def _optional_arg_hint(tool_id: str, used_args: Dict[str, Any], tool_meta: Dict[str, Any]) -> str:
    schema = tool_meta.get("parameters") or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    unused_optional = [k for k in props if k not in required and k not in _CREDENTIAL_FIELDS and k not in used_args]
    if not unused_optional or len(unused_optional) > 3:
        return ""
    params = ", ".join(f"`{k}`" for k in unused_optional)
    return f"\n\nOptional filters you could also specify: {params}."


# -----------------------------------------------------------------------------
# Clarification loop (Step 7) — ask for missing required args, resume next turn
# -----------------------------------------------------------------------------
def _missing_required_fields(args: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Required schema fields absent or empty in args, excluding injected credential fields.

    Planners sometimes send `""` or null for a value the user never provided instead
    of omitting the key — treat those as missing so they trigger clarification, not
    a format-validation hard block.
    """
    required = schema.get("required") or []

    def _is_missing(f: str) -> bool:
        if f not in args:
            return True
        v = args[f]
        return v is None or (isinstance(v, str) and not v.strip())

    return [f for f in required if f not in _CREDENTIAL_FIELDS and _is_missing(f)]


def _curated_optionals(schema: Dict[str, Any], filled_args: Dict[str, Any], limit: int = 3) -> List[str]:
    """Up to `limit` optional, non-credential, currently-unfilled params (same selection as the hint)."""
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    optionals = [k for k in props if k not in required and k not in _CREDENTIAL_FIELDS and k not in filled_args]
    return optionals[:limit]


def _tool_def_for(tool_id: str) -> Optional[Dict[str, Any]]:
    """Build a single OpenAI-style tool definition from the registry, for resume planning."""
    meta = TOOL_REGISTRY.get(tool_id)
    if not meta:
        return None
    return {
        "type": "function",
        "function": {
            "name": tool_id,
            "description": meta.get("description") or "",
            "parameters": _planner_schema(meta.get("parameters") or {}),
        },
    }


async def _generate_clarification_question(
    tool_id: str,
    meta: Dict[str, Any],
    missing_fields: List[str],
    filled_args: Dict[str, Any],
    trace_id: Optional[str],
) -> str:
    """Ask the LLM for one friendly question covering the missing required fields
    (and curated optionals). Falls back to a deterministic template on any failure."""
    schema = meta.get("parameters") or {}
    props = schema.get("properties") or {}
    optionals = _curated_optionals(schema, filled_args)

    def _desc(field: str) -> str:
        return (props.get(field) or {}).get("description") or field

    req_lines = "\n".join(f"- {_desc(f)}" for f in missing_fields)
    user_parts = [f"Operation purpose: {meta.get('description', '')}", "Required information still needed:", req_lines]
    if optionals:
        opt_lines = "\n".join(f"- {_desc(o)}" for o in optionals)
        user_parts += ["Optional extras the user could also provide:", opt_lines]
    user_msg = "\n".join(user_parts)

    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": CLARIFY_QUESTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
            trace_id=trace_id,
        )
        content, _ = _pick_tool_calls_from_llm_response(data)
        if content and content.strip():
            return content.strip()
    except Exception as exc:  # noqa: BLE001 — any LLM failure falls back to the template
        logger.warning("Clarification question generation failed (%s); using template.", exc)

    # Deterministic fallback — describes the missing fields in plain words.
    req_part = "; ".join(_desc(f) for f in missing_fields)
    msg = f"I need a bit more information to do that: {req_part}."
    if optionals:
        msg += f" You can optionally also provide: {', '.join(_desc(o) for o in optionals)}."
    return msg


def _clarification_giveup_message(tool_id: str, meta: Dict[str, Any], missing_fields: List[str]) -> str:
    """Terminal message after the clarification attempt cap is exhausted."""
    props = (meta.get("parameters") or {}).get("properties") or {}
    fields = "; ".join((props.get(f) or {}).get("description") or f for f in missing_fields)
    return (
        "I still don't have everything I need to do that. "
        f"The required information is: {fields}. "
        "Please send a new request with those details included."
    )


async def _generate_mutation_explanation(
    tool_id: str,
    meta: Dict[str, Any],
    args: Dict[str, Any],
    trace_id: Optional[str],
) -> str:
    """Plain-English description of what a mutating tool will do, for the approval
    dialog. One LLM call; falls back to a generic-but-safe template on failure.
    Credential fields are stripped before the args reach the LLM."""
    safe_args = {k: v for k, v in (args or {}).items() if k not in _CREDENTIAL_FIELDS}
    user_msg = (
        f"Operation purpose: {meta.get('description', '')}\n"
        f"It will run with these details: {json.dumps(safe_args, ensure_ascii=False)}"
    )
    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": MUTATION_EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
            trace_id=trace_id,
        )
        content, _ = _pick_tool_calls_from_llm_response(data)
        if content and content.strip():
            return content.strip()
    except Exception as exc:  # noqa: BLE001 — any LLM failure falls back to the template
        logger.warning("Mutation explanation generation failed (%s); using template.", exc)

    purpose = (meta.get("description") or "").rstrip(".")
    base = purpose or "This will modify your Sisense deployment"
    return f"{base}. Review the details below before approving."


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
async def call_llm_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    mcp_client: McpClient,
    approved_mutations: Optional[Set[Tuple[str, str]]] = None,
    allow_summarization: Optional[bool] = None,
    pending_clarification: Optional[Dict[str, Any]] = None,
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
    allow_summarization:
        Per-turn override from the API layer.
        Global env ALLOW_SUMMARIZATION is still a hard cap.
    pending_clarification:
        Carried-over clarification state from the previous turn (Step 7). When set,
        the router is skipped and the planner re-runs constrained to the pinned tool
        to merge the user's answer. Shape: {tool_id, missing_fields, filled_args, attempts}.
    """
    global LAST_TOOL_RESULT, LAST_PENDING_CLARIFICATION

    approved_mutations = approved_mutations or set()

    # Global is a hard cap; per-turn can only further restrict.
    if allow_summarization is None:
        allow_summarization_flag = ALLOW_SUMMARIZATION
    else:
        allow_summarization_flag = ALLOW_SUMMARIZATION and bool(allow_summarization)

    LAST_TOOL_RESULT = None
    # Cleared each turn; set again only if this turn pauses for clarification.
    LAST_PENDING_CLARIFICATION = None

    latest_user_message = _extract_latest_user_message(messages)
    user_text = str(latest_user_message.get("content", ""))

    mode = _infer_mode_from_tools(tools)
    planning_context = MIGRATION_PLANNING_CONTEXT_PROMPT if mode == "migration" else CHAT_PLANNING_CONTEXT_PROMPT
    summary_system_prompt = SUMMARY_SYSTEM_PROMPT_MIGRATION if mode == "migration" else SUMMARY_SYSTEM_PROMPT_CHAT

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
        "routing_module": "",
        "routing_latency_ms": 0,
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

    # -------------------------------------------------------------------------
    # 1) Build planning history (last-N-turns context)
    # -------------------------------------------------------------------------
    _history = _build_planning_history(messages, latest_user_message, LLM_PLANNING_HISTORY_TURNS)
    logger.debug("Planning history: %d prior messages (max turns=%d)", len(_history), LLM_PLANNING_HISTORY_TURNS)

    # -------------------------------------------------------------------------
    # 1b) Clarification resume (Step 7): skip routing, re-plan the pinned tool.
    # On the resume turn the latest user message is the answer to a prior
    # clarifying question, so routing on it alone would be unreliable. Instead we
    # re-run the planner constrained to the one tool we were resolving, with
    # tool_choice="auto" so a decline (the answer wasn't really an answer →
    # topic change) cleanly falls back to fresh routing.
    # -------------------------------------------------------------------------
    clarify_attempts_base = 0
    planning_content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []
    _resumed = False
    _resume_declined = False

    if pending_clarification:
        _pc_tool_id = pending_clarification.get("tool_id")
        _pc_def = _tool_def_for(_pc_tool_id) if _pc_tool_id else None
        if _pc_def is None:
            logger.warning("Resume: pending tool %s not in registry — dropping clarification.", _pc_tool_id)
        else:
            clarify_attempts_base = int(pending_clarification.get("attempts", 1))
            # Anchor the re-plan with the stored clarifying question if the client
            # didn't echo it in history (the UI does; bare API clients may not).
            # Without it the answer ("admin@x.com") floats context-free and the
            # planner has nothing tying it to the pinned tool's missing field.
            _pc_question = pending_clarification.get("question") or ""
            _needs_q = _pc_question and not any(
                m.get("role") == "assistant" and m.get("content") == _pc_question for m in _history
            )
            _resume_messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "system", "content": planning_context},
                *_history,
                *([{"role": "assistant", "content": _pc_question}] if _needs_q else []),
                latest_user_message,
            ]
            logger.info("Resume: re-planning pinned tool %s (attempt base=%d).", _pc_tool_id, clarify_attempts_base)
            try:
                _rdata = await call_llm_raw(
                    _resume_messages, tools=[_pc_def], trace_id=turn_trace_id, tool_choice="auto"
                )
                planning_content, tool_calls = _pick_tool_calls_from_llm_response(_rdata)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Resume planning failed (%s); falling back to fresh routing.", exc)
                tool_calls = []
            if tool_calls:
                _resumed = True
                tools = [_pc_def]
                _trace["routing_module"] = "resume"
                _trace["tool_selected"] = (tool_calls[0].get("function") or {}).get("name", "")
            else:
                # Planner declined the pinned tool → likely a topic change; route
                # fresh. If routing then finds no intent either, it wasn't a topic
                # change — it was a non-answer ("I'm not sure"), handled below.
                logger.info("Resume: planner declined %s — topic change, routing fresh.", _pc_tool_id)
                _resume_declined = True
                clarify_attempts_base = 0

    # -------------------------------------------------------------------------
    # 1a) Tool selection: migration fast-path or 3-level navigation
    # -------------------------------------------------------------------------
    if not _resumed:
        if mode == "migration":
            # Migration has ~9 tools — load all directly, no navigation needed.
            _nav_tools = _load_all_package_tools("migration")
            _nav_pkg, _nav_mixin, _routing_ms = "migration", "all", 0
            if _nav_tools:
                tools = _nav_tools
                logger.info("Migration fast-path: %d tools loaded directly", len(tools))
            else:
                logger.warning("Migration fast-path: registry files missing, using passed tools")
        else:
            # Chat mode: 3-level navigation (package → mixin → tools).
            _nav_tools, _nav_pkg, _nav_mixin, _routing_ms = await _navigate_to_tools(
                latest_user_message, _history, turn_trace_id
            )
            if _nav_pkg == "__unclear__":
                if _resume_declined and pending_clarification:
                    # The "topic change" had no topic — the user gave a non-answer
                    # to the clarifying question ("I'm not sure"). Keep the
                    # clarification alive: re-ask, count the attempt, cap applies.
                    _pc_tool_id = pending_clarification.get("tool_id") or ""
                    _pc_meta = TOOL_REGISTRY.get(_pc_tool_id) or {}
                    _pc_missing = pending_clarification.get("missing_fields") or []
                    _pc_filled = pending_clarification.get("filled_args") or {}
                    attempts = int(pending_clarification.get("attempts", 1)) + 1
                    if attempts > CLARIFY_MAX_ATTEMPTS:
                        logger.info(
                            "Clarification cap (%d) reached for %s after non-answer; giving up.",
                            CLARIFY_MAX_ATTEMPTS,
                            _pc_tool_id,
                        )
                        _trace["outcome"] = "clarification_exhausted"
                        _write_llm_trace(_trace)
                        return _clarification_giveup_message(_pc_tool_id, _pc_meta, _pc_missing)
                    question = await _generate_clarification_question(
                        _pc_tool_id, _pc_meta, _pc_missing, _pc_filled, turn_trace_id
                    )
                    LAST_PENDING_CLARIFICATION = {
                        "tool_id": _pc_tool_id,
                        "missing_fields": _pc_missing,
                        "filled_args": _pc_filled,
                        "attempts": attempts,
                        "question": question,
                    }
                    logger.info("Clarification re-asked after non-answer: tool=%s attempt=%d", _pc_tool_id, attempts)
                    _trace["outcome"] = "awaiting_clarification"
                    _write_llm_trace(_trace)
                    return question
                _trace["outcome"] = "unclear_intent"
                _write_llm_trace(_trace)
                return (
                    "I didn't quite understand that. What would you like me to help with? "
                    "For example: 'show all users', 'list dashboards', or 'get all datamodels'."
                )
            if _nav_tools:
                tools = _nav_tools
                logger.info(
                    "Navigation: %s → %s → %d tools (was %d)",
                    _nav_pkg,
                    _nav_mixin,
                    len(tools),
                    _trace["tools_available"],
                )
            else:
                logger.warning(
                    "Navigation failed (%s/%s), falling back to full tool list (%d)",
                    _nav_pkg,
                    _nav_mixin,
                    len(tools),
                )

        _trace["routing_module"] = f"{_nav_pkg}/{_nav_mixin}" if _nav_mixin else _nav_pkg
        _trace["routing_latency_ms"] = _routing_ms

        planning_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "system", "content": planning_context},
            *_history,
            latest_user_message,
        ]

        # ---------------------------------------------------------------------
        # 2) Planning call
        # ---------------------------------------------------------------------
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

    if not tool_calls:
        _trace["outcome"] = "no_tool"
        _write_llm_trace(_trace)
        return planning_content or ""

    # -------------------------------------------------------------------------
    # 3) Execute tools via MCP with mutation approval gating
    # -------------------------------------------------------------------------
    tool_messages_for_llm: List[Dict[str, Any]] = []

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

        # Validate args against the tool's JSON schema; block on mismatch rather than proceeding silently.
        tool_schema = meta.get("parameters")
        if tool_schema:
            try:
                jsonschema.validate(
                    instance=args,
                    schema=tool_schema,
                    format_checker=jsonschema.FormatChecker(),
                )
            except jsonschema.ValidationError as _ve:
                missing = _missing_required_fields(args, tool_schema)
                if missing:
                    # Missing required arg → clarification loop (Step 7), not a dead end.
                    attempts = clarify_attempts_base + 1
                    if attempts > CLARIFY_MAX_ATTEMPTS:
                        logger.info(
                            "Clarification cap (%d) reached for %s; giving up. missing=%s",
                            CLARIFY_MAX_ATTEMPTS,
                            tool_id,
                            missing,
                        )
                        _trace["outcome"] = "clarification_exhausted"
                        _write_llm_trace(_trace)
                        return _clarification_giveup_message(tool_id, meta, missing)

                    filled = {
                        k: v for k, v in args.items() if not (v is None or (isinstance(v, str) and not v.strip()))
                    }
                    question = await _generate_clarification_question(tool_id, meta, missing, filled, turn_trace_id)
                    LAST_PENDING_CLARIFICATION = {
                        "tool_id": tool_id,
                        "missing_fields": missing,
                        "filled_args": filled,
                        "attempts": attempts,
                        "question": question,
                    }
                    logger.info("Clarification needed: tool=%s missing=%s attempt=%d", tool_id, missing, attempts)
                    _trace["outcome"] = "awaiting_clarification"
                    _write_llm_trace(_trace)
                    return question

                # Value present but wrong (format/type/enum) → hard block, no loop.
                logger.error("Tool %s arg validation failed: %s", tool_id, _ve.message)
                _trace["outcome"] = "validation_failed"
                _write_llm_trace(_trace)
                return (
                    f"I couldn't call `{tool_id}` — a required argument is invalid or missing: "
                    f"{_ve.message}. Please provide more details."
                )

        logger.info("Tool selected: %s (mutates=%s)", tool_id, mutates)
        _log_json_truncated("Tool args (from planner)", args)

        if mutates and REQUIRE_MUTATION_CONFIRM:
            key = _approval_key(tool_id, args)
            if key not in approved_mutations:
                # Plain-English description of what will change, for the approval dialog.
                explanation = await _generate_mutation_explanation(tool_id, meta, args, turn_trace_id)
                pending = {
                    "tool_id": tool_id,
                    "arguments": args,
                    "reason": explanation,
                }
                LAST_TOOL_RESULT = {"ok": False, "pending_confirmation": pending}
                logger.info(
                    "Pending mutation approval tool=%s args=%s",
                    tool_id,
                    json.dumps(_scrub_secrets(args), ensure_ascii=False),
                )
                _trace["outcome"] = "pending_mutation"
                _write_llm_trace(_trace)
                return explanation

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

    if not tool_messages_for_llm:
        _trace["outcome"] = "no_execution"
        _write_llm_trace(_trace)
        return planning_content or ""

    # -------------------------------------------------------------------------
    # 4) Summarize (optional) or return local-only message if disabled
    # -------------------------------------------------------------------------
    if not allow_summarization_flag:
        last_name = tool_messages_for_llm[-1].get("name", "unknown")
        _trace["outcome"] = "summarization_disabled"
        _write_llm_trace(_trace)
        return _describe_tool_result(last_name, LAST_TOOL_RESULT) + _optional_arg_hint(tool_id, args, meta)

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
    return (final_content or "") + _optional_arg_hint(tool_id, args, meta)
