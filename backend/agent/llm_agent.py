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
    MAX_AGENT_STEPS,
    REQUIRE_MUTATION_CONFIRM,
    _log_json_truncated,
    _scrub_secrets,
    _write_llm_trace,
    audit_logger,
    logger,
)

# --- _prompts ---
from ._prompts import (
    AGENT_DECIDE_SYSTEM_PROMPT,
    AGENT_FIRST_STEP_SYSTEM_PROMPT,
    CHAT_PLANNING_CONTEXT_PROMPT,
    CLARIFY_QUESTION_SYSTEM_PROMPT,
    MIGRATION_PLANNING_CONTEXT_PROMPT,
    MUTATION_EXPLAIN_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
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

# Step 8: set when the agentic loop pauses mid-turn for a mutation approval.
# runtime.py persists it per session; the approval turn resumes the loop from
# the paused step instead of re-planning from scratch (Option A semantics).
# Shape: {transcript, steps_executed, tool_id, arguments}.
LAST_PENDING_LOOP: Optional[Dict[str, Any]] = None

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
# Agentic loop (Step 8) — decide → route → plan → execute, until done or capped
# -----------------------------------------------------------------------------
async def _decompose_first_step(user_text: str, trace_id: str) -> str:
    """Return the single first operation to route/plan for this turn.

    Compound requests ("all datamodels and all groups") confuse step-1 routing
    and planning, which see two intents and mis-pick. This returns just the
    first sub-task so step 1 routes on a clean single intent (like every
    continuation step does); the decide loop handles the remaining parts. For a
    single-intent message it returns the message ~unchanged. Falls back to the
    original text on any failure — never blocks the turn."""
    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": AGENT_FIRST_STEP_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            tools=None,
            trace_id=trace_id,
        )
        text, _ = _pick_tool_calls_from_llm_response(data)
        text = (text or "").strip()
        return text or user_text
    except Exception as exc:  # noqa: BLE001
        logger.warning("First-step decomposition failed (%s); using full message.", exc)
        return user_text


async def _emit_agent_progress(event: Dict[str, Any]) -> None:
    """Publish a loop progress event to the turn's SSE stream (best-effort)."""
    try:
        # Lazy import: runtime imports this module, so a top-level import would be circular.
        from backend import runtime as runtime_mod

        await runtime_mod.publish_progress({"type": "agent_progress", **event})
    except Exception:  # noqa: BLE001 — progress is cosmetic, never break the turn
        logger.debug("agent progress emit failed", exc_info=True)


def _loop_partial_message(steps_executed: int, remains: str, reason: str) -> str:
    """Terminal message when the loop stops before the goal is complete."""
    return (
        f"I completed {steps_executed} step(s) but stopped before finishing ({reason}). "
        f"Still to do: {remains} "
        "The results so far are shown above — send a follow-up message to continue."
    )


async def _finalize_from_transcript(
    *,
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    turn_trace_id: str,
) -> str:
    """Force a final answer from the results gathered so far — used when the loop
    must stop (e.g. a continued step overreached into a tool needing info the user
    never gave). One LLM call; falls back to describing the last result."""
    messages = [
        {"role": "system", "content": AGENT_DECIDE_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "The turn is ending now. Answer the user based only on the results already gathered. "
            "Do NOT reply CONTINUE.",
        },
        *history,
        latest_user_message,
        *transcript,
    ]
    try:
        data = await call_llm_raw(messages, tools=None, trace_id=turn_trace_id)
        text, _ = _pick_tool_calls_from_llm_response(data)
        text = (text or "").strip()
        # Strip a stray CONTINUE if the model ignores the instruction.
        if text and not text.upper().startswith("CONTINUE:"):
            return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Loop finalize call failed (%s); describing last result.", exc)
    last_name = next((m.get("name", "unknown") for m in reversed(transcript) if m.get("role") == "tool"), "unknown")
    return _describe_tool_result(last_name, LAST_TOOL_RESULT)


async def _agent_continuation_loop(
    *,
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    planning_context: str,
    transcript: List[Dict[str, Any]],
    steps_executed: int,
    mcp_client: McpClient,
    approved_mutations: Set[Tuple[str, str]],
    turn_trace_id: str,
    trace: Dict[str, Any],
    first_tool_hint: Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]] = None,
) -> str:
    """
    Continue an agent turn after at least one tool has executed.

    Each iteration: a decide call checks the goal against the transcript — a
    plain reply is the final answer; "CONTINUE: <next thing>" drives one more
    route → plan → execute step. The CONTINUE sentence (not the original
    compound message) is what gets routed, so each step's routing input is a
    clean single intent.

    Exits: final answer, step cap, mutation pause (state saved for Option A
    resume), missing-arg clarification, or a routing/planning dead end — every
    exit returns something readable, never a silent stop.
    """
    global LAST_TOOL_RESULT, LAST_PENDING_CLARIFICATION, LAST_PENDING_LOOP

    while True:
        # ------------------------------------------------------------------ decide
        await _emit_agent_progress({"phase": "deciding", "step": steps_executed, "max_steps": MAX_AGENT_STEPS})
        decide_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_DECIDE_SYSTEM_PROMPT},
            *history,
            latest_user_message,
            *transcript,
        ]
        try:
            decide_data = await call_llm_raw(decide_messages, tools=None, trace_id=turn_trace_id)
            decide_text, _ = _pick_tool_calls_from_llm_response(decide_data)
            decide_text = (decide_text or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent decide call failed (%s); describing last result.", exc)
            trace["outcome"] = "decide_failed"
            trace["agent_steps"] = steps_executed
            _write_llm_trace(trace)
            last_name = next(
                (m.get("name", "unknown") for m in reversed(transcript) if m.get("role") == "tool"),
                "unknown",
            )
            return _describe_tool_result(last_name, LAST_TOOL_RESULT)

        # Models sometimes hedge — a partial answer AND a trailing "CONTINUE:"
        # line. If any line signals CONTINUE, more work is wanted: continue wins
        # and none of the hedged text leaks to the user.
        _continue_line = next(
            (ln.strip() for ln in decide_text.splitlines() if ln.strip().upper().startswith("CONTINUE:")),
            None,
        )

        if _continue_line is None:
            # Goal satisfied — the decide reply IS the final answer.
            trace["outcome"] = "ok"
            trace["summarization_used"] = True
            trace["agent_steps"] = steps_executed
            _write_llm_trace(trace)
            hint = ""
            if steps_executed == 1 and first_tool_hint:
                _ht, _ha, _hm = first_tool_hint
                hint = _optional_arg_hint(_ht, _ha, _hm)
            return decide_text + hint

        remains = _continue_line.split(":", 1)[1].strip()
        logger.info("Agent loop step %d done; continuing: %s", steps_executed, remains[:200])

        if steps_executed >= MAX_AGENT_STEPS:
            trace["outcome"] = "step_cap"
            trace["agent_steps"] = steps_executed
            _write_llm_trace(trace)
            return _loop_partial_message(steps_executed, remains, "per-turn step limit reached")

        # ------------------------------------------------------------------ route
        step_number = steps_executed + 1
        await _emit_agent_progress({"phase": "planning", "step": step_number, "max_steps": MAX_AGENT_STEPS})
        step_message = {"role": "user", "content": remains}

        nav_tools, nav_pkg, nav_mixin, _ms = await _navigate_to_tools(step_message, [], turn_trace_id)
        if (not nav_tools) and nav_pkg and nav_pkg != "__unclear__":
            # Backtrack: mixin-level miss — retry with the whole package once.
            nav_tools = _load_all_package_tools(nav_pkg)
        if not nav_tools:
            trace["outcome"] = "loop_routing_dead_end"
            trace["agent_steps"] = steps_executed
            _write_llm_trace(trace)
            return await _finalize_from_transcript(
                latest_user_message=latest_user_message,
                history=history,
                transcript=transcript,
                turn_trace_id=turn_trace_id,
            )

        # ------------------------------------------------------------------ plan
        planning_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "system", "content": planning_context},
            latest_user_message,
            *transcript,
            step_message,
        ]
        try:
            plan_data = await call_llm_raw(planning_messages, tools=nav_tools, trace_id=turn_trace_id)
            _content, calls = _pick_tool_calls_from_llm_response(plan_data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent loop planning failed (%s).", exc)
            calls = []

        if not calls and nav_pkg and nav_pkg != "__unclear__":
            # Backtrack: planner declined the mixin's tools — retry once with the whole package.
            full = _load_all_package_tools(nav_pkg)
            if full and len(full) != len(nav_tools):
                logger.info("Agent loop backtrack: retrying step %d with all %s tools", step_number, nav_pkg)
                try:
                    plan_data = await call_llm_raw(planning_messages, tools=full, trace_id=turn_trace_id)
                    _content, calls = _pick_tool_calls_from_llm_response(plan_data)
                except Exception:  # noqa: BLE001
                    calls = []

        if not calls:
            trace["outcome"] = "loop_planning_dead_end"
            trace["agent_steps"] = steps_executed
            _write_llm_trace(trace)
            return await _finalize_from_transcript(
                latest_user_message=latest_user_message,
                history=history,
                transcript=transcript,
                turn_trace_id=turn_trace_id,
            )

        # One operation per loop step — keeps mutation gating and progress legible.
        call = calls[0]
        fn = call.get("function") or {}
        tool_id = str(fn.get("name") or "")
        args = _safe_json_loads(fn.get("arguments", "{}"), default={})
        if not isinstance(args, dict):
            args = {}
        meta = TOOL_REGISTRY.get(tool_id) or {}

        # ------------------------------------------------------------------ validate
        tool_schema = meta.get("parameters")
        if tool_schema:
            try:
                jsonschema.validate(instance=args, schema=tool_schema, format_checker=jsonschema.FormatChecker())
            except jsonschema.ValidationError as _ve:
                missing = _missing_required_fields(args, tool_schema)
                if missing:
                    # A continued step needs info the user never gave → the decide
                    # call overreached (drilled into a detail the request didn't
                    # ask for). Don't pause to ask a confusing question mid-loop;
                    # stop and answer from what's already gathered.
                    logger.info(
                        "Agent loop overreach at step %d: %s needs %s the user didn't provide; finalizing.",
                        step_number,
                        tool_id,
                        missing,
                    )
                    trace["outcome"] = "loop_overreach_finalized"
                    trace["agent_steps"] = steps_executed
                    _write_llm_trace(trace)
                    return await _finalize_from_transcript(
                        latest_user_message=latest_user_message,
                        history=history,
                        transcript=transcript,
                        turn_trace_id=turn_trace_id,
                    )
                trace["outcome"] = "loop_validation_failed"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return _loop_partial_message(steps_executed, remains, f"invalid argument for {tool_id}: {_ve.message}")

        # ------------------------------------------------------------------ gate
        if bool(meta.get("mutates")) and REQUIRE_MUTATION_CONFIRM:
            key = _approval_key(tool_id, args)
            if key not in approved_mutations:
                explanation = await _generate_mutation_explanation(tool_id, meta, args, turn_trace_id)
                LAST_TOOL_RESULT = {
                    "ok": False,
                    "pending_confirmation": {"tool_id": tool_id, "arguments": args, "reason": explanation},
                }
                LAST_PENDING_LOOP = {
                    "transcript": transcript,
                    "steps_executed": steps_executed,
                    "tool_id": tool_id,
                    "arguments": args,
                }
                logger.info("Agent loop paused for mutation approval at step %d: %s", step_number, tool_id)
                trace["outcome"] = "loop_pending_mutation"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return explanation

        if bool(meta.get("mutates")):
            audit_logger.info(
                "EXECUTING mutation tool=%s args=%s",
                tool_id,
                json.dumps(_scrub_secrets(args), ensure_ascii=False),
            )

        # ------------------------------------------------------------------ execute
        await _emit_agent_progress(
            {"phase": "executing", "step": step_number, "max_steps": MAX_AGENT_STEPS, "tool_id": tool_id}
        )
        result = await mcp_client.invoke_tool(tool_id, args)
        LAST_TOOL_RESULT = result

        shrunk = _shrink_for_llm(result)
        transcript.append({"role": "assistant", "content": "", "tool_calls": [call]})
        transcript.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": tool_id,
                "content": json.dumps(shrunk, ensure_ascii=False),
            }
        )
        steps_executed += 1
        await _emit_agent_progress(
            {
                "phase": "completed",
                "step": step_number,
                "max_steps": MAX_AGENT_STEPS,
                "tool_id": tool_id,
                "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
            }
        )


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
    pending_loop: Optional[Dict[str, Any]] = None,
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
    pending_loop:
        Carried-over agentic-loop state from a turn that paused mid-loop for a
        mutation approval (Step 8). When set AND this turn approves that exact
        tool+args, the gated tool executes directly and the loop resumes from the
        paused step. Shape: {transcript, steps_executed, tool_id, arguments}.
    """
    global LAST_TOOL_RESULT, LAST_PENDING_CLARIFICATION, LAST_PENDING_LOOP

    approved_mutations = approved_mutations or set()

    # Global is a hard cap; per-turn can only further restrict.
    if allow_summarization is None:
        allow_summarization_flag = ALLOW_SUMMARIZATION
    else:
        allow_summarization_flag = ALLOW_SUMMARIZATION and bool(allow_summarization)

    LAST_TOOL_RESULT = None
    # Cleared each turn; set again only if this turn pauses for clarification.
    LAST_PENDING_CLARIFICATION = None
    # Cleared each turn; set again only if this turn pauses mid-loop for approval.
    LAST_PENDING_LOOP = None

    latest_user_message = _extract_latest_user_message(messages)
    user_text = str(latest_user_message.get("content", ""))

    mode = _infer_mode_from_tools(tools)
    planning_context = MIGRATION_PLANNING_CONTEXT_PROMPT if mode == "migration" else CHAT_PLANNING_CONTEXT_PROMPT

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
    # 1a-loop) Agentic loop resume (Step 8): a previous turn paused mid-loop for
    # a mutation approval. If this turn approves that exact tool+args, execute it
    # directly (no re-plan — deterministic) and hand back to the continuation
    # loop with the saved transcript. Any other input drops the paused loop and
    # processes normally (typing something else = implicit cancel).
    # -------------------------------------------------------------------------
    if pending_loop:
        _pl_tool_id = str(pending_loop.get("tool_id") or "")
        _pl_args = pending_loop.get("arguments") or {}
        _pl_key = _approval_key(_pl_tool_id, _pl_args)
        if _pl_key in approved_mutations and _pl_tool_id in TOOL_REGISTRY:
            _pl_meta = TOOL_REGISTRY.get(_pl_tool_id) or {}
            audit_logger.info(
                "EXECUTING mutation (loop resume) tool=%s args=%s",
                _pl_tool_id,
                json.dumps(_scrub_secrets(_pl_args), ensure_ascii=False),
            )
            _pl_step = int(pending_loop.get("steps_executed", 0)) + 1
            await _emit_agent_progress(
                {"phase": "executing", "step": _pl_step, "max_steps": MAX_AGENT_STEPS, "tool_id": _pl_tool_id}
            )
            result = await mcp_client.invoke_tool(_pl_tool_id, _pl_args)
            LAST_TOOL_RESULT = result
            await _emit_agent_progress(
                {
                    "phase": "completed",
                    "step": _pl_step,
                    "max_steps": MAX_AGENT_STEPS,
                    "tool_id": _pl_tool_id,
                    "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
                }
            )

            if not allow_summarization_flag:
                _trace["outcome"] = "loop_resumed_summarization_disabled"
                _write_llm_trace(_trace)
                return _describe_tool_result(_pl_tool_id, result)

            _resume_transcript = list(pending_loop.get("transcript") or [])
            _resume_transcript.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"resume-{turn_trace_id[:8]}",
                            "type": "function",
                            "function": {
                                "name": _pl_tool_id,
                                "arguments": json.dumps(_pl_args, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            _resume_transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": f"resume-{turn_trace_id[:8]}",
                    "name": _pl_tool_id,
                    "content": json.dumps(_shrink_for_llm(result), ensure_ascii=False),
                }
            )
            _trace["routing_module"] = "loop_resume"
            _trace["tool_selected"] = _pl_tool_id
            return await _agent_continuation_loop(
                latest_user_message=latest_user_message,
                history=_history,
                planning_context=planning_context,
                transcript=_resume_transcript,
                steps_executed=_pl_step,
                mcp_client=mcp_client,
                approved_mutations=approved_mutations,
                turn_trace_id=turn_trace_id,
                trace=_trace,
            )
        logger.info("Dropping paused agent loop (no matching approval this turn).")

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
        # The message step-1 planning is built around. Chat mode may narrow this
        # to a decomposed first sub-task below; migration always uses the full one.
        step1_message = latest_user_message
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
            # Chat mode. When the loop is enabled (summarization on), decompose a
            # possibly-compound request to its first sub-task so step-1 routing +
            # planning see a single clean intent — the decide loop handles the
            # rest. Single-intent messages pass through ~unchanged.
            if allow_summarization_flag:
                _first = await _decompose_first_step(user_text, turn_trace_id)
                if _first and _first.strip() and _first.strip() != user_text.strip():
                    logger.info("First-step decomposition: %r -> %r", user_text[:120], _first[:120])
                    step1_message = {"role": "user", "content": _first}

            # 3-level navigation (package → mixin → tools).
            _nav_tools, _nav_pkg, _nav_mixin, _routing_ms = await _navigate_to_tools(
                step1_message, _history, turn_trace_id
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
            step1_message,
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
                # Save loop state so the approval turn executes this exact call
                # directly (no re-plan) and can continue the turn afterwards.
                LAST_PENDING_LOOP = {
                    "transcript": list(tool_messages_for_llm),
                    "steps_executed": 0,
                    "tool_id": tool_id,
                    "arguments": args,
                }
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

        # Step 1 executes here (the loop's own steps emit from within the loop).
        await _emit_agent_progress({"phase": "executing", "step": 1, "max_steps": MAX_AGENT_STEPS, "tool_id": tool_id})
        result = await mcp_client.invoke_tool(tool_id, args)
        LAST_TOOL_RESULT = result
        await _emit_agent_progress(
            {
                "phase": "completed",
                "step": 1,
                "max_steps": MAX_AGENT_STEPS,
                "tool_id": tool_id,
                "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
            }
        )

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
    # 4) Continue the turn (Step 8) — decide whether the goal is met; if not,
    # loop route → plan → execute until done, paused, or capped. The decide call
    # doubles as the summarizer: its non-CONTINUE reply is the final answer.
    # -------------------------------------------------------------------------
    if not allow_summarization_flag:
        # Kill-switch: tool results must not reach the LLM, so neither the decide
        # call nor multi-step continuation is possible — single-shot behaviour.
        last_name = tool_messages_for_llm[-1].get("name", "unknown")
        _trace["outcome"] = "summarization_disabled"
        _write_llm_trace(_trace)
        return _describe_tool_result(last_name, LAST_TOOL_RESULT) + _optional_arg_hint(tool_id, args, meta)

    transcript: List[Dict[str, Any]] = [planning_assistant_message, *tool_messages_for_llm]
    return await _agent_continuation_loop(
        latest_user_message=latest_user_message,
        history=_history,
        planning_context=planning_context,
        transcript=transcript,
        steps_executed=1,
        mcp_client=mcp_client,
        approved_mutations=approved_mutations,
        turn_trace_id=turn_trace_id,
        trace=_trace,
        first_tool_hint=(tool_id, args, meta),
    )
