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
    MAX_REPLANS,
    REQUIRE_MUTATION_CONFIRM,
    VERIFY_GOAL,
    VERIFY_MAX_RECHECKS,
    _log_json_truncated,
    _scrub_secrets,
    _write_llm_trace,
    audit_logger,
    logger,
    set_current_turn,
)

# --- _prompts ---
from ._prompts import (
    AGENT_DECIDE_NODATA_SYSTEM_PROMPT,
    AGENT_DECIDE_SYSTEM_PROMPT,
    AGENT_PLAN_SYSTEM_PROMPT,
    AGENT_REPLAN_SYSTEM_PROMPT,
    CHAT_PLANNING_CONTEXT_PROMPT,
    CLARIFY_QUESTION_SYSTEM_PROMPT,
    MIGRATION_PLANNING_CONTEXT_PROMPT,
    MUTATION_EXPLAIN_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    VERIFY_GOAL_SYSTEM_PROMPT,
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

# Every tool result of the current turn, in order: [{step, tool_id, result}].
# LAST_TOOL_RESULT is only the LAST one (a single slot); this keeps the whole
# chain so the UI can show each step's output instead of just the final table.
# Reset at turn start; appended at each execution. Read by the API layer.
LAST_STEP_RESULTS: List[Dict[str, Any]] = []

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
            label="clarify",
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
            label="mutation_explain",
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
def _capability_catalog(mode: str) -> str:
    """One line per tool — `tool_id: first line of description` — for the
    strategist (plan/replan). NO schemas: the strategist writes prose steps, it
    never emits tool calls, so the compact full catalog is safe where showing
    119 schemas to the CALLING planner would not be. Mode-filtered the same way
    the registry is (migration tools only in migration mode)."""
    lines: List[str] = []
    for tid in sorted(TOOL_REGISTRY):
        meta = TOOL_REGISTRY[tid]
        is_migration = meta.get("module") == "migration"
        if (mode == "migration") != is_migration:
            continue
        desc = (meta.get("description") or "").strip().splitlines()
        lines.append(f"- {tid}: {desc[0] if desc else ''}")
    return "\n".join(lines)


def _parse_plan_lines(text: str) -> List[str]:
    """Extract the numbered steps from a strategist reply."""
    import re

    steps = []
    for ln in (text or "").splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", ln)
        if m and m.group(1).strip():
            steps.append(m.group(1).strip())
    return steps


async def _make_plan(user_text: str, mode: str, history: List[Dict[str, Any]], trace_id: str) -> List[str]:
    """The upfront strategist call: request + capability catalog → ordered plan
    (a list of one-operation instructions). Falls back to [user_text] on any
    failure — planning must never block a turn. Privacy-safe in both summ modes:
    it reads only the request text and the catalog, never tool results."""
    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": AGENT_PLAN_SYSTEM_PROMPT},
                {"role": "system", "content": f"Operation catalog:\n{_capability_catalog(mode)}"},
                *history,
                {"role": "user", "content": user_text},
            ],
            tools=None,
            trace_id=trace_id,
            label="strategy",
        )
        text, _ = _pick_tool_calls_from_llm_response(data)
        steps = _parse_plan_lines(text or "")
        if steps:
            return steps
        # A bare unnumbered one-liner still counts as a single-step plan.
        text = (text or "").strip()
        return [text] if text else [user_text]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plan call failed (%s); using the raw message as a single step.", exc)
        return [user_text]


async def _replan(
    user_text: str,
    mode: str,
    transcript: List[Dict[str, Any]],
    reason: str,
    trace_id: str,
) -> Tuple[List[str], str]:
    """The recovery strategist: request + what ran (with outcomes) + why the
    executor gave up + the catalog → a REVISED plan for the remaining work, or
    ("GIVEUP", <user-facing sentence>) when no alternative exists. Returns
    (steps, giveup_message) — one of the two is empty."""
    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": AGENT_REPLAN_SYSTEM_PROMPT},
                {"role": "system", "content": f"Operation catalog:\n{_capability_catalog(mode)}"},
                {"role": "user", "content": user_text},
                *transcript,
                {"role": "user", "content": f"The executor gave up on the current plan because: {reason}"},
            ],
            tools=None,
            trace_id=trace_id,
            label="replan",
        )
        text, _ = _pick_tool_calls_from_llm_response(data)
        text = (text or "").strip()
        if text.upper().startswith("GIVEUP"):
            msg = text.split(":", 1)[1].strip() if ":" in text else ""
            return [], msg or "I couldn't find another way to do this with the available operations."
        steps = _parse_plan_lines(text)
        return (steps, "") if steps else ([], "I couldn't find another way to do this with the available operations.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Replan call failed (%s).", exc)
        return [], "I couldn't work out an alternative approach."


def _metadata_record(tool_id: str, result: Any) -> Dict[str, Any]:
    """The privacy-safe view of a tool result for summarization-OFF turns: what
    ran and whether it worked — never the data itself. This is what goes into the
    LLM's history when data must not reach the model."""
    rec: Dict[str, Any] = {"tool": tool_id, "ok": bool(result.get("ok")) if isinstance(result, dict) else False}
    if isinstance(result, dict):
        payload = result.get("result")
        if isinstance(payload, list):
            rec["count"] = len(payload)
        if not result.get("ok"):
            rec["error"] = result.get("error")
    return rec


def _describe_results_local(raw_results: List[Tuple[str, Any]]) -> str:
    """Render collected results locally (no LLM) for summarization-OFF final
    answers — the raw data never leaves the process."""
    if not raw_results:
        return "No results."
    return "\n\n".join(_describe_tool_result(tid, res) for tid, res in raw_results)


def _transcript_step(call: Dict[str, Any], tool_id: str, result: Any, summ_on: bool) -> List[Dict[str, Any]]:
    """The two messages (assistant tool_call + tool result) appended to the
    LLM-visible history for one executed step. Content is the full result when
    summarization is on, metadata only when off — this is the single point where
    the privacy boundary is enforced in code."""
    content = _shrink_for_llm(result) if summ_on else _metadata_record(tool_id, result)
    return [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call_id": call.get("id"),
            "name": tool_id,
            "content": json.dumps(content, ensure_ascii=False, default=str),
        },
    ]


async def _verify_goal_complete(
    latest_user_message: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    turn_trace_id: str,
) -> Tuple[bool, str]:
    """Independent goal check (verify #3): a separate adversarial LLM call decides
    whether the whole request is actually done. Returns (complete, missing_op).

    This is the *checker* half of maker/checker — the decide call is the maker.
    Scoped to goal completion only; per-step verify (schema, ok flag) is
    deterministic code and needs no second opinion. Sees the same transcript the
    decide call saw, so summarization-off keeps its metadata-only privacy. Any
    failure defaults to 'complete' — the checker must never block a good answer."""
    messages = [
        {"role": "system", "content": VERIFY_GOAL_SYSTEM_PROMPT},
        latest_user_message,
        *transcript,
    ]
    try:
        data = await call_llm_raw(messages, tools=None, trace_id=turn_trace_id, label="verify")
        text, _ = _pick_tool_calls_from_llm_response(data)
        text = (text or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Goal checker failed (%s); accepting the answer.", exc)
        return True, ""
    if text.upper().startswith("INCOMPLETE"):
        missing = text.split(":", 1)[1].strip() if ":" in text else ""
        return (False, missing) if missing else (True, "")
    return True, ""


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
    raw_results: List[Tuple[str, Any]],
    summ_on: bool,
    turn_trace_id: str,
) -> str:
    """Force a final answer from the results gathered so far — used when the loop
    must stop (e.g. a continued step overreached into a tool needing info the user
    never gave).

    Summarization off: render the raw results locally — data must not reach the
    LLM. Summarization on: one LLM call to summarise; fall back to a local
    description on failure."""
    if not summ_on:
        return _describe_results_local(raw_results)
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
        data = await call_llm_raw(messages, tools=None, trace_id=turn_trace_id, label="finalize")
        text, _ = _pick_tool_calls_from_llm_response(data)
        text = (text or "").strip()
        # Strip a stray CONTINUE if the model ignores the instruction.
        if text and not text.upper().startswith("CONTINUE:"):
            return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Loop finalize call failed (%s); describing results locally.", exc)
    return _describe_results_local(raw_results)


async def _reask_clarification_or_giveup(resume_clar: Dict[str, Any], turn_trace_id: str, trace: Dict[str, Any]) -> str:
    """Declined-clarification resume whose answer had no clear intent either — it
    was a non-answer ("I'm not sure"), not a topic change. Re-ask (counting the
    attempt) or give up at the cap. Sets LAST_PENDING_CLARIFICATION when re-asking."""
    global LAST_PENDING_CLARIFICATION
    pc_tool_id = resume_clar.get("tool_id") or ""
    pc_meta = TOOL_REGISTRY.get(pc_tool_id) or {}
    pc_missing = resume_clar.get("missing_fields") or []
    pc_filled = resume_clar.get("filled_args") or {}
    attempts = int(resume_clar.get("attempts", 1)) + 1
    if attempts > CLARIFY_MAX_ATTEMPTS:
        logger.info(
            "Clarification cap (%d) reached for %s after non-answer; giving up.", CLARIFY_MAX_ATTEMPTS, pc_tool_id
        )
        trace["outcome"] = "clarification_exhausted"
        _write_llm_trace(trace)
        return _clarification_giveup_message(pc_tool_id, pc_meta, pc_missing)
    question = await _generate_clarification_question(pc_tool_id, pc_meta, pc_missing, pc_filled, turn_trace_id)
    LAST_PENDING_CLARIFICATION = {
        "tool_id": pc_tool_id,
        "missing_fields": pc_missing,
        "filled_args": pc_filled,
        "attempts": attempts,
        "question": question,
    }
    logger.info("Clarification re-asked after non-answer: tool=%s attempt=%d", pc_tool_id, attempts)
    trace["outcome"] = "awaiting_clarification"
    _write_llm_trace(trace)
    return question


async def _reactive_loop(
    *,
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    planning_context: str,
    mode: str,
    passed_tools: List[Dict[str, Any]],
    user_text: str,
    mcp_client: McpClient,
    approved_mutations: Set[Tuple[str, str]],
    summ_on: bool,
    turn_trace_id: str,
    trace: Dict[str, Any],
    transcript: Optional[List[Dict[str, Any]]] = None,
    raw_results: Optional[List[Tuple[str, Any]]] = None,
    steps_executed: int = 0,
    seed_call: Optional[Dict[str, Any]] = None,
    clarify_attempts_base: int = 0,
    resume_clarification: Optional[Dict[str, Any]] = None,
) -> str:
    """
    The single reactive loop for an entire turn — step 1 is not special.

    One iteration = decide-what's-next → route → plan → validate → gate → execute.
    The "what's next" differs only by where we are:
      - step 0, fresh          → decompose the request to its first sub-task
      - step 0, clarify-resolved → use the pinned tool call directly (`seed_call`)
      - step > 0               → the decide call reads goal + history

    Runs in BOTH summarization modes: the flag only controls what the decide call
    and planner see of each result (full data vs action metadata) — enforced by
    `_transcript_step`. Final answer is LLM prose (summ on) or a local render of
    `raw_results` (summ off, so data never reaches the model).

    Step-0-only exits (a real conversation is possible): clarification (ask the
    user), unclear-intent short-circuit, planning-failure fallback. Later-step
    exits stop-and-summarise instead (`_finalize_from_transcript`). Every exit
    returns readable text — never a silent stop.
    """
    global LAST_TOOL_RESULT, LAST_PENDING_CLARIFICATION, LAST_PENDING_LOOP, LAST_STEP_RESULTS

    transcript = transcript if transcript is not None else []
    raw_results = raw_results if raw_results is not None else []
    first_tool_hint: Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]] = None
    pending_seed = seed_call  # consumed on the first iteration only
    checker_overrides = 0  # times the goal checker has pushed a "done" back into the loop
    replans_used = 0  # times the strategist revised the plan this turn
    next_op_override: Optional[str] = None  # set by a replan; consumed instead of a decide call

    def _done(answer: str) -> str:
        trace["outcome"] = "ok"
        trace["summarization_used"] = summ_on
        trace["agent_steps"] = steps_executed
        _write_llm_trace(trace)
        hint = ""
        if steps_executed == 1 and first_tool_hint:
            _ht, _ha, _hm = first_tool_hint
            hint = _optional_arg_hint(_ht, _ha, _hm)
        return answer + hint

    async def _finalize() -> str:
        return await _finalize_from_transcript(
            latest_user_message=latest_user_message,
            history=history,
            transcript=transcript,
            raw_results=raw_results,
            summ_on=summ_on,
            turn_trace_id=turn_trace_id,
        )

    async def _attempt_replan(reason: str) -> Tuple[Optional[str], str]:
        """Ask the strategist for a revised plan after the current approach failed.
        Returns (next_op, giveup_msg): next_op None = no viable alternative
        (budget spent, or the strategist gave up)."""
        nonlocal replans_used
        if replans_used >= MAX_REPLANS:
            return None, ""
        replans_used += 1
        trace["replans"] = replans_used
        await _emit_agent_progress({"phase": "replanning", "step": steps_executed, "max_steps": MAX_AGENT_STEPS})
        new_steps, giveup = await _replan(user_text, mode, transcript, reason, turn_trace_id)
        if not new_steps:
            return None, giveup
        plan_text = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(new_steps))
        transcript.append({"role": "assistant", "content": f"REVISED PLAN (after: {reason}):\n{plan_text}"})
        await _emit_agent_progress(
            {"phase": "replanned", "step": steps_executed, "max_steps": MAX_AGENT_STEPS, "plan": plan_text}
        )
        logger.info(
            "Replanned (%d/%d) after: %s -> next: %s", replans_used, MAX_REPLANS, reason[:120], new_steps[0][:120]
        )
        return new_steps[0], ""

    while True:
        is_first = steps_executed == 0
        step_number = steps_executed + 1
        calls: List[Dict[str, Any]] = []
        remains = ""

        # ============================================================ what's next
        if is_first and pending_seed is not None:
            # Clarification resolved on a prior turn → the pinned tool is already
            # planned with the user's answer; skip decompose/route/plan.
            calls = [pending_seed]
            pending_seed = None

        elif is_first:
            # Fresh turn: the strategist drafts the full plan (request + capability
            # catalog, no schemas), the loop executes its first operation. The plan
            # is stashed in the transcript so decide/verify follow it, and emitted
            # to the UI for transparency.
            await _emit_agent_progress({"phase": "planning", "step": 1, "max_steps": MAX_AGENT_STEPS})
            plan_steps = await _make_plan(user_text, mode, history, turn_trace_id)
            if len(plan_steps) > 1:
                _plan_text = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(plan_steps))
                transcript.append({"role": "assistant", "content": f"PLAN:\n{_plan_text}"})
                await _emit_agent_progress(
                    {"phase": "planned", "step": 1, "max_steps": MAX_AGENT_STEPS, "plan": _plan_text}
                )
            first_op = plan_steps[0] if plan_steps else user_text
            step_message = {"role": "user", "content": first_op if (first_op or "").strip() else user_text}

            if mode == "migration":
                nav_tools = _load_all_package_tools("migration") or passed_tools
                nav_pkg, nav_mixin = "migration", "all"
            else:
                nav_tools, nav_pkg, nav_mixin, _routing_ms = await _navigate_to_tools(
                    step_message, history, turn_trace_id
                )
                if nav_pkg == "__unclear__":
                    if resume_clarification:
                        # Declined clarification + no fresh intent = a non-answer.
                        return await _reask_clarification_or_giveup(resume_clarification, turn_trace_id, trace)
                    trace["outcome"] = "unclear_intent"
                    _write_llm_trace(trace)
                    return (
                        "I didn't quite understand that. What would you like me to help with? "
                        "For example: 'show all users', 'list dashboards', or 'get all datamodels'."
                    )
                if not nav_tools:
                    nav_tools = passed_tools

            _trace_pkg = f"{nav_pkg}/{nav_mixin}" if nav_mixin else nav_pkg
            trace["routing_module"] = _trace_pkg
            planning_messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "system", "content": planning_context},
                *history,
                step_message,
            ]
            try:
                plan_data = await call_llm_raw(planning_messages, tools=nav_tools, trace_id=turn_trace_id, label="plan")
                content, calls = _pick_tool_calls_from_llm_response(plan_data)
            except Exception as exc:  # noqa: BLE001 — planning failure → keyword fallback
                logger.warning("Planning LLM call failed (%s). Using fallback direct tool.", exc)
                trace["outcome"] = "fallback"
                summary, result = await _fallback_direct_tool(user_text, mcp_client)
                LAST_TOOL_RESULT = result
                LAST_STEP_RESULTS.append({"step": 1, "tool_id": result.get("tool_id", "fallback"), "result": result})
                _write_llm_trace(trace)
                return summary
            if not calls:
                # Planner chose to answer in natural language (no tool fits).
                trace["outcome"] = "no_tool"
                _write_llm_trace(trace)
                return content or ""

        elif next_op_override is not None:
            # A replan already chose the next operation — skip the decide call.
            remains = next_op_override
            next_op_override = None

        else:
            # ---------------------------------------------------------- decide
            await _emit_agent_progress({"phase": "deciding", "step": steps_executed, "max_steps": MAX_AGENT_STEPS})
            decide_prompt = AGENT_DECIDE_SYSTEM_PROMPT if summ_on else AGENT_DECIDE_NODATA_SYSTEM_PROMPT
            decide_messages = [{"role": "system", "content": decide_prompt}, *history, latest_user_message, *transcript]
            try:
                decide_data = await call_llm_raw(decide_messages, tools=None, trace_id=turn_trace_id, label="decide")
                decide_text, _ = _pick_tool_calls_from_llm_response(decide_data)
                decide_text = (decide_text or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent decide call failed (%s); rendering results locally.", exc)
                trace["outcome"] = "decide_failed"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return _describe_results_local(raw_results)

            # Hedge handling: an action line anywhere wins over surrounding prose.
            dlines = [ln.strip() for ln in decide_text.splitlines() if ln.strip()]
            continue_line = next((ln for ln in dlines if ln.upper().startswith("CONTINUE:")), None)
            replan_line = next((ln for ln in dlines if ln.upper().startswith("REPLAN:")), None)
            blocked_line = next((ln for ln in dlines if ln.upper().startswith("BLOCKED:")), None)

            if continue_line is not None:
                remains = continue_line.split(":", 1)[1].strip()
                logger.info("Agent loop step %d done; continuing: %s", steps_executed, remains[:200])
            elif replan_line is not None:
                # The last step's outcome contradicts the plan → strategist revises
                # with the capability catalog (a retry that CHANGES approach).
                reason = replan_line.split(":", 1)[1].strip()
                op, giveup = await _attempt_replan(reason)
                if op is None:
                    trace["outcome"] = "replan_giveup"
                    trace["agent_steps"] = steps_executed
                    _write_llm_trace(trace)
                    prefix = f"{giveup}\n\n" if giveup else ""
                    return prefix + await _finalize()
                remains = op
            elif (not summ_on) and blocked_line is not None:
                reason = blocked_line.split(":", 1)[1].strip()
                trace["outcome"] = "loop_blocked_no_data"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return (
                    "I did the parts I can without reading returned data, but the rest needs a value "
                    f"from an earlier step that I can't see with summarization off ({reason}). "
                    "Turn summarization on to let me continue.\n\n" + _describe_results_local(raw_results)
                )
            else:
                # VERIFY #3 (goal): the decide call (maker) thinks it's done. An
                # independent checker re-reads the whole request against the
                # results and can push one more step if something was missed.
                # Summarization-on only: judging whether the goal was actually
                # ACHIEVED needs the result data. With summarization off the
                # checker would see only metadata — which the decide call already
                # checked — so it adds a call for no real depth; skip it.
                answer = decide_text if summ_on else _describe_results_local(raw_results)
                if summ_on and VERIFY_GOAL and checker_overrides < VERIFY_MAX_RECHECKS:
                    await _emit_agent_progress(
                        {"phase": "verifying", "step": steps_executed, "max_steps": MAX_AGENT_STEPS}
                    )
                    complete, missing = await _verify_goal_complete(latest_user_message, transcript, turn_trace_id)
                    if not complete and missing:
                        checker_overrides += 1
                        trace["goal_rechecks"] = checker_overrides
                        logger.info("Goal checker: INCOMPLETE → continuing with: %s", missing[:160])
                        remains = missing
                    else:
                        return _done(answer)
                else:
                    return _done(answer)

        # ------------------------------------------------ route + plan (steps > 0)
        # Runs for both a decide CONTINUE and a replan-injected op (`calls` is
        # already set on the first step / clarification-seed paths).
        if not calls:
            if steps_executed >= MAX_AGENT_STEPS:
                trace["outcome"] = "step_cap"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return _loop_partial_message(steps_executed, remains, "per-turn step limit reached")

            await _emit_agent_progress({"phase": "planning", "step": step_number, "max_steps": MAX_AGENT_STEPS})
            step_message = {"role": "user", "content": remains}
            nav_tools, nav_pkg, nav_mixin, _ms = await _navigate_to_tools(step_message, [], turn_trace_id)
            if (not nav_tools) and nav_pkg and nav_pkg != "__unclear__":
                nav_tools = _load_all_package_tools(nav_pkg)
            if not nav_tools:
                # No drawer fits this op — let the strategist rephrase/reroute once.
                op, giveup = await _attempt_replan(f"no matching operation found for: {remains}")
                if op:
                    next_op_override = op
                    continue
                trace["outcome"] = "loop_routing_dead_end"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return (f"{giveup}\n\n" if giveup else "") + await _finalize()

            planning_messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "system", "content": planning_context},
                latest_user_message,
                *transcript,
                step_message,
            ]
            try:
                plan_data = await call_llm_raw(planning_messages, tools=nav_tools, trace_id=turn_trace_id, label="plan")
                _content, calls = _pick_tool_calls_from_llm_response(plan_data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent loop planning failed (%s).", exc)
                calls = []
            if not calls and nav_pkg and nav_pkg != "__unclear__":
                full = _load_all_package_tools(nav_pkg)
                if full and len(full) != len(nav_tools):
                    logger.info("Agent loop backtrack: retrying step %d with all %s tools", step_number, nav_pkg)
                    try:
                        plan_data = await call_llm_raw(
                            planning_messages, tools=full, trace_id=turn_trace_id, label="plan"
                        )
                        _content, calls = _pick_tool_calls_from_llm_response(plan_data)
                    except Exception:  # noqa: BLE001
                        calls = []
            if not calls:
                # The planner couldn't pick a tool for this op — strategist retry.
                op, giveup = await _attempt_replan(f"could not pick an operation for: {remains}")
                if op:
                    next_op_override = op
                    continue
                trace["outcome"] = "loop_planning_dead_end"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return (f"{giveup}\n\n" if giveup else "") + await _finalize()

        # ============================================ one tool for this iteration
        call = calls[0]
        fn = call.get("function") or {}
        tool_id = str(fn.get("name") or "")
        if not tool_id:
            trace["outcome"] = "no_execution"
            _write_llm_trace(trace)
            return ""
        args = _safe_json_loads(fn.get("arguments", "{}"), default={})
        if not isinstance(args, dict):
            args = {}
        meta = TOOL_REGISTRY.get(tool_id) or {}
        trace["tool_selected"] = tool_id

        # -------------------------------------------------------------- validate
        tool_schema = meta.get("parameters")
        if tool_schema:
            try:
                jsonschema.validate(instance=args, schema=tool_schema, format_checker=jsonschema.FormatChecker())
            except jsonschema.ValidationError as _ve:
                missing = _missing_required_fields(args, tool_schema)
                if missing and is_first:
                    # First step missing a required arg the user never gave → ask.
                    attempts = clarify_attempts_base + 1
                    if attempts > CLARIFY_MAX_ATTEMPTS:
                        logger.info("Clarification cap (%d) reached for %s; giving up.", CLARIFY_MAX_ATTEMPTS, tool_id)
                        trace["outcome"] = "clarification_exhausted"
                        _write_llm_trace(trace)
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
                    trace["outcome"] = "awaiting_clarification"
                    _write_llm_trace(trace)
                    return question
                if missing:
                    # Mid-loop: the decide call overreached into a detail the user
                    # didn't ask for → stop and answer from what we have.
                    logger.info(
                        "Agent loop overreach at step %d: %s needs %s; finalizing.", step_number, tool_id, missing
                    )
                    trace["outcome"] = "loop_overreach_finalized"
                    trace["agent_steps"] = steps_executed
                    _write_llm_trace(trace)
                    return await _finalize()
                # Value present but wrong (format/type/enum) → hard block.
                logger.error("Tool %s arg validation failed: %s", tool_id, _ve.message)
                if is_first:
                    trace["outcome"] = "validation_failed"
                    _write_llm_trace(trace)
                    return (
                        f"I couldn't call `{tool_id}` — a required argument is invalid or missing: "
                        f"{_ve.message}. Please provide more details."
                    )
                trace["outcome"] = "loop_validation_failed"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return _loop_partial_message(steps_executed, remains, f"invalid argument for {tool_id}: {_ve.message}")

        logger.info("Tool selected: %s (mutates=%s)", tool_id, bool(meta.get("mutates")))
        _log_json_truncated("Tool args (from planner)", args)

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
                    "raw_results": raw_results,
                    "steps_executed": steps_executed,
                    "tool_id": tool_id,
                    "arguments": args,
                }
                logger.info("Agent loop paused for mutation approval at step %d: %s", step_number, tool_id)
                trace["outcome"] = "loop_pending_mutation" if not is_first else "pending_mutation"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return explanation

        if bool(meta.get("mutates")):
            audit_logger.info(
                "EXECUTING mutation tool=%s args=%s",
                tool_id,
                json.dumps(_scrub_secrets(args), ensure_ascii=False),
            )

        # --------------------------------------------------------------- execute
        await _emit_agent_progress(
            {"phase": "executing", "step": step_number, "max_steps": MAX_AGENT_STEPS, "tool_id": tool_id}
        )
        result = await mcp_client.invoke_tool(tool_id, args)
        LAST_TOOL_RESULT = result
        LAST_STEP_RESULTS.append({"step": step_number, "tool_id": tool_id, "result": result})
        raw_results.append((tool_id, result))
        transcript.extend(_transcript_step(call, tool_id, result, summ_on))
        if is_first:
            first_tool_hint = (tool_id, args, meta)
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
    global LAST_TOOL_RESULT, LAST_PENDING_CLARIFICATION, LAST_PENDING_LOOP, LAST_STEP_RESULTS

    approved_mutations = approved_mutations or set()

    # Global is a hard cap; per-turn can only further restrict.
    if allow_summarization is None:
        allow_summarization_flag = ALLOW_SUMMARIZATION
    else:
        allow_summarization_flag = ALLOW_SUMMARIZATION and bool(allow_summarization)

    LAST_TOOL_RESULT = None
    LAST_STEP_RESULTS = []
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

    # Stamp every LLM call this turn makes with this id + the user message, for
    # the per-call log (llm_calls.csv). Task-isolated, so no reset needed.
    set_current_turn(turn_trace_id, user_text)

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
            LAST_STEP_RESULTS.append({"step": _pl_step, "tool_id": _pl_tool_id, "result": result})
            await _emit_agent_progress(
                {
                    "phase": "completed",
                    "step": _pl_step,
                    "max_steps": MAX_AGENT_STEPS,
                    "tool_id": _pl_tool_id,
                    "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
                }
            )

            _resume_call = {
                "id": f"resume-{turn_trace_id[:8]}",
                "type": "function",
                "function": {"name": _pl_tool_id, "arguments": json.dumps(_pl_args, ensure_ascii=False)},
            }
            _resume_transcript = list(pending_loop.get("transcript") or [])
            _resume_transcript.extend(_transcript_step(_resume_call, _pl_tool_id, result, allow_summarization_flag))
            _resume_raw = list(pending_loop.get("raw_results") or [])
            _resume_raw.append((_pl_tool_id, result))
            _trace["routing_module"] = "loop_resume"
            _trace["tool_selected"] = _pl_tool_id
            return await _reactive_loop(
                latest_user_message=latest_user_message,
                history=_history,
                planning_context=planning_context,
                mode=mode,
                passed_tools=tools,
                user_text=user_text,
                mcp_client=mcp_client,
                approved_mutations=approved_mutations,
                summ_on=allow_summarization_flag,
                turn_trace_id=turn_trace_id,
                trace=_trace,
                transcript=_resume_transcript,
                raw_results=_resume_raw,
                steps_executed=_pl_step,
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
    # -------------------------------------------------------------------------
    # 1b) Clarification resume (Step 7): the latest user message is the answer to
    # a prior clarifying question. Re-plan the pinned tool constrained to it
    # (tool_choice="auto" so a decline = topic change). A resolved re-plan becomes
    # the loop's first step (seed_call); a decline enters the loop fresh, carrying
    # the pending state so a non-answer ("I'm not sure") re-asks instead of routing.
    # -------------------------------------------------------------------------
    seed_call: Optional[Dict[str, Any]] = None
    clarify_attempts_base = 0
    resume_clarification: Optional[Dict[str, Any]] = None

    if pending_clarification:
        _pc_tool_id = pending_clarification.get("tool_id")
        _pc_def = _tool_def_for(_pc_tool_id) if _pc_tool_id else None
        if _pc_def is None:
            logger.warning("Resume: pending tool %s not in registry — dropping clarification.", _pc_tool_id)
        else:
            clarify_attempts_base = int(pending_clarification.get("attempts", 1))
            # Anchor the re-plan with the stored clarifying question if the client
            # didn't echo it in history (the UI does; bare API clients may not).
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
            _tool_calls: List[Dict[str, Any]] = []
            try:
                _rdata = await call_llm_raw(
                    _resume_messages, tools=[_pc_def], trace_id=turn_trace_id, tool_choice="auto", label="plan_resume"
                )
                _pc_content, _tool_calls = _pick_tool_calls_from_llm_response(_rdata)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Resume planning failed (%s); falling back to fresh routing.", exc)
                _tool_calls = []
            if _tool_calls:
                seed_call = _tool_calls[0]
                _trace["routing_module"] = "resume"
            else:
                # Declined the pinned tool → maybe a topic change (enter fresh), or
                # a non-answer (fresh routing will be unclear → re-ask via this state).
                logger.info("Resume: planner declined %s — routing fresh.", _pc_tool_id)
                resume_clarification = pending_clarification
                clarify_attempts_base = 0

    # -------------------------------------------------------------------------
    # 2) The single reactive loop — handles step 1 through N (both summ modes).
    # -------------------------------------------------------------------------
    return await _reactive_loop(
        latest_user_message=latest_user_message,
        history=_history,
        planning_context=planning_context,
        mode=mode,
        passed_tools=tools,
        user_text=user_text,
        mcp_client=mcp_client,
        approved_mutations=approved_mutations,
        summ_on=allow_summarization_flag,
        turn_trace_id=turn_trace_id,
        trace=_trace,
        seed_call=seed_call,
        clarify_attempts_base=clarify_attempts_base,
        resume_clarification=resume_clarification,
    )
