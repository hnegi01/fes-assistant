"""
LangGraph engine — the alternative harness for the agent turn.

Selected by ``FES_AGENT_ENGINE=langgraph`` (default ``custom`` keeps the
hand-rolled ``_reactive_loop``). Both engines are thin control flow over the
SAME organs in ``llm_agent`` — planner, router, tool selection, validation,
mutation gate, execution, decide/replan, critic, finalize — accessed here via
module attributes (``A.<helper>``) so unit-test monkeypatches on ``llm_agent``
apply identically to both engines. The 150-test suite therefore doubles as the
parity harness: run it with the env flag flipped.

Graph shape (mirrors AGENT_ARCHITECTURE.md → "Mapping to LangGraph"):

    entry ─┬─ seed (clarify-resolved pinned call) ──────────────┐
           └─ planner ──┬─ Send fan-out → branch* → join ──┐    │
                        └────────────── first_select ──────┤    │
                                                           ▼    ▼
              ┌─────────────────────────────────────── validator
              │   next_select ◀── decide(replanner) ◀── tools ◀── gate
              │        ▲                │   │
              │        └── evaluator ◀──┘   └──▶ END (reply set)
              └───────────────────────────────────────────────▶ END

No checkpointer, no database, no files — pauses (mutation approval,
clarification) END the graph run and persist via the existing SessionEntry
mechanism, exactly like the custom loop; a resume enters through
``call_llm_with_tools``'s prelude paths.

State is one mutable dict passed through nodes. Only ``branch_results`` needs a
reducer (parallel Send branches append concurrently); everything else is
written by exactly one node at a time. ``transcript``/``raw_results``/``trace``
are mutated in place — same objects the resume paths persist.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Dict, List, Optional, Tuple

import jsonschema
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from . import llm_agent as A
from ._config import logger

# A generous ceiling: MAX_AGENT_STEPS laps x ~6 node hops each, plus slack.
_RECURSION_LIMIT = 300


class GraphState(TypedDict, total=False):
    # ---- turn inputs (set once) ----
    latest_user_message: Dict[str, Any]
    history: List[Dict[str, Any]]
    planning_context: str
    mode: str
    passed_tools: List[Dict[str, Any]]
    user_text: str
    mcp_client: Any
    approved_mutations: Any  # Set[Tuple[str, str]]
    summ_on: bool
    turn_trace_id: str
    trace: Dict[str, Any]
    clarify_attempts_base: int
    resume_clarification: Optional[Dict[str, Any]]
    seed_call: Optional[Dict[str, Any]]
    # ---- working state ----
    transcript: List[Dict[str, Any]]
    raw_results: List[Tuple[str, Any]]
    steps_executed: int
    first_tool_hint: Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]
    checker_overrides: int
    replans_used: int
    blocked_tail: List[str]
    plan_steps: List[str]
    independent_steps: List[str]
    remains: str
    calls: List[Dict[str, Any]]
    branch_results: Annotated[List[Dict[str, Any]], operator.add]
    disposition: str  # decide outcome: continue | done | end
    # ---- output ----
    reply: Optional[str]


# ---------------------------------------------------------------------------
# Shared bits of glue (state-level equivalents of the loop's closures)
# ---------------------------------------------------------------------------
def _write_trace(s: GraphState, outcome: str, with_steps: bool = True) -> None:
    s["trace"]["outcome"] = outcome
    if with_steps:
        s["trace"]["agent_steps"] = s["steps_executed"]
    A._write_llm_trace(s["trace"])


def _done_reply(s: GraphState, answer: str) -> str:
    """Equivalent of the loop's `_done`: trace + optional-arg hint + tail note."""
    s["trace"]["outcome"] = "ok"
    s["trace"]["summarization_used"] = s["summ_on"]
    s["trace"]["agent_steps"] = s["steps_executed"]
    A._write_llm_trace(s["trace"])
    hint = ""
    if s["steps_executed"] == 1 and s.get("first_tool_hint"):
        _ht, _ha, _hm = s["first_tool_hint"]
        hint = A._optional_arg_hint(_ht, _ha, _hm)
    tail_note = ""
    if s.get("blocked_tail"):
        skipped = "; ".join(s["blocked_tail"])
        tail_note = (
            "\n\n⏭️ Skipped (needs a value from an earlier result, which I can't read "
            f"with summarization off): {skipped}. Turn summarization on to run it."
        )
    return answer + hint + tail_note


async def _finalize_reply(s: GraphState) -> str:
    return await A._finalize_from_transcript(
        latest_user_message=s["latest_user_message"],
        history=s["history"],
        transcript=s["transcript"],
        raw_results=s["raw_results"],
        summ_on=s["summ_on"],
        turn_trace_id=s["turn_trace_id"],
    )


async def _attempt_replan(s: GraphState, reason: str) -> Tuple[Optional[str], str, int]:
    """Planner revision after a failed approach — same budget/semantics as the loop.
    Returns (next_op, giveup_msg, replans_used) — scalars must be RETURNED as
    LangGraph state updates; in-place writes to the node's input dict are lost."""
    used = s["replans_used"]
    if used >= A.MAX_REPLANS:
        return None, "", used
    used += 1
    s["trace"]["replans"] = used
    await A._emit_agent_progress({"phase": "replanning", "step": s["steps_executed"], "max_steps": A.MAX_AGENT_STEPS})
    new_steps, giveup = await A._replan(s["user_text"], s["mode"], s["transcript"], reason, s["turn_trace_id"])
    if not new_steps:
        return None, giveup, used
    plan_text = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(new_steps))
    if s["summ_on"]:
        for _st in new_steps:
            A.mark_tainted(_st)
        A.mark_tainted(reason)
    s["transcript"].append({"role": "assistant", "content": f"REVISED PLAN (after: {reason}):\n{plan_text}"})
    await A._emit_agent_progress(
        {"phase": "replanned", "step": s["steps_executed"], "max_steps": A.MAX_AGENT_STEPS, "plan": plan_text}
    )
    logger.info("Replanned (%d/%d) after: %s -> next: %s", used, A.MAX_REPLANS, reason[:120], new_steps[0][:120])
    return new_steps[0], "", used


def _record_execution(
    s: GraphState,
    call: Dict[str, Any],
    tool_id: str,
    args: Dict[str, Any],
    meta: Dict[str, Any],
    result: Any,
    steps_executed: int,
    first_tool_hint,
):
    """Shared bookkeeping after any tool execution. Mutates the SHARED objects
    (transcript/raw_results — same references the resume paths persist) but
    returns the scalar counters, which each node must emit as state updates."""
    A.LAST_TOOL_RESULT = result
    A.LAST_STEP_RESULTS.append({"step": steps_executed + 1, "tool_id": tool_id, "result": result})
    s["raw_results"].append((tool_id, result))
    s["transcript"].extend(A._transcript_step(call, tool_id, result, s["summ_on"]))
    if steps_executed == 0 and first_tool_hint is None:
        first_tool_hint = (tool_id, args, meta)
    return steps_executed + 1, first_tool_hint


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
async def node_seed(s: GraphState) -> Dict[str, Any]:
    """Clarify-resolved turn: the pinned tool call is already planned."""
    return {"calls": [s["seed_call"]], "seed_call": None}


async def node_planner(s: GraphState) -> Dict[str, Any]:
    """The PLANNER: drafts the dependency-ordered plan (catalog, no schemas)."""
    await A._emit_agent_progress({"phase": "planning", "step": 1, "max_steps": A.MAX_AGENT_STEPS})
    raw_plan = await A._make_plan(s["user_text"], s["mode"], s["history"], s["turn_trace_id"])
    independent_steps, dependent_steps = A._split_dependent_tail(raw_plan)
    if len(independent_steps) + len(dependent_steps) == 1 and not s["history"]:
        # Faithfulness guard: fresh single-step request → the user's words ARE the step.
        independent_steps, dependent_steps = [s["user_text"]], []
    if s["summ_on"]:
        plan_steps = independent_steps + dependent_steps
        blocked_tail: List[str] = []
    else:
        plan_steps = independent_steps
        blocked_tail = dependent_steps
        if blocked_tail:
            logger.info("Summ-off dependency gate: skipping %d dependent step(s).", len(blocked_tail))
    if len(plan_steps) > 1:
        plan_text = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(plan_steps))
        s["transcript"].append({"role": "assistant", "content": f"PLAN:\n{plan_text}"})
        await A._emit_agent_progress({"phase": "planned", "step": 1, "max_steps": A.MAX_AGENT_STEPS, "plan": plan_text})
    return {"plan_steps": plan_steps, "independent_steps": independent_steps, "blocked_tail": blocked_tail}


def route_after_planner(s: GraphState):
    """Fan out independent steps via Send, or go sequential."""
    fan = s["independent_steps"][: A.MAX_PARALLEL_STEPS] if A.MAX_PARALLEL_STEPS > 1 else []
    if s["mode"] != "migration" and len(fan) >= 2:
        logger.info("Fan-out: running %d independent steps concurrently.", len(fan))
        return [
            Send("branch", {**s, "branch_op": op, "branch_step": i + 1, "branch_results": []})
            for i, op in enumerate(fan)
        ]
    return "first_select"


async def node_branch(s: Dict[str, Any]) -> Dict[str, Any]:
    """One fan-out branch: route→select→validate→execute, blind to siblings.
    Mutations / missing args / dead ends DEFER to the sequential path."""
    op_text: str = s["branch_op"]
    branch_step: int = s["branch_step"]
    step_message = {"role": "user", "content": op_text}
    try:
        nav_tools, nav_pkg, _nm, _ms = await A._navigate_to_tools(step_message, [], s["turn_trace_id"])
        if (not nav_tools) and nav_pkg and nav_pkg != "__unclear__":
            nav_tools = A._load_all_package_tools(nav_pkg)
        if not nav_tools:
            return {"branch_results": [{"status": "deferred", "step": branch_step, "op": op_text, "why": "no route"}]}

        planning_messages = [
            {"role": "system", "content": A.PLANNING_SYSTEM_PROMPT},
            {"role": "system", "content": s["planning_context"]},
            s["latest_user_message"],
            step_message,
        ]
        plan_data = await A.call_llm_raw(planning_messages, tools=nav_tools, trace_id=s["turn_trace_id"], label="plan")
        _bc, bcalls = A._pick_tool_calls_from_llm_response(plan_data)
        if not bcalls:
            return {"branch_results": [{"status": "deferred", "step": branch_step, "op": op_text, "why": "no tool"}]}

        bcall = bcalls[0]
        bfn = bcall.get("function") or {}
        btool_id = str(bfn.get("name") or "")
        bargs = A._safe_json_loads(bfn.get("arguments", "{}"), default={})
        if not isinstance(bargs, dict):
            bargs = {}
        bmeta = A.TOOL_REGISTRY.get(btool_id) or {}

        bschema = bmeta.get("parameters")
        if bschema:
            try:
                A._validate_tool_args(bschema, bargs)
            except jsonschema.ValidationError:
                missing = A._missing_required_fields(bargs, bschema)
                if missing:
                    filled = {
                        k: v for k, v in bargs.items() if not (v is None or (isinstance(v, str) and not v.strip()))
                    }
                    return {
                        "branch_results": [
                            {
                                "status": "missing",
                                "step": branch_step,
                                "op": op_text,
                                "tool_id": btool_id,
                                "meta": bmeta,
                                "missing": missing,
                                "filled": filled,
                            }
                        ]
                    }
                return {
                    "branch_results": [{"status": "deferred", "step": branch_step, "op": op_text, "why": "bad args"}]
                }

        if bool(bmeta.get("mutates")) and A.REQUIRE_MUTATION_CONFIRM:
            if not A._consume_approval(s["approved_mutations"], btool_id, bargs):
                return {
                    "branch_results": [
                        {"status": "deferred", "step": branch_step, "op": op_text, "why": "mutation gate"}
                    ]
                }
        if bool(bmeta.get("mutates")):
            A.audit_logger.info(
                "EXECUTING mutation tool=%s args=%s", btool_id, json.dumps(A._scrub_secrets(bargs), ensure_ascii=False)
            )

        await A._emit_agent_progress(
            {"phase": "executing", "step": branch_step, "max_steps": A.MAX_AGENT_STEPS, "tool_id": btool_id}
        )
        bresult = await A._invoke_tool_traced(s["mcp_client"], btool_id, bargs, s["mode"])
        await A._emit_agent_progress(
            {
                "phase": "completed",
                "step": branch_step,
                "max_steps": A.MAX_AGENT_STEPS,
                "tool_id": btool_id,
                "ok": bool(bresult.get("ok")) if isinstance(bresult, dict) else False,
            }
        )
        return {
            "branch_results": [
                {
                    "status": "executed",
                    "step": branch_step,
                    "op": op_text,
                    "tool_id": btool_id,
                    "args": bargs,
                    "meta": bmeta,
                    "call": bcall,
                    "result": bresult,
                }
            ]
        }
    except Exception as exc:  # noqa: BLE001 — a failed branch defers, never kills the turn
        logger.warning("Fan-out branch %d failed (%s); deferring to sequential loop.", branch_step, exc)
        return {"branch_results": [{"status": "deferred", "step": branch_step, "op": op_text, "why": "error"}]}


async def node_join(s: GraphState) -> Dict[str, Any]:
    """Fan-in: apply branch results in plan order; clarify pauses the turn."""
    clarify_branch = None
    n, hint = s["steps_executed"], s.get("first_tool_hint")
    for br in sorted(s.get("branch_results") or [], key=lambda b: b["step"]):
        if br["status"] == "executed":
            n, hint = _record_execution(s, br["call"], br["tool_id"], br["args"], br["meta"], br["result"], n, hint)
            s["trace"]["tool_selected"] = s["trace"]["tool_selected"] or br["tool_id"]
        elif br["status"] == "missing" and clarify_branch is None:
            clarify_branch = br
    s["steps_executed"] = n  # for _write_trace below; ALSO returned as an update
    if clarify_branch is not None:
        question = await A._generate_clarification_question(
            clarify_branch["tool_id"],
            clarify_branch["meta"],
            clarify_branch["missing"],
            clarify_branch["filled"],
            s["turn_trace_id"],
        )
        A.LAST_PENDING_CLARIFICATION = {
            "tool_id": clarify_branch["tool_id"],
            "missing_fields": clarify_branch["missing"],
            "filled_args": clarify_branch["filled"],
            "attempts": s["clarify_attempts_base"] + 1,
            "question": question,
        }
        _write_trace(s, "awaiting_clarification")
        return {"reply": question, "steps_executed": n, "first_tool_hint": hint}
    return {"steps_executed": n, "first_tool_hint": hint}


def route_after_join(s: GraphState):
    if s.get("reply") is not None:
        return END
    if s["steps_executed"] > 0:
        return "decide"  # deferred branches / dependent tail / goal check
    return "first_select"  # all branches deferred → sequential fallback


async def node_first_select(s: GraphState) -> Dict[str, Any]:
    """Fresh-turn sequential path: route the first op, pick ONE tool."""
    plan_steps = s.get("plan_steps") or []
    first_op = plan_steps[0] if plan_steps else s["user_text"]
    step_message = {"role": "user", "content": first_op if (first_op or "").strip() else s["user_text"]}

    if s["mode"] == "migration":
        nav_tools = list(s["passed_tools"]) or A._load_all_package_tools("migration")
        nav_pkg, nav_mixin = "migration", "all"
    else:
        nav_tools, nav_pkg, nav_mixin, _ms = await A._navigate_to_tools(step_message, s["history"], s["turn_trace_id"])
        if nav_pkg == "__unclear__":
            if s.get("resume_clarification"):
                reply = await A._reask_clarification_or_giveup(
                    s["resume_clarification"], s["turn_trace_id"], s["trace"], s["user_text"]
                )
                return {"reply": reply}
            _write_trace(s, "unclear_intent", with_steps=False)
            return {
                "reply": (
                    "I didn't quite understand that. What would you like me to help with? "
                    "For example: 'show all users', 'list dashboards', or 'get all datamodels'."
                )
            }
        if not nav_tools:
            nav_tools = s["passed_tools"]

    s["trace"]["routing_module"] = f"{nav_pkg}/{nav_mixin}" if nav_mixin else nav_pkg
    # The request goes alongside the step text — same reasoning as the custom
    # loop's first-select path (see llm_agent). `history` is prior TURNS, so
    # without this the call sees only the planner's sentence, and anything the
    # planner left out of it is unrecoverable. Skipped when they are identical
    # (the single-step faithfulness guard makes them so).
    _same = (step_message.get("content") or "").strip() == (
        (s["latest_user_message"] or {}).get("content") or ""
    ).strip()
    planning_messages = [
        {"role": "system", "content": A.PLANNING_SYSTEM_PROMPT},
        {"role": "system", "content": s["planning_context"]},
        *s["history"],
        *([] if _same else [s["latest_user_message"]]),
        step_message,
    ]
    try:
        plan_data = await A.call_llm_raw(planning_messages, tools=nav_tools, trace_id=s["turn_trace_id"], label="plan")
        content, calls = A._pick_tool_calls_from_llm_response(plan_data)
    except Exception as exc:  # noqa: BLE001 — planning failure → keyword fallback
        logger.warning("Planning LLM call failed (%s). Using fallback direct tool.", exc)
        s["trace"]["outcome"] = "fallback"
        summary, result = await A._fallback_direct_tool(s["user_text"], s["mcp_client"], s["mode"])
        A.LAST_TOOL_RESULT = result
        A.LAST_STEP_RESULTS.append({"step": 1, "tool_id": result.get("tool_id", "fallback"), "result": result})
        A._write_llm_trace(s["trace"])
        return {"reply": summary}
    if not calls:
        _write_trace(s, "no_tool", with_steps=False)
        return {"reply": content or ""}
    return {"calls": calls}


async def node_decide(s: GraphState) -> Dict[str, Any]:
    """The REPLANNER role: continue / replan / blocked / done (with critic gate)."""
    await A._emit_agent_progress({"phase": "deciding", "step": s["steps_executed"], "max_steps": A.MAX_AGENT_STEPS})
    decide_prompt = A.AGENT_DECIDE_SYSTEM_PROMPT if s["summ_on"] else A.AGENT_DECIDE_NODATA_SYSTEM_PROMPT
    decide_messages = [
        {"role": "system", "content": decide_prompt},
        *s["history"],
        s["latest_user_message"],
        *s["transcript"],
    ]
    try:
        decide_data = await A.call_llm_raw(decide_messages, tools=None, trace_id=s["turn_trace_id"], label="decide")
        decide_text, _ = A._pick_tool_calls_from_llm_response(decide_data)
        decide_text = (decide_text or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent decide call failed (%s); rendering results locally.", exc)
        _write_trace(s, "decide_failed")
        return {"reply": A._describe_results_local(s["raw_results"]), "disposition": "end"}

    dlines = [ln.strip() for ln in decide_text.splitlines() if ln.strip()]
    continue_line = next((ln for ln in dlines if ln.upper().startswith("CONTINUE:")), None)
    replan_line = next((ln for ln in dlines if ln.upper().startswith("REPLAN:")), None)
    blocked_line = next((ln for ln in dlines if ln.upper().startswith("BLOCKED:")), None)

    if continue_line is not None:
        remains = continue_line.split(":", 1)[1].strip()
        if s["summ_on"]:
            A.mark_tainted(remains)
        logger.info("Agent loop step %d done; continuing: %s", s["steps_executed"], remains[:200])
        return {"remains": remains, "disposition": "continue"}

    if replan_line is not None:
        reason = replan_line.split(":", 1)[1].strip()
        op, giveup, used = await _attempt_replan(s, reason)
        if op is None:
            _write_trace(s, "replan_giveup")
            prefix = f"{giveup}\n\n" if giveup else ""
            return {"reply": prefix + await _finalize_reply(s), "disposition": "end", "replans_used": used}
        return {"remains": op, "disposition": "continue", "replans_used": used}

    if (not s["summ_on"]) and blocked_line is not None:
        reason = blocked_line.split(":", 1)[1].strip()
        _write_trace(s, "loop_blocked_no_data")
        return {
            "reply": (
                "I did the parts I can without reading returned data, but the rest needs a value "
                f"from an earlier step that I can't see with summarization off ({reason}). "
                "Turn summarization on to let me continue.\n\n" + A._describe_results_local(s["raw_results"])
            ),
            "disposition": "end",
        }

    # Done per the maker — the critic (evaluator) may push one more step.
    answer = decide_text if s["summ_on"] else A._describe_results_local(s["raw_results"])
    if s["summ_on"] and A.VERIFY_GOAL and s["checker_overrides"] < A.VERIFY_MAX_RECHECKS:
        await A._emit_agent_progress(
            {"phase": "verifying", "step": s["steps_executed"], "max_steps": A.MAX_AGENT_STEPS}
        )
        complete, missing = await A._verify_goal_complete(s["latest_user_message"], s["transcript"], s["turn_trace_id"])
        if not complete and missing:
            overrides = s["checker_overrides"] + 1
            s["trace"]["goal_rechecks"] = overrides
            logger.info("Goal checker: INCOMPLETE → continuing with: %s", missing[:160])
            return {"remains": missing, "disposition": "continue", "checker_overrides": overrides}
    return {"reply": _done_reply(s, answer), "disposition": "end"}


def route_after_decide(s: GraphState):
    return END if s.get("disposition") == "end" else "next_select"


async def node_next_select(s: GraphState) -> Dict[str, Any]:
    """Steps > 0: step-cap check, route + tool-select (+ backtrack, replan dead-ends)."""
    remains = s.get("remains") or ""
    if s["steps_executed"] >= A.MAX_AGENT_STEPS:
        _write_trace(s, "step_cap")
        return {"reply": A._loop_partial_message(s["steps_executed"], remains, "per-turn step limit reached")}

    step_number = s["steps_executed"] + 1
    await A._emit_agent_progress({"phase": "planning", "step": step_number, "max_steps": A.MAX_AGENT_STEPS})
    step_message = {"role": "user", "content": remains}
    nav_tools, nav_pkg, nav_mixin, _ms = await A._navigate_for_step(
        step_message, s["mode"], s["turn_trace_id"], s["passed_tools"]
    )
    if (not nav_tools) and nav_pkg and nav_pkg != "__unclear__":
        nav_tools = A._load_all_package_tools(nav_pkg)
    if not nav_tools:
        op, giveup, used = await _attempt_replan(s, f"no matching operation found for: {remains}")
        if op:
            return {"remains": op, "calls": [], "replans_used": used}  # loops back via edge
        _write_trace(s, "loop_routing_dead_end")
        return {"reply": (f"{giveup}\n\n" if giveup else "") + await _finalize_reply(s), "replans_used": used}

    planning_messages = [
        {"role": "system", "content": A.PLANNING_SYSTEM_PROMPT},
        {"role": "system", "content": s["planning_context"]},
        s["latest_user_message"],
        *s["transcript"],
        step_message,
    ]
    calls: List[Dict[str, Any]] = []
    try:
        plan_data = await A.call_llm_raw(planning_messages, tools=nav_tools, trace_id=s["turn_trace_id"], label="plan")
        _content, calls = A._pick_tool_calls_from_llm_response(plan_data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent loop planning failed (%s).", exc)
        calls = []
    if not calls and nav_pkg and nav_pkg != "__unclear__":
        full = A._load_all_package_tools(nav_pkg)
        if full and len(full) != len(nav_tools):
            logger.info("Agent loop backtrack: retrying step %d with all %s tools", step_number, nav_pkg)
            try:
                plan_data = await A.call_llm_raw(
                    planning_messages, tools=full, trace_id=s["turn_trace_id"], label="plan"
                )
                _content, calls = A._pick_tool_calls_from_llm_response(plan_data)
            except Exception:  # noqa: BLE001
                calls = []
    if not calls:
        op, giveup, used = await _attempt_replan(s, f"could not pick an operation for: {remains}")
        if op:
            return {"remains": op, "calls": [], "replans_used": used}
        _write_trace(s, "loop_planning_dead_end")
        return {"reply": (f"{giveup}\n\n" if giveup else "") + await _finalize_reply(s), "replans_used": used}
    return {"calls": calls}


def route_after_next_select(s: GraphState):
    if s.get("reply") is not None:
        return END
    if not s.get("calls"):
        return "next_select"  # replan injected a new op — try selecting for it
    return "validator"


async def node_validator(s: GraphState) -> Dict[str, Any]:
    """Code checks: schema validation → clarify (step 0) / overreach / hard block."""
    is_first = s["steps_executed"] == 0
    call = s["calls"][0]
    fn = call.get("function") or {}
    tool_id = str(fn.get("name") or "")
    if not tool_id:
        _write_trace(s, "no_execution", with_steps=False)
        return {"reply": ""}
    args = A._safe_json_loads(fn.get("arguments", "{}"), default={})
    if not isinstance(args, dict):
        args = {}
    meta = A.TOOL_REGISTRY.get(tool_id) or {}
    s["trace"]["tool_selected"] = tool_id

    tool_schema = meta.get("parameters")
    if tool_schema:
        try:
            A._validate_tool_args(tool_schema, args)
        except jsonschema.ValidationError as _ve:
            missing = A._missing_required_fields(args, tool_schema)
            if missing and is_first:
                attempts = s["clarify_attempts_base"] + 1
                if attempts > A.CLARIFY_MAX_ATTEMPTS:
                    logger.info("Clarification cap (%d) reached for %s; giving up.", A.CLARIFY_MAX_ATTEMPTS, tool_id)
                    _write_trace(s, "clarification_exhausted", with_steps=False)
                    return {"reply": A._clarification_giveup_message(tool_id, meta, missing)}
                filled = {k: v for k, v in args.items() if not (v is None or (isinstance(v, str) and not v.strip()))}
                question = await A._generate_clarification_question(tool_id, meta, missing, filled, s["turn_trace_id"])
                A.LAST_PENDING_CLARIFICATION = {
                    "tool_id": tool_id,
                    "missing_fields": missing,
                    "filled_args": filled,
                    "attempts": attempts,
                    "question": question,
                }
                logger.info("Clarification needed: tool=%s missing=%s attempt=%d", tool_id, missing, attempts)
                _write_trace(s, "awaiting_clarification", with_steps=False)
                return {"reply": question}
            if missing:
                logger.info(
                    "Agent loop overreach at step %d: %s needs %s; finalizing.",
                    s["steps_executed"] + 1,
                    tool_id,
                    missing,
                )
                _write_trace(s, "loop_overreach_finalized")
                return {"reply": await _finalize_reply(s)}
            logger.error("Tool %s arg validation failed: %s", tool_id, _ve.message)
            if is_first:
                _write_trace(s, "validation_failed", with_steps=False)
                return {
                    "reply": (
                        f"I couldn't call `{tool_id}` — a required argument is invalid or missing: "
                        f"{_ve.message}. Please provide more details."
                    )
                }
            _write_trace(s, "loop_validation_failed")
            return {
                "reply": A._loop_partial_message(
                    s["steps_executed"], s.get("remains") or "", f"invalid argument for {tool_id}: {_ve.message}"
                )
            }

    logger.info("Tool selected: %s (mutates=%s)", tool_id, bool(meta.get("mutates")))
    A._log_json("Tool args (from tool selection)", args)
    return {}


def route_after_validator(s: GraphState):
    return END if s.get("reply") is not None else "gate"


async def node_gate(s: GraphState) -> Dict[str, Any]:
    """Human-in-the-loop mutation gate: pause the turn (END) until approved."""
    is_first = s["steps_executed"] == 0
    call = s["calls"][0]
    fn = call.get("function") or {}
    tool_id = str(fn.get("name") or "")
    args = A._safe_json_loads(fn.get("arguments", "{}"), default={})
    if not isinstance(args, dict):
        args = {}
    meta = A.TOOL_REGISTRY.get(tool_id) or {}

    if bool(meta.get("mutates")) and A.REQUIRE_MUTATION_CONFIRM:
        if not A._consume_approval(s["approved_mutations"], tool_id, args):
            explanation = await A._generate_mutation_explanation(tool_id, meta, args, s["turn_trace_id"])
            A.LAST_TOOL_RESULT = {
                "ok": False,
                "pending_confirmation": {"tool_id": tool_id, "arguments": args, "reason": explanation},
            }
            A.LAST_PENDING_LOOP = {
                "transcript": s["transcript"],
                "raw_results": s["raw_results"],
                "steps_executed": s["steps_executed"],
                "tool_id": tool_id,
                "arguments": args,
            }
            logger.info("Agent loop paused for mutation approval at step %d: %s", s["steps_executed"] + 1, tool_id)
            _write_trace(s, "loop_pending_mutation" if not is_first else "pending_mutation")
            return {"reply": explanation}
    if bool(meta.get("mutates")):
        A.audit_logger.info(
            "EXECUTING mutation tool=%s args=%s", tool_id, json.dumps(A._scrub_secrets(args), ensure_ascii=False)
        )
    return {}


def route_after_gate(s: GraphState):
    return END if s.get("reply") is not None else "tools"


async def node_tools(s: GraphState) -> Dict[str, Any]:
    """Execute exactly ONE tool via MCP, record everything, loop to decide."""
    step_number = s["steps_executed"] + 1
    call = s["calls"][0]
    fn = call.get("function") or {}
    tool_id = str(fn.get("name") or "")
    args = A._safe_json_loads(fn.get("arguments", "{}"), default={})
    if not isinstance(args, dict):
        args = {}
    meta = A.TOOL_REGISTRY.get(tool_id) or {}

    await A._emit_agent_progress(
        {"phase": "executing", "step": step_number, "max_steps": A.MAX_AGENT_STEPS, "tool_id": tool_id}
    )
    result = await A._invoke_tool_traced(s["mcp_client"], tool_id, args, s["mode"])
    n, hint = _record_execution(s, call, tool_id, args, meta, result, s["steps_executed"], s.get("first_tool_hint"))
    await A._emit_agent_progress(
        {
            "phase": "completed",
            "step": step_number,
            "max_steps": A.MAX_AGENT_STEPS,
            "tool_id": tool_id,
            "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
        }
    )
    return {"calls": [], "steps_executed": n, "first_tool_hint": hint}


def route_entry(s: GraphState):
    if s.get("seed_call") is not None:
        return "seed"
    if s["steps_executed"] > 0:
        return "decide"  # mutation-approval resume: transcript restored, keep going
    return "planner"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def _build_graph():
    g = StateGraph(GraphState)
    g.add_node("seed", node_seed)
    g.add_node("planner", node_planner)
    g.add_node("branch", node_branch)
    g.add_node("join", node_join)
    g.add_node("first_select", node_first_select)
    g.add_node("decide", node_decide)
    g.add_node("next_select", node_next_select)
    g.add_node("validator", node_validator)
    g.add_node("gate", node_gate)
    g.add_node("tools", node_tools)

    g.add_conditional_edges(START, route_entry, ["seed", "planner", "decide"])
    g.add_edge("seed", "validator")
    g.add_conditional_edges("planner", route_after_planner, ["branch", "first_select"])
    g.add_edge("branch", "join")
    g.add_conditional_edges("join", route_after_join, ["decide", "first_select", END])
    g.add_conditional_edges(
        "first_select", lambda s: END if s.get("reply") is not None else "validator", ["validator", END]
    )
    g.add_conditional_edges("decide", route_after_decide, ["next_select", END])
    g.add_conditional_edges("next_select", route_after_next_select, ["validator", "next_select", END])
    g.add_conditional_edges("validator", route_after_validator, ["gate", END])
    g.add_conditional_edges("gate", route_after_gate, ["tools", END])
    g.add_edge("tools", "decide")
    return g.compile()  # no checkpointer: in-memory, single invocation per turn


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


async def run_graph_loop(**kwargs: Any) -> str:
    """Drop-in replacement for `_reactive_loop` — same signature, same contract."""
    state: GraphState = {
        "latest_user_message": kwargs["latest_user_message"],
        "history": kwargs["history"],
        "planning_context": kwargs["planning_context"],
        "mode": kwargs["mode"],
        "passed_tools": kwargs["passed_tools"],
        "user_text": kwargs["user_text"],
        "mcp_client": kwargs["mcp_client"],
        "approved_mutations": kwargs["approved_mutations"],
        "summ_on": kwargs["summ_on"],
        "turn_trace_id": kwargs["turn_trace_id"],
        "trace": kwargs["trace"],
        "transcript": kwargs.get("transcript") if kwargs.get("transcript") is not None else [],
        "raw_results": kwargs.get("raw_results") if kwargs.get("raw_results") is not None else [],
        "steps_executed": kwargs.get("steps_executed", 0),
        "seed_call": kwargs.get("seed_call"),
        "clarify_attempts_base": kwargs.get("clarify_attempts_base", 0),
        "resume_clarification": kwargs.get("resume_clarification"),
        "first_tool_hint": None,
        "checker_overrides": 0,
        "replans_used": 0,
        "blocked_tail": [],
        "branch_results": [],
        "calls": [],
        "remains": "",
        "reply": None,
    }
    final = await _graph().ainvoke(state, config={"recursion_limit": _RECURSION_LIMIT})
    reply = final.get("reply")
    return reply if isinstance(reply, str) else ""
