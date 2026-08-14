"""
Migration turn flow — plan once, confirm once, execute in sequence.

Migration mode is not the chat loop with different tools; it is a different
shape of problem, so it gets its own path rather than bending the reactive loop
around it.

    chat       plan → execute → "what next?" → execute → "what next?" → …
    migration  plan (ONE call, all 9 tools) → ONE approval → execute in order

The chat loop re-asks the LLM after every step because a step's RESULT can change
the plan: "get user X, then list others with that same role" cannot name step 2's
argument until step 1 has run. Nothing in migration works that way. No migration
tool consumes a value another one produces, and migration mode has no read tools,
so the whole plan is knowable from the request alone. Those per-step calls buy
nothing and cost latency plus a chance for the model to drift.

**Order comes from the planner, reviewed by a human.** The dependency rule
(groups → users → datamodels → dashboards) lives in
MIGRATION_PLANNING_CONTEXT_PROMPT, not in a rank table here. A table keyed on
tool names would have to be edited every time a migration tool is added, and
would silently mis-rank anything it did not recognise. Getting the order wrong
fails QUIETLY — `migrate_users` preserves group assignments, so users migrated
before their groups exist arrive without them — which is exactly why the
approval dialog lists the sequence: the person approving is the check.

**One approval per request, not per step.** The steps are sequential, not
dependent: nothing in step 1's result can change whether step 2 is a good idea,
so approving them one at a time asks the same question repeatedly with no new
information between asks. The dialog names every operation and its arguments, so
this is still explicit consent — just gathered once. The approval is still
SINGLE USE (`A._consume_approval`) and still covers exactly the plan shown.

On failure it stops. Migrating users into groups that failed to migrate produces
a half-configured target that is worse than either outcome, so the reply names
what ran, what failed, and what was not attempted, and leaves the rest to you.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema

from . import llm_agent as A
from ._config import logger
from .mcp_client import McpClient

# Synthetic tool_id for a whole-plan approval. It is not a real tool and is never
# dispatched — it exists so a plan can be keyed, stored and consumed by the same
# approval machinery a single mutation uses (`A._approval_key`), including the
# UI's identical key function.
PLAN_TOOL_ID = "migration.plan"


def _tool_id_of(call: Dict[str, Any]) -> str:
    return str(((call or {}).get("function") or {}).get("name") or "")


def _args_of(call: Dict[str, Any]) -> Dict[str, Any]:
    raw = ((call or {}).get("function") or {}).get("arguments", "{}")
    args = A._safe_json_loads(raw, default={})
    return args if isinstance(args, dict) else {}


def plan_arguments(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The approval payload: every step, in order, with its arguments.

    This IS the thing being approved, so it doubles as the approval key — change
    a step, an argument, or the order, and the key changes and the plan must be
    approved again. Credentials are stripped; they are injected downstream and
    have no business in a dialog or an audit line.
    """
    return {
        "steps": [
            {"step": i + 1, "tool": _tool_id_of(c), "arguments": A._scrub_secrets(_args_of(c))}
            for i, c in enumerate(calls)
        ]
    }


def _plan_text(calls: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{i + 1}. {_tool_id_of(c)}" for i, c in enumerate(calls))


def _step_label(tool_id: str) -> str:
    """A human name for one step, for the approval dialog.

    `migration.migrate_all_groups` is an identifier, not something to put in
    front of an admin — MUTATION_EXPLAIN_SYSTEM_PROMPT already forbids naming
    tools and parameters in the prose, and the step list should hold the same
    line. The registry's description is generated from the SDK docstring, so it
    is human-written AND code-derived: no drift between what is shown and what
    runs.

    Every description ends "... from the source (environment) to the target
    environment ...", which the sentence above the list has already said, so it
    is trimmed. Anything unexpected falls back to the description in full, and
    then to a de-prefixed id — never to nothing.
    """
    desc = ((A.TOOL_REGISTRY.get(tool_id) or {}).get("description") or "").strip().splitlines()
    text = desc[0].strip() if desc else ""
    for marker in (" from the source", " from source"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break
    text = text.rstrip(" .")
    return text or tool_id.split(".", 1)[-1].replace("_", " ").capitalize()


def _humanise_args(args: Dict[str, Any]) -> str:
    """Arguments as readable pairs rather than a JSON blob.

    `{"group_name_list": ["Sales Team"]}` reads as `group name list: Sales Team`.
    The parameter NAMES stay recognisable on purpose — they are what the user has
    to say back to us to change one ("...with action overwrite").
    """
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        if isinstance(value, list):
            shown = ", ".join(str(v) for v in value)
        elif isinstance(value, bool):
            shown = "yes" if value else "no"
        else:
            shown = str(value)
        parts.append(f"{key.replace('_', ' ')}: {shown}")
    return " · ".join(parts)


def _report(done: List[str], failed: Optional[Tuple[str, str]], not_attempted: List[str], body: str) -> str:
    """Stop-and-report: what ran, what broke, what was deliberately left alone."""
    lines = []
    if failed:
        tool_id, reason = failed
        lines.append(f"**Stopped** — `{tool_id}` failed: {reason}")
    if done:
        lines.append("**Completed:** " + ", ".join(f"`{t}`" for t in done))
    if not_attempted:
        lines.append(
            "**Not attempted:** "
            + ", ".join(f"`{t}`" for t in not_attempted)
            + " — these come after the step that failed, so running them would leave "
            "the target half-configured. Fix the failure and ask again."
        )
    head = "\n\n".join(lines)
    return f"{head}\n\n{body}" if body else head


async def _missing_kinds(user_text: str, calls: List[Dict[str, Any]], trace_id: Optional[str]) -> List[str]:
    """Asset kinds the user asked for that this plan omits, per a fresh reader.

    The planner emits every call in ONE response, and that gets less reliable as
    the count rises — a four-kind request returned only two calls in 2 of 6 runs
    (2026-08-08), always stopping at exactly two. Prompt emphasis fixed the
    three-kind case and not the four-kind one, so this is a second pair of eyes
    whose only job is to COUNT. Cheap prompt, no tools, one line back.

    Returns [] on anything unexpected: a checker that cannot answer must not be
    able to derail a plan the planner was happy with.
    """
    planned = ", ".join(_tool_id_of(c) for c in calls) or "(none)"
    try:
        data = await A.call_llm_raw(
            [
                {"role": "system", "content": A.MIGRATION_COMPLETENESS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"User's request:\n{user_text}\n\nOperations the plan will run:\n{planned}",
                },
            ],
            tools=None,
            trace_id=trace_id,
            label="migration_completeness",
        )
        content, _ = A._pick_tool_calls_from_llm_response(data)
    except Exception as exc:  # noqa: BLE001 — a failed check must never block a valid plan
        logger.warning("Migration completeness check failed (%s); accepting the plan as planned.", exc)
        return []

    text = (content or "").strip()
    if not text or text.upper().startswith("COMPLETE"):
        return []
    if ":" not in text:
        logger.warning("Completeness check gave an unparseable answer (%r); accepting the plan.", text[:120])
        return []
    kinds = [k.strip().lower() for k in text.split(":", 1)[1].split(",") if k.strip()]
    return [k for k in kinds if k]


def _render_plan_dialog(calls: List[Dict[str, Any]]) -> str:
    """The approval dialog, built entirely in code — zero LLM calls.

    This used to open with an LLM-written sentence (`migration_plan_explain`,
    one call per turn). Cut 2026-08-10: the code-built step list below already
    names every operation and every argument, so the prose added warmth, not
    information — and for a dialog someone approves destructive work from,
    deterministic beats warm. Chat's single-tool dialogs keep their LLM
    explanation; there the prose IS the dialog.
    """
    steps = plan_arguments(calls)["steps"]
    single = len(calls) == 1

    lines = [
        "This will migrate the following from the source to the target environment."
        if single
        else "This will run the following migrations, in this order, against the target environment.",
        "",
    ]
    if not single:
        lines.append(f"**{len(steps)} operations will run in this order** — approving covers all of them:")
    for s in steps:
        args = _humanise_args(s["arguments"])
        lines.append(f"{s['step']}. **{_step_label(s['tool'])}**" + (f" — {args}" if args else ""))
    lines.append("")

    # The per-tool options note ends with its own "Approve to run as described…"
    # line, written for a single-tool dialog. Inside a plan that is the second
    # such sentence on screen, so it is suppressed here. In a multi-step plan
    # each block is headed by its step, so params from one tool are never read
    # as applying to another.
    disclosure = "\n\n".join(
        n
        for n in (
            A._approval_disclosure(
                s["tool"],
                A.TOOL_REGISTRY.get(s["tool"]) or {},
                s["arguments"],
                with_call_to_action=False,
                heading=None
                if single
                else f"**Optional settings for step {s['step']} — {_step_label(s['tool'])} (not set)**",
            ).strip()
            for s in steps
        )
        if n
    )
    closing = (
        "Approve to run this migration, or cancel and ask again with any changes"
        if single
        else "Approve to run the whole sequence, or cancel and ask again with any changes"
    )
    # Point at the optional settings only when there are some to point at.
    if disclosure:
        closing += " — you can include any of the optional settings below in your request."
    else:
        closing += "."
    lines.append(closing)
    return "\n".join(lines) + (f"\n\n{disclosure}" if disclosure else "")


async def run(
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
    pending_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """One migration turn. Returns the reply; pauses via A.LAST_PENDING_LOOP."""
    transcript = transcript if transcript is not None else []
    raw_results = raw_results if raw_results is not None else []
    calls: Optional[List[Dict[str, Any]]] = None

    def _finish(outcome: str, reply: str) -> str:
        trace["outcome"] = outcome
        trace["agent_steps"] = steps_executed
        A._write_llm_trace(trace)
        return reply

    # ---------------------------------------------------------------- resume
    # A plan the user approved on a previous turn. Never replanned: re-asking
    # the planner the same question can produce a different plan from the one
    # that was actually shown and agreed to.
    if pending_plan:
        saved = pending_plan.get("plan") or []
        if A._consume_approval(approved_mutations, PLAN_TOOL_ID, pending_plan.get("plan_arguments") or {}):
            calls = saved
            logger.info("Resuming approved migration plan (%d step(s)).", len(calls))
        else:
            logger.info("Dropping paused migration plan (no matching approval this turn).")

    # ------------------------------------------------------------------ plan
    # An approved pending_plan (calls already set above) skips this whole block,
    # gate included — it was gated last turn. EVERYTHING else, seed included,
    # goes through validation and the gate below.
    skip_completeness = False
    if calls is None:
        if seed_call is not None:
            # Clarify resolved: the prelude re-planned the PINNED tool with the
            # user's values. That IS the plan — no planning call, no completeness
            # check (it would judge the user's ANSWER text as if it were a request).
            # Found unused 2026-08-10: this flow accepted seed_call and ignored it.
            calls = [seed_call]
            skip_completeness = True
        else:
            await A._emit_agent_progress({"phase": "planning", "step": 1, "max_steps": A.MAX_AGENT_STEPS})
            # Migration's own planning prompt, not the tool-SELECTION one: that prompt
            # exists to pick a SINGLE tool for a SINGLE step, and asking it for an
            # ordered multi-call plan left the ordering rule arguing with its primary
            # instruction (users-before-groups, 12 runs out of 12).
            # NO conversation history, deliberately. Migration plans every call in
            # one response, and ANY preceding turn roughly halves how many it emits.
            # Measured 2026-08-10 on "migrate the dashboards, the users and the
            # groups", 8 runs each:
            #
            #     no history                          8/8 complete
            #     a cancellation notice               3/8
            #     the previous turn's plan listing    5/8
            #     a plain answer ("Migrated 1 group") 3/8
            #
            # Note the last row: a neutral one-line answer containing no plan hurts
            # as much as anything else, so this is not about the model mistaking an
            # earlier proposal for completed work — that was a guess, and it was
            # wrong. The mechanism is not understood; the measurement is consistent.
            # A prompt rule telling it to plan from scratch measured no better
            # (5/12). Withholding history measured 12/12.
            #
            # Affordable here in a way it would not be in chat: migration requests
            # name their own assets, so there is no "its members" or "that datamodel"
            # to resolve against an earlier turn. The cost is that a bare follow-up
            # relying on prior context is not understood — it must name what to
            # migrate, which these requests do anyway.
            # ONE system message, everything stated once. (An earlier two-message
            # arrangement was defended here on 6–12-run samples; 9/12 vs 12/12 is
            # not significant, and a clean single prompt had never been tested.)
            messages = [
                {"role": "system", "content": A.MIGRATION_PLAN_SYSTEM_PROMPT},
                latest_user_message,
            ]
            try:
                data = await A.call_llm_raw(
                    messages, tools=passed_tools, trace_id=turn_trace_id, label="migration_plan"
                )
                content, calls = A._pick_tool_calls_from_llm_response(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Migration planning call failed (%s).", exc)
                summary, result = await A._fallback_direct_tool(user_text, mcp_client, mode)
                A.LAST_TOOL_RESULT = result
                return _finish("fallback", summary)

            if not calls:
                if resume_clarification:
                    # A pending clarification, and the reply resolved nothing —
                    # re-ask (bounded by the attempt cap) instead of answering.
                    return await A._reask_clarification_or_giveup(resume_clarification, turn_trace_id, trace, user_text)
                # No tool fits, or the model is asking which assets to migrate.
                return _finish("no_tool", content or "")

            if resume_clarification and all(
                _tool_id_of(c) != (resume_clarification.get("tool_id") or "") for c in calls
            ):
                # The user was asked for values for ONE pinned tool and replied with
                # something that planned into DIFFERENT operations. Live 2026-08-10:
                # a question about the pending clarification ("is change_ownership a
                # flag?") became a gated migrate_all_dashboards plan. A question must
                # never mutate into a different mutation — re-ask. Costs a genuine
                # topic change one extra turn (the attempt cap then clears it).
                logger.info(
                    "Clarification resume planned %s instead of the pinned %s — re-asking.",
                    [_tool_id_of(c) for c in calls],
                    resume_clarification.get("tool_id"),
                )
                return await A._reask_clarification_or_giveup(resume_clarification, turn_trace_id, trace, user_text)

            trace["routing_module"] = "migration/all"
            logger.info("Migration plan (%d call(s)): %s", len(calls), [_tool_id_of(c) for c in calls])

            # ---------------------------------------------------- completeness check
            # A fresh reader counts the asset kinds. If it names one the plan omits,
            # the PLANNER gets one more attempt with the omission spelled out —
            # rather than us appending a call ourselves, which would have to guess
            # where in the dependency order it belongs. Bounded to a single retry:
            # two chances at each kind, never a loop.
            if A.MIGRATION_COMPLETENESS_CHECK and not skip_completeness:
                missing = await _missing_kinds(user_text, calls, turn_trace_id)
                if missing:
                    logger.info("Completeness check: plan omits %s — replanning once.", ", ".join(missing))
                    trace["migration_completeness_retry"] = 1
                    retry_messages = messages + [
                        {
                            "role": "assistant",
                            "content": "Planned: " + ", ".join(_tool_id_of(c) for c in calls),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"That plan omits {', '.join(missing)}, which I asked for. "
                                "Emit the COMPLETE plan again — every kind I named, one tool call each, "
                                "in dependency order."
                            ),
                        },
                    ]
                    try:
                        retry_data = await A.call_llm_raw(
                            retry_messages, tools=passed_tools, trace_id=turn_trace_id, label="migration_plan_retry"
                        )
                        _c, retry_calls = A._pick_tool_calls_from_llm_response(retry_data)
                        if retry_calls and len(retry_calls) > len(calls):
                            calls = retry_calls
                            logger.info("Replanned to %d call(s): %s", len(calls), [_tool_id_of(c) for c in calls])
                        else:
                            # A retry that is no more complete is not an improvement;
                            # keep what the planner produced first.
                            logger.info("Retry was not more complete; keeping the original plan.")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Completeness retry failed (%s); keeping the original plan.", exc)

        # -------------------------------------------------------------- validate
        # Every call is checked BEFORE anything is proposed. Approving a plan
        # whose third step cannot run wastes the approval, and at this point
        # nothing has been written.
        for call in calls:
            tool_id = _tool_id_of(call)
            meta = A.TOOL_REGISTRY.get(tool_id) or {}
            schema = meta.get("parameters")
            if not schema:
                continue
            args = _args_of(call)
            try:
                A._validate_tool_args(schema, args)
            except jsonschema.ValidationError as ve:
                missing = A._missing_required_fields(args, schema)
                if missing:
                    if clarify_attempts_base + 1 > A.CLARIFY_MAX_ATTEMPTS:
                        return _finish(
                            "clarification_exhausted", A._clarification_giveup_message(tool_id, meta, missing)
                        )
                    filled = {k: v for k, v in args.items() if A._is_filled(v)}
                    question = await A._generate_clarification_question(tool_id, meta, missing, filled, turn_trace_id)
                    A.LAST_PENDING_CLARIFICATION = {
                        "tool_id": tool_id,
                        "missing_fields": missing,
                        "filled_args": filled,
                        "attempts": clarify_attempts_base + 1,
                        "question": question,
                    }
                    return _finish("awaiting_clarification", question)
                logger.error("Migration arg validation failed for %s: %s", tool_id, ve.message)
                return _finish(
                    "validation_failed",
                    f"I couldn't run `{tool_id}` — {ve.message} Please rephrase with the details corrected.",
                )

        text = _plan_text(calls)
        if len(calls) > 1:
            transcript.append({"role": "assistant", "content": f"PLAN:\n{text}"})
        await A._emit_agent_progress({"phase": "planned", "step": 1, "max_steps": A.MAX_AGENT_STEPS, "plan": text})

        # ------------------------------------------------------------ approve
        # One gate for the whole plan. Every migration tool mutates, so there is
        # always something to approve; a read-only plan would skip this entirely.
        if any((A.TOOL_REGISTRY.get(_tool_id_of(c)) or {}).get("mutates") for c in calls):
            if A.REQUIRE_MUTATION_CONFIRM:
                plan_args = plan_arguments(calls)
                if not A._consume_approval(approved_mutations, PLAN_TOOL_ID, plan_args):
                    explanation = _render_plan_dialog(calls)
                    A.LAST_TOOL_RESULT = {
                        "ok": False,
                        "pending_confirmation": {
                            "tool_id": PLAN_TOOL_ID,
                            "arguments": plan_args,
                            "reason": explanation,
                        },
                    }
                    A.LAST_PENDING_LOOP = {
                        "transcript": transcript,
                        "raw_results": raw_results,
                        "steps_executed": steps_executed,
                        "tool_id": PLAN_TOOL_ID,
                        "arguments": plan_args,
                        "plan": calls,
                        "plan_arguments": plan_args,
                    }
                    logger.info("Migration plan awaiting approval (%d step(s)).", len(calls))
                    return _finish("pending_mutation", explanation)

    # --------------------------------------------------------------- execute
    done: List[str] = []
    for idx, call in enumerate(calls):
        tool_id = _tool_id_of(call)
        args = _args_of(call)
        meta = A.TOOL_REGISTRY.get(tool_id) or {}
        step_number = steps_executed + 1

        if steps_executed >= A.MAX_AGENT_STEPS:
            pending = [_tool_id_of(c) for c in calls[idx:]]
            return _finish(
                "step_cap",
                _report(done, None, pending, "Per-turn step limit reached — send a follow-up to continue."),
            )

        if bool(meta.get("mutates")):
            A.audit_logger.info(
                "EXECUTING mutation tool=%s args=%s",
                tool_id,
                json.dumps(A._scrub_secrets(args), ensure_ascii=False),
            )

        await A._emit_agent_progress(
            {"phase": "executing", "step": step_number, "max_steps": A.MAX_AGENT_STEPS, "tool_id": tool_id}
        )
        result = await A._invoke_tool_traced(mcp_client, tool_id, args, mode)
        ok = bool(result.get("ok")) if isinstance(result, dict) else False

        A.LAST_TOOL_RESULT = result
        A.LAST_STEP_RESULTS.append({"step": step_number, "tool_id": tool_id, "result": result})
        raw_results.append((tool_id, result))
        transcript.extend(A._transcript_step(call, tool_id, result, summ_on))
        steps_executed += 1
        trace["tool_selected"] = trace.get("tool_selected") or tool_id
        await A._emit_agent_progress(
            {
                "phase": "completed",
                "step": step_number,
                "max_steps": A.MAX_AGENT_STEPS,
                "tool_id": tool_id,
                "ok": ok,
            }
        )

        if not ok:
            reason = str((result or {}).get("error") or "no reason reported").strip()
            skipped = [_tool_id_of(c) for c in calls[idx + 1 :]]
            logger.warning("Migration stopped at %s: %s (skipping %d)", tool_id, reason[:200], len(skipped))
            body = A._describe_results_local(raw_results) if not summ_on else ""
            return _finish("migration_failed", _report(done, (tool_id, reason), skipped, body))

        done.append(tool_id)

    # ----------------------------------------------------------------- done
    if not summ_on:
        return _finish("ok", A._describe_results_local(raw_results))
    answer = await A._finalize_from_transcript(
        latest_user_message=latest_user_message,
        history=history,
        transcript=transcript,
        raw_results=raw_results,
        summ_on=summ_on,
        turn_trace_id=turn_trace_id,
    )
    trace["summarization_used"] = True
    return _finish("ok", answer)
