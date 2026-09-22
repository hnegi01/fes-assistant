"""
skill_flow — the typed plan, its validation, the one approval, and the runtime.

What must hold (docs/design/skills.md §6–§9):

  - REFERENCES resolve in code: `steps[<id>].result...` and loop variables,
    scoped so a loop body sees its own iteration and everything before the loop.
  - VALIDATION rejects, before anything is shown: a tool outside the skill's
    list, a reference to a step that has not run, a loop variable out of scope,
    a swap of a dashboard this run did not create without a validation gate.
  - THE APPROVAL KEY is the canonical plan: same plan → same key; any edit →
    a different key. One approval, single use, `skill.plan` as the tool id.
  - THE RUNTIME expands loops from live results, skips a step whose `when` is
    false, stops on the first failure, and runs the skill's declared
    COMPENSATIONS in reverse — skipping a copy-swap whose copy is about to be
    deleted, and stopping the unwind loudly if a compensation itself fails.
  - RESUME runs exactly the approved plan; a skill that changed version
    underneath it refuses rather than running an old plan against new rules.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import backend.agent.llm_agent as A
import backend.agent.skill_flow as F
from backend.agent._skills import Skill


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# A fixture registry: three reads, three writes, minimal schemas
# ---------------------------------------------------------------------------
def _schema(*required, **props):
    """Required params are strings; keyword params are string unless given None (untyped)."""
    return {
        "type": "object",
        "properties": {p: ({} if v is None else {"type": "string"}) for p, v in props.items()}
        | {p: {"type": "string"} for p in required},
        "required": list(required),
    }


REGISTRY = {
    "dm.analyze": {"mutates": False, "description": "Analyze the model.", "parameters": _schema("datamodel")},
    "dm.create": {
        "mutates": True,
        "description": "Create a perspective.",
        "parameters": _schema("datamodel", "name", tables=None),
    },
    "dm.delete": {"mutates": True, "description": "Delete a perspective.", "parameters": _schema("perspective")},
    "dm.deploy": {"mutates": True, "description": "Build a model.", "parameters": _schema("datamodel_name")},
    "db.list": {"mutates": False, "description": "Find dashboards on a model.", "parameters": _schema("datamodel")},
    "db.duplicate_dashboard": {"mutates": True, "description": "Copy a dashboard.", "parameters": _schema("dashboard")},
    "db.replace_datasource": {
        "mutates": True,
        "description": "Swap a dashboard's datasource.",
        "parameters": _schema("dashboard", "datasource"),
    },
    "db.validate_dashboard_queries": {
        "mutates": False,
        "description": "Validate widgets.",
        "parameters": _schema("dashboard"),
    },
    "db.delete_dashboard": {
        "mutates": True,
        "description": "Delete a dashboard.",
        "parameters": _schema("dashboard_id", "title"),
    },
    "db.compare_dashboard_values": {
        "mutates": False,
        "description": "Compare a dashboard's values on two datasources.",
        "parameters": _schema("dashboard", "datasource_a", "datasource_b"),
    },
    "access_management.get_my_user": {"mutates": False, "description": "Who am I.", "parameters": {"type": "object"}},
}

SKILL = Skill(
    name="fixture",
    description="A fixture procedure.",
    version=2,
    tools=tuple(t for t in REGISTRY if t != "access_management.get_my_user"),
    body="## Procedure\n1. things",
    path=Path("/fixture/fixture/SKILL.md"),
    requires_role="dataDesigner",
    guardrails=({"id": "validate-before-swap", "rule": "..."},),
    compensations={
        "dm.create": {"tool": "dm.delete", "args": {"perspective": "{args.name}"}},
        "db.duplicate_dashboard": {
            "tool": "db.delete_dashboard",
            "args": {"dashboard_id": "{result.oid}", "title": "{result.title}"},
        },
        "db.replace_datasource": {
            "tool": "db.replace_datasource",
            "args": {"dashboard": "{args.dashboard}", "datasource": "{result.previous_datasource}"},
        },
    },
)

# The full write plan, as the planner would emit it.
FULL_PLAN = {
    "steps": [
        {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
        {
            "id": "2",
            "tool": "dm.create",
            "args": {"datamodel": "M", "name": "M_AI"},
            "args_from": {"tables": "steps[1].result.tables"},
        },
        {"id": "3", "tool": "db.list", "args": {"datamodel": "M"}},
        {
            "id": "4",
            "for_each": "steps[3].result[*]",
            "as": "dash",
            "steps": [
                {"id": "4a", "tool": "db.duplicate_dashboard", "args_from": {"dashboard": "dash.oid"}},
                {
                    "id": "4b",
                    "tool": "db.replace_datasource",
                    "args": {"datasource": "M_AI"},
                    "args_from": {"dashboard": "steps[4a].result.oid"},
                },
                {
                    "id": "4c",
                    "tool": "db.validate_dashboard_queries",
                    "args_from": {"dashboard": "steps[4a].result.oid"},
                },
                {
                    "id": "4d",
                    "tool": "db.replace_datasource",
                    "args": {"datasource": "M_AI"},
                    "args_from": {"dashboard": "dash.oid"},
                    "when": "steps[4c].result.failed == 0",
                },
                {
                    "id": "4e",
                    "tool": "db.delete_dashboard",
                    "args_from": {"dashboard_id": "steps[4a].result.oid", "title": "steps[4a].result.title"},
                },
            ],
        },
    ]
}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(A, "TOOL_REGISTRY", dict(REGISTRY))
    monkeypatch.setattr(A, "REQUIRE_MUTATION_CONFIRM", True)
    monkeypatch.setattr(A, "_write_llm_trace", lambda trace: None)
    monkeypatch.setattr(A, "_emit_agent_progress", AsyncMock())
    monkeypatch.setattr(A, "_describe_results_local", lambda raw: f"({len(raw)} results)")
    monkeypatch.setattr(A, "_record_tool_result", lambda r: None)
    monkeypatch.setattr(A, "_record_step", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# The path language
# ---------------------------------------------------------------------------
class TestPaths:
    def test_step_result_and_dotted_keys(self):
        sc = F.Scope()
        sc.results["1"] = {"a": {"b": [10, 20]}}
        assert F.resolve_path("steps[1].result.a.b[1]", sc) == 20
        assert F.resolve_path("steps[1].result", sc) == {"a": {"b": [10, 20]}}

    def test_star_over_list_and_map_over_records(self):
        sc = F.Scope()
        sc.results["3"] = [{"oid": "x"}, {"oid": "y"}]
        assert F.resolve_path("steps[3].result[*]", sc) == [{"oid": "x"}, {"oid": "y"}]
        assert F.resolve_path("steps[3].result[*].oid", sc) == ["x", "y"]

    def test_loop_variable_and_scope_chain(self):
        root = F.Scope()
        root.results["1"] = {"n": 1}
        child = F.Scope(root)
        child.vars["dash"] = {"oid": "d1"}
        child.results["4a"] = {"oid": "copy1"}
        assert F.resolve_path("dash.oid", child) == "d1"
        assert F.resolve_path("steps[1].result.n", child) == 1, "parent results are visible"
        assert F.resolve_path("steps[4a].result.oid", child) == "copy1"
        with pytest.raises(F.PlanError, match="has not run yet"):
            F.resolve_path("steps[4a].result.oid", root)  # iteration-local, not global

    @pytest.mark.parametrize(
        "expr, msg",
        [
            ("steps[9].result", "has not run yet"),
            ("steps[1].result.nope", "no field"),
            ("steps[1].result[3]", "out of range"),
            ("dash.oid", "not a loop variable"),
            ("steps[1]result", "malformed"),
        ],
    )
    def test_errors_are_specific(self, expr, msg):
        sc = F.Scope()
        sc.results["1"] = {"a": [1]}
        with pytest.raises(F.PlanError, match=msg):
            F.resolve_path(expr, sc)

    def test_when(self):
        sc = F.Scope()
        sc.results["4c"] = {"failed": 0, "ok": 3, "name": "x"}
        assert F.eval_when("steps[4c].result.failed == 0", sc) is True
        assert F.eval_when("steps[4c].result.failed != 0", sc) is False
        assert F.eval_when('steps[4c].result.name == "x"', sc) is True
        assert F.eval_when("steps[4c].result.ok", sc) is True
        assert F.eval_when("steps[4c].result.failed", sc) is False


# ---------------------------------------------------------------------------
# Parse + validate
# ---------------------------------------------------------------------------
class TestValidation:
    def test_full_plan_is_valid(self):
        items = F.parse_plan(FULL_PLAN)
        F.validate_plan(items, SKILL)
        assert F.has_mutation(items)

    def test_tool_outside_the_skill_is_rejected(self):
        plan = {"steps": [{"id": "1", "tool": "access_management.get_my_user", "args": {}}]}
        with pytest.raises(F.PlanError, match="does not allow"):
            F.validate_plan(F.parse_plan(plan), SKILL)

    def test_reference_to_a_later_step_is_rejected(self):
        plan = {
            "steps": [
                {
                    "id": "1",
                    "tool": "dm.create",
                    "args": {"datamodel": "M", "name": "N"},
                    "args_from": {"tables": "steps[2].result.t"},
                },
                {"id": "2", "tool": "dm.analyze", "args": {"datamodel": "M"}},
            ]
        }
        with pytest.raises(F.PlanError, match="does not run before it"):
            F.validate_plan(F.parse_plan(plan), SKILL)

    def test_loop_variable_outside_its_loop_is_rejected(self):
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {"id": "2", "tool": "db.duplicate_dashboard", "args_from": {"dashboard": "dash.oid"}},
            ]
        }
        with pytest.raises(F.PlanError, match="not in scope"):
            F.validate_plan(F.parse_plan(plan), SKILL)

    def test_missing_required_literal_is_named(self):
        plan = {"steps": [{"id": "1", "tool": "dm.analyze", "args": {}}]}
        with pytest.raises(F.PlanError, match="missing required value.*datamodel"):
            F.validate_plan(F.parse_plan(plan), SKILL)

    def test_unknown_parameter_is_rejected(self):
        plan = {"steps": [{"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M", "bogus": 1}}]}
        with pytest.raises(F.PlanError, match="has no parameter"):
            F.validate_plan(F.parse_plan(plan), SKILL)

    def test_nested_loops_are_refused_at_parse(self):
        plan = {
            "steps": [
                {
                    "id": "1",
                    "for_each": "steps[0].result[*]",
                    "as": "a",
                    "steps": [{"id": "2", "for_each": "a.x", "as": "b", "steps": []}],
                }
            ]
        }
        with pytest.raises(F.PlanError, match="nested"):
            F.parse_plan(plan)

    def test_guardrail_rejects_swap_of_original_without_validation(self):
        plan = {
            "steps": [
                {"id": "3", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "4",
                    "for_each": "steps[3].result[*]",
                    "as": "dash",
                    "steps": [
                        {
                            "id": "4d",
                            "tool": "db.replace_datasource",
                            "args": {"datasource": "M_AI"},
                            "args_from": {"dashboard": "dash.oid"},
                        },
                    ],
                },
            ]
        }
        with pytest.raises(F.PlanError, match="validate-before-swap"):
            F.validate_plan(F.parse_plan(plan), SKILL)

    def test_guardrail_allows_swapping_the_copy(self):
        plan = {
            "steps": [
                {"id": "3", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "4",
                    "for_each": "steps[3].result[*]",
                    "as": "dash",
                    "steps": [
                        {"id": "4a", "tool": "db.duplicate_dashboard", "args_from": {"dashboard": "dash.oid"}},
                        {
                            "id": "4b",
                            "tool": "db.replace_datasource",
                            "args": {"datasource": "M_AI"},
                            "args_from": {"dashboard": "steps[4a].result.oid"},
                        },
                    ],
                },
            ]
        }
        F.validate_plan(F.parse_plan(plan), SKILL)  # no raise

    def test_step_ceiling(self):
        plan = {"steps": [{"id": str(i), "tool": "dm.analyze", "args": {"datamodel": "M"}} for i in range(5)]}
        with pytest.raises(F.PlanError, match="ceiling"):
            F.validate_plan(F.parse_plan(plan), SKILL, max_steps=4)


# ---------------------------------------------------------------------------
# Approval payload + dialog
# ---------------------------------------------------------------------------
class TestApprovalPayload:
    def test_same_plan_same_key_edited_plan_new_key(self):
        a = F.plan_arguments(F.parse_plan(FULL_PLAN), SKILL)
        b = F.plan_arguments(F.parse_plan(FULL_PLAN), SKILL)
        assert A._approval_key(F.PLAN_TOOL_ID, a) == A._approval_key(F.PLAN_TOOL_ID, b)
        edited = {"steps": [dict(FULL_PLAN["steps"][0], args={"datamodel": "OTHER"})] + FULL_PLAN["steps"][1:]}
        c = F.plan_arguments(F.parse_plan(edited), SKILL)
        assert A._approval_key(F.PLAN_TOOL_ID, a) != A._approval_key(F.PLAN_TOOL_ID, c)
        assert a["skill"] == "fixture" and a["skill_version"] == 2

    def test_dialog_lists_writes_and_summarises_reads(self):
        text = F.render_dialog(F.parse_plan(FULL_PLAN), SKILL)
        assert "Create a perspective" in text and "Delete a dashboard" in text
        assert "Analyze the model" not in text, "reads are not itemised"
        assert "3 read-only steps" in text
        assert text.startswith("Using the skill **fixture**") and "v2" not in text
        assert "only if steps[4c].result.failed == 0" in text
        assert "from step 1's result" in text, "derived args say where they come from"
        assert "undoes what it created" in text


# ---------------------------------------------------------------------------
# Role gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "have, need, expected",
    [
        ("viewer", "dataDesigner", False),
        ("dataDesigner", "dataDesigner", True),
        ("admin", "dataDesigner", True),
        ("sysAdmin", "admin", True),
        ("mystery", "admin", None),
        ("viewer", None, True),
    ],
)
def test_role_satisfies(have, need, expected):
    assert F.role_satisfies(have, need) is expected


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------
def _invoker(results_by_tool, fail_on=None, fail_comp=None):
    """A fake _invoke_tool_traced: per-tool canned payloads, optional failures."""
    calls = []

    async def invoke(mcp, tool_id, args, mode):
        calls.append((tool_id, dict(args)))
        if fail_on and tool_id == fail_on[0] and len([c for c in calls if c[0] == tool_id]) == fail_on[1]:
            return {"tool_id": tool_id, "ok": False, "error": "boom"}
        if fail_comp and tool_id == fail_comp:
            return {"tool_id": tool_id, "ok": False, "error": "cannot undo"}
        payload = results_by_tool[tool_id]
        payload = payload(args) if callable(payload) else payload
        return {"tool_id": tool_id, "ok": True, "result": payload}

    invoke.calls = calls
    return invoke


RESULTS = {
    "dm.analyze": {"tables": [{"table": "t1", "columns": "all"}]},
    "dm.create": {"name": "M_AI"},
    "db.list": [{"oid": "d1", "title": "One"}, {"oid": "d2", "title": "Two"}],
    "db.duplicate_dashboard": lambda a: {"oid": f"copy-{a['dashboard']}", "title": f"{a['dashboard']}_stage"},
    "db.replace_datasource": lambda a: {"previous_datasource": "M", "published": True},
    "db.validate_dashboard_queries": lambda a: {"failed": 1 if a["dashboard"] == "copy-d2" else 0},
    "db.delete_dashboard": {},
    "dm.delete": {},
    "dm.deploy": {},
    "db.compare_dashboard_values": lambda a: {"all_match": a["dashboard"] != "d2", "compared": 2, "skipped": 0},
}


def _ctx():
    return dict(
        mcp_client=object(), mode="chat", summ_on=False, transcript=[], raw_results=[], steps_executed=0, trace={}
    )


class TestRuntime:
    def test_happy_path_expands_loop_resolves_refs_and_skips_on_when(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        items = F.parse_plan(FULL_PLAN)
        report, n, _, _ = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))

        tools = [c[0] for c in inv.calls]
        assert report.failed is None and report.outcome == "ok"
        # 1 analyze, 1 create, 1 list, then per dashboard: dup, swap-copy, validate, (swap-original?), delete
        assert tools[:3] == ["dm.analyze", "dm.create", "db.list"]
        assert tools.count("db.duplicate_dashboard") == 2
        # d1 validated clean → original swapped; d2 had a failure → skipped
        swaps = [c for c in inv.calls if c[0] == "db.replace_datasource"]
        assert [c[1]["dashboard"] for c in swaps] == ["copy-d1", "d1", "copy-d2"]
        assert report.skipped == ["db.replace_datasource"]
        # args_from resolved: create got tables from analyze's result
        create = next(c for c in inv.calls if c[0] == "dm.create")
        assert create[1]["tables"] == [{"table": "t1", "columns": "all"}]
        assert n == len(inv.calls)

    def test_failure_stops_and_compensates_in_reverse_skipping_copy_swaps(self, monkeypatch):
        # Fail the SECOND duplicate (d2). Completed mutations by then: create, dup(d1),
        # swap(copy-d1), swap(d1 original), delete(copy-d1).
        inv = _invoker(RESULTS, fail_on=("db.duplicate_dashboard", 2))
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        items = F.parse_plan(FULL_PLAN)
        report, _, _, _ = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))

        assert report.failed == ("db.duplicate_dashboard", "boom")
        assert report.outcome == "skill_failed"
        # not attempted: the rest of iteration 2 (4b..4e) after the failed 4a
        assert report.not_attempted == [
            "db.replace_datasource",
            "db.validate_dashboard_queries",
            "db.replace_datasource",
            "db.delete_dashboard",
        ]
        # Compensation order = reverse of completed mutations that declare one:
        #   delete(copy-d1)   — no compensation declared, and it is what REMOVED copy-d1
        #   swap(d1 original) — reverted to its previous datasource, from the result
        #   swap(copy-d1)     — SKIPPED: the copy is gone
        #   dup(d1)           — SKIPPED: its copy was already deleted in-plan (else a 404 would stop the unwind)
        #   create            — perspective deleted
        comps = inv.calls[inv.calls.index(("db.duplicate_dashboard", {"dashboard": "d2"})) + 1 :]
        assert [c[0] for c in comps] == ["db.replace_datasource", "dm.delete"]
        assert comps[0] == ("db.replace_datasource", {"dashboard": "d1", "datasource": "M"})
        assert report.compensated == ["db.replace_datasource", "dm.delete"] and report.compensation_failed is None
        assert ("dm.delete", {"perspective": "M_AI"}) in comps, "template {args.name} resolved from the step's args"

    def test_failed_compensation_stops_the_unwind_loudly(self, monkeypatch):
        inv = _invoker(RESULTS, fail_on=("db.list", 1), fail_comp="dm.delete")
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        items = F.parse_plan(FULL_PLAN)
        report, _, _, _ = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))
        assert report.failed == ("db.list", "boom")
        assert report.compensation_failed == ("dm.delete", "cannot undo")
        assert "Could not revert" in report.text("")

    def test_unresolvable_reference_at_runtime_is_a_reported_failure(self, monkeypatch):
        results = dict(RESULTS, **{"dm.analyze": {"nothing": 1}})  # no `tables`
        inv = _invoker(results)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        items = F.parse_plan(FULL_PLAN)
        report, _, _, _ = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))
        assert report.outcome == "skill_failed"
        assert report.failed[0] == "plan" and "no field 'tables'" in report.failed[1]
        assert [c[0] for c in inv.calls] == ["dm.analyze"], "stopped before the create, nothing to compensate"


# ---------------------------------------------------------------------------
# run(): gate, resume, version pin
# ---------------------------------------------------------------------------
def _run_kwargs(**over):
    base = dict(
        latest_user_message={"role": "user", "content": "make M ready"},
        history=[],
        planning_context="",
        mode="chat",
        passed_tools=[],
        user_text="make M ready",
        mcp_client=object(),
        approved_mutations=set(),
        summ_on=False,
        turn_trace_id="t",
        trace={},
    )
    base.update(over)
    return base


class TestRun:
    def test_fresh_plan_with_writes_gates_once_and_records_pending(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(A, "_record_pending_loop", lambda v: recorded.setdefault("loop", v))
        # The look-first read records its own tool result first; the gate's
        # pending_confirmation is the LAST thing recorded.
        monkeypatch.setattr(A, "_record_tool_result", lambda v: recorded.__setitem__("result", v))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)

        reply = _run(F.run(skill=SKILL, plan=FULL_PLAN, **_run_kwargs()))
        assert "Approve to run the whole sequence" in reply
        # Look first: the read that precedes the first write ran; no write did.
        assert [c[0] for c in inv.calls] == ["dm.analyze"], "reads before the gate, never a write"
        pl = recorded["loop"]
        assert pl["tool_id"] == F.PLAN_TOOL_ID
        assert pl["skill"] == {"name": "fixture", "version": 2}
        assert pl["plan"]["steps"][0]["tool"] == "dm.analyze"
        assert pl["consumed"] == 1 and pl["ran"] == ["dm.analyze"]
        assert pl["scope"]["results"]["1"] == RESULTS["dm.analyze"], "the paused scope carries the read result"
        pc = recorded["result"]["pending_confirmation"]
        assert pc["tool_id"] == F.PLAN_TOOL_ID
        assert "Using the skill **fixture**" in pc["details"], "the exact operation list rides `details`"
        assert "v2" not in reply and "procedure" not in reply.lower()

    def test_read_only_plan_runs_without_a_gate(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {"id": "2", "tool": "db.list", "args": {"datamodel": "M"}},
            ]
        }
        reply = _run(F.run(skill=SKILL, plan=plan, **_run_kwargs()))
        assert [c[0] for c in inv.calls] == ["dm.analyze", "db.list"]
        assert "Done" in reply

    def test_viewer_gets_the_reads_and_a_handoff(self, monkeypatch):
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="viewer"))
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        reply = _run(F.run(skill=SKILL, plan=FULL_PLAN, **_run_kwargs()))
        assert "needs the **dataDesigner** role" in reply and "your token has **viewer**" in reply
        assert all(not REGISTRY[c[0]]["mutates"] for c in inv.calls), "no write ran"
        assert [c[0] for c in inv.calls] == [
            "dm.analyze",
            "db.list",
            "db.validate_dashboard_queries",
            "db.validate_dashboard_queries",
        ] or [c[0] for c in inv.calls][:2] == ["dm.analyze", "db.list"]

    def test_resume_with_matching_approval_executes(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        items = F.parse_plan(FULL_PLAN)
        plan_args = F.plan_arguments(items, SKILL)
        pending = {
            "tool_id": F.PLAN_TOOL_ID,
            "plan": {"steps": plan_args["steps"]},
            "plan_arguments": plan_args,
            "skill": {"name": "fixture", "version": 2},
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, plan_args)}
        reply = _run(F.run(pending_plan=pending, **_run_kwargs(approved_mutations=approved)))
        assert inv.calls and inv.calls[0][0] == "dm.analyze"
        assert "Done" in reply
        assert approved == set(), "the approval was consumed"

    def test_resume_without_approval_runs_nothing(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        plan_args = F.plan_arguments(F.parse_plan(FULL_PLAN), SKILL)
        pending = {
            "tool_id": F.PLAN_TOOL_ID,
            "plan": {"steps": plan_args["steps"]},
            "plan_arguments": plan_args,
            "skill": {"name": "fixture", "version": 2},
        }
        reply = _run(F.run(pending_plan=pending, **_run_kwargs()))
        assert inv.calls == [] and "not approved" in reply

    def test_resume_refuses_when_the_skill_changed_version(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})  # v2 on disk
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        plan_args = F.plan_arguments(F.parse_plan(FULL_PLAN), SKILL)
        pending = {
            "tool_id": F.PLAN_TOOL_ID,
            "plan": {"steps": plan_args["steps"]},
            "plan_arguments": plan_args,
            "skill": {"name": "fixture", "version": 1},
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, plan_args)}
        reply = _run(F.run(pending_plan=pending, **_run_kwargs(approved_mutations=approved)))
        assert inv.calls == []
        assert "changed after this plan was proposed" in reply and "v1" not in reply

    def test_invalid_plan_is_reported_not_run(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        plan = {"steps": [{"id": "1", "tool": "access_management.get_my_user", "args": {}}]}
        reply = _run(F.run(skill=SKILL, plan=plan, **_run_kwargs()))
        assert inv.calls == [] and "could not build a safe plan" in reply and "does not allow" in reply


# ---------------------------------------------------------------------------
# The response names the skill — set by the runtime, fresh and resume alike
# ---------------------------------------------------------------------------
class TestSkillInTurnOutput:
    def _with_turn(self, tid):
        from backend.agent._config import begin_turn_output, set_current_turn

        set_current_turn(tid, "make M ready")
        begin_turn_output(tid)

    def test_fresh_run_records_the_skill(self, monkeypatch):
        from backend.agent._config import pop_turn_output

        self._with_turn("t-skill-fresh")
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        plan = {"steps": [{"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}}]}
        _run(F.run(skill=SKILL, plan=plan, **_run_kwargs(turn_trace_id="t-skill-fresh")))
        assert pop_turn_output("t-skill-fresh")["skill"] == {"name": "fixture", "version": 2}

    def test_resume_records_the_skill(self, monkeypatch):
        from backend.agent._config import pop_turn_output

        self._with_turn("t-skill-resume")
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        plan_args = F.plan_arguments(F.parse_plan(FULL_PLAN), SKILL)
        pending = {
            "tool_id": F.PLAN_TOOL_ID,
            "plan": {"steps": plan_args["steps"]},
            "plan_arguments": plan_args,
            "skill": {"name": "fixture", "version": 2},
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, plan_args)}
        _run(F.run(pending_plan=pending, **_run_kwargs(approved_mutations=approved, turn_trace_id="t-skill-resume")))
        assert pop_turn_output("t-skill-resume")["skill"] == {"name": "fixture", "version": 2}

    def test_ordinary_turn_output_defaults_to_no_skill(self):
        from backend.agent._config import pop_turn_output

        assert pop_turn_output("never-opened")["skill"] is None


class TestTemplateRendering:
    def test_dotted_result_path_reaches_a_nested_field(self):
        payload = {"previous_datasource": {"title": "Sales Root", "id": "x"}, "title": "Copy"}
        assert F._render_template("{result.previous_datasource.title}", {}, payload) == "Sales Root"
        assert F._render_template("{result.title}", {}, payload) == "Copy"
        assert F._render_template("{args.dashboard}", {"dashboard": "D"}, payload) == "D"

    def test_missing_segment_is_none_not_a_crash(self):
        assert F._render_template("{result.previous_datasource.title}", {}, {"previous_datasource": "M"}) is None
        assert F._render_template("{result.nope.deeper}", {}, {}) is None
        assert F._render_template("literal", {}, {}) == "literal"


class TestArgsReferences:
    """`steps[<id>].args.<param>` reads back what an earlier step ran with."""

    def test_resolves_from_the_scope_that_ran_the_step(self):
        sc = F.Scope()
        sc.args["2"] = {"name": "M_AI", "datamodel": "M"}
        assert F.resolve_path("steps[2].args.name", sc) == "M_AI"
        child = F.Scope(sc)
        assert F.resolve_path("steps[2].args.datamodel", child) == "M", "parent args are visible"

    def test_unrun_step_and_malformed_paths_are_named_plainly(self):
        sc = F.Scope()
        with pytest.raises(F.PlanError, match="has not run yet"):
            F.resolve_path("steps[9].args.name", sc)
        with pytest.raises(F.PlanError, match="malformed"):
            F.resolve_path("steps[2].argz.name", sc)
        assert F._referenced_var("steps[2].argz.name") is None, "a steps path is never a loop variable"

    def test_validation_accepts_a_backward_args_reference_and_rejects_forward(self):
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.create", "args": {"datamodel": "M", "name": "M_AI", "tables": ["t"]}},
                {"id": "2", "tool": "dm.deploy", "args_from": {"datamodel_name": "steps[1].args.name"}},
            ]
        }
        F.validate_plan(F.parse_plan(plan), SKILL)
        bad = {
            "steps": [
                {"id": "1", "tool": "dm.deploy", "args_from": {"datamodel_name": "steps[2].args.name"}},
                {"id": "2", "tool": "dm.create", "args": {"datamodel": "M", "name": "M_AI", "tables": ["t"]}},
            ]
        }
        with pytest.raises(F.PlanError, match="does not run before it"):
            F.validate_plan(F.parse_plan(bad), SKILL)

    def test_runtime_feeds_the_created_name_into_the_build(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.create", "args": {"datamodel": "M", "name": "M_AI", "tables": ["t"]}},
                {"id": "2", "tool": "dm.deploy", "args_from": {"datamodel_name": "steps[1].args.name"}},
            ]
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, F.plan_arguments(F.parse_plan(plan), SKILL))}
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        _run(F.run(skill=SKILL, plan=plan, **_run_kwargs(approved_mutations=approved)))
        assert ("dm.deploy", {"datamodel_name": "M_AI"}) in [(c[0], c[1]) for c in inv.calls]


class TestAskBeforePlanning:
    def test_question_goes_out_through_the_clarification_channel(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(A, "_record_pending_clarification", lambda v: recorded.setdefault("pc", v))
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        reply = _run(
            F.run(skill=SKILL, plan={"ask": "What should the perspective be called?", "attempts": 1}, **_run_kwargs())
        )
        assert reply == "What should the perspective be called?", "one missing value → one direct question"
        assert recorded["pc"]["tool_id"] == F.PLAN_TOOL_ID
        assert recorded["pc"]["skill"] == {"name": "fixture", "version": 2}
        assert recorded["pc"]["attempts"] == 1 and recorded["pc"]["question"] == reply
        assert inv.calls == [], "nothing runs while a question is open"

    def test_cap_reached_gives_up_plainly(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(A, "_record_pending_clarification", lambda v: recorded.setdefault("pc", v))
        monkeypatch.setattr(A, "CLARIFY_MAX_ATTEMPTS", 2)
        reply = _run(F.run(skill=SKILL, plan={"ask": "The name?", "attempts": 3}, **_run_kwargs()))
        assert "can't do this without that information" in reply and "The name?" in reply
        assert "pc" not in recorded, "no further question is pinned once the cap is hit"


class TestLiteralReferencePromotion:
    def test_a_steps_path_under_args_becomes_args_from(self):
        (step,) = F.parse_plan(
            {
                "steps": [
                    {
                        "id": "2",
                        "tool": "dm.create",
                        "args": {"datamodel": "M", "name": "N", "tables": "steps[1].result.tables"},
                    }
                ]
            }
        )
        assert step.args == {"datamodel": "M", "name": "N"}
        assert step.args_from == {"tables": "steps[1].result.tables"}

    def test_ordinary_literals_are_left_alone(self):
        (step,) = F.parse_plan(
            {"steps": [{"id": "1", "tool": "dm.analyze", "args": {"datamodel": "steps and stairs"}}]}
        )
        assert step.args == {"datamodel": "steps and stairs"} and step.args_from == {}


# ---------------------------------------------------------------------------
# Skipped steps, loop conditions, look-first gate, approval summary, resume
# ---------------------------------------------------------------------------
class TestSkippedSemantics:
    def test_condition_on_a_skipped_step_is_false_and_args_from_it_is_an_error(self):
        sc = F.Scope()
        sc.results["2"] = F.SKIPPED
        assert F.eval_when("steps[2].result", sc) is False
        assert F.eval_when("steps[2].result.ok == true", sc) is False
        assert F.eval_when("steps[2].result.ok != true", sc) is False, "any operator on a skipped step is false"
        with pytest.raises(F.PlanError, match="was skipped"):
            F.resolve_path("steps[2].result.name", sc)

    def test_chain_collapses_when_the_first_link_is_skipped(self, monkeypatch):
        results = dict(RESULTS, **{"dm.analyze": {"errors": ["unresolved_reference"], "tables": []}})
        inv = _invoker(results)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "tool": "dm.create",
                    "args": {"datamodel": "M", "name": "N"},
                    "args_from": {"tables": "steps[1].result.tables"},
                    "when": "steps[1].result.errors == []",
                },
                {
                    "id": "3",
                    "tool": "dm.deploy",
                    "args_from": {"datamodel_name": "steps[2].args.name"},
                    "when": "steps[2].result",
                },
                {"id": "4", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "5",
                    "for_each": "steps[4].result[*]",
                    "as": "d",
                    "when": "steps[1].result.errors == []",
                    "steps": [{"id": "5a", "tool": "db.duplicate_dashboard", "args_from": {"dashboard": "d.oid"}}],
                },
            ]
        }
        items = F.parse_plan(plan)
        F.validate_plan(items, SKILL)
        report, n, root, consumed = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))
        assert [c[0] for c in inv.calls] == ["dm.analyze", "db.list"], "no write ran"
        assert report.outcome == "ok" and report.skipped == ["dm.create", "dm.deploy", "loop 5"]
        assert report.skipped_because[0] == "steps[1].result.errors == []"
        assert consumed == 5

    def test_args_from_a_skipped_step_stops_the_run(self, monkeypatch):
        results = dict(RESULTS, **{"dm.analyze": {"errors": ["x"], "tables": []}})
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(results))
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "tool": "dm.create",
                    "args": {"datamodel": "M", "name": "N"},
                    "args_from": {"tables": "steps[1].result.tables"},
                    "when": "steps[1].result.errors == []",
                },
                {"id": "3", "tool": "dm.deploy", "args_from": {"datamodel_name": "steps[2].args.name"}},
            ]
        }
        report, *_ = _run(F._execute(F.parse_plan(plan), SKILL, report=F._Report(), **_ctx()))
        assert report.outcome == "skill_failed"
        assert report.failed[0] == "plan" and "was skipped" in report.failed[1]

    def test_loop_when_is_validated_and_keyed(self):
        bad = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "d",
                    "when": "steps[9].result",
                    "steps": [{"id": "2a", "tool": "db.duplicate_dashboard", "args_from": {"dashboard": "d.oid"}}],
                },
            ]
        }
        with pytest.raises(F.PlanError, match="does not run before it"):
            F.validate_plan(F.parse_plan(bad), SKILL)
        good = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "d",
                    "when": "steps[1].result",
                    "steps": [{"id": "2a", "tool": "db.duplicate_dashboard", "args_from": {"dashboard": "d.oid"}}],
                },
            ]
        }
        assert F.plan_arguments(F.parse_plan(good), SKILL)["steps"][1]["when"] == "steps[1].result"


class TestLookFirstGate:
    def test_all_writes_already_skipped_means_a_report_not_a_dialog(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(A, "_record_pending_loop", lambda v: recorded.setdefault("loop", v))
        monkeypatch.setattr(A, "_record_tool_result", lambda v: recorded.setdefault("result", v))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        results = dict(RESULTS, **{"dm.analyze": {"errors": ["unresolved_reference"], "tables": []}})
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(results))
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "tool": "dm.create",
                    "args": {"datamodel": "M", "name": "N"},
                    "args_from": {"tables": "steps[1].result.tables"},
                    "when": "steps[1].result.errors == []",
                },
            ]
        }
        reply = _run(F.run(skill=SKILL, plan=plan, **_run_kwargs()))
        assert "loop" not in recorded, "nothing to approve"
        assert "Skipped, condition not met" in reply and "steps[1].result.errors == []" not in reply, (
            "conditions stay in the logs"
        )

    def test_read_failure_before_the_gate_is_reported_not_gated(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(A, "_record_pending_loop", lambda v: recorded.setdefault("loop", v))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS, fail_on=("dm.analyze", 1)))
        reply = _run(F.run(skill=SKILL, plan=FULL_PLAN, **_run_kwargs()))
        assert "loop" not in recorded and "**Stopped**" in reply

    def test_resume_continues_after_the_consumed_reads(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        items = F.parse_plan(FULL_PLAN)
        plan_args = F.plan_arguments(items, SKILL)
        pending = {
            "tool_id": F.PLAN_TOOL_ID,
            "plan": {"steps": plan_args["steps"]},
            "plan_arguments": plan_args,
            "skill": {"name": "fixture", "version": 2},
            "scope": {"results": {"1": RESULTS["dm.analyze"]}, "args": {"1": {"datamodel": "M"}}},
            "consumed": 1,
            "ran": ["dm.analyze"],
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, plan_args)}
        reply = _run(F.run(pending_plan=pending, **_run_kwargs(approved_mutations=approved)))
        tools = [c[0] for c in inv.calls]
        assert tools[0] == "dm.create", "the analyze read is not repeated"
        create = next(c for c in inv.calls if c[0] == "dm.create")
        assert create[1]["tables"] == RESULTS["dm.analyze"]["tables"], "resolved from the restored scope"
        assert "Done" in reply and "Analyze the model" in reply, "the report still lists the pre-gate read"


class TestApprovalSummary:
    def _skill(self, approval):
        return Skill(**{**SKILL.__dict__, "approval": approval})

    def test_placeholders_fill_from_plan_args_and_read_results(self):
        items = F.parse_plan(FULL_PLAN)
        root = F.Scope()
        root.results["1"] = {"tables": [{"table": "t1", "columns": "all"}], "summary": {"model_tables": 26}}
        root.args["1"] = {"datamodel": "M"}
        root.results["3"] = [{"oid": "d1", "title": "One"}, {"oid": "d2", "title": "Two"}]
        sk = self._skill(
            "Create **{create.args.name}** on {analyze.args.datamodel} ({analyze.result.summary.model_tables} tables). "
            "Move {list.count} dashboard{list.count|s}: {list.result[*].title}. Approve?"
        )
        assert F.render_approval(items, sk, root) == (
            "Create **M_AI** on M (26 tables). Move 2 dashboards: One, Two. Approve?"
        )

    def test_unknowns_render_as_question_marks_and_cta_is_added(self):
        items = F.parse_plan(FULL_PLAN)
        sk = self._skill("Name: {create.args.name}; rows: {list.count}; nope: {ghost.args.x}.")
        text = F.render_approval(items, sk, F.Scope())
        assert text.startswith("Name: M_AI; rows: ?; nope: ?.")
        assert "Approve to proceed" in text

    def test_no_approval_section_falls_back_to_the_operation_list(self):
        text = F.render_approval(F.parse_plan(FULL_PLAN), SKILL, F.Scope())
        assert text.startswith("Using the skill **fixture**")


# ---------------------------------------------------------------------------
# args_ask: asked after the reads, filled from the answer, then gated
# ---------------------------------------------------------------------------
ASK_PLAN = {
    "steps": [
        {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
        {
            "id": "2",
            "tool": "dm.create",
            "args": {"datamodel": "M"},
            "args_from": {"tables": "steps[1].result.tables"},
            "args_ask": {"name": "What should the perspective be called?"},
        },
        {"id": "3", "tool": "db.list", "args": {"datamodel": "M"}},
    ]
}


class TestArgsAsk:
    def test_parse_and_validate_accept_an_asked_required_param(self):
        items = F.parse_plan(ASK_PLAN)
        F.validate_plan(items, SKILL)
        assert F.pending_asks(items) == [(items[1], "name")]
        assert F.plan_arguments(items, SKILL)["steps"][1]["args_ask"] == {
            "name": "What should the perspective be called?"
        }

    def test_a_param_given_twice_is_rejected(self):
        bad = {
            "steps": [
                {
                    "id": "1",
                    "tool": "dm.create",
                    "args": {"datamodel": "M", "name": "N", "tables": []},
                    "args_ask": {"name": "?"},
                }
            ]
        }
        with pytest.raises(F.PlanError, match="more than one of"):
            F.validate_plan(F.parse_plan(bad), SKILL)

    def test_fresh_run_asks_after_the_reads_with_the_findings(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(A, "_record_pending_clarification", lambda v: recorded.setdefault("pc", v))
        monkeypatch.setattr(A, "_record_pending_loop", lambda v: recorded.setdefault("loop", v))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        sk = Skill(**{**SKILL.__dict__, "ask": "Found {analyze.result.tables[0].table}. Name for the perspective"})
        reply = _run(F.run(skill=sk, plan=ASK_PLAN, **_run_kwargs()))
        assert [c[0] for c in inv.calls] == ["dm.analyze"], "the read before the first write ran first"
        assert reply == "Found t1. Name for the perspective?", "the skill's Ask text, filled from the read"
        pc = recorded["pc"]
        assert pc["tool_id"] == F.PLAN_TOOL_ID and pc["ask"] == {"step_id": "2", "param": "name"}
        assert pc["consumed"] == 1 and pc["scope"]["results"]["1"] == RESULTS["dm.analyze"]
        assert "loop" not in recorded, "no approval yet — the plan is not complete"

    def _pending(self, **over):
        items = F.parse_plan(ASK_PLAN)
        base = {
            **F._paused_state(
                SKILL,
                items,
                F.Scope.restore({"results": {"1": RESULTS["dm.analyze"]}, "args": {"1": {"datamodel": "M"}}}),
                1,
                F._Report(ran=["dm.analyze"]),
                [],
                [],
                1,
            ),
            "missing_fields": ["name"],
            "filled_args": {},
            "attempts": 1,
            "question": "What should the perspective be called?",
            "ask": {"step_id": "2", "param": "name"},
        }
        base.update(over)
        return base

    def _answer_kwargs(self, text, **over):
        base = dict(
            latest_user_message={"role": "user", "content": text},
            user_text=text,
            mode="chat",
            mcp_client=object(),
            approved_mutations=set(),
            summ_on=False,
            turn_trace_id="t",
            trace={},
        )
        base.update(over)
        return base

    def test_answer_fills_the_value_and_reaches_the_gate_without_rerunning_reads(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(A, "_record_pending_loop", lambda v: recorded.setdefault("loop", v))
        monkeypatch.setattr(A, "_record_tool_result", lambda v: recorded.__setitem__("result", v))
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(F, "_interpret_answer", AsyncMock(return_value="Sales_AI"))
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        reply = _run(F.answer(self._pending(), **self._answer_kwargs("call it Sales_AI")))
        assert inv.calls == [], "no read re-ran, no write ran"
        assert "Approve" in reply
        pl = recorded["loop"]
        create = pl["plan"]["steps"][1]
        assert create["args"]["name"] == "Sales_AI" and "args_ask" not in create, "the answer is now a literal arg"
        assert pl["consumed"] == 1 and pl["ran"] == ["dm.analyze"]

    def test_non_answer_reasks_and_a_new_request_returns_none(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(A, "_record_pending_clarification", lambda v: recorded.append(v))
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(F, "_interpret_answer", AsyncMock(return_value=None))
        monkeypatch.setattr(A, "_navigate_to_tools", AsyncMock(return_value=([], "__unclear__", None, 0)))
        reply = _run(F.answer(self._pending(), **self._answer_kwargs("hmm why?")))
        assert reply.endswith("?") and recorded[-1]["attempts"] == 2, "re-asked, counted"
        monkeypatch.setattr(A, "_navigate_to_tools", AsyncMock(return_value=([], "access_management", "users", 0)))
        assert _run(F.answer(self._pending(), **self._answer_kwargs("list all users"))) is None, "a new request"

    def test_cap_gives_up(self, monkeypatch):
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(F, "_interpret_answer", AsyncMock(return_value=None))
        monkeypatch.setattr(A, "_navigate_to_tools", AsyncMock(return_value=([], "__unclear__", None, 0)))
        monkeypatch.setattr(A, "CLARIFY_MAX_ATTEMPTS", 2)
        reply = _run(F.answer(self._pending(attempts=2), **self._answer_kwargs("dunno")))
        assert "can't continue without that" in reply

    def test_changed_skill_refuses(self, monkeypatch):
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})  # v2
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        reply = _run(F.answer(self._pending(skill={"name": "fixture", "version": 1}), **self._answer_kwargs("X")))
        assert "changed before you answered" in reply

    def test_run_step_refuses_an_unanswered_ask(self, monkeypatch):
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        report, *_ = _run(F._execute(F.parse_plan(ASK_PLAN), SKILL, report=F._Report(), **_ctx()))
        assert report.outcome == "skill_failed" and "still needs the user's answer" in report.failed[1]


class TestApprovalTextFromReferences:
    def test_unrun_step_args_given_by_reference_resolve_from_the_reads(self):
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "tool": "dm.create",
                    "args": {"name": "N"},
                    "args_from": {"datamodel": "steps[1].args.datamodel", "tables": "steps[1].result.tables"},
                },
            ]
        }
        items = F.parse_plan(plan)
        root = F.Scope()
        root.args["1"] = {"datamodel": "M"}
        root.results["1"] = RESULTS["dm.analyze"]
        sk = Skill(**{**SKILL.__dict__, "approval": "Create {create.args.name} on {create.args.datamodel}. Approve?"})
        assert F.render_approval(items, sk, root) == "Create N on M. Approve?"


class TestUnparsableSkillPass:
    def test_error_handoff_ends_honestly_and_runs_nothing(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        reply = _run(F.run(skill=SKILL, plan={"error": "1. list models"}, **_run_kwargs()))
        assert inv.calls == [] and "Nothing was changed" in reply and "rephrase" in reply


class TestLoopVariableNormalisation:
    def test_a_wrong_bare_name_inside_a_loop_means_the_loop_variable(self):
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "dashboard",
                    "steps": [
                        {"id": "2a", "tool": "db.duplicate_dashboard", "args_from": {"dashboard": "item.oid"}},
                        {
                            "id": "2b",
                            "tool": "db.validate_dashboard_queries",
                            "args_from": {"dashboard": "steps[2a].result.oid"},
                            "when": 'item.title != "x"',
                        },
                    ],
                },
            ]
        }
        items = F.parse_plan(plan)
        F.validate_plan(items, SKILL)
        assert items[1].steps[0].args_from == {"dashboard": "dashboard.oid"}
        assert items[1].steps[1].when == 'dashboard.title != "x"'
        assert items[1].steps[1].args_from == {"dashboard": "steps[2a].result.oid"}, "step references untouched"


class TestPackageSlipNormalisation:
    def test_right_method_wrong_package_is_corrected(self):
        items = F.parse_plan({"steps": [{"id": "1", "tool": "dashboard.analyze", "args": {"datamodel": "M"}}]})
        F.validate_plan(items, SKILL)
        assert items[0].tool == "dm.analyze"

    def test_unknown_method_is_still_rejected(self):
        items = F.parse_plan({"steps": [{"id": "1", "tool": "dm.explode", "args": {"datamodel": "M"}}]})
        with pytest.raises(F.PlanError, match="does not allow"):
            F.validate_plan(items, SKILL)


# ---------------------------------------------------------------------------
# Choosing references, tolerant conditions, text filters (analysis 75eaa9a shape)
# ---------------------------------------------------------------------------
class TestChoosingReferences:
    def _scope(self, warnings):
        sc = F.Scope()
        sc.results["1"] = {
            "perspective_tables": [{"table": "A", "columns": ["x"]}],
            "perspective_tables_all_paths": [{"table": "A", "columns": ["x"]}, {"table": "B", "columns": ["k"]}],
            "warnings": warnings,
            "errors": [],
        }
        return sc

    EXPR = (
        "steps[1].result.warnings.ambiguous_join_path != null ? "
        "steps[1].result.perspective_tables_all_paths : steps[1].result.perspective_tables"
    )

    def test_absent_warning_key_picks_the_exact_list(self):
        assert F.resolve_path(self.EXPR, self._scope({})) == [{"table": "A", "columns": ["x"]}]

    def test_present_warning_picks_the_safe_superset(self):
        assert len(F.resolve_path(self.EXPR, self._scope({"ambiguous_join_path": 1}))) == 2

    def test_condition_on_a_missing_key_is_false_not_an_error(self):
        sc = self._scope({})
        assert F.eval_when("steps[1].result.warnings.ambiguous_join_path", sc) is False
        assert F.eval_when("steps[1].result.warnings.ambiguous_join_path == null", sc) is True
        with pytest.raises(F.PlanError, match="no field"):
            F.resolve_path("steps[1].result.warnings.ambiguous_join_path", sc), "args_from stays strict"

    def test_validation_sees_every_step_a_ternary_names(self):
        ok = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "tool": "dm.create",
                    "args": {"datamodel": "M", "name": "N"},
                    "args_from": {"tables": self.EXPR},
                },
            ]
        }
        F.validate_plan(F.parse_plan(ok), SKILL)
        forward = {
            "steps": [
                {
                    "id": "1",
                    "tool": "dm.create",
                    "args": {"datamodel": "M", "name": "N"},
                    "args_from": {"tables": "steps[2].result.x == 1 ? steps[2].result.a : steps[2].result.b"},
                },
                {"id": "2", "tool": "dm.analyze", "args": {"datamodel": "M"}},
            ]
        }
        with pytest.raises(F.PlanError, match="does not run before it"):
            F.validate_plan(F.parse_plan(forward), SKILL)

    def test_runtime_feeds_the_chosen_list_into_create(self, monkeypatch):
        results = dict(RESULTS, **{"dm.analyze": self._scope({"ambiguous_join_path": 2}).results["1"]})
        inv = _invoker(results)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "tool": "dm.create",
                    "args": {"datamodel": "M", "name": "N"},
                    "args_from": {"tables": self.EXPR},
                },
            ]
        }
        report, *_ = _run(F._execute(F.parse_plan(plan), SKILL, report=F._Report(), **_ctx()))
        assert report.outcome == "ok"
        create = next(c for c in inv.calls if c[0] == "dm.create")
        assert len(create[1]["tables"]) == 2, "the superset, because the analysis flagged an ambiguous join"


class TestTextFilters:
    def test_count_and_or_filters(self):
        items = F.parse_plan(
            {
                "steps": [
                    {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                    {
                        "id": "2",
                        "tool": "dm.create",
                        "args": {"datamodel": "M", "name": "N", "tables": [{"table": "A"}, {"table": "B"}]},
                    },
                ]
            }
        )
        root = F.Scope()
        root.results["1"] = {"warnings": {}, "summary": {"model_tables": 26}}
        sk = Skill(
            **{
                **SKILL.__dict__,
                "approval": (
                    "{create.args.tables|count} tables; "
                    "ambiguous: {analyze.result.warnings.ambiguous_join_path|or:none}; "
                    "all-paths: {analyze.result.summary.tables_required_all_paths|or:?}; "
                    "{ghost.args.x|or:n/a}. Approve?"
                ),
            }
        )
        assert F.render_approval(items, sk, root) == "2 tables; ambiguous: none; all-paths: ?; n/a. Approve?"


class TestCompareBeforeSwap:
    """The two-step per-dashboard block: compare on both datasources, then swap
    the original only on all_match. No copy, no validate, no delete."""

    PLAN = {
        "steps": [
            {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
            {
                "id": "2",
                "for_each": "steps[1].result[*]",
                "as": "d",
                "steps": [
                    {
                        "id": "2a",
                        "tool": "db.compare_dashboard_values",
                        "args": {"datasource_a": "M", "datasource_b": "M_AI"},
                        "args_from": {"dashboard": "d.oid"},
                    },
                    {
                        "id": "2b",
                        "tool": "db.replace_datasource",
                        "args": {"datasource": "M_AI"},
                        "args_from": {"dashboard": "d.oid"},
                        "when": "steps[2a].result.all_match == true",
                    },
                ],
            },
        ]
    }

    def test_compare_satisfies_the_guardrail(self):
        items = F.parse_plan(self.PLAN)
        F.validate_plan(items, SKILL)  # would raise if the guardrail did not accept compare as proof

    def test_swap_without_any_proof_is_still_rejected(self):
        bad = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "d",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.replace_datasource",
                            "args": {"datasource": "M_AI"},
                            "args_from": {"dashboard": "d.oid"},
                        },
                    ],
                },
            ]
        }
        with pytest.raises(F.PlanError, match="compare_dashboard_values result"):
            F.validate_plan(F.parse_plan(bad), SKILL)

    def test_runtime_swaps_only_matching_dashboards(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        report, *_ = _run(F._execute(F.parse_plan(self.PLAN), SKILL, report=F._Report(), **_ctx()))
        swaps = [c[1]["dashboard"] for c in inv.calls if c[0] == "db.replace_datasource"]
        assert swaps == ["d1"], "d2 did not match, so its original was left alone"
        assert report.skipped == ["db.replace_datasource"] and report.outcome == "ok"


class TestLiteralLoopItemPromotion:
    def test_item_field_under_args_becomes_a_loop_reference(self):
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "dashboard",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.compare_dashboard_values",
                            "args": {"dashboard": "item.oid", "datasource_a": "M", "datasource_b": "M_AI"},
                        },
                    ],
                },
            ]
        }
        items = F.parse_plan(plan)
        F.validate_plan(items, SKILL)
        st = items[1].steps[0]
        assert st.args == {"datasource_a": "M", "datasource_b": "M_AI"}
        assert st.args_from == {"dashboard": "dashboard.oid"}

    def test_a_dotted_literal_that_is_not_the_item_stays_literal(self):
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "d",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.compare_dashboard_values",
                            "args": {"datasource_a": "Sales.Overview", "datasource_b": "M_AI"},
                            "args_from": {"dashboard": "d.oid"},
                        },
                    ],
                },
            ]
        }
        st = F.parse_plan(plan)[1].steps[0]
        assert st.args["datasource_a"] == "Sales.Overview"


class TestProgressStages:
    """The stages a skill run announces, in order, from the code that performs them."""

    def _events(self, monkeypatch):
        seen = []

        async def _emit(ev):
            seen.append(ev)

        monkeypatch.setattr(A, "_emit_agent_progress", _emit)
        return seen

    def test_ask_run_announces_labelled_reads_then_waiting_for_answer(self, monkeypatch):
        seen = self._events(monkeypatch)
        monkeypatch.setattr(A, "_record_pending_clarification", lambda v: None)
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        sk = Skill(**{**SKILL.__dict__, "step_labels": {"dm.analyze": "Analysing the model"}})
        _run(F.run(skill=sk, plan=ASK_PLAN, **_run_kwargs()))
        phases = [e["phase"] for e in seen]
        assert phases[:1] == ["planned"] and phases[-1] == "awaiting_answer"
        ex = next(e for e in seen if e["phase"] == "executing")
        assert ex["label"] == "Analysing the model" and ex["skill"] == "fixture"
        done = next(e for e in seen if e["phase"] == "completed")
        assert done["label"] == "Analysing the model"

    def test_gate_announces_waiting_for_approval(self, monkeypatch):
        seen = self._events(monkeypatch)
        monkeypatch.setattr(A, "_record_pending_loop", lambda v: None)
        monkeypatch.setattr(A, "_record_tool_result", lambda v: None)
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        _run(F.run(skill=SKILL, plan=FULL_PLAN, **_run_kwargs()))
        assert [e["phase"] for e in seen][-1] == "awaiting_approval"

    def test_loop_steps_carry_the_loop_variable_and_done_closes(self, monkeypatch):
        seen = self._events(monkeypatch)
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "dashboard",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.validate_dashboard_queries",
                            "args_from": {"dashboard": "dashboard.oid"},
                        },
                    ],
                },
            ]
        }
        _run(F.run(skill=SKILL, plan=plan, **_run_kwargs()))
        loop_ex = [e for e in seen if e["phase"] == "executing" and e.get("loop_index")]
        assert [(e["loop_index"], e["loop_total"], e["loop_var"]) for e in loop_ex] == [
            (1, 2, "dashboard"),
            (2, 2, "dashboard"),
        ]
        assert seen[-1]["phase"] == "done" and seen[-1]["outcome"] == "ok"

    def test_compensation_is_announced(self, monkeypatch):
        seen = self._events(monkeypatch)
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS, fail_on=("db.duplicate_dashboard", 2)))
        _run(F._execute(F.parse_plan(FULL_PLAN), SKILL, report=F._Report(), **_ctx()))
        comp = [e for e in seen if e["phase"] == "compensating"]
        assert comp and all(e["tool_id"] in ("dm.delete", "db.delete_dashboard", "db.replace_datasource") for e in comp)


class TestHeartbeat:
    def test_ticks_while_a_tool_runs(self, monkeypatch):
        seen = []

        async def _emit(ev):
            seen.append(ev)

        monkeypatch.setattr(A, "_emit_agent_progress", _emit)
        monkeypatch.setattr(A, "PROGRESS_HEARTBEAT_SECONDS", 0.01)

        async def _go():
            import time as _t

            task = asyncio.ensure_future(A._heartbeat("x.y", _t.perf_counter()))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        _run(_go())
        assert len(seen) >= 2 and all(e["phase"] == "running" and e["tool_id"] == "x.y" for e in seen)


class TestNoQuestionForASkippedStep:
    def test_ask_is_skipped_when_the_steps_condition_is_already_false(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(A, "_record_pending_clarification", lambda v: recorded.setdefault("pc", v))
        monkeypatch.setattr(A, "_record_pending_loop", lambda v: recorded.setdefault("loop", v))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        results = dict(RESULTS, **{"db.list": []})  # no dashboards
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(results))
        plan = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {"id": "2", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "3",
                    "tool": "dm.create",
                    "args": {"datamodel": "M"},
                    "args_from": {"tables": "steps[1].result.tables"},
                    "args_ask": {"name": "Name?"},
                    "when": "steps[2].result != []",
                },
            ]
        }
        reply = _run(F.run(skill=SKILL, plan=plan, **_run_kwargs()))
        assert "pc" not in recorded and "loop" not in recorded, "no question, no approval"
        assert "Skipped, condition not met" in reply and "steps[2].result != []" not in reply


class TestNoPlanBlockForSkills:
    def test_planned_event_carries_no_plan_text(self, monkeypatch):
        seen = []

        async def _emit(ev):
            seen.append(ev)

        monkeypatch.setattr(A, "_emit_agent_progress", _emit)
        monkeypatch.setattr(A, "_record_pending_loop", lambda v: None)
        monkeypatch.setattr(A, "_record_tool_result", lambda v: None)
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        _run(F.run(skill=SKILL, plan=FULL_PLAN, **_run_kwargs()))
        planned = next(e for e in seen if e["phase"] == "planned")
        assert "plan" not in planned and planned["skill"] == "fixture"


class TestConjunctiveConditions:
    def test_and_requires_every_clause(self):
        sc = F.Scope()
        sc.results["1"] = {"errors": []}
        sc.results["2"] = [{"oid": "d1"}]
        assert F.eval_when("steps[1].result.errors == [] && steps[2].result != []", sc) is True
        sc.results["2"] = []
        assert F.eval_when("steps[1].result.errors == [] && steps[2].result != []", sc) is False

    def test_a_condition_the_runtime_cannot_judge_is_rejected_at_validation(self):
        bad = {
            "steps": [
                {"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}},
                {"id": "2", "tool": "db.list", "args": {"datamodel": "M"}, "when": "steps[1].result.errors == empty"},
            ]
        }
        with pytest.raises(F.PlanError, match="not understood"):
            F.validate_plan(F.parse_plan(bad), SKILL)

    def test_loop_var_normaliser_handles_each_clause(self):
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "d",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.validate_dashboard_queries",
                            "args_from": {"dashboard": "d.oid"},
                            "when": 'item.title != "x" && steps[1].result != []',
                        },
                    ],
                },
            ]
        }
        st = F.parse_plan(plan)[1].steps[0]
        assert st.when == 'd.title != "x" && steps[1].result != []'


class TestReportInUsersWords:
    def test_labels_replace_tool_ids_and_conditions_are_not_shown(self):
        r = F._Report(ran=["dm.analyze", "db.list"], skipped=["dm.create", "loop 5"], skipped_because=["a == []", "b"])
        text = r.text(
            "",
            lambda t: {
                "dm.analyze": "Analysing",
                "db.list": "Finding dashboards",
                "dm.create": "Creating the perspective",
            }.get(t, t),
        )
        assert "**Done:** Analysing, Finding dashboards" in text
        assert "**Skipped, condition not met:** Creating the perspective" in text
        assert "dm." not in text and "== []" not in text and "loop 5" not in text

    def test_step_records_carry_the_label(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(A, "_record_step", lambda *a, **k: recorded.append((a, k)))
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        sk = Skill(**{**SKILL.__dict__, "step_labels": {"dm.analyze": "Analysing the model"}})
        plan = {"steps": [{"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}}]}
        _run(F.run(skill=sk, plan=plan, **_run_kwargs()))
        assert recorded and recorded[0][0][1] == "dm.analyze" and recorded[0][1] == {"label": "Analysing the model"}


class TestOutcomeReport:
    def test_report_lists_one_outcome_per_step_with_loop_item_titles(self, monkeypatch):
        results = dict(
            RESULTS,
            **{
                "db.compare_dashboard_values": lambda a: {"all_match": a["dashboard"] != "d2", "compared": 2},
            },
        )
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(results))
        sk = Skill(
            **{
                **SKILL.__dict__,
                "step_labels": {"db.list": "Finding dashboards", "db.compare_dashboard_values": "Comparing widgets"},
            }
        )
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "d",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.compare_dashboard_values",
                            "args": {"datasource_a": "M", "datasource_b": "M_AI"},
                            "args_from": {"dashboard": "d.oid"},
                        },
                    ],
                },
            ]
        }
        reply = _run(F.run(skill=sk, plan=plan, **_run_kwargs()))
        assert "- Finding dashboards — 2 rows" in reply
        assert "- One: Comparing widgets — all 2 widgets match" in reply
        assert "- Two: Comparing widgets — differences found" in reply
        assert "succeeded. Details shown above" not in reply and "db." not in reply

    def test_pre_gate_outcomes_survive_the_approval_round_trip(self, monkeypatch):
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        monkeypatch.setattr(F, "load_skills", lambda **kw: {"fixture": SKILL})
        monkeypatch.setattr(A, "all_registry_tool_ids", lambda: set(REGISTRY))
        monkeypatch.setattr(A, "exposed_tool_ids", lambda: set(REGISTRY))
        items = F.parse_plan(FULL_PLAN)
        plan_args = F.plan_arguments(items, SKILL)
        pending = {
            "tool_id": F.PLAN_TOOL_ID,
            "plan": {"steps": plan_args["steps"]},
            "plan_arguments": plan_args,
            "skill": {"name": "fixture", "version": 2},
            "scope": {"results": {"1": RESULTS["dm.analyze"]}, "args": {"1": {"datamodel": "M"}}},
            "consumed": 1,
            "ran": ["dm.analyze"],
            "outcomes": ["Analyze the model — done"],
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, plan_args)}
        reply = _run(F.run(pending_plan=pending, **_run_kwargs(approved_mutations=approved)))
        assert reply.index("- Analyze the model — done") < reply.index("- Create a perspective — done")


class TestPublishAwareOutcome:
    def test_swap_outcome_reflects_the_publish_flag(self):
        assert (
            F._outcome_line("db.replace_datasource", True, {"published": True, "widgets_updated": 2}, "Moving")
            == "Moving — done and published"
        )
        assert (
            F._outcome_line("db.replace_datasource", True, {"published": False, "widgets_updated": 2}, "Moving")
            == "Moving — done for the owner; publish pending"
        )
        line = F._outcome_line(
            "db.replace_datasource",
            True,
            {"published": False, "widgets_updated": 2, "publish_error": "shared copy still on Root"},
            "Moving",
        )
        assert line.endswith("publish pending — shared copy still on Root")
        line = F._outcome_line(
            "db.replace_datasource",
            True,
            {"published": False, "widgets_updated": 2, "publish_error": "403", "owner": "o@x.com"},
            "Moving",
        )
        assert line.endswith("publish pending — only the owner can publish it (o@x.com)")


class TestConditionalTextBlocks:
    def _sk(self, text):
        return Skill(**{**SKILL.__dict__, "approval": text})

    def _root(self, m2m):
        root = F.Scope()
        root.results["1"] = {
            "errors": [],
            "warnings": {"ambiguous_join_path": 0, "many_to_many_in_perspective": len(m2m)},
            "many_to_many": m2m,
            "summary": {"model_tables": 26},
        }
        root.args["1"] = {"datamodel": "M"}
        return root

    def test_zero_or_empty_hides_the_block_and_bang_shows_it(self):
        items = F.parse_plan(FULL_PLAN)
        text = (
            "{?analyze.result.warnings.many_to_many_in_perspective}M2M!{/}"
            "{!analyze.result.warnings.many_to_many_in_perspective}no m2m{/}"
            "{?analyze.result.errors|count}broken{/}{!analyze.result.errors|count}clean{/}"
        )
        assert F.render_text(text, items, self._sk(text), self._root([])) == "no m2mclean"

    def test_nonzero_shows_the_block_and_pairs_render(self):
        items = F.parse_plan(FULL_PLAN)
        m2m = [
            {
                "table_a": "DimCountries",
                "columns_a": ["CountryCode"],
                "table_b": "Fact_Sale_orders",
                "columns_b": ["CountryCode"],
            },
            {"table_a": "A", "columns_a": ["x", "y"], "table_b": "B", "columns_b": ["x", "y"]},
        ]
        text = "{?analyze.result.warnings.many_to_many_in_perspective}- M2M: {analyze.result.many_to_many|pairs}{/}"
        assert (
            F.render_text(text, items, self._sk(text), self._root(m2m))
            == "- M2M: DimCountries ↔ Fact_Sale_orders (CountryCode); A ↔ B (x, y)"
        )

    def test_unresolvable_condition_hides_the_block(self):
        items = F.parse_plan(FULL_PLAN)
        text = "{?ghost.result.x}never{/}{!ghost.result.x}shown{/}tail"
        assert F.render_text(text, items, self._sk(text), F.Scope()) == "showntail"


class TestHeadFilter:
    def test_head_lists_first_n_then_counts_the_rest(self):
        items = F.parse_plan(FULL_PLAN)
        root = F.Scope()
        root.results["3"] = [{"oid": str(i), "title": f"D{i}"} for i in range(8)]
        sk = Skill(**{**SKILL.__dict__, "approval": "{list.result[*].title|head:3}"})
        assert F.render_text(sk.approval, items, sk, root) == "D0, D1, D2 … and 5 more"
        root.results["3"] = [{"oid": "1", "title": "Only"}]
        assert F.render_text(sk.approval, items, sk, root) == "Only"


class TestUniqueFilterAndRoster:
    def test_unique_owners(self):
        items = F.parse_plan(FULL_PLAN)
        root = F.Scope()
        root.results["3"] = [
            {"oid": "1", "title": "A", "owner_email": "x@s"},
            {"oid": "2", "title": "B", "owner_email": "x@s"},
            {"oid": "3", "title": "C", "owner_email": "y@s"},
        ]
        sk = Skill(**{**SKILL.__dict__, "approval": "{list.result[*].owner_email|unique}"})
        assert F.render_text(sk.approval, items, sk, root) == "x@s, y@s"

    def test_details_list_the_full_roster_when_the_reads_ran(self):
        items = F.parse_plan(FULL_PLAN)
        root = F.Scope()
        root.results["3"] = [{"oid": "1", "title": "A"}, {"oid": "2", "title": "B"}]
        text = F.render_dialog(items, SKILL, root)
        assert "**Dashs in scope (2):** A, B" in text
        assert "in scope" not in F.render_dialog(items, SKILL), "no roster without the reads"


class TestEndOfRunSummary:
    def _skill(self):
        return Skill(
            **{
                **SKILL.__dict__,
                "report": (
                    "Moved: {replace_datasource.result|items_ran}. Left: {replace_datasource.result|items_skipped}. "
                    "Owners: {replace_datasource.result|items_ran_owners}."
                ),
            }
        )

    def test_summary_lists_dashboards_per_outcome_with_owners(self, monkeypatch):
        results = dict(
            RESULTS,
            **{
                "db.list": [
                    {"oid": "d1", "title": "One", "owner_email": "a@s"},
                    {"oid": "d2", "title": "Two", "owner_email": "b@s"},
                ],
                "db.compare_dashboard_values": lambda a: {"all_match": a["dashboard"] == "d1", "compared": 2},
            },
        )
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(results))
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "dashboard",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.compare_dashboard_values",
                            "args": {"datasource_a": "M", "datasource_b": "M_AI"},
                            "args_from": {"dashboard": "dashboard.oid"},
                        },
                        {
                            "id": "2b",
                            "tool": "db.replace_datasource",
                            "args": {"datasource": "M_AI"},
                            "args_from": {"dashboard": "dashboard.oid"},
                            "when": "steps[2a].result.all_match == true",
                        },
                    ],
                },
            ]
        }
        sk = self._skill()
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        approved = {A._approval_key(F.PLAN_TOOL_ID, F.plan_arguments(F.parse_plan(plan), sk))}
        reply = _run(F.run(skill=sk, plan=plan, **_run_kwargs(approved_mutations=approved)))
        assert reply.startswith("Moved: One (a@s). Left: Two (b@s). Owners: a@s.")
        assert "**Done**" in reply, "a step was skipped, so the per-step list follows the summary"

    def test_no_report_section_means_no_summary(self, monkeypatch):
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(RESULTS))
        plan = {"steps": [{"id": "1", "tool": "dm.analyze", "args": {"datamodel": "M"}}]}
        reply = _run(F.run(skill=SKILL, plan=plan, **_run_kwargs()))
        assert reply.startswith("**Done**")


class TestConditionalOnItems:
    def test_blocks_test_the_per_item_lists(self, monkeypatch):
        sk = Skill(
            **{
                **SKILL.__dict__,
                "report": (
                    "{?replace_datasource.result|items_skipped}LEFT: {replace_datasource.result|items_skipped}{/}"
                    "{!replace_datasource.result|items_skipped}NONE LEFT{/}"
                ),
            }
        )
        results = dict(
            RESULTS,
            **{
                "db.list": [{"oid": "d1", "title": "One"}],
                "db.compare_dashboard_values": lambda a: {"all_match": True, "compared": 2},
            },
        )
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(results))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "dashboard",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.compare_dashboard_values",
                            "args": {"datasource_a": "M", "datasource_b": "M_AI"},
                            "args_from": {"dashboard": "dashboard.oid"},
                        },
                        {
                            "id": "2b",
                            "tool": "db.replace_datasource",
                            "args": {"datasource": "M_AI"},
                            "args_from": {"dashboard": "dashboard.oid"},
                            "when": "steps[2a].result.all_match == true",
                        },
                    ],
                },
            ]
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, F.plan_arguments(F.parse_plan(plan), sk))}
        reply = _run(F.run(skill=sk, plan=plan, **_run_kwargs(approved_mutations=approved)))
        assert reply == "NONE LEFT", "clean run: summary alone, no step list"


class TestItemListCap:
    def test_long_item_lists_are_capped(self):
        items = F.parse_plan(FULL_PLAN)
        rep = F._Report()
        for i in range(13):
            rep.note_item("db.replace_datasource", "ran", {"oid": str(i), "title": f"D{i}", "owner_email": "o@s"})
        sk = Skill(**{**SKILL.__dict__, "report": "{replace_datasource.result|items_ran}"})
        text = F.render_report(items, sk, F.Scope(), rep)
        assert text.startswith("D0 (o@s), D1 (o@s)") and text.endswith("D9 (o@s) … and 3 more")


class TestNestedConditionals:
    def test_inner_blocks_resolve_inside_outer_ones(self):
        items = F.parse_plan(FULL_PLAN)
        tmpl = (
            "{!analyze.result.choices|count}none found{/}"
            "{?analyze.result.choices|count}{analyze.result.choices|count} found — "
            "{!analyze.result.warnings.amb}all resolved{/}"
            "{?analyze.result.warnings.amb}{analyze.result.warnings.amb} unresolved{/}{/}"
        )
        sk = Skill(**{**SKILL.__dict__, "approval": tmpl})

        def root(choices, amb):
            r = F.Scope()
            r.results["1"] = {"choices": choices, "warnings": {"amb": amb}}
            return r

        assert F.render_text(tmpl, items, sk, root([], 0)) == "none found"
        assert F.render_text(tmpl, items, sk, root([{}, {}], 0)) == "2 found — all resolved"
        assert F.render_text(tmpl, items, sk, root([{}, {}, {}], 1)) == "3 found — 1 unresolved"


class TestOnBehalfSummary:
    def test_dashboards_moved_on_the_owners_behalf_get_their_own_sentence(self, monkeypatch):
        results = dict(
            RESULTS,
            **{
                "db.list": [{"oid": "d1", "title": "One", "owner_email": "a@s"}],
                "db.compare_dashboard_values": lambda a: {"all_match": True, "compared": 2},
                "db.replace_datasource": lambda a: {
                    "previous_datasource": "M",
                    "published": True,
                    "ownership_transferred_temporarily": True,
                    "co_authoring": True,
                },
            },
        )
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(results))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        sk = Skill(
            **{
                **SKILL.__dict__,
                "report": (
                    "{?replace_datasource.result|items_on_behalf}OWNERS LAG: "
                    "{replace_datasource.result|items_on_behalf}{/}"
                    "{!replace_datasource.result|items_on_behalf}ALL DIRECT{/}"
                ),
            }
        )
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "dashboard",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.compare_dashboard_values",
                            "args": {"datasource_a": "M", "datasource_b": "M_AI"},
                            "args_from": {"dashboard": "dashboard.oid"},
                        },
                        {
                            "id": "2b",
                            "tool": "db.replace_datasource",
                            "args": {"datasource": "M_AI"},
                            "args_from": {"dashboard": "dashboard.oid"},
                            "when": "steps[2a].result.all_match == true",
                        },
                    ],
                },
            ]
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, F.plan_arguments(F.parse_plan(plan), sk))}
        assert _run(F.run(skill=sk, plan=plan, **_run_kwargs(approved_mutations=approved))) == "OWNERS LAG: One (a@s)"


class TestOnBehalfNeedsCoAuthoring:
    def test_feature_off_reads_like_an_owner_run(self, monkeypatch):
        results = dict(
            RESULTS,
            **{
                "db.list": [{"oid": "d1", "title": "One", "owner_email": "a@s"}],
                "db.compare_dashboard_values": lambda a: {"all_match": True, "compared": 2},
                "db.replace_datasource": lambda a: {
                    "previous_datasource": "M",
                    "published": True,
                    "ownership_transferred_temporarily": True,
                    "co_authoring": False,
                },
            },
        )
        monkeypatch.setattr(A, "_invoke_tool_traced", _invoker(results))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        sk = Skill(
            **{
                **SKILL.__dict__,
                "report": (
                    "{?replace_datasource.result|items_on_behalf}LAG{/}"
                    "{!replace_datasource.result|items_on_behalf}SAME AS OWNER{/}"
                ),
            }
        )
        plan = {
            "steps": [
                {"id": "1", "tool": "db.list", "args": {"datamodel": "M"}},
                {
                    "id": "2",
                    "for_each": "steps[1].result[*]",
                    "as": "dashboard",
                    "steps": [
                        {
                            "id": "2a",
                            "tool": "db.compare_dashboard_values",
                            "args": {"datasource_a": "M", "datasource_b": "M_AI"},
                            "args_from": {"dashboard": "dashboard.oid"},
                        },
                        {
                            "id": "2b",
                            "tool": "db.replace_datasource",
                            "args": {"datasource": "M_AI"},
                            "args_from": {"dashboard": "dashboard.oid"},
                            "when": "steps[2a].result.all_match == true",
                        },
                    ],
                },
            ]
        }
        approved = {A._approval_key(F.PLAN_TOOL_ID, F.plan_arguments(F.parse_plan(plan), sk))}
        assert _run(F.run(skill=sk, plan=plan, **_run_kwargs(approved_mutations=approved))) == "SAME AS OWNER"
