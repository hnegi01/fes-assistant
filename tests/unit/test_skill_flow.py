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
    monkeypatch.setattr(A, "_record_step", lambda *a: None)


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
        with pytest.raises(F.PlanError, match="no result yet"):
            F.resolve_path("steps[4a].result.oid", root)  # iteration-local, not global

    @pytest.mark.parametrize(
        "expr, msg",
        [
            ("steps[9].result", "no result yet"),
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
        report, n = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))

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
        report, _ = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))

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
        report, _ = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))
        assert report.failed == ("db.list", "boom")
        assert report.compensation_failed == ("dm.delete", "cannot undo")
        assert "Could not revert" in report.text("")

    def test_unresolvable_reference_at_runtime_is_a_reported_failure(self, monkeypatch):
        results = dict(RESULTS, **{"dm.analyze": {"nothing": 1}})  # no `tables`
        inv = _invoker(results)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)
        items = F.parse_plan(FULL_PLAN)
        report, _ = _run(F._execute(items, SKILL, report=F._Report(), **_ctx()))
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
        monkeypatch.setattr(A, "_record_tool_result", lambda v: recorded.setdefault("result", v))
        monkeypatch.setattr(F, "_current_role", AsyncMock(return_value="admin"))
        inv = _invoker(RESULTS)
        monkeypatch.setattr(A, "_invoke_tool_traced", inv)

        reply = _run(F.run(skill=SKILL, plan=FULL_PLAN, **_run_kwargs()))
        assert "Approve to run the whole sequence" in reply
        assert inv.calls == [], "nothing executes before approval"
        pl = recorded["loop"]
        assert pl["tool_id"] == F.PLAN_TOOL_ID
        assert pl["skill"] == {"name": "fixture", "version": 2}
        assert pl["plan"]["steps"][0]["tool"] == "dm.analyze"
        assert recorded["result"]["pending_confirmation"]["tool_id"] == F.PLAN_TOOL_ID

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
        assert "Completed" in reply

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
        assert "Completed" in reply
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
        assert "changed (v1 → v2)" in reply

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
