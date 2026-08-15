"""
Unit tests for the single-shot migration flow (backend/agent/migration_flow.py).

Migration plans everything in ONE LLM call, takes ONE approval for the whole
plan, then executes in the planner's order — no per-step "what next?" call and
no per-step dialog, because no migration tool consumes a value another produces.

Ordering is the planner's, stated in MIGRATION_PLANNING_CONTEXT_PROMPT and
reviewed by the human who approves the sequence. There is deliberately no rank
table here: it would need editing for every new migration tool and would
mis-rank anything it did not recognise.

Covered:
  - exactly one planning call, whatever the plan's length
  - the planner's order is preserved verbatim into execution
  - approval: ONE gate covering the whole plan; the dialog names every step
  - the approval key covers the exact plan — reorder or edit it and it re-gates
  - resume: runs the approved plan, never replans
  - failure: stops, and the reply names ran / failed / not-attempted
  - validation happens BEFORE the plan is even proposed
  - the flow is migration-only, and the kill switch routes back to the loop
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

import backend.agent.llm_agent as m
import backend.agent.migration_flow as mf

GROUPS_SCHEMA = {
    "type": "object",
    "properties": {"group_name_list": {"type": "array", "items": {"type": "string"}}},
    "required": ["group_name_list"],
}
USERS_SCHEMA = {
    "type": "object",
    "properties": {"user_name_list": {"type": "array", "items": {"type": "string"}}},
    "required": ["user_name_list"],
}
DASH_SCHEMA = {
    "type": "object",
    "properties": {
        "dashboard_names": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": ["skip", "overwrite", "duplicate"]},
    },
    "required": [],
}


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    def _meta(method, schema):
        return {
            "module": "migration",
            "method": method,
            "mutates": True,
            "description": f"Migrate {method}.",
            "parameters": schema,
        }

    monkeypatch.setattr(
        m,
        "TOOL_REGISTRY",
        {
            "migration.migrate_groups": _meta("migrate_groups", GROUPS_SCHEMA),
            "migration.migrate_users": _meta("migrate_users", USERS_SCHEMA),
            "migration.migrate_dashboards": _meta("migrate_dashboards", DASH_SCHEMA),
            "migration.migrate_all_datamodels": _meta(
                "migrate_all_datamodels", {"type": "object", "properties": {}, "required": []}
            ),
        },
    )
    monkeypatch.setattr(m, "ALLOW_SUMMARIZATION", True)
    monkeypatch.setattr(m, "REQUIRE_MUTATION_CONFIRM", True)
    monkeypatch.setattr(m, "VERIFY_GOAL", False)
    # Off by default here: these tests pin exact LLM response lists, and the
    # completeness check consumes one. Its own behaviour is covered in
    # TestCompletenessCheck (test_migration_flow.py).
    monkeypatch.setattr(m, "MIGRATION_COMPLETENESS_CHECK", False)


@pytest.fixture(autouse=True)
def reset_globals():
    m.LAST_PENDING_LOOP = None
    m.LAST_PENDING_CLARIFICATION = None
    m.LAST_TOOL_RESULT = None
    m.LAST_STEP_RESULTS = []
    yield
    m.LAST_PENDING_LOOP = None
    m.LAST_PENDING_CLARIFICATION = None
    m.LAST_TOOL_RESULT = None
    m.LAST_STEP_RESULTS = []


def _call(tool_id, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": tool_id, "arguments": json.dumps(args)}}


def _multi_resp(*calls):
    """One assistant message carrying SEVERAL tool_calls — the whole point."""
    return {
        "choices": [{"message": {"content": None, "tool_calls": list(calls)}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _text_resp(text):
    return {"choices": [{"message": {"content": text, "tool_calls": []}}], "usage": {}}


def _tool_def(name, schema):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": schema}}


MIGRATION_TOOLS = [
    _tool_def("migration.migrate_groups", GROUPS_SCHEMA),
    _tool_def("migration.migrate_users", USERS_SCHEMA),
    _tool_def("migration.migrate_dashboards", DASH_SCHEMA),
]

GROUP_ARGS = {"group_name_list": ["Sales Team"]}
USER_ARGS = {"user_name_list": ["jane@x.com"]}


def _turn(llm_responses, client, approved=None, prompt="migrate the groups and users", **kw):
    raw = AsyncMock(side_effect=llm_responses)
    with patch.object(m, "call_llm_raw", new=raw), patch.object(m, "_navigate_to_tools", new=AsyncMock()):
        reply = run(
            m.call_llm_with_tools(
                [{"role": "user", "content": prompt}],
                MIGRATION_TOOLS,
                client,
                approved_mutations=approved or set(),
                allow_summarization=True,
                **kw,
            )
        )
    return reply, raw


def _client(*results):
    """No results given → every call succeeds, however many there are."""
    c = AsyncMock()
    if results:
        c.invoke_tool = AsyncMock(side_effect=list(results))
    else:
        c.invoke_tool = AsyncMock(return_value={"ok": True, "result": {"migrated": 1}})
    return c


# ---------------------------------------------------------------------------
# 1) The planner's order is preserved, not re-sorted
# ---------------------------------------------------------------------------
class TestOrderIsThePlanners:
    """No rank table: a new migration tool must not need a code change, and a
    hardcoded list would silently mis-rank what it doesn't recognise. The
    dependency rule is in the prompt; the approval dialog is the human check."""

    def test_no_rank_table_exists(self):
        assert not hasattr(mf, "dependency_rank")
        assert not hasattr(mf, "DEPENDENCY_ORDER")

    def test_execution_follows_the_planner_order(self):
        client = _client()
        plan_calls = [
            _call("migration.migrate_groups", GROUP_ARGS, "c1"),
            _call("migration.migrate_users", USER_ARGS, "c2"),
        ]
        approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
        _turn([_multi_resp(*plan_calls)], client, approved=approved)
        assert [c.args[0] for c in client.invoke_tool.await_args_list] == [
            "migration.migrate_groups",
            "migration.migrate_users",
        ]

    def test_the_dialog_shows_the_order_the_user_is_approving(self):
        """The step list is built in code from the calls that will actually run,
        so it can never drift from the sequence being approved."""
        client = _client()
        _turn(
            [
                _multi_resp(
                    _call("migration.migrate_groups", GROUP_ARGS, "c1"),
                    _call("migration.migrate_users", USER_ARGS, "c2"),
                ),
                _text_resp("This copies a group and a user to the target."),
            ],
            client,
        )
        reason = m.LAST_TOOL_RESULT["pending_confirmation"]["reason"]
        assert reason.index("migrate_groups") < reason.index("migrate_users")
        assert "2 operations will run in this order" in reason


# ---------------------------------------------------------------------------
# 2) One planning call for the whole plan
# ---------------------------------------------------------------------------
def test_a_three_step_plan_costs_one_planning_call():
    """The reactive loop would spend a plan call plus a decide call per step."""
    client = _client()
    _reply, raw = _turn(
        [
            _multi_resp(
                _call("migration.migrate_groups", GROUP_ARGS, "c1"),
                _call("migration.migrate_users", USER_ARGS, "c2"),
                _call("migration.migrate_dashboards", {"dashboard_names": ["Sales"]}, "c3"),
            ),
            _text_resp("This will copy the Sales Team group."),  # gate explanation
        ],
        client,
    )
    labels = [c.kwargs.get("label") for c in raw.await_args_list]
    assert labels.count("migration_plan") == 1
    assert "decide" not in labels
    assert "route" not in labels


def test_the_plan_is_emitted_for_the_ui():
    client = _client()
    seen = []

    async def _capture(event):
        seen.append(event)

    with patch.object(m, "_emit_agent_progress", new=_capture):
        _turn(
            [
                _multi_resp(
                    _call("migration.migrate_groups", GROUP_ARGS, "c1"),
                    _call("migration.migrate_users", USER_ARGS, "c2"),
                ),
                _text_resp("This will copy the Sales Team group."),
            ],
            client,
        )
    planned = [e for e in seen if e.get("phase") == "planned"]
    assert planned, "the UI checklist needs the plan"
    assert "migration.migrate_groups" in planned[0]["plan"]


# ---------------------------------------------------------------------------
# 3) One approval for the whole plan
# ---------------------------------------------------------------------------
def test_a_multi_step_plan_asks_once_and_runs_nothing_yet():
    client = _client()
    _turn(
        [
            _multi_resp(
                _call("migration.migrate_groups", GROUP_ARGS, "c1"),
                _call("migration.migrate_users", USER_ARGS, "c2"),
            ),
            _text_resp("This copies a group and a user to the target."),
        ],
        client,
    )
    client.invoke_tool.assert_not_awaited()
    pending = m.LAST_TOOL_RESULT["pending_confirmation"]
    assert pending["tool_id"] == mf.PLAN_TOOL_ID
    steps = pending["arguments"]["steps"]
    assert [s["tool"] for s in steps] == ["migration.migrate_groups", "migration.migrate_users"]
    assert m.LAST_PENDING_LOOP["plan"], "the plan must survive the pause"


def test_the_dialog_names_every_operation_and_its_arguments():
    client = _client()
    _turn(
        [
            _multi_resp(
                _call("migration.migrate_groups", GROUP_ARGS, "c1"),
                _call("migration.migrate_users", USER_ARGS, "c2"),
            ),
            _text_resp("This copies a group and a user to the target."),
        ],
        client,
    )
    reason = m.LAST_TOOL_RESULT["pending_confirmation"]["reason"]
    assert "Sales Team" in reason and "jane@x.com" in reason
    assert "Approve to run the whole sequence" in reason


def test_approving_runs_every_step_without_asking_again():
    client = _client({"ok": True, "result": {}}, {"ok": True, "result": {}})
    plan_calls = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),
        _call("migration.migrate_users", USER_ARGS, "c2"),
    ]
    approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
    reply, raw = _turn([_multi_resp(*plan_calls)], client, approved=approved)
    assert client.invoke_tool.await_count == 2
    assert m.LAST_PENDING_LOOP is None
    # The final answer is code-built (2026-08-14): deterministic per-step lines,
    # and the finalize LLM call is gone — planning was the ONLY llm call.
    assert "`migration.migrate_groups` succeeded" in reply
    assert "`migration.migrate_users` succeeded" in reply
    assert raw.await_count == 1, "plan call only — no LLM finalize for migration replies"


def test_resume_runs_the_approved_plan_without_replanning():
    client = _client({"ok": True, "result": {}}, {"ok": True, "result": {}})
    plan_calls = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),
        _call("migration.migrate_users", USER_ARGS, "c2"),
    ]
    plan_args = mf.plan_arguments(plan_calls)
    pending_loop = {
        "transcript": [],
        "raw_results": [],
        "steps_executed": 0,
        "tool_id": mf.PLAN_TOOL_ID,
        "arguments": plan_args,
        "plan": plan_calls,
        "plan_arguments": plan_args,
    }
    reply, raw = _turn(
        [],  # resume needs NO llm at all: no replan, no finalize
        client,
        approved={m._approval_key(mf.PLAN_TOOL_ID, plan_args)},
        pending_loop=pending_loop,
    )
    assert client.invoke_tool.await_count == 2
    assert raw.await_count == 0, "an approved resume is zero LLM calls end to end"
    assert "`migration.migrate_groups` succeeded" in reply
    assert "`migration.migrate_users` succeeded" in reply


def test_the_approval_covers_that_exact_plan_only():
    """Approving [groups, users] must not authorise [groups, dashboards]: the
    key is the ordered step list, so any edit re-gates."""
    approved_plan = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),
        _call("migration.migrate_users", USER_ARGS, "c2"),
    ]
    different = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),
        _call("migration.migrate_dashboards", {"dashboard_names": ["Sales"]}, "c2"),
    ]
    reordered = list(reversed(approved_plan))
    key = m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(approved_plan))
    assert key != m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(different))
    assert key != m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(reordered))


def test_a_stale_plan_without_approval_is_dropped_and_replanned():
    client = _client()
    stale = [_call("migration.migrate_groups", GROUP_ARGS, "c1")]
    pending_loop = {
        "transcript": [],
        "raw_results": [],
        "steps_executed": 0,
        "tool_id": mf.PLAN_TOOL_ID,
        "arguments": mf.plan_arguments(stale),
        "plan": stale,
        "plan_arguments": mf.plan_arguments(stale),
    }
    _reply, raw = _turn(
        [
            _multi_resp(_call("migration.migrate_users", USER_ARGS, "c9")),
            _text_resp("This copies jane@x.com to the target."),
        ],
        client,
        approved=set(),  # user typed something else instead of approving
        pending_loop=pending_loop,
    )
    client.invoke_tool.assert_not_awaited()
    assert "migration_plan" in [c.kwargs.get("label") for c in raw.await_args_list]


def test_the_plan_approval_is_single_use():
    plan_calls = [_call("migration.migrate_groups", GROUP_ARGS, "c1")]
    approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
    assert m._consume_approval(approved, mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls)) is True
    assert m._consume_approval(approved, mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls)) is False


# ---------------------------------------------------------------------------
# 4) Failure stops the run and says what was skipped
# ---------------------------------------------------------------------------
def test_failure_stops_and_names_ran_failed_and_not_attempted():
    client = _client(
        {"ok": False, "error": "target group endpoint returned 500"},
    )
    plan_calls = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),
        _call("migration.migrate_users", USER_ARGS, "c2"),
    ]
    approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
    reply, _ = _turn([_multi_resp(*plan_calls)], client, approved=approved)
    assert client.invoke_tool.await_count == 1, "the dependent step must not run"
    assert "Stopped" in reply
    assert "target group endpoint returned 500" in reply
    assert "migration.migrate_users" in reply
    assert "Not attempted" in reply


def test_completed_steps_are_listed_when_a_later_one_fails():
    client = _client(
        {"ok": True, "result": {"migrated": 2}},
        {"ok": False, "error": "user already exists"},
    )
    plan_calls = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),
        _call("migration.migrate_users", USER_ARGS, "c2"),
        _call("migration.migrate_dashboards", {"dashboard_names": ["Sales"]}, "c3"),
    ]
    approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
    reply, _ = _turn([_multi_resp(*plan_calls)], client, approved=approved)
    assert client.invoke_tool.await_count == 2
    assert "Completed:" in reply and "migrate_groups" in reply
    assert "user already exists" in reply
    assert "migrate_dashboards" in reply.split("Not attempted")[-1]


# ---------------------------------------------------------------------------
# 5) Validation happens before anything runs
# ---------------------------------------------------------------------------
def test_a_bad_argument_in_step_three_prevents_step_one_from_running():
    """Half a migration is worse than none, and nothing has been written yet
    at plan time — so every call is checked up front."""
    client = _client()
    plan_calls = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),
        _call("migration.migrate_dashboards", {"action": "replace"}, "c2"),
    ]
    approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
    reply, _ = _turn([_multi_resp(*plan_calls)], client, approved=approved)
    client.invoke_tool.assert_not_awaited()
    assert "replace" in reply


def test_a_missing_required_argument_asks_before_running_anything():
    client = _client()
    with patch.object(
        m, "_generate_clarification_question", new=AsyncMock(return_value="Which groups should I migrate?")
    ):
        reply, _ = _turn(
            [_multi_resp(_call("migration.migrate_groups", {}, "c1"))],
            client,
        )
    client.invoke_tool.assert_not_awaited()
    assert reply == "Which groups should I migrate?"
    assert m.LAST_PENDING_CLARIFICATION["tool_id"] == "migration.migrate_groups"


def test_no_tool_selected_returns_the_models_question():
    client = _client()
    reply, _ = _turn([_text_resp("Which dashboards would you like to migrate?")], client)
    client.invoke_tool.assert_not_awaited()
    assert reply == "Which dashboards would you like to migrate?"


# ---------------------------------------------------------------------------
# 6) Scope: migration only, and the kill switch works
# ---------------------------------------------------------------------------
def test_chat_mode_does_not_use_this_flow(monkeypatch):
    """Chat needs the per-step loop: a step's RESULT can change the next step."""
    monkeypatch.setattr(
        m,
        "TOOL_REGISTRY",
        {
            "access_management.get_user": {
                "module": "access_management",
                "mutates": False,
                "description": "Get a user.",
                "parameters": {"type": "object", "properties": {"user_email": {"type": "string"}}, "required": []},
            }
        },
    )
    called = AsyncMock(return_value="unused")
    chat_tools = [_tool_def("access_management.get_user", {"type": "object", "properties": {}})]
    with (
        patch.object(mf, "run", new=called),
        patch.object(
            m, "_navigate_to_tools", new=AsyncMock(return_value=(chat_tools, "access_management", "users", 0))
        ),
        patch.object(m, "call_llm_raw", new=AsyncMock(return_value=_text_resp("Done."))),
        patch.object(m, "_make_plan", new=AsyncMock(return_value=["look up a user"])),
    ):
        run(
            m.call_llm_with_tools(
                [{"role": "user", "content": "show me a user"}], chat_tools, _client(), allow_summarization=True
            )
        )
    called.assert_not_awaited()


def test_kill_switch_routes_migration_back_to_the_reactive_loop(monkeypatch):
    monkeypatch.setattr(m, "MIGRATION_SINGLE_SHOT", False)
    monkeypatch.setenv("FES_AGENT_ENGINE", "custom")
    loop = AsyncMock(return_value="from the loop")
    flow = AsyncMock(return_value="from the flow")
    with patch.object(m, "_reactive_loop", new=loop), patch.object(mf, "run", new=flow):
        out = run(m._run_loop_engine(mode="migration"))
    assert out == "from the loop"
    flow.assert_not_awaited()


def test_single_shot_is_on_by_default():
    assert m.MIGRATION_SINGLE_SHOT is True


def test_the_plan_key_survives_the_json_round_trip_to_the_ui():
    """The key is a hash of the whole plan, and the plan crosses the wire twice:
    backend → UI as JSON in pending_confirmation, then UI → backend as an
    approved key. If serialisation changes the dict at all, the key stops
    matching and Approve silently does nothing. Much bigger surface than a
    single tool's flat arguments, so it is worth pinning."""
    plan_calls = [
        _call("migration.migrate_groups", {"group_name_list": ["Sales Team", "Ops"]}, "c1"),
        _call("migration.migrate_dashboards", {"dashboard_names": ["Revenue — Q4"], "republish": True}, "c2"),
    ]
    backend_args = mf.plan_arguments(plan_calls)
    # What the UI receives and hands back, having been through HTTP.
    ui_args = json.loads(json.dumps(backend_args))
    assert m._approval_key(mf.PLAN_TOOL_ID, ui_args) == m._approval_key(mf.PLAN_TOOL_ID, backend_args)

    # And the round-tripped key actually opens the gate.
    approved = {m._approval_key(mf.PLAN_TOOL_ID, ui_args)}
    assert m._consume_approval(approved, mf.PLAN_TOOL_ID, backend_args) is True


def test_the_ui_agrees_on_the_plan_tool_id():
    """The UI hides its raw-JSON expander for plan approvals by comparing against
    this id. It runs in a separate process and imports nothing from the backend,
    so the constant is duplicated — if the two drift, the UI silently starts
    showing `migration.plan` and a nested step blob again."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "frontend" / "app.py").read_text(encoding="utf-8")
    found = re.search(r'^MIGRATION_PLAN_TOOL_ID\s*=\s*"([^"]+)"', src, re.M)
    assert found, "frontend/app.py must define MIGRATION_PLAN_TOOL_ID"
    assert found.group(1) == mf.PLAN_TOOL_ID


# ---------------------------------------------------------------------------
# 7) Completeness check — a second pair of eyes that only counts
# ---------------------------------------------------------------------------
class TestCompletenessCheck:
    """The planner emits every call in ONE response and gets less reliable as the
    count rises: measured 2026-08-08, a four-kind request returned only two calls
    in 2 of 6 runs, always stopping at exactly two. There is no second chance
    inside the plan, so a dropped call silently loses part of the request. A
    fresh reader counts the kinds; anything it names triggers ONE re-plan."""

    @pytest.fixture(autouse=True)
    def enable(self, monkeypatch):
        monkeypatch.setattr(m, "MIGRATION_COMPLETENESS_CHECK", True)

    # --- parsing the checker's one-line answer --------------------------
    @pytest.mark.parametrize("answer", ["COMPLETE", "complete", "  COMPLETE  ", ""])
    def test_complete_answers_mean_nothing_missing(self, answer):
        with patch.object(m, "call_llm_raw", new=AsyncMock(return_value=_text_resp(answer))):
            assert run(mf._missing_kinds("migrate the groups", [], "t")) == []

    def test_missing_line_is_parsed_into_kinds(self):
        with patch.object(m, "call_llm_raw", new=AsyncMock(return_value=_text_resp("MISSING: dashboards, datamodels"))):
            assert run(mf._missing_kinds("migrate everything", [], "t")) == ["dashboards", "datamodels"]

    @pytest.mark.parametrize("answer", ["I think it looks fine", "no colon here", "???"])
    def test_unparseable_answers_never_derail_a_plan(self, answer):
        """A checker that cannot answer must not be able to reject a plan the
        planner was happy with."""
        with patch.object(m, "call_llm_raw", new=AsyncMock(return_value=_text_resp(answer))):
            assert run(mf._missing_kinds("migrate the groups", [], "t")) == []

    def test_an_llm_failure_never_blocks_a_plan(self):
        with patch.object(m, "call_llm_raw", new=AsyncMock(side_effect=RuntimeError("boom"))):
            assert run(mf._missing_kinds("migrate the groups", [], "t")) == []

    # --- the retry it drives -------------------------------------------
    def test_an_omission_triggers_one_replan_and_the_fuller_plan_wins(self):
        client = _client()
        short = [_call("migration.migrate_groups", GROUP_ARGS, "c1")]
        full = [
            _call("migration.migrate_groups", GROUP_ARGS, "c1"),
            _call("migration.migrate_users", USER_ARGS, "c2"),
        ]
        _reply, raw = _turn(
            [
                _multi_resp(*short),  # plan: only groups
                _text_resp("MISSING: users"),  # checker spots the gap
                _multi_resp(*full),  # re-plan: both
                _text_resp("This copies a group and a user."),  # dialog prose
            ],
            client,
            prompt="migrate the groups and the users",
        )
        steps = [s["tool"] for s in m.LAST_TOOL_RESULT["pending_confirmation"]["arguments"]["steps"]]
        assert steps == ["migration.migrate_groups", "migration.migrate_users"]
        assert "migration_plan_retry" in [c.kwargs.get("label") for c in raw.await_args_list]

    def test_a_retry_that_is_not_fuller_is_discarded(self):
        """A retry returning the same count (or fewer) is not an improvement —
        keep what the planner produced first rather than churning."""
        client = _client()
        short = [_call("migration.migrate_groups", GROUP_ARGS, "c1")]
        _turn(
            [
                _multi_resp(*short),
                _text_resp("MISSING: users"),
                _multi_resp(*short),  # retry: no better
                _text_resp("This copies a group."),
            ],
            client,
            prompt="migrate the groups and the users",
        )
        steps = [s["tool"] for s in m.LAST_TOOL_RESULT["pending_confirmation"]["arguments"]["steps"]]
        assert steps == ["migration.migrate_groups"]

    def test_the_retry_is_bounded_to_one_attempt(self):
        """Never a loop: the checker runs once, so one retry is the ceiling."""
        client = _client()
        short = [_call("migration.migrate_groups", GROUP_ARGS, "c1")]
        _reply, raw = _turn(
            [
                _multi_resp(*short),
                _text_resp("MISSING: users"),
                _multi_resp(*short),
                _text_resp("This copies a group."),
            ],
            client,
            prompt="migrate the groups and the users",
        )
        labels = [c.kwargs.get("label") for c in raw.await_args_list]
        assert labels.count("migration_completeness") == 1
        assert labels.count("migration_plan_retry") == 1

    def test_a_complete_plan_costs_the_check_and_no_retry(self):
        client = _client()
        full = [
            _call("migration.migrate_groups", GROUP_ARGS, "c1"),
            _call("migration.migrate_users", USER_ARGS, "c2"),
        ]
        _reply, raw = _turn(
            [_multi_resp(*full), _text_resp("COMPLETE"), _text_resp("This copies a group and a user.")],
            client,
        )
        labels = [c.kwargs.get("label") for c in raw.await_args_list]
        assert labels.count("migration_completeness") == 1
        assert "migration_plan_retry" not in labels

    def test_the_flag_switches_the_whole_thing_off(self, monkeypatch):
        monkeypatch.setattr(m, "MIGRATION_COMPLETENESS_CHECK", False)
        client = _client()
        full = [_call("migration.migrate_groups", GROUP_ARGS, "c1")]
        _reply, raw = _turn([_multi_resp(*full), _text_resp("This copies a group.")], client)
        assert "migration_completeness" not in [c.kwargs.get("label") for c in raw.await_args_list]


# ---------------------------------------------------------------------------
# 8) The migration planner is not shown conversation history
# ---------------------------------------------------------------------------
def test_history_is_withheld_from_the_migration_planner():
    """Measured 2026-08-10: "migrate the dashboards, the users and the groups"
    after one prior migration turn planned all 3 calls in 6/6 runs with no
    history and 3/6 with it, once collapsing to a single call. The prior
    assistant message LISTS a plan, and the planner reads it as work already
    accounted for. A prompt rule saying to ignore earlier turns measured no
    better (5/12); withholding history measured 12/12.

    Affordable only because migration requests name their own assets — there is
    no "its members" to resolve against an earlier turn, as there is in chat.
    """
    client = _client()
    history_marker = "PREVIOUSLY-PLANNED-migrate_groups"
    messages = [
        {"role": "user", "content": "migrate the Sales Team group"},
        {"role": "assistant", "content": f"2 operations will run: {history_marker}"},
        {"role": "user", "content": "migrate the dashboards, the users and the groups"},
    ]
    raw = AsyncMock(
        side_effect=[
            _multi_resp(_call("migration.migrate_groups", GROUP_ARGS, "c1")),
            _text_resp("This copies a group."),
        ]
    )
    with patch.object(m, "call_llm_raw", new=raw), patch.object(m, "_navigate_to_tools", new=AsyncMock()):
        run(
            m.call_llm_with_tools(messages, MIGRATION_TOOLS, client, approved_mutations=set(), allow_summarization=True)
        )

    sent = raw.await_args_list[0].args[0]
    contents = " ".join(str(msg.get("content") or "") for msg in sent)
    assert history_marker not in contents, "the planner must not see previously-proposed plans"
    assert "migrate the dashboards, the users and the groups" in contents, "the current request must be there"
    # ONE system prompt + the request. Nothing else.
    assert [msg["role"] for msg in sent] == ["system", "user"]


# ---------------------------------------------------------------------------
# 9) The dialog speaks English, not identifiers
# ---------------------------------------------------------------------------
class TestDialogReadability:
    """MUTATION_EXPLAIN_SYSTEM_PROMPT forbids naming tools and parameters in the
    prose; the step list has to hold the same line. Labels come from the
    registry description — human-written by the SDK authors, and code-derived,
    so what is shown cannot drift from what runs."""

    def test_label_is_never_a_tool_id(self):
        """The fixture's description is terse; the point is that the label comes
        from the description at all, rather than the dotted id."""
        assert not mf._step_label("migration.migrate_groups").startswith("migration.")

    def test_the_redundant_environment_clause_is_trimmed(self, monkeypatch):
        monkeypatch.setitem(
            m.TOOL_REGISTRY,
            "migration.migrate_groups",
            {
                "description": "Migrate groups from the source environment to the target environment"
                " using the bulk endpoint."
            },
        )
        assert mf._step_label("migration.migrate_groups") == "Migrate groups"

    def test_unknown_tool_still_gets_a_readable_label(self, monkeypatch):
        monkeypatch.setattr(m, "TOOL_REGISTRY", {})
        assert mf._step_label("migration.migrate_widgets") == "Migrate widgets"

    def test_arguments_render_without_json_punctuation(self):
        out = mf._humanise_args({"group_name_list": ["Sales Team", "Ops"], "migrate_share": True})
        assert out == "group names: Sales Team, Ops · migrate share: yes"
        for ch in '{}[]"':
            assert ch not in out

    def test_list_suffix_dropped_and_singular_for_one_value(self):
        """`_list` is a code artifact, not information (user feedback, live M5
        2026-08-14: 'group name list: assaf_test_2' reads weird)."""
        assert mf._humanise_args({"group_name_list": ["assaf_test_2"]}) == "group name: assaf_test_2"
        assert mf._humanise_args({"dashboard_ids": ["d1", "d2"]}) == "dashboard ids: d1, d2"


class TestPayloadVerdictStopsTheFlow:
    """The SDK can fail by RETURNING a failure report instead of raising —
    found live 2026-08-14: migrate_all_users wrote nothing (66/66 'already
    exists', payload ok:false), the wrapper said ok:true, the run log said
    'succeeded', and step 2 ran anyway. The payload's own verdict must fail
    the step so stop-on-failure and honest reporting fire."""

    FAILED_PAYLOAD = {
        "ok": False,
        "status": "failed",
        "success_count": 0,
        "failed_count": 66,
        "raw_error": {"error": {"message": "username/email already exists", "status": 400}},
    }

    def test_payload_verdict_fails_the_step_and_skips_the_rest(self):
        client = _client({"ok": True, "result": dict(self.FAILED_PAYLOAD)})
        plan_calls = [
            _call("migration.migrate_users", USER_ARGS, "c1"),
            _call("migration.migrate_groups", GROUP_ARGS, "c2"),
        ]
        approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
        reply, _ = _turn([_multi_resp(*plan_calls)], client, approved=approved)
        # Step 1 executed, step 2 never attempted.
        assert client.invoke_tool.await_count == 1
        assert "Stopped" in reply and "migration.migrate_users" in reply
        assert "username/email already exists" in reply, "the SDK's own words, not ours"
        assert "migration.migrate_groups" in reply, "the not-attempted step is named"

    def test_single_step_partial_failure_reports_without_stopped(self):
        """'Stopped' is only meaningful when later steps were cut short. A
        single-step plan that partially succeeded (SDK migrates what it can)
        gets the per-step counters line alone — live 2026-08-14: 'Stopped —
        failed: 232 of 295 succeeded' read as a contradiction."""
        payload = {"ok": False, "status": "failed", "succeeded_count": 232, "failed_count": 63, "total_count": 295}
        client = _client({"ok": True, "result": payload})
        plan_calls = [_call("migration.migrate_users", USER_ARGS, "c1")]
        approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
        reply, _ = _turn([_multi_resp(*plan_calls)], client, approved=approved)
        assert "Stopped" not in reply
        assert "Not attempted" not in reply
        assert "completed with failures — 232 succeeded, 63 failed" in reply

    def test_failure_body_is_deterministic_with_summarization_on(self):
        """The failure report carries the per-step lines in BOTH summ modes —
        summ-on used to return the Stopped header with an empty body."""
        client = _client({"ok": True, "result": dict(self.FAILED_PAYLOAD)})
        plan_calls = [
            _call("migration.migrate_users", USER_ARGS, "c1"),
            _call("migration.migrate_groups", GROUP_ARGS, "c2"),
        ]
        approved = {m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(plan_calls))}
        # _turn runs with allow_summarization=True — exactly the mode whose
        # failure body used to be empty.
        reply, _ = _turn([_multi_resp(*plan_calls)], client, approved=approved)
        assert "Stopped" in reply, "later steps were cut short — header applies"
        assert "`migration.migrate_users` failed — username/email already exists" in reply

    def test_no_arguments_renders_as_nothing(self):
        assert mf._humanise_args({}) == ""

    def test_the_dialog_carries_no_tool_ids_or_json(self):
        client = _client()
        _turn(
            [
                _multi_resp(
                    _call("migration.migrate_groups", GROUP_ARGS, "c1"),
                    _call("migration.migrate_users", USER_ARGS, "c2"),
                ),
                _text_resp("This copies a group and a user to the target."),
            ],
            client,
        )
        reason = m.LAST_TOOL_RESULT["pending_confirmation"]["reason"]
        plan_block = reason.split("Approve to run the whole sequence")[0]
        assert "migration.migrate_" not in plan_block, "step list must not show tool ids"
        assert '{"' not in plan_block, "step list must not show raw JSON"
        # The values the user gave are still visible — that is the point of it.
        assert "Sales Team" in plan_block and "jane@x.com" in plan_block

    def test_only_one_approve_sentence_in_a_plan_dialog(self):
        client = _client()
        _turn(
            [
                _multi_resp(_call("migration.migrate_dashboards", {"dashboard_names": ["Sales"]}, "c1")),
                _text_resp("This copies a dashboard."),
            ],
            client,
        )
        reason = m.LAST_TOOL_RESULT["pending_confirmation"]["reason"]
        assert reason.lower().count("approve to run") == 1

    def test_single_tool_dialogs_keep_their_own_approve_line(self):
        """Chat's per-tool dialog has no plan above it, so it still needs one."""
        meta = m.TOOL_REGISTRY["migration.migrate_dashboards"]
        note = m._approval_disclosure("migration.migrate_dashboards", meta, {"dashboard_names": ["Sales"]})
        assert "Approve to run as described" in note


def test_completeness_check_is_off_by_default():
    """The human reading the numbered dialog is the completeness check in
    interactive use; the extra call is opt-in insurance for unattended runs.
    (Asserted on the config module — the class fixture above forces it on.)"""
    import backend.agent._config as cfg

    assert cfg.MIGRATION_COMPLETENESS_CHECK is False


def test_the_dialog_needs_no_llm_at_all():
    """The whole dialog is code — an LLM outage cannot degrade or delay the
    approval screen for destructive work."""
    import backend.agent.migration_flow as flow

    calls = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),
        _call("migration.migrate_users", USER_ARGS, "c2"),
    ]
    with patch.object(m, "call_llm_raw", new=AsyncMock(side_effect=RuntimeError("LLM down"))):
        text = flow._render_plan_dialog(calls)  # not async, nothing to await
    assert "2 operations will run in this order" in text
    assert "Sales Team" in text and "jane@x.com" in text
    assert "Approve to run the whole sequence" in text


def test_a_single_step_dialog_does_not_say_sequence():
    """One task is not a "whole sequence" — the closing line is singular-aware,
    like the opening line already was (caught by the user on a live M3 run)."""
    import backend.agent.migration_flow as flow

    text = flow._render_plan_dialog([_call("migration.migrate_all_datamodels", {}, "c1")])
    assert "Approve to run this migration" in text
    assert "whole sequence" not in text
    assert "operations will run in this order" not in text


def test_options_blocks_are_attributed_to_their_step():
    """In a multi-step dialog every options block is headed by its step number
    and label — found live on M2 (2026-08-14): all six unset params belonged to
    the dashboards step, and nothing said so."""
    import backend.agent.migration_flow as flow

    calls = [
        _call("migration.migrate_groups", GROUP_ARGS, "c1"),  # fully specified — no block
        _call("migration.migrate_dashboards", {}, "c2"),  # dashboard_names + action unset
    ]
    text = flow._render_plan_dialog(calls)
    label = flow._step_label("migration.migrate_dashboards")
    assert f"Optional settings for step 2 — {label} (not set)" in text
    # The generic single-tool heading must not appear: an unattributed block is
    # exactly the ambiguity this fixes.
    assert "Optional settings, not set" not in text
    # Steps without unset optionals contribute no block.
    assert "Optional settings for step 1" not in text


def test_closing_line_points_at_optionals_only_when_there_are_some():
    import backend.agent.migration_flow as flow

    with_opts = flow._render_plan_dialog(
        [_call("migration.migrate_groups", GROUP_ARGS, "c1"), _call("migration.migrate_dashboards", {}, "c2")]
    )
    assert "optional settings below" in with_opts

    # Both calls fully specified — nothing to point at, the line stays short.
    without = flow._render_plan_dialog(
        [_call("migration.migrate_groups", GROUP_ARGS, "c1"), _call("migration.migrate_users", USER_ARGS, "c2")]
    )
    assert "optional settings below" not in without
    assert "Approve to run the whole sequence, or cancel and ask again with any changes." in without


def test_single_step_dialog_keeps_the_plain_heading():
    """One step needs no attribution — 'for step 1' would be noise."""
    import backend.agent.migration_flow as flow

    text = flow._render_plan_dialog([_call("migration.migrate_dashboards", {}, "c1")])
    assert "Optional settings, not set" in text
    assert "Optional settings for step" not in text
    assert "optional settings below" in text


# ---------------------------------------------------------------------------
# 10) Clarification resume — found broken live 2026-08-10
# ---------------------------------------------------------------------------
def _flow_kwargs(client, **over):
    base = dict(
        latest_user_message={"role": "user", "content": over.pop("user_text_msg", "dash1 and dashA")},
        history=[],
        planning_context="",
        mode="migration",
        passed_tools=MIGRATION_TOOLS,
        user_text="dash1 and dashA",
        mcp_client=client,
        approved_mutations=set(),
        summ_on=False,
        turn_trace_id="t",
        trace={"outcome": "unknown"},
    )
    base.update(over)
    return base


def test_a_resolved_clarification_is_gated_not_executed():
    """The seed IS the plan — but it still goes through validation and the
    approval gate. (First draft of this fix routed seeds around the gate —
    straight to execution — which unit tests only catch if this test exists.)"""
    import backend.agent.migration_flow as mf2

    client = _client()
    seed = _call("migration.migrate_groups", GROUP_ARGS, "seed1")
    raw = AsyncMock(side_effect=AssertionError("no LLM call belongs in the seed path"))
    with patch.object(m, "call_llm_raw", new=raw):
        reply = run(mf2.run(seed_call=seed, **_flow_kwargs(client)))

    client.invoke_tool.assert_not_awaited(), "a seed must never execute without approval"
    pending = m.LAST_TOOL_RESULT["pending_confirmation"]
    assert pending["tool_id"] == mf2.PLAN_TOOL_ID
    assert pending["arguments"]["steps"] == [{"step": 1, "tool": "migration.migrate_groups", "arguments": GROUP_ARGS}]
    assert "Sales Team" in reply


def test_a_question_about_the_clarification_cannot_become_a_different_plan():
    """Live bug: pinned tool was migrate_dashboard_shares; the user asked what
    change_ownership meant; the planner turned the question into a gated
    migrate_all_dashboards plan. The pinned-tool guard re-asks instead."""
    import backend.agent.migration_flow as mf2

    client = _client()
    resume = {
        "tool_id": "migration.migrate_dashboard_shares",
        "missing_fields": ["source_dashboard_ids", "target_dashboard_ids"],
        "filled_args": {},
        "attempts": 1,
        "question": "Which dashboards' shares should I migrate?",
    }
    raw = AsyncMock(return_value=_multi_resp(_call("migration.migrate_all_datamodels", {}, "c9")))
    with patch.object(m, "call_llm_raw", new=raw):
        run(
            mf2.run(
                resume_clarification=resume, **_flow_kwargs(client, user_text_msg="is change_ownership a yes/no flag?")
            )
        )

    client.invoke_tool.assert_not_awaited()
    assert m.LAST_TOOL_RESULT is None or not (m.LAST_TOOL_RESULT or {}).get("pending_confirmation"), (
        "no plan may be gated from a non-answer"
    )
    assert m.LAST_PENDING_CLARIFICATION is not None, "it must re-ask"
    assert m.LAST_PENDING_CLARIFICATION["tool_id"] == "migration.migrate_dashboard_shares"
    assert m.LAST_PENDING_CLARIFICATION["attempts"] == 2, "the re-ask counts against the cap"


def test_a_resumed_plan_matching_the_pinned_tool_proceeds_normally():
    """The guard must not block the actual answer-path: a resume that plans the
    pinned tool gates it as usual."""
    import backend.agent.migration_flow as mf2

    client = _client()
    resume = {
        "tool_id": "migration.migrate_groups",
        "missing_fields": ["group_name_list"],
        "filled_args": {},
        "attempts": 1,
        "question": "Which groups?",
    }
    raw = AsyncMock(return_value=_multi_resp(_call("migration.migrate_groups", GROUP_ARGS, "c1")))
    with patch.object(m, "call_llm_raw", new=raw):
        run(mf2.run(resume_clarification=resume, **_flow_kwargs(client)))
    assert m.LAST_TOOL_RESULT["pending_confirmation"]["arguments"]["steps"][0]["tool"] == "migration.migrate_groups"


def test_no_calls_on_a_resume_reasks_instead_of_chatting():
    import backend.agent.migration_flow as mf2

    client = _client()
    resume = {
        "tool_id": "migration.migrate_groups",
        "missing_fields": ["group_name_list"],
        "filled_args": {},
        "attempts": 1,
        "question": "Which groups?",
    }
    raw = AsyncMock(return_value=_text_resp("Could you tell me which groups you mean?"))
    with patch.object(m, "call_llm_raw", new=raw):
        run(mf2.run(resume_clarification=resume, **_flow_kwargs(client, user_text_msg="hmm not sure")))
    assert m.LAST_PENDING_CLARIFICATION is not None
    assert m.LAST_PENDING_CLARIFICATION["attempts"] == 2


def test_a_question_about_the_clarification_gets_an_answer_before_the_reask():
    """The user's meta-question ("is change_ownership a yes/no flag?") is
    answered from the tool's own definition — one LLM call, schema only — and
    the structured question follows. Live gap 2026-08-10: the guard re-asked
    verbatim and the question went unanswered."""
    import backend.agent.migration_flow as mf2

    client = _client()
    resume = {
        "tool_id": "migration.migrate_dashboards",
        "missing_fields": ["dashboard_names"],
        "filled_args": {},
        "attempts": 1,
        "question": "Which dashboards?",
    }
    raw = AsyncMock(
        side_effect=[
            _multi_resp(_call("migration.migrate_all_datamodels", {}, "c9")),  # planner's wrong guess
            _text_resp("Yes — it's a yes/no setting and it defaults to no."),  # the answer call
        ]
    )
    with patch.object(m, "call_llm_raw", new=raw):
        reply = run(
            mf2.run(
                resume_clarification=resume, **_flow_kwargs(client, user_text_msg="is change_ownership a yes/no flag?")
            )
        )

    assert reply.startswith("Yes — it's a yes/no setting"), "their question is answered first"
    assert "I need a bit more information" in reply, "…then the structured re-ask follows"
    labels = [c.kwargs.get("label") for c in raw.await_args_list]
    assert "clarify_answer" in labels
    # and the answer call saw only the DEFINITION — never results or history
    answer_call = raw.await_args_list[labels.index("clarify_answer")].args[0]
    assert "Operation definition" in answer_call[-1]["content"]


def test_the_answer_call_failing_still_reasks():
    import backend.agent.migration_flow as mf2

    client = _client()
    resume = {
        "tool_id": "migration.migrate_groups",
        "missing_fields": ["group_name_list"],
        "filled_args": {},
        "attempts": 1,
        "question": "Which groups?",
    }
    raw = AsyncMock(
        side_effect=[
            _multi_resp(_call("migration.migrate_all_datamodels", {}, "c9")),
            RuntimeError("LLM down"),
        ]
    )
    with patch.object(m, "call_llm_raw", new=raw):
        reply = run(mf2.run(resume_clarification=resume, **_flow_kwargs(client, user_text_msg="what does this do?")))
    assert "I need a bit more information" in reply
    assert m.LAST_PENDING_CLARIFICATION is not None
