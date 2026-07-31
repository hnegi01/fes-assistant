"""
Unit tests for the Step 8 agentic loop in call_llm_with_tools.

Behaviour covered (LLM + router mocked; no creds, no network):
  - Single step: tool executes, decide call returns the final answer.
  - Two-step chain: decide says CONTINUE → second route/plan/execute → answer.
  - Step cap: decide keeps saying CONTINUE → partial "still to do" message.
  - Mutation mid-loop: gate pauses the loop, saves LAST_PENDING_LOOP.
  - Loop resume: approved tool executes directly (no re-plan) and the loop
    continues from the saved transcript.
  - ALLOW_SUMMARIZATION=false: no decide call — single-shot behaviour.
  - Backtrack: planner declines the mixin's tools → one retry with the whole
    package.

The decide protocol: a call_llm_raw text reply starting with "CONTINUE:" drives
another loop step; any other text is the final answer.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

import backend.agent.llm_agent as m

GET_USER_SCHEMA = {
    "type": "object",
    "properties": {"user_email": {"type": "string", "format": "email", "description": "Email of the user."}},
    "required": ["user_email"],
}
GET_DASHBOARDS_SCHEMA = {"type": "object", "properties": {}, "required": []}
DELETE_USER_SCHEMA = {
    "type": "object",
    "properties": {"user_name": {"type": "string", "description": "Name of the user to delete."}},
    "required": ["user_name"],
}


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    monkeypatch.setattr(
        m,
        "TOOL_REGISTRY",
        {
            "access_management.get_user": {
                "module": "access_management",
                "mutates": False,
                "description": "Retrieve a user by email.",
                "parameters": GET_USER_SCHEMA,
            },
            "dashboard.get_dashboards": {
                "module": "dashboard",
                "mutates": False,
                "description": "List dashboards.",
                "parameters": GET_DASHBOARDS_SCHEMA,
            },
            "access_management.delete_user": {
                "module": "access_management",
                "mutates": True,
                "description": "Delete a Sisense user.",
                "parameters": DELETE_USER_SCHEMA,
            },
        },
    )
    # The loop requires summarization (results go to the LLM) and gating on.
    monkeypatch.setattr(m, "ALLOW_SUMMARIZATION", True)
    monkeypatch.setattr(m, "REQUIRE_MUTATION_CONFIRM", True)


@pytest.fixture(autouse=True)
def reset_pending():
    m.LAST_PENDING_CLARIFICATION = None
    m.LAST_PENDING_LOOP = None
    yield
    m.LAST_PENDING_CLARIFICATION = None
    m.LAST_PENDING_LOOP = None


@pytest.fixture(autouse=True)
def no_decompose(monkeypatch):
    """Step-1 decomposition is an extra LLM call on summarization-on chat turns.
    Neutralise it here (identity) so tests drive call_llm_raw with fixed
    side-effect lists; decomposition behaviour is covered separately."""

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    monkeypatch.setattr(m, "_make_plan", _one_step_plan)


@pytest.fixture(autouse=True)
def disable_goal_checker(monkeypatch):
    """The goal checker adds an LLM call at each 'done'. Off by default so most
    tests keep tight side-effect lists; the checker tests re-enable it."""
    monkeypatch.setattr(m, "VERIFY_GOAL", False)


def _tool_def(name, schema):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": schema}}


def _plan_resp(tool_id, arguments_json, call_id="c1"):
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": call_id, "function": {"name": tool_id, "arguments": arguments_json}}],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _text_resp(text):
    return {
        "choices": [{"message": {"content": text, "tool_calls": []}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _fake_client(results=None):
    client = AsyncMock()
    if results:
        client.invoke_tool = AsyncMock(side_effect=results)
    else:
        client.invoke_tool = AsyncMock(return_value={"ok": True, "result": [{"id": 1}]})
    return client


GET_USER_TOOLS = [_tool_def("access_management.get_user", GET_USER_SCHEMA)]
DASHBOARD_TOOLS = [_tool_def("dashboard.get_dashboards", GET_DASHBOARDS_SCHEMA)]


def _run_turn(*, llm_responses, nav_side_effect, client, user_msg="do the thing", **kwargs):
    messages = [{"role": "user", "content": user_msg}]
    nav = AsyncMock(side_effect=nav_side_effect)
    raw = AsyncMock(side_effect=llm_responses)
    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=raw):
        reply = run(m.call_llm_with_tools(messages, GET_USER_TOOLS, client, allow_summarization=True, **kwargs))
    return reply, nav, raw


# ---------------------------------------------------------------------------
# 1) Single step: execute → decide returns final answer
# ---------------------------------------------------------------------------


def test_single_step_decide_answers():
    client = _fake_client()
    reply, _nav, raw = _run_turn(
        llm_responses=[
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),
            _text_resp("Found the user a@b.com — role admin."),  # decide: final answer
        ],
        nav_side_effect=[(GET_USER_TOOLS, "access_management", "users", 0)],
        client=client,
    )

    client.invoke_tool.assert_awaited_once()
    assert reply.startswith("Found the user a@b.com")
    assert m.LAST_PENDING_LOOP is None


# ---------------------------------------------------------------------------
# 2) Two-step chain: CONTINUE drives a second route/plan/execute
# ---------------------------------------------------------------------------


def test_two_step_chain_executes_both_then_answers():
    client = _fake_client(
        results=[
            {"ok": True, "result": {"email": "a@b.com"}},
            {"ok": True, "result": [{"title": "Sales"}]},
        ]
    )
    reply, nav, _raw = _run_turn(
        llm_responses=[
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),
            _text_resp("CONTINUE: list the dashboards"),  # decide after step 1
            _plan_resp("dashboard.get_dashboards", "{}", call_id="c2"),  # step-2 plan
            _text_resp("User a@b.com found; 1 dashboard: Sales."),  # decide: final
        ],
        nav_side_effect=[
            (GET_USER_TOOLS, "access_management", "users", 0),  # step-1 routing
            (DASHBOARD_TOOLS, "dashboard", "core", 0),  # step-2 routing (from CONTINUE text)
        ],
        client=client,
    )

    assert client.invoke_tool.await_count == 2
    second_tool = client.invoke_tool.await_args_list[1].args[0]
    assert second_tool == "dashboard.get_dashboards"
    # Step-2 routing input is the CONTINUE sentence, not the original message.
    step2_msg = nav.await_args_list[1].args[0]
    assert step2_msg["content"] == "list the dashboards"
    assert reply.startswith("User a@b.com found")


# ---------------------------------------------------------------------------
# 3) Step cap: CONTINUE past the cap → partial message
# ---------------------------------------------------------------------------


def test_step_cap_returns_partial(monkeypatch):
    monkeypatch.setattr(m, "MAX_AGENT_STEPS", 2)
    client = _fake_client()
    reply, _nav, _raw = _run_turn(
        llm_responses=[
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),
            _text_resp("CONTINUE: list dashboards"),  # decide after step 1
            _plan_resp("dashboard.get_dashboards", "{}", call_id="c2"),
            _text_resp("CONTINUE: check the folders too"),  # decide after step 2 → cap
        ],
        nav_side_effect=[
            (GET_USER_TOOLS, "access_management", "users", 0),
            (DASHBOARD_TOOLS, "dashboard", "core", 0),
        ],
        client=client,
    )

    assert client.invoke_tool.await_count == 2
    assert "check the folders too" in reply
    assert "step limit" in reply


# ---------------------------------------------------------------------------
# 4) Mutation mid-loop: gate pauses, LAST_PENDING_LOOP saved
# ---------------------------------------------------------------------------


def test_mutation_mid_loop_pauses_and_saves_state():
    client = _fake_client()
    delete_tools = [_tool_def("access_management.delete_user", DELETE_USER_SCHEMA)]
    reply, _nav, _raw = _run_turn(
        llm_responses=[
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),
            _text_resp("CONTINUE: delete that user"),  # decide after step 1
            _plan_resp("access_management.delete_user", '{"user_name":"a@b.com"}', call_id="c2"),
            _text_resp("This will permanently delete the user a@b.com."),  # mutation explanation
        ],
        nav_side_effect=[
            (GET_USER_TOOLS, "access_management", "users", 0),
            (delete_tools, "access_management", "users", 0),
        ],
        client=client,
    )

    # Only the read tool ran; the delete is gated.
    client.invoke_tool.assert_awaited_once()
    assert "delete" in reply.lower()
    assert m.LAST_TOOL_RESULT["pending_confirmation"]["tool_id"] == "access_management.delete_user"
    assert m.LAST_PENDING_LOOP is not None
    assert m.LAST_PENDING_LOOP["tool_id"] == "access_management.delete_user"
    assert m.LAST_PENDING_LOOP["steps_executed"] == 1
    assert len(m.LAST_PENDING_LOOP["transcript"]) == 2  # step-1 assistant + tool messages


# ---------------------------------------------------------------------------
# 5) Loop resume: approved tool executes directly, loop continues
# ---------------------------------------------------------------------------


def test_loop_resume_executes_approved_and_continues():
    client = _fake_client()
    pending_loop = {
        "transcript": [],
        "steps_executed": 0,
        "tool_id": "access_management.delete_user",
        "arguments": {"user_name": "a@b.com"},
    }
    key = m._approval_key("access_management.delete_user", {"user_name": "a@b.com"})

    messages = [{"role": "user", "content": "delete the user a@b.com"}]
    nav = AsyncMock()
    raw = AsyncMock(side_effect=[_text_resp("Deleted user a@b.com.")])  # decide: final
    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=raw):
        reply = run(
            m.call_llm_with_tools(
                messages,
                GET_USER_TOOLS,
                client,
                approved_mutations={key},
                allow_summarization=True,
                pending_loop=pending_loop,
            )
        )

    # Executed directly — no routing, no re-plan of the gated tool.
    nav.assert_not_called()
    client.invoke_tool.assert_awaited_once()
    called_tool, called_args = client.invoke_tool.await_args.args
    assert called_tool == "access_management.delete_user"
    assert called_args == {"user_name": "a@b.com"}
    assert reply == "Deleted user a@b.com."
    assert m.LAST_PENDING_LOOP is None  # cleared after resume


# ---------------------------------------------------------------------------
# 6) ALLOW_SUMMARIZATION=false → still loops, but on metadata; final answer is
#    rendered locally (no result data ever sent to the LLM).
# ---------------------------------------------------------------------------


def test_summarization_off_runs_metadata_loop(monkeypatch):
    """Two-step chain with summ off: independent steps still run; the decide call
    sees metadata (nodata prompt), replies DONE, and the answer is local."""
    sent_to_llm = []

    client = _fake_client(
        results=[{"ok": True, "result": [{"secretcol": 1}]}, {"ok": True, "result": [{"othercol": 2}]}]
    )
    messages = [{"role": "user", "content": "list users and list dashboards"}]
    nav = AsyncMock(
        side_effect=[
            (GET_USER_TOOLS, "access_management", "users", 0),
            (DASHBOARD_TOOLS, "dashboard", "core", 0),
        ]
    )

    async def _capturing_raw(msgs, **kw):
        sent_to_llm.append(msgs)
        # planning step 1 → nodata decide CONTINUE → planning step 2 → nodata decide DONE
        seq = [
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),
            _text_resp("CONTINUE: list dashboards"),
            _plan_resp("dashboard.get_dashboards", "{}", call_id="c2"),
            _text_resp("DONE"),
        ]
        return seq[len(sent_to_llm) - 1]

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    with (
        patch.object(m, "_navigate_to_tools", new=nav),
        patch.object(m, "call_llm_raw", new=_capturing_raw),
        patch.object(m, "_make_plan", new=_one_step_plan),
    ):
        reply = run(m.call_llm_with_tools(messages, GET_USER_TOOLS, client, allow_summarization=False))

    # Both independent steps executed (summ off is no longer single-shot).
    assert client.invoke_tool.await_count == 2

    # Inspect every tool-role message that reached the LLM.
    tool_msgs = [msg for call in sent_to_llm for msg in call if msg.get("role") == "tool"]
    assert tool_msgs, "expected tool results in the LLM context"
    for tm in tool_msgs:
        payload = json.loads(tm["content"])
        # Privacy: metadata only — the actual column data never left the process.
        assert set(payload.keys()) <= {"tool", "ok", "count", "error"}, f"data leaked to LLM: {payload}"
        assert "secretcol" not in tm["content"] and "othercol" not in tm["content"]

    assert isinstance(reply, str) and reply


# ---------------------------------------------------------------------------
# 6b) Summ off + adaptive: the decide call says BLOCKED (needs a value it can't
#     see); the loop stops gracefully and explains, no data leaked.
# ---------------------------------------------------------------------------


def test_summarization_off_adaptive_blocks_gracefully():
    client = _fake_client(results=[{"ok": True, "result": [{"id": "hidden"}]}])
    messages = [{"role": "user", "content": "find the datamodels owned by john"}]
    nav = AsyncMock(return_value=(GET_USER_TOOLS, "access_management", "users", 0))

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    raw = AsyncMock(
        side_effect=[
            _plan_resp("access_management.get_user", '{"user_email":"john@acme.com"}'),  # step 1
            _text_resp("BLOCKED: need john's user id from step 1 to list his datamodels"),  # nodata decide
        ]
    )
    with (
        patch.object(m, "_navigate_to_tools", new=nav),
        patch.object(m, "call_llm_raw", new=raw),
        patch.object(m, "_make_plan", new=_one_step_plan),
    ):
        reply = run(m.call_llm_with_tools(messages, GET_USER_TOOLS, client, allow_summarization=False))

    client.invoke_tool.assert_awaited_once()  # step 1 ran; the blocked step did not
    low = reply.lower()
    assert "summarization" in low and ("can't" in low or "cannot" in low or "turn summarization on" in low)


# ---------------------------------------------------------------------------
# 8) Overreach: a continued step needs an arg the user never gave → the loop
#    stops and answers from what it has (no confusing mid-loop clarification).
# ---------------------------------------------------------------------------


def test_continued_step_missing_arg_finalizes_not_clarifies():
    client = _fake_client()
    get_user_tools = [_tool_def("access_management.get_user", GET_USER_SCHEMA)]
    reply, _nav, _raw = _run_turn(
        llm_responses=[
            _plan_resp("dashboard.get_dashboards", "{}"),  # step 1: list dashboards
            _text_resp("CONTINUE: get that user's details"),  # decide overreaches
            _plan_resp("access_management.get_user", "{}"),  # step-2 plan: no email (user never gave one)
            _text_resp("Here are your 3 dashboards: A, B, C."),  # finalize call answers from what we have
        ],
        nav_side_effect=[
            (DASHBOARD_TOOLS, "dashboard", "core", 0),
            (get_user_tools, "access_management", "users", 0),
        ],
        client=client,
    )

    # Only the first tool ran; the overreaching second step never executed.
    client.invoke_tool.assert_awaited_once()
    # It answered — did NOT pause for a "which user's email?" clarification.
    assert m.LAST_PENDING_CLARIFICATION is None
    assert "dashboards" in reply.lower()


# ---------------------------------------------------------------------------
# 9) Backtrack: planner declines mixin tools → retried with whole package
# ---------------------------------------------------------------------------


def test_backtrack_retries_with_full_package(monkeypatch):
    client = _fake_client()
    full_package = [
        _tool_def("dashboard.get_dashboards", GET_DASHBOARDS_SCHEMA),
        _tool_def("access_management.get_user", GET_USER_SCHEMA),
    ]
    monkeypatch.setattr(m, "_load_all_package_tools", lambda pkg: full_package)

    reply, _nav, raw = _run_turn(
        llm_responses=[
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),
            _text_resp("CONTINUE: list dashboards"),
            _text_resp("I can't find a matching tool."),  # step-2 plan declines (mixin tools)
            _plan_resp("dashboard.get_dashboards", "{}", call_id="c2"),  # retry with full package
            _text_resp("Done: 1 dashboard."),  # decide: final
        ],
        nav_side_effect=[
            (GET_USER_TOOLS, "access_management", "users", 0),
            ([_tool_def("dashboard.get_widget", GET_DASHBOARDS_SCHEMA)], "dashboard", "widgets", 0),
        ],
        client=client,
    )

    assert client.invoke_tool.await_count == 2
    assert reply == "Done: 1 dashboard."


# ---------------------------------------------------------------------------
# 10) Goal checker (verify #3): a "done" is re-checked by an independent call.
# ---------------------------------------------------------------------------


def test_goal_checker_pushes_incomplete_then_finishes(monkeypatch):
    """decide says done → checker says INCOMPLETE → the loop runs one more step,
    then the (capped) checker accepts the next done."""
    monkeypatch.setattr(m, "VERIFY_GOAL", True)
    monkeypatch.setattr(m, "VERIFY_MAX_RECHECKS", 1)
    client = _fake_client(results=[{"ok": True, "result": {"e": "a@b.com"}}, {"ok": True, "result": [{"t": "Sales"}]}])
    reply, _nav, _raw = _run_turn(
        llm_responses=[
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),  # step 1 plan
            _text_resp("Here is the user."),  # decide 1: looks done...
            _text_resp("INCOMPLETE: list the user's dashboards"),  # checker overrides
            _plan_resp("dashboard.get_dashboards", "{}", call_id="c2"),  # step 2 plan
            _text_resp("User and 1 dashboard shown."),  # decide 2: done (checker now capped)
        ],
        nav_side_effect=[
            (GET_USER_TOOLS, "access_management", "users", 0),
            (DASHBOARD_TOOLS, "dashboard", "core", 0),
        ],
        client=client,
    )
    # The checker forced a second step that the maker had skipped.
    assert client.invoke_tool.await_count == 2
    assert reply == "User and 1 dashboard shown."


def test_goal_checker_confirms_complete(monkeypatch):
    """decide says done → checker says COMPLETE → answer returned, no extra step."""
    monkeypatch.setattr(m, "VERIFY_GOAL", True)
    monkeypatch.setattr(m, "VERIFY_MAX_RECHECKS", 1)
    client = _fake_client()
    reply, _nav, _raw = _run_turn(
        llm_responses=[
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),
            _text_resp("Found the user a@b.com."),  # decide: done
            _text_resp("COMPLETE"),  # checker: agrees
        ],
        nav_side_effect=[(GET_USER_TOOLS, "access_management", "users", 0)],
        client=client,
    )
    client.invoke_tool.assert_awaited_once()
    assert reply == "Found the user a@b.com."


# ---------------------------------------------------------------------------
# 11) Plan→replan: decide says REPLAN → strategist revises → new op executes.
# ---------------------------------------------------------------------------


def _plan_text_resp(text):
    return _text_resp(text)


def test_decide_replan_revises_and_continues(monkeypatch):
    monkeypatch.setattr(m, "MAX_REPLANS", 1)
    client = _fake_client(
        results=[
            {"ok": False, "error": "Group 'x' not found."},  # step 1: wrong approach fails
            {"ok": True, "result": {"email": "a@b.com", "groups": ["Everyone"]}},  # step 2 after replan
        ]
    )
    reply, _nav, raw = _run_turn(
        llm_responses=[
            _plan_resp("dashboard.get_dashboards", "{}"),  # step-1 plan (wrong approach)
            _text_resp("REPLAN: that approach failed; still need the user's group"),  # decide
            _plan_text_resp("1. Get the user record for a@b.com"),  # strategist replan
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}', call_id="c2"),  # new plan call
            _text_resp("a@b.com belongs to the Everyone group."),  # decide: final
        ],
        nav_side_effect=[
            (DASHBOARD_TOOLS, "dashboard", "core", 0),
            (GET_USER_TOOLS, "access_management", "users", 0),
        ],
        client=client,
    )

    assert client.invoke_tool.await_count == 2
    second_tool = client.invoke_tool.await_args_list[1].args[0]
    assert second_tool == "access_management.get_user"
    assert reply == "a@b.com belongs to the Everyone group."


def test_replan_budget_exhausted_gives_up_gracefully(monkeypatch):
    monkeypatch.setattr(m, "MAX_REPLANS", 0)  # replanning disabled
    client = _fake_client(results=[{"ok": False, "error": "not found"}])
    reply, _nav, _raw = _run_turn(
        llm_responses=[
            _plan_resp("dashboard.get_dashboards", "{}"),  # step 1 fails
            _text_resp("REPLAN: approach failed"),  # decide wants a replan
            _text_resp("Here is what I found so far."),  # finalize call
        ],
        nav_side_effect=[(DASHBOARD_TOOLS, "dashboard", "core", 0)],
        client=client,
    )

    client.invoke_tool.assert_awaited_once()  # nothing further executed
    assert isinstance(reply, str) and reply


def test_replan_strategist_giveup_message_surfaces(monkeypatch):
    monkeypatch.setattr(m, "MAX_REPLANS", 1)
    client = _fake_client(results=[{"ok": False, "error": "not found"}])
    reply, _nav, _raw = _run_turn(
        llm_responses=[
            _plan_resp("dashboard.get_dashboards", "{}"),  # step 1 fails
            _text_resp("REPLAN: approach failed"),  # decide
            _text_resp("GIVEUP: There is no operation that can retrieve this."),  # strategist
            _text_resp("Summary of what ran."),  # finalize
        ],
        nav_side_effect=[(DASHBOARD_TOOLS, "dashboard", "core", 0)],
        client=client,
    )

    assert "no operation that can retrieve this" in reply


# ---------------------------------------------------------------------------
# 12) Summ-off dependency gate: plan steps tagged [needs-prior-result] are
#     skipped up front — no doomed call — and the reply says why.
# ---------------------------------------------------------------------------


def test_summ_off_dependency_gate_skips_tagged_tail(monkeypatch):
    async def _tagged_plan(user_text, mode, history, trace_id):
        return [
            "Get the user record for a@b.com",
            "List all users in that group [needs-prior-result]",
        ]

    monkeypatch.setattr(m, "_make_plan", _tagged_plan)

    client = _fake_client(results=[{"ok": True, "result": [{"USER": "a@b.com", "GROUPS": ["Everyone"]}]}])
    messages = [{"role": "user", "content": "which group does a@b.com belong to and show its users"}]
    nav = AsyncMock(return_value=(GET_USER_TOOLS, "access_management", "users", 0))
    raw = AsyncMock(
        side_effect=[
            _plan_resp("access_management.get_user", '{"user_email":"a@b.com"}'),  # step-1 plan call
            _text_resp("DONE"),  # nodata decide: prefix complete
        ]
    )
    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=raw):
        reply = run(m.call_llm_with_tools(messages, GET_USER_TOOLS, client, allow_summarization=False))

    # Only the independent prefix executed; the dependent tail never ran.
    client.invoke_tool.assert_awaited_once()
    low = reply.lower()
    assert "skipped" in low and "summarization" in low
    assert "list all users in that group" in low  # names what was skipped, marker stripped
    assert "[needs-prior-result]" not in low
