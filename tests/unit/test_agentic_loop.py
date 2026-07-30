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
# 6) ALLOW_SUMMARIZATION=false → single-shot, no decide call
# ---------------------------------------------------------------------------


def test_summarization_disabled_degrades_to_single_shot(monkeypatch):
    client = _fake_client()
    messages = [{"role": "user", "content": "get user a@b.com"}]
    nav = AsyncMock(return_value=(GET_USER_TOOLS, "access_management", "users", 0))
    raw = AsyncMock(side_effect=[_plan_resp("access_management.get_user", '{"user_email":"a@b.com"}')])
    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=raw):
        reply = run(m.call_llm_with_tools(messages, GET_USER_TOOLS, client, allow_summarization=False))

    client.invoke_tool.assert_awaited_once()
    assert raw.await_count == 1  # planning only — no decide call ever happened
    assert isinstance(reply, str) and reply  # local description, not an LLM answer


# ---------------------------------------------------------------------------
# 7) Backtrack: planner declines mixin tools → retried with whole package
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
