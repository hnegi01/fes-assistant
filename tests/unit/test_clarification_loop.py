"""
Unit tests for the Step 7 clarification loop in call_llm_with_tools.

Behaviour covered (planner LLM mocked; no creds, no network):
  - Fresh turn, missing required arg → asks a question, records pending state,
    does NOT execute the tool.
  - Resume turn whose answer fills the field → executes, clears pending state,
    skips the router.
  - Resume turn still missing past the attempt cap → terminal give-up message,
    clears pending state.
  - Resume turn where the planner declines the pinned tool (topic change) →
    falls back to fresh routing and runs the new tool.

call_llm_raw is patched on backend.agent.llm_agent (where the names resolve) and
driven with side_effect lists because a single turn can call it multiple times
(plan → clarification-question; or resume-plan → fresh-plan).
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
GET_USERS_ALL_SCHEMA = {"type": "object", "properties": {}, "required": []}
DELETE_USER_SCHEMA = {
    "type": "object",
    "properties": {"user_name": {"type": "string", "description": "Name of the user to delete."}},
    "required": ["user_name"],
}
# Two required fields, for the multiple-missing case.
SCHEDULE_SCHEMA = {
    "type": "object",
    "properties": {
        "datamodel_name": {"type": "string", "description": "Datamodel to schedule."},
        "hour": {"type": "integer", "description": "Hour of day to run."},
    },
    "required": ["datamodel_name", "hour"],
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
            "access_management.get_users_all": {
                "module": "access_management",
                "mutates": False,
                "description": "List all users.",
                "parameters": GET_USERS_ALL_SCHEMA,
            },
            "access_management.delete_user": {
                "module": "access_management",
                "mutates": True,
                "description": "Delete a Sisense user.",
                "parameters": DELETE_USER_SCHEMA,
            },
            "access_management.create_schedule_build": {
                "module": "access_management",
                "mutates": True,
                "description": "Schedule a datamodel build.",
                "parameters": SCHEDULE_SCHEMA,
            },
        },
    )


@pytest.fixture(autouse=True)
def reset_pending():
    m.LAST_PENDING_CLARIFICATION = None
    m.LAST_PENDING_LOOP = None
    yield
    m.LAST_PENDING_CLARIFICATION = None
    m.LAST_PENDING_LOOP = None


@pytest.fixture(autouse=True)
def no_decompose(monkeypatch):
    """Step-1 decomposition is an extra LLM call on every chat turn now (both
    summ modes). Neutralise it (identity, no call_llm_raw) so tests drive
    call_llm_raw with fixed side-effect lists."""

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    monkeypatch.setattr(m, "_make_plan", _one_step_plan)


def _tool_def(name, schema):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": schema}}


def _plan_resp(tool_id, arguments_json):
    """Planner response that calls a tool."""
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": tool_id, "arguments": arguments_json}}],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _text_resp(text):
    """Planner/question response that is plain text (no tool call) — also a 'decline'."""
    return {
        "choices": [{"message": {"content": text, "tool_calls": []}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _fake_client(result=None):
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value=result or {"ok": True, "data": [{"id": 1}]})
    return client


def _run_turn(*, pending, llm_responses, nav_tools, client, user_msg="answer"):
    """Drive one call_llm_with_tools turn with router + planner mocked."""
    tools = [_tool_def("access_management.get_user", GET_USER_SCHEMA)]
    messages = [{"role": "user", "content": user_msg}]
    nav = AsyncMock(return_value=(nav_tools, "access_management", "core", 0))
    raw = AsyncMock(side_effect=llm_responses)
    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=raw):
        reply = run(
            m.call_llm_with_tools(messages, tools, client, allow_summarization=False, pending_clarification=pending)
        )
    return reply, nav, raw


# ---------------------------------------------------------------------------
# 1) Fresh turn, missing required → ask + record pending, no execution
# ---------------------------------------------------------------------------


def test_missing_required_asks_and_records_pending():
    client = _fake_client()
    reply, _nav, _raw = _run_turn(
        pending=None,
        # 1st call_llm_raw = planning (tool chosen, no args). The clarification
        # question itself is rendered in code — no second LLM call.
        llm_responses=[_plan_resp("access_management.get_user", "{}")],
        nav_tools=[_tool_def("access_management.get_user", GET_USER_SCHEMA)],
        client=client,
    )

    client.invoke_tool.assert_not_called()
    assert reply.startswith("I need a bit more information")
    assert "Email of the user" in reply, "the bullet comes from the schema's own description"
    assert m.LAST_PENDING_CLARIFICATION is not None
    assert m.LAST_PENDING_CLARIFICATION["tool_id"] == "access_management.get_user"
    assert m.LAST_PENDING_CLARIFICATION["missing_fields"] == ["user_email"]
    assert m.LAST_PENDING_CLARIFICATION["attempts"] == 1


# ---------------------------------------------------------------------------
# 2) Resume turn whose answer fills the field → execute, clear, skip router
# ---------------------------------------------------------------------------


def test_resume_with_answer_executes_and_clears():
    client = _fake_client()
    pending = {
        "tool_id": "access_management.get_user",
        "missing_fields": ["user_email"],
        "filled_args": {},
        "attempts": 1,
    }
    reply, nav, _raw = _run_turn(
        pending=pending,
        # Only the resume planning call — it now fills user_email from the answer.
        llm_responses=[_plan_resp("access_management.get_user", '{"user_email":"jane@example.com"}')],
        nav_tools=[_tool_def("access_management.get_user", GET_USER_SCHEMA)],
        client=client,
        user_msg="jane@example.com",
    )

    # Router skipped on resume.
    nav.assert_not_called()
    # Tool executed with the merged arg.
    client.invoke_tool.assert_awaited_once()
    called_tool, called_args = client.invoke_tool.await_args.args
    assert called_tool == "access_management.get_user"
    assert called_args["user_email"] == "jane@example.com"
    # Pending cleared.
    assert m.LAST_PENDING_CLARIFICATION is None


# ---------------------------------------------------------------------------
# 3) Resume still missing past the cap → terminal give-up, clear
# ---------------------------------------------------------------------------


def test_resume_exhausts_cap_and_gives_up():
    client = _fake_client()
    # attempts base = 2 → next failure makes attempt 3 > CLARIFY_MAX_ATTEMPTS (2).
    pending = {
        "tool_id": "access_management.get_user",
        "missing_fields": ["user_email"],
        "filled_args": {},
        "attempts": 2,
    }
    reply, _nav, _raw = _run_turn(
        pending=pending,
        llm_responses=[_plan_resp("access_management.get_user", "{}")],  # resume plan, still missing
        nav_tools=[_tool_def("access_management.get_user", GET_USER_SCHEMA)],
        client=client,
        user_msg="dunno",
    )

    client.invoke_tool.assert_not_called()
    assert "still don't have" in reply.lower()
    # State cleared so the user isn't stuck in the loop.
    assert m.LAST_PENDING_CLARIFICATION is None


# ---------------------------------------------------------------------------
# 4) Resume where planner declines pinned tool (topic change) → fresh routing
# ---------------------------------------------------------------------------


def test_resume_decline_falls_back_to_fresh_routing():
    client = _fake_client()
    pending = {
        "tool_id": "access_management.get_user",
        "missing_fields": ["user_email"],
        "filled_args": {},
        "attempts": 1,
    }
    reply, nav, _raw = _run_turn(
        pending=pending,
        # 1st call = resume plan DECLINES (plain text, no tool_call) → topic change.
        # 2nd call = fresh planning after routing → picks the list-all tool.
        llm_responses=[_text_resp("sure"), _plan_resp("access_management.get_users_all", "{}")],
        nav_tools=[_tool_def("access_management.get_users_all", GET_USERS_ALL_SCHEMA)],
        client=client,
        user_msg="actually just list all users",
    )

    # Two router calls: the fresh-intent test (does this message stand alone?),
    # then the real routing for the new request. The tool ran.
    assert nav.await_count == 2
    assert nav.await_args_list[0].args[1] == [], "the intent test must route WITHOUT history"
    client.invoke_tool.assert_awaited_once()
    called_tool, _ = client.invoke_tool.await_args.args
    assert called_tool == "access_management.get_users_all"
    # Old clarification cleared.
    assert m.LAST_PENDING_CLARIFICATION is None


def test_resume_decline_with_no_fresh_intent_reasks_instead_of_replanning():
    """A NON-answer ("i'm not sure") must re-ask the clarification, not become a
    new request. The planner sees conversation history and will happily turn a
    non-answer into a confident step — live 2026-08-18, "i'm not sure" after a
    delete-user clarification was planned as "List all users" and EXECUTED,
    silently killing the clarification so the attempt cap never fired. The
    fresh-intent test (route the message alone; no history) runs before the
    planner and catches it."""
    client = _fake_client()
    pending = {
        "tool_id": "access_management.get_user",
        "missing_fields": ["user_email"],
        "filled_args": {},
        "attempts": 1,
    }
    nav = AsyncMock(return_value=([], "__unclear__", "", 0))
    raw = AsyncMock(side_effect=[_text_resp("hmm")])  # resume plan declines; nothing after
    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=raw):
        reply = run(
            m.call_llm_with_tools(
                [{"role": "user", "content": "i'm not sure"}],
                [_tool_def("access_management.get_user", GET_USER_SCHEMA)],
                client,
                allow_summarization=False,
                pending_clarification=pending,
            )
        )

    client.invoke_tool.assert_not_called(), "a non-answer must never execute anything"
    assert reply.startswith("I need a bit more information")
    assert m.LAST_PENDING_CLARIFICATION is not None
    assert m.LAST_PENDING_CLARIFICATION["attempts"] == 2, "the attempt must count toward the cap"


def test_resume_decline_no_fresh_intent_gives_up_at_the_cap():
    """At the cap, the terminal message states what is still required and the
    clarification is dropped — no endless asking."""
    client = _fake_client()
    pending = {
        "tool_id": "access_management.get_user",
        "missing_fields": ["user_email"],
        "filled_args": {},
        "attempts": m.CLARIFY_MAX_ATTEMPTS,  # one more non-answer exceeds it
    }
    nav = AsyncMock(return_value=([], "__unclear__", "", 0))
    raw = AsyncMock(side_effect=[_text_resp("hmm")])
    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=raw):
        reply = run(
            m.call_llm_with_tools(
                [{"role": "user", "content": "still don't know"}],
                [_tool_def("access_management.get_user", GET_USER_SCHEMA)],
                client,
                allow_summarization=False,
                pending_clarification=pending,
            )
        )

    client.invoke_tool.assert_not_called()
    low = reply.lower()
    assert "need" in low or "required" in low, f"expected a terminal requirements message, got: {reply!r}"


# ---------------------------------------------------------------------------
# 5) Multiple missing required fields → all asked for in one question
# ---------------------------------------------------------------------------


def test_multiple_missing_fields_recorded_together():
    client = _fake_client()
    reply, _nav, _raw = _run_turn(
        pending=None,
        llm_responses=[
            _plan_resp("access_management.create_schedule_build", "{}"),  # both required missing
            _text_resp("Which datamodel, and at what hour?"),
        ],
        nav_tools=[_tool_def("access_management.create_schedule_build", SCHEDULE_SCHEMA)],
        client=client,
        user_msg="schedule a build",
    )

    client.invoke_tool.assert_not_called()
    assert m.LAST_PENDING_CLARIFICATION is not None
    assert set(m.LAST_PENDING_CLARIFICATION["missing_fields"]) == {"datamodel_name", "hour"}


# ---------------------------------------------------------------------------
# 6) Mutating tool with all args → approval gate fires with English reason
# ---------------------------------------------------------------------------


def test_mutating_tool_pending_confirmation_has_english_reason():
    client = _fake_client()
    reply, _nav, _raw = _run_turn(
        pending=None,
        llm_responses=[
            _plan_resp("access_management.delete_user", '{"user_name":"bob"}'),  # planning
            _text_resp("This will permanently delete the user bob."),  # mutation explanation
        ],
        nav_tools=[_tool_def("access_management.delete_user", DELETE_USER_SCHEMA)],
        client=client,
        user_msg="delete user bob",
    )

    # Gate fires — tool NOT executed without approval.
    client.invoke_tool.assert_not_called()
    assert m.LAST_TOOL_RESULT is not None
    pc = m.LAST_TOOL_RESULT.get("pending_confirmation")
    assert pc is not None
    assert pc["tool_id"] == "access_management.delete_user"
    # The generated English explanation is surfaced as the reason and the reply.
    assert pc["reason"] == "This will permanently delete the user bob."
    assert reply == "This will permanently delete the user bob."


# ---------------------------------------------------------------------------
# 7) Clarify-then-mutate: missing arg on a mutating tool clarifies first,
#    then the approval gate fires once the arg is supplied.
# ---------------------------------------------------------------------------


def test_clarify_then_mutation_gate():
    # Turn 1 (fresh): missing required on a mutating tool → clarify, no gate yet.
    client = _fake_client()
    _run_turn(
        pending=None,
        llm_responses=[_plan_resp("access_management.delete_user", "{}"), _text_resp("Which user should I delete?")],
        nav_tools=[_tool_def("access_management.delete_user", DELETE_USER_SCHEMA)],
        client=client,
        user_msg="delete a user",
    )
    client.invoke_tool.assert_not_called()
    assert m.LAST_PENDING_CLARIFICATION is not None
    pending = m.LAST_PENDING_CLARIFICATION

    # Turn 2 (resume): answer fills the arg → validation passes → mutation gate fires.
    client2 = _fake_client()
    reply2, _nav2, _raw2 = _run_turn(
        pending=pending,
        llm_responses=[
            _plan_resp("access_management.delete_user", '{"user_name":"bob"}'),  # resume plan
            _text_resp("This will permanently delete the user bob."),  # mutation explanation
        ],
        nav_tools=[_tool_def("access_management.delete_user", DELETE_USER_SCHEMA)],
        client=client2,
        user_msg="bob",
    )

    # Still not executed — now blocked by the approval gate, not by clarification.
    client2.invoke_tool.assert_not_called()
    assert m.LAST_PENDING_CLARIFICATION is None  # clarification resolved
    pc = m.LAST_TOOL_RESULT.get("pending_confirmation")
    assert pc is not None and pc["arguments"]["user_name"] == "bob"


# ---------------------------------------------------------------------------
# 8) Format error (value present, wrong type/format) → hard block, NOT clarified
# ---------------------------------------------------------------------------


def test_format_error_is_hard_blocked_not_clarified():
    """user_email='john' passes as present but fails format:email → hard block only.

    The clarification loop must NOT fire when the field is present but wrong-format.
    Verified two ways: LAST_PENDING_CLARIFICATION stays None, and only 1 LLM call
    is made (the clarification-question generator is never reached).
    """
    client = _fake_client()
    reply, _nav, raw = _run_turn(
        pending=None,
        llm_responses=[_plan_resp("access_management.get_user", '{"user_email":"john"}')],
        nav_tools=[_tool_def("access_management.get_user", GET_USER_SCHEMA)],
        client=client,
        user_msg="show me user john",
    )

    client.invoke_tool.assert_not_called()
    assert "couldn't call" in reply.lower()
    # Clarification loop must NOT have fired.
    assert m.LAST_PENDING_CLARIFICATION is None
    # Only the planning call — no second LLM call for a clarification question.
    assert raw.call_count == 1


# ---------------------------------------------------------------------------
# 9) Off-topic / unclear intent → __unclear__ short-circuit before planner
# ---------------------------------------------------------------------------


def test_unclear_intent_short_circuits():
    """_navigate_to_tools returning __unclear__ exits before any planner call.

    Zero calls to call_llm_raw, zero MCP calls, no pending state written.
    """
    client = _fake_client()
    tools = [_tool_def("access_management.get_user", GET_USER_SCHEMA)]
    messages = [{"role": "user", "content": "what's the capital of France?"}]
    nav = AsyncMock(return_value=([], "__unclear__", "", 0))
    raw = AsyncMock()

    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=raw):
        reply = run(
            m.call_llm_with_tools(messages, tools, client, allow_summarization=False, pending_clarification=None)
        )

    nav.assert_awaited_once()
    raw.assert_not_called()
    client.invoke_tool.assert_not_called()
    assert "understand" in reply.lower()
    assert m.LAST_PENDING_CLARIFICATION is None


# ---------------------------------------------------------------------------
# 10) Approved mutation executes and summarizes (Scenario 9)
# ---------------------------------------------------------------------------


def test_approved_mutation_executes():
    """When the approval key is in approved_mutations the gate passes and the tool executes.

    Flow: fresh routing → planner picks delete_user(user_name='bob') → gate checks
    approved_mutations → key present → MCP call made → summarizer runs.
    """
    import json

    client = _fake_client(result={"ok": True, "result": "User bob deleted."})

    args = {"user_name": "bob"}
    # Reproduce exactly how _approval_key() constructs the key.
    approval_key = ("access_management.delete_user", json.dumps(args, sort_keys=True))

    tools = [_tool_def("access_management.delete_user", DELETE_USER_SCHEMA)]
    messages = [{"role": "user", "content": "delete user bob"}]
    nav = AsyncMock(return_value=(tools, "access_management", "users", 0))
    raw = AsyncMock(
        side_effect=[
            _plan_resp("access_management.delete_user", json.dumps(args)),  # planning
            _text_resp("Done. User bob has been deleted."),  # decide → final answer
        ]
    )

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    with (
        patch.object(m, "_navigate_to_tools", new=nav),
        patch.object(m, "call_llm_raw", new=raw),
        patch.object(m, "_make_plan", new=_one_step_plan),
    ):
        reply = run(
            m.call_llm_with_tools(
                messages,
                tools,
                client,
                approved_mutations={approval_key},
                allow_summarization=True,
                pending_clarification=None,
            )
        )

    # Gate passed — tool executed.
    client.invoke_tool.assert_awaited_once()
    called_tool, called_args = client.invoke_tool.await_args.args
    assert called_tool == "access_management.delete_user"
    assert called_args["user_name"] == "bob"
    # Summarizer ran and its output reached the caller.
    assert "deleted" in reply.lower()
    # No pending state — clean turn.
    assert m.LAST_PENDING_CLARIFICATION is None
    assert m.LAST_TOOL_RESULT is not None
    assert m.LAST_TOOL_RESULT.get("ok") is True


def test_clarification_shows_the_registry_example_as_a_phrasing_template():
    """ "Paired positionally" tells the user little; the curated example[0]
    query shows the shape to copy. Code-appended from the registry — no LLM."""
    import asyncio

    meta = {
        "description": "Migrate shares for specific dashboards from the source to the target environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_dashboard_ids": {"type": "array", "description": "Dashboard IDs from the source environment."},
                "target_dashboard_ids": {"type": "array", "description": "Dashboard IDs from the target environment."},
            },
            "required": ["source_dashboard_ids", "target_dashboard_ids"],
        },
        "examples": [
            {
                "user_query": "Migrate shares from source dashboards dash123 to target dashboards dashABC.",
                "arguments": {"source_dashboard_ids": ["dash123"], "target_dashboard_ids": ["dashABC"]},
            }
        ],
    }
    q = asyncio.get_event_loop().run_until_complete(
        m._generate_clarification_question(
            "migration.migrate_dashboard_shares", meta, ["source_dashboard_ids", "target_dashboard_ids"], {}, "t"
        )
    )
    # Same renderer as the approval dialog (_example_hint) — one format, no drift.
    assert '*For example, you could ask: "Migrate shares from source dashboards dash123' in q


def test_clarification_without_examples_omits_the_example_line():
    meta = {"description": "Get a user.", "parameters": GET_USER_SCHEMA}
    import asyncio

    q = asyncio.get_event_loop().run_until_complete(
        m._generate_clarification_question("access_management.get_user", meta, ["user_email"], {}, "t")
    )
    assert "For example" not in q


def test_empty_containers_count_as_missing_required():
    """The model expresses "the user gave me nothing" three ways: omitting the
    key, null/"", or an EMPTY LIST. All three must clarify. Seen live
    2026-08-14: migrate_dashboard_shares gated with source/target id lists of
    [] — schema-valid, missing-check-blind, and an approval dialog proposing a
    migration of nothing. Emptiness now shares _is_filled with the optionals."""
    schema = {
        "type": "object",
        "properties": {
            "source_dashboard_ids": {"type": "array"},
            "target_dashboard_ids": {"type": "array"},
            "change_ownership": {"type": "boolean"},
        },
        "required": ["source_dashboard_ids", "target_dashboard_ids"],
    }
    assert m._missing_required_fields({}, schema) == ["source_dashboard_ids", "target_dashboard_ids"]
    assert m._missing_required_fields({"source_dashboard_ids": [], "target_dashboard_ids": []}, schema) == [
        "source_dashboard_ids",
        "target_dashboard_ids",
    ]
    assert m._missing_required_fields({"source_dashboard_ids": ["d1"], "target_dashboard_ids": []}, schema) == [
        "target_dashboard_ids"
    ]
    assert m._missing_required_fields({"source_dashboard_ids": ["d1"], "target_dashboard_ids": ["dA"]}, schema) == []


def test_validate_tool_args_rejects_present_but_empty_required():
    """JSON Schema `required` only demands the KEY exists, so explicit empties
    ({"ids": []}) pass jsonschema.validate — and every clarify path used to sit
    inside `except ValidationError`, unreachable for schema-valid emptiness.
    Seen live 2026-08-18: the same bare prompt clarified when the model omitted
    the keys but gated blank id lists when it emitted []. The one validation
    entry point must raise on both, so all callers fall into their existing
    clarification handling."""
    import jsonschema

    schema = {
        "type": "object",
        "properties": {
            "source_dashboard_ids": {"type": "array"},
            "target_dashboard_ids": {"type": "array"},
        },
        "required": ["source_dashboard_ids", "target_dashboard_ids"],
    }
    with pytest.raises(jsonschema.ValidationError, match="empty"):
        m._validate_tool_args(schema, {"source_dashboard_ids": [], "target_dashboard_ids": []})
    str_schema = {"type": "object", "properties": {"user_email": {"type": "string"}}, "required": ["user_email"]}
    with pytest.raises(jsonschema.ValidationError, match="empty"):
        m._validate_tool_args(str_schema, {"user_email": "   "})
    # filled args still pass; empty OPTIONAL values never block
    m._validate_tool_args(schema, {"source_dashboard_ids": ["d1"], "target_dashboard_ids": ["dA"]})
    opt = {"type": "object", "properties": {"action": {"type": "string"}}, "required": []}
    m._validate_tool_args(opt, {"action": ""})
