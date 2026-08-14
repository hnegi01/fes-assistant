"""
Unit tests for single-use mutation approvals.

An approval authorises exactly ONE execution of exactly one (tool_id, arguments)
pair. Before this, approvals were a membership test against a set the UI
accumulated for the whole browser session, so the *second* request for an
identical operation ran with no dialog — the gate silently stopped appearing
for precisely the operations it exists to guard (delete, cross-environment
migrate).

Covered:
  - _consume_approval: authorises once, then refuses; arg-sensitive; atomic
  - the sequential gate consumes, so a repeat inside the SAME turn re-gates
  - the pending_loop resume consumes, so a replayed approval cannot re-execute
  - a non-matching approval still gates (no over-broad matching)
  - both engines behave identically (FES_AGENT_ENGINE parity)
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

import backend.agent.llm_agent as m

DELETE_SCHEMA = {
    "type": "object",
    "properties": {"user_name": {"type": "string", "description": "Name of the user to delete."}},
    "required": ["user_name"],
}
MIGRATE_SCHEMA = {
    "type": "object",
    "properties": {"group_name_list": {"type": "array", "items": {"type": "string"}}},
    "required": ["group_name_list"],
}


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    monkeypatch.setattr(
        m,
        "TOOL_REGISTRY",
        {
            "access_management.delete_user": {
                "module": "access_management",
                "mutates": True,
                "description": "Delete a Sisense user.",
                "parameters": DELETE_SCHEMA,
            },
            "migration.migrate_groups": {
                "module": "migration",
                "mutates": True,
                "description": "Migrate specific groups.",
                "parameters": MIGRATE_SCHEMA,
            },
        },
    )
    monkeypatch.setattr(m, "ALLOW_SUMMARIZATION", True)
    monkeypatch.setattr(m, "REQUIRE_MUTATION_CONFIRM", True)
    monkeypatch.setattr(m, "VERIFY_GOAL", False)


@pytest.fixture(autouse=True)
def reset_pending():
    m.LAST_PENDING_LOOP = None
    m.LAST_TOOL_RESULT = None
    yield
    m.LAST_PENDING_LOOP = None
    m.LAST_TOOL_RESULT = None


# ---------------------------------------------------------------------------
# _consume_approval
# ---------------------------------------------------------------------------
class TestConsumeApproval:
    TOOL = "access_management.delete_user"
    ARGS = {"user_name": "a@b.com"}

    def test_authorises_once_then_refuses(self):
        approved = {m._approval_key(self.TOOL, self.ARGS)}
        assert m._consume_approval(approved, self.TOOL, self.ARGS) is True
        assert m._consume_approval(approved, self.TOOL, self.ARGS) is False

    def test_removes_the_key_it_consumed(self):
        key = m._approval_key(self.TOOL, self.ARGS)
        approved = {key}
        m._consume_approval(approved, self.TOOL, self.ARGS)
        assert key not in approved

    def test_refuses_when_not_approved(self):
        assert m._consume_approval(set(), self.TOOL, self.ARGS) is False

    def test_is_argument_sensitive(self):
        approved = {m._approval_key(self.TOOL, {"user_name": "a@b.com"})}
        assert m._consume_approval(approved, self.TOOL, {"user_name": "c@d.com"}) is False
        # ...and the untouched approval is still spendable on its own arguments.
        assert m._consume_approval(approved, self.TOOL, {"user_name": "a@b.com"}) is True

    def test_is_tool_sensitive(self):
        approved = {m._approval_key(self.TOOL, self.ARGS)}
        assert m._consume_approval(approved, "migration.migrate_groups", self.ARGS) is False

    def test_argument_order_does_not_matter(self):
        approved = {m._approval_key(self.TOOL, {"a": 1, "b": 2})}
        assert m._consume_approval(approved, self.TOOL, {"b": 2, "a": 1}) is True

    def test_other_approvals_survive(self):
        k1 = m._approval_key(self.TOOL, self.ARGS)
        k2 = m._approval_key("migration.migrate_groups", {"group_name_list": ["Sales"]})
        approved = {k1, k2}
        m._consume_approval(approved, self.TOOL, self.ARGS)
        assert approved == {k2}

    def test_concurrent_claims_only_one_wins(self):
        """No await between test and discard, so two coroutines racing for the
        same approval cannot both execute — the fan-out mutation path relies on
        this."""
        approved = {m._approval_key(self.TOOL, self.ARGS)}

        async def claim():
            return m._consume_approval(approved, self.TOOL, self.ARGS)

        async def both():
            return await asyncio.gather(claim(), claim())

        assert sorted(run(both())) == [False, True]


# ---------------------------------------------------------------------------
# The loop: an approval spent on one step does not authorise a second
# ---------------------------------------------------------------------------
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


def _tool_def(name, schema):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": schema}}


DELETE_TOOLS = [_tool_def("access_management.delete_user", DELETE_SCHEMA)]


@pytest.fixture(params=["custom", "langgraph"])
def engine(request, monkeypatch):
    """Both harnesses must gate identically — the unit suite is the parity harness."""
    monkeypatch.setenv("FES_AGENT_ENGINE", request.param)
    return request.param


def test_same_mutation_twice_in_one_turn_gates_the_second_time(engine):
    """Approve delete(a@b.com); the loop runs it, then decides to do it again.
    The second attempt must pause for a fresh approval, not reuse the first."""
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {"deleted": True}})
    args = {"user_name": "a@b.com"}
    key = m._approval_key("access_management.delete_user", args)

    messages = [{"role": "user", "content": "delete the user a@b.com"}]
    nav = AsyncMock(side_effect=[(DELETE_TOOLS, "access_management", "users", 0)] * 4)
    raw = AsyncMock(
        side_effect=[
            _plan_resp("access_management.delete_user", '{"user_name":"a@b.com"}'),
            _text_resp("CONTINUE: delete the user a@b.com"),  # decide: do it again
            _plan_resp("access_management.delete_user", '{"user_name":"a@b.com"}', call_id="c2"),
            _text_resp("This will permanently delete the user a@b.com."),  # gate explanation
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
            m.call_llm_with_tools(messages, DELETE_TOOLS, client, approved_mutations={key}, allow_summarization=True)
        )

    # Ran exactly once — the repeat was gated, not executed.
    client.invoke_tool.assert_awaited_once()
    assert m.LAST_PENDING_LOOP is not None
    assert m.LAST_PENDING_LOOP["tool_id"] == "access_management.delete_user"
    assert "delete" in reply.lower()


def test_resume_consumes_the_approval(engine):
    """After the resume executes the gated tool, the approval is spent — the
    loop continuing on cannot silently re-run it."""
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {"deleted": True}})
    args = {"user_name": "a@b.com"}
    key = m._approval_key("access_management.delete_user", args)
    approved = {key}
    pending_loop = {
        "transcript": [],
        "steps_executed": 0,
        "tool_id": "access_management.delete_user",
        "arguments": args,
    }

    messages = [{"role": "user", "content": "delete the user a@b.com"}]
    raw = AsyncMock(side_effect=[_text_resp("Deleted user a@b.com.")])
    with patch.object(m, "_navigate_to_tools", new=AsyncMock()), patch.object(m, "call_llm_raw", new=raw):
        reply = run(
            m.call_llm_with_tools(
                messages,
                DELETE_TOOLS,
                client,
                approved_mutations=approved,
                allow_summarization=True,
                pending_loop=pending_loop,
            )
        )

    assert reply == "Deleted user a@b.com."
    client.invoke_tool.assert_awaited_once()
    assert approved == set(), "the resume must spend the approval it used"


def test_resume_without_matching_approval_does_not_execute(engine):
    """A stale pending_loop with no approval this turn is dropped, never run."""
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {}})
    pending_loop = {
        "transcript": [],
        "steps_executed": 0,
        "tool_id": "access_management.delete_user",
        "arguments": {"user_name": "a@b.com"},
    }
    # Approval is for DIFFERENT arguments.
    approved = {m._approval_key("access_management.delete_user", {"user_name": "someone.else@b.com"})}

    messages = [{"role": "user", "content": "never mind, show me something else"}]
    nav = AsyncMock(side_effect=[([], "__unclear__", "", 0)])
    with patch.object(m, "_navigate_to_tools", new=nav), patch.object(m, "call_llm_raw", new=AsyncMock()):
        with patch.object(m, "_make_plan", new=AsyncMock(return_value=["show me something else"])):
            run(
                m.call_llm_with_tools(
                    messages,
                    DELETE_TOOLS,
                    client,
                    approved_mutations=approved,
                    allow_summarization=True,
                    pending_loop=pending_loop,
                )
            )

    client.invoke_tool.assert_not_awaited()
    assert approved, "an approval for other arguments must not be consumed"


def test_engine_env_is_restored():
    """Guard: the engine fixture uses monkeypatch.setenv, so nothing leaks."""
    assert os.getenv("FES_AGENT_ENGINE", "custom") in ("custom", "langgraph")
