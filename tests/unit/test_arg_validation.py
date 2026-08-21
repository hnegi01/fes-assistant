"""
Unit tests for blocking argument validation.

Covers two regressions guarded together:

  1. Registry: `user_email` parameters carry `format: "email"` so a non-email
     string ("john") is a schema violation, not a valid value.
  2. Call site: call_llm_with_tools() validates planner args with
     jsonschema + a FormatChecker BEFORE executing the tool, and blocks
     (returns a user-facing message, never calls the MCP client) on a
     mismatch. Removing the FormatChecker would silently re-allow "john".

These run in CI with no LLM and no Sisense credentials: the planner LLM call
is mocked to return a chosen tool + args, and the MCP client is a stub whose
invoke_tool we assert is (not) called.

Patching note: llm_agent.py does `from ._routing import _navigate_to_tools,
call_llm_raw`, so those names resolve in the llm_agent module namespace —
patch them on backend.agent.llm_agent, not on _routing.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import backend.agent.llm_agent as m

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "tools.registry.with_examples.json"


def run(coro):
    """Run a coroutine synchronously (avoids pytest-asyncio dependency)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1) Registry guard — the email fields actually declare format: email
# ---------------------------------------------------------------------------


class TestRegistryEmailFormat:
    """If someone drops `format: email` from the registry, blocking can't fire."""

    def _load_registry(self):
        with REGISTRY_PATH.open() as fh:
            return {t["tool_id"]: t for t in json.load(fh)}

    @pytest.mark.parametrize(
        "tool_id",
        ["access_management.get_user", "access_management.update_user"],
    )
    def test_user_email_declares_email_format(self, tool_id):
        registry = self._load_registry()
        props = registry[tool_id]["parameters"]["properties"]
        assert props["user_email"]["format"] == "email", (
            f"{tool_id}.user_email must declare format: email so a non-email value is blocked before execution"
        )


# ---------------------------------------------------------------------------
# Fixtures for the wiring tests
# ---------------------------------------------------------------------------

GET_USER_SCHEMA = {
    "type": "object",
    "properties": {
        "user_email": {
            "type": "string",
            "format": "email",
            "description": "Email address of the user to retrieve.",
        }
    },
    "required": ["user_email"],
}


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    monkeypatch.setattr(
        m,
        "TOOL_REGISTRY",
        {
            "access_management.get_user": {
                "module": "access_management",
                "mutates": False,
                "parameters": GET_USER_SCHEMA,
            }
        },
    )


def _tool_def(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _planning_response(tool_id, arguments_json):
    """Shape returned by call_llm_raw → parsed by _pick_tool_calls_from_llm_response."""
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "call_1", "function": {"name": tool_id, "arguments": arguments_json}}],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _fake_mcp_client(result=None):
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value=result or {"ok": True, "data": [{"id": 1}]})
    return client


def _call(tool_id, arguments_json, mcp_client, user_message="test prompt"):
    """Drive a single chat turn with routing mocked and the planner forced.

    `user_message` matters: a required value the user never typed and that
    looks invented is treated as missing (see _is_fabricated), so tests that
    expect execution must put the value in the user's own words — exactly as a
    real conversation would.
    """
    tools = [_tool_def("access_management.get_user")]
    messages = [{"role": "user", "content": user_message}]
    with (
        patch.object(
            m,
            "_navigate_to_tools",
            new=AsyncMock(return_value=(tools, "access_management", "core", 0)),
        ),
        patch.object(
            m,
            "call_llm_raw",
            new=AsyncMock(return_value=_planning_response(tool_id, arguments_json)),
        ),
    ):
        # allow_summarization=False → no second LLM call; returns local description.
        return run(m.call_llm_with_tools(messages, tools, mcp_client, allow_summarization=False))


# ---------------------------------------------------------------------------
# 2) Call-site wiring — block on bad value, block on missing, allow on valid
# ---------------------------------------------------------------------------


class TestBlockingValidationWiring:
    def test_non_email_value_is_blocked_and_not_executed(self):
        """'john' fails format: email → user-facing block, MCP never called."""
        client = _fake_mcp_client()
        reply = _call("access_management.get_user", json.dumps({"user_email": "john"}), client)

        assert "couldn't call" in reply.lower()
        assert "access_management.get_user" in reply
        client.invoke_tool.assert_not_called()

    def test_missing_required_arg_triggers_clarification_not_execution(self):
        """No user_email → Step 7 clarification: ask, don't execute, don't hard-block.

        (A missing required arg used to be a dead-end block; it now pauses for
        clarification. The not-executed guarantee is unchanged.)
        """
        client = _fake_mcp_client()
        reply = _call("access_management.get_user", json.dumps({}), client)

        # Tool must NOT run while a required arg is missing.
        client.invoke_tool.assert_not_called()
        # Pending clarification recorded for the next turn.
        assert m.LAST_PENDING_CLARIFICATION is not None
        assert m.LAST_PENDING_CLARIFICATION["tool_id"] == "access_management.get_user"
        assert "user_email" in m.LAST_PENDING_CLARIFICATION["missing_fields"]
        # This is a question, not the wrong-value hard block.
        assert "couldn't call" not in reply.lower()

    def test_valid_email_is_executed(self):
        """A real email the user typed passes validation → tool runs (no over-blocking)."""
        client = _fake_mcp_client()
        _call(
            "access_management.get_user",
            json.dumps({"user_email": "jane@acme.com"}),
            client,
            user_message="look up jane@acme.com",
        )

        client.invoke_tool.assert_awaited_once()
        called_tool, called_args = client.invoke_tool.await_args.args
        assert called_tool == "access_management.get_user"
        assert called_args["user_email"] == "jane@acme.com"

    def test_invented_placeholder_email_clarifies_instead_of_executing(self):
        """The third kind of "not given": a value the model MADE UP.

        "look up a single user by their email address" names no address, and
        the tool-selection call answers with user_email="user@example.com" —
        non-empty and schema-valid (format: email), so every emptiness check
        passes it through. Live 2026-08-18 it executed and reported "User
        'user@example.com' not found" for a user nobody asked about. The
        request must clarify instead.
        """
        client = _fake_mcp_client()
        reply = _call(
            "access_management.get_user",
            json.dumps({"user_email": "user@example.com"}),
            client,
            user_message="look up a single user by their email address",
        )

        client.invoke_tool.assert_not_called()
        assert m.LAST_PENDING_CLARIFICATION is not None
        assert "user_email" in m.LAST_PENDING_CLARIFICATION["missing_fields"]
        # and the invented value is not echoed back as something already supplied
        assert "example.com" not in json.dumps(m.LAST_PENDING_CLARIFICATION.get("filled_args") or {})
        assert "example.com" not in reply

    def test_a_reserved_domain_the_user_typed_is_honoured(self):
        """The rule is "did the user say it", not "does it look fake". The
        integration suite deletes nobody@fes-test.invalid — a reserved TLD,
        typed deliberately. Pattern-matching alone would block it."""
        client = _fake_mcp_client()
        _call(
            "access_management.get_user",
            json.dumps({"user_email": "nobody@fes-test.invalid"}),
            client,
            user_message="look up the user nobody@fes-test.invalid",
        )

        client.invoke_tool.assert_awaited_once()
        _t, called_args = client.invoke_tool.await_args.args
        assert called_args["user_email"] == "nobody@fes-test.invalid"
