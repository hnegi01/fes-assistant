"""
Unit tests for two-stage routing helpers.

Covers:
  - _get_module_tools: correct grouping by registry module
  - _parse_module_from_response: exact match, substring match, unknown, empty
  - _route_to_module: mocked call_llm_raw for success, fallback, unknown response

After the llm_config split, _route_to_module and call_llm_raw live in
backend.agent.llm_routing. Patching call_llm_raw must target that module
(where _route_to_module's call resolves), not the re-export on llm_agent.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import backend.agent.llm_agent as m
import backend.agent._routing as routing_m


def run(coro):
    """Run a coroutine synchronously (avoids pytest-asyncio dependency)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_tool_registry(monkeypatch):
    """Seed TOOL_REGISTRY with a small set of tools across three modules."""
    registry = {
        "access_management.get_users_all": {"module": "access_management", "mutates": False},
        "access_management.get_groups_all": {"module": "access_management", "mutates": False},
        "dashboard.get_dashboards_all": {"module": "dashboard", "mutates": False},
        "datamodel.get_all_datamodel": {"module": "datamodel", "mutates": False},
        "datamodel.get_datasecurity": {"module": "datamodel", "mutates": False},
    }
    monkeypatch.setattr(m, "TOOL_REGISTRY", registry)


def _make_tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


ALL_TOOLS = [
    _make_tool("access_management.get_users_all"),
    _make_tool("access_management.get_groups_all"),
    _make_tool("dashboard.get_dashboards_all"),
    _make_tool("datamodel.get_all_datamodel"),
    _make_tool("datamodel.get_datasecurity"),
]

MODULES = {
    "access_management": "user and group management",
    "dashboard": "dashboard operations",
    "datamodel": "data models and data sources",
}


# ---------------------------------------------------------------------------
# _get_module_tools
# ---------------------------------------------------------------------------

class TestGetModuleTools:
    def test_groups_by_module(self):
        result = m._get_module_tools(ALL_TOOLS)

        assert set(result.keys()) == {"access_management", "dashboard", "datamodel"}
        assert len(result["access_management"]) == 2
        assert len(result["dashboard"]) == 1
        assert len(result["datamodel"]) == 2

    def test_unknown_module_falls_back_to_unknown_key(self, monkeypatch):
        monkeypatch.setattr(m, "TOOL_REGISTRY", {})  # no registry entries
        tools = [_make_tool("mystery.tool")]

        result = m._get_module_tools(tools)

        assert "unknown" in result

    def test_empty_tools_returns_empty(self):
        assert m._get_module_tools([]) == {}


# ---------------------------------------------------------------------------
# _parse_module_from_response
# ---------------------------------------------------------------------------

class TestParseModuleFromResponse:
    def test_exact_match(self):
        assert m._parse_module_from_response("datamodel", MODULES) == "datamodel"

    def test_case_insensitive_exact(self):
        assert m._parse_module_from_response("  DataModel  ", MODULES) == "datamodel"

    def test_substring_match(self):
        # LLM returned a sentence instead of a single word
        assert m._parse_module_from_response(
            "I would choose the datamodel module.", MODULES
        ) == "datamodel"

    def test_returns_none_for_unknown(self):
        assert m._parse_module_from_response("migration", MODULES) is None

    def test_returns_none_for_empty(self):
        assert m._parse_module_from_response("", MODULES) is None

    def test_returns_none_for_none_content(self):
        assert m._parse_module_from_response(None, MODULES) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _route_to_module (mocked call_llm_raw on llm_routing — where it is called)
# ---------------------------------------------------------------------------

def _llm_response(content: str) -> dict:
    """Build a minimal OpenAI-style response dict."""
    return {
        "choices": [{"message": {"content": content, "tool_calls": None}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 5},
    }


class TestRouteToModule:
    def test_returns_correct_module_on_success(self):
        with patch.object(routing_m, "call_llm_raw", new=AsyncMock(return_value=_llm_response("datamodel"))):
            chosen, latency = run(m._route_to_module(
                {"role": "user", "content": "show all data models"},
                [],
                MODULES,
                trace_id=None,
            ))
        assert chosen == "datamodel"
        assert latency >= 0

    def test_returns_none_on_llm_failure(self):
        with patch.object(routing_m, "call_llm_raw", new=AsyncMock(side_effect=RuntimeError("timeout"))):
            chosen, latency = run(m._route_to_module(
                {"role": "user", "content": "show all data models"},
                [],
                MODULES,
                trace_id=None,
            ))
        assert chosen is None
        assert latency >= 0

    def test_returns_none_for_unrecognised_response(self):
        with patch.object(routing_m, "call_llm_raw", new=AsyncMock(return_value=_llm_response("I don't know"))):
            chosen, _ = run(m._route_to_module(
                {"role": "user", "content": "do something"},
                [],
                MODULES,
                trace_id=None,
            ))
        assert chosen is None

    def test_substring_response_still_resolves(self):
        with patch.object(
            routing_m, "call_llm_raw",
            new=AsyncMock(return_value=_llm_response("The access_management module handles this.")),
        ):
            chosen, _ = run(m._route_to_module(
                {"role": "user", "content": "list all users"},
                [],
                MODULES,
                trace_id=None,
            ))
        assert chosen == "access_management"
