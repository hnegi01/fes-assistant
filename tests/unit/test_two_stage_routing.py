"""
Unit tests for two-stage routing helpers.

Covers:
  - _get_module_tools: correct grouping by registry module
  - _parse_module_from_response: exact match, substring match, unknown, empty
  - _route_to_module: mocked call_llm_raw for success, fallback, unknown response

After the module split, _parse_module_from_response, _route_to_module, and
call_llm_raw live in backend.agent._routing (imported here as routing_m).
Call and patch them there — _get_module_tools stays on llm_agent (m).
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import backend.agent._routing as routing_m
import backend.agent.llm_agent as m


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
        assert routing_m._parse_module_from_response("datamodel", MODULES) == "datamodel"

    def test_case_insensitive_exact(self):
        assert routing_m._parse_module_from_response("  DataModel  ", MODULES) == "datamodel"

    def test_substring_match(self):
        # LLM returned a sentence instead of a single word
        assert routing_m._parse_module_from_response("I would choose the datamodel module.", MODULES) == "datamodel"

    def test_returns_none_for_unknown(self):
        assert routing_m._parse_module_from_response("migration", MODULES) is None

    def test_returns_none_for_empty(self):
        assert routing_m._parse_module_from_response("", MODULES) is None

    def test_returns_none_for_none_content(self):
        assert routing_m._parse_module_from_response(None, MODULES) is None  # type: ignore[arg-type]


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
            chosen, latency = run(
                routing_m._route_to_module(
                    {"role": "user", "content": "show all data models"},
                    [],
                    MODULES,
                    trace_id=None,
                )
            )
        assert chosen == "datamodel"
        assert latency >= 0

    def test_returns_none_on_llm_failure(self):
        with patch.object(routing_m, "call_llm_raw", new=AsyncMock(side_effect=RuntimeError("timeout"))):
            chosen, latency = run(
                routing_m._route_to_module(
                    {"role": "user", "content": "show all data models"},
                    [],
                    MODULES,
                    trace_id=None,
                )
            )
        assert chosen is None
        assert latency >= 0

    def test_returns_none_for_unrecognised_response(self):
        with patch.object(routing_m, "call_llm_raw", new=AsyncMock(return_value=_llm_response("I don't know"))):
            chosen, _ = run(
                routing_m._route_to_module(
                    {"role": "user", "content": "do something"},
                    [],
                    MODULES,
                    trace_id=None,
                )
            )
        assert chosen is None

    def test_substring_response_still_resolves(self):
        with patch.object(
            routing_m,
            "call_llm_raw",
            new=AsyncMock(return_value=_llm_response("The access_management module handles this.")),
        ):
            chosen, _ = run(
                routing_m._route_to_module(
                    {"role": "user", "content": "list all users"},
                    [],
                    MODULES,
                    trace_id=None,
                )
            )
        assert chosen == "access_management"


class TestProviderToolCap:
    """OpenAI rejects a tools array longer than 128 with HTTP 400, which turns
    the whole turn into a keyword-fallback answer. Exposing 146 tools put chat
    mode at 137, so every path that hands over the unrouted full list failed
    hard (live 2026-08-29). Routing normally keeps calls near 10 tools; this
    cap only bites on the fallback paths."""

    def _call(self, monkeypatch, n_tools):
        import asyncio

        import backend.agent._routing as r

        seen = {}

        async def fake_acompletion(**kwargs):
            seen["n"] = len(kwargs.get("tools") or [])

            class _M:
                def model_dump(self):
                    return {"choices": [{"message": {"content": "x", "tool_calls": []}}]}

            return _M()

        monkeypatch.setattr(r.litellm, "acompletion", fake_acompletion)
        tools = [{"type": "function", "function": {"name": f"t{i}", "parameters": {}}} for i in range(n_tools)]
        asyncio.get_event_loop().run_until_complete(r.call_llm_raw([{"role": "user", "content": "hi"}], tools=tools))
        return seen["n"]

    def test_oversized_list_is_capped(self, monkeypatch):
        assert self._call(monkeypatch, 137) == r_cap()

    def test_normal_routed_list_is_untouched(self, monkeypatch):
        assert self._call(monkeypatch, 10) == 10

    def test_exactly_at_the_cap_is_untouched(self, monkeypatch):
        assert self._call(monkeypatch, r_cap()) == r_cap()


def r_cap():
    from backend.agent._routing import MAX_TOOLS_PER_CALL

    return MAX_TOOLS_PER_CALL
