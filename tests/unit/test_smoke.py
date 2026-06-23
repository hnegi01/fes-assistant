"""
Smoke tests for pure utility functions in llm_agent.py.

These tests require no running services — no LLM, no MCP server, no Sisense.
They cover the functions that have no side effects and are safe to call anywhere.

What to add next (as the codebase grows):
  - Registry loading tests (mock the JSON file path)
  - Mutation approval key round-trip tests
  - MCP client credential-injection tests (mock httpx)
"""

from backend.agent.llm_agent import _approval_key, _safe_json_loads, _shrink_for_llm


class TestShrinkForLlm:
    def test_long_string_is_truncated(self):
        result = _shrink_for_llm("a" * 500, max_string_length=100)
        assert isinstance(result, str)
        assert "truncated" in result

    def test_large_list_is_capped(self):
        result = _shrink_for_llm(list(range(100)), max_list_items=10)
        assert isinstance(result, list)
        assert any("omitted" in str(item) for item in result)

    def test_small_dict_passes_through_unchanged(self):
        data = {"name": "test", "count": 42}
        result = _shrink_for_llm(data)
        assert result["name"] == "test"
        assert result["count"] == 42

    def test_none_passes_through(self):
        assert _shrink_for_llm(None) is None

    def test_booleans_pass_through(self):
        assert _shrink_for_llm(True) is True
        assert _shrink_for_llm(False) is False

    def test_empty_list_passes_through(self):
        assert _shrink_for_llm([]) == []

    def test_empty_dict_passes_through(self):
        assert _shrink_for_llm({}) == {}


class TestSafeJsonLoads:
    def test_valid_json_object(self):
        assert _safe_json_loads('{"key": "val"}', default={}) == {"key": "val"}

    def test_valid_json_array(self):
        assert _safe_json_loads("[1, 2, 3]", default=[]) == [1, 2, 3]

    def test_invalid_json_returns_default(self):
        assert _safe_json_loads("not valid {{", default={"fallback": True}) == {"fallback": True}

    def test_empty_string_returns_default(self):
        assert _safe_json_loads("", default=None) is None

    def test_none_input_returns_default(self):
        assert _safe_json_loads(None, default="fallback") == "fallback"


class TestApprovalKey:
    def test_key_is_stable_regardless_of_arg_insertion_order(self):
        k1 = _approval_key("tool.foo", {"b": 2, "a": 1})
        k2 = _approval_key("tool.foo", {"a": 1, "b": 2})
        assert k1 == k2

    def test_different_tool_ids_produce_different_keys(self):
        k1 = _approval_key("tool.foo", {"x": 1})
        k2 = _approval_key("tool.bar", {"x": 1})
        assert k1 != k2

    def test_different_args_produce_different_keys(self):
        k1 = _approval_key("tool.foo", {"x": 1})
        k2 = _approval_key("tool.foo", {"x": 99})
        assert k1 != k2

    def test_returns_tuple_of_two_strings(self):
        key = _approval_key("tool.foo", {"a": 1})
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert all(isinstance(part, str) for part in key)
