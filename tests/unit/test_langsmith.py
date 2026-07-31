"""
Unit tests for LangSmith observability wiring.

Contract under test:
  - LiteLLM's bundled "langsmith" callback is NEVER registered — it drops
    custom metadata and posts isolated root runs. Reporting goes through
    backend/agent/_tracing.py (RunTree: root agent_turn + llm/tool children).
  - _tracing enforces the data boundaries: content gated by
    FES_LANGSMITH_LOG_CONTENT; tool runs metadata-only; no-ops when disabled.
"""

import litellm
import pytest

import backend.agent._config as config_m
import backend.agent._tracing as tracing_m
import backend.agent.llm_agent as m


class TestConfigureLangsmithTracing:
    def test_litellm_callback_never_registered_even_when_tracing_enabled(self, monkeypatch):
        # The lossy bundled logger stays retired — tracing goes via _tracing.py.
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        config_m._configure_langsmith_tracing()

        assert "langsmith" not in litellm.success_callback
        assert "langsmith" not in litellm.failure_callback

    def test_callbacks_cleared_when_tracing_disabled(self, monkeypatch):
        # Pre-set to simulate a stale previous registration
        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]

        monkeypatch.setenv("LANGSMITH_TRACING", "false")

        config_m._configure_langsmith_tracing()

        assert "langsmith" not in litellm.success_callback
        assert "langsmith" not in litellm.failure_callback

    def test_callbacks_cleared_when_tracing_unset(self, monkeypatch):
        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

        config_m._configure_langsmith_tracing()

        assert "langsmith" not in litellm.success_callback
        assert "langsmith" not in litellm.failure_callback

    @pytest.fixture(autouse=True)
    def reset_litellm_callbacks(self):
        """Restore litellm callback lists after each test."""
        orig_success = list(litellm.success_callback)
        orig_failure = list(litellm.failure_callback)
        yield
        litellm.success_callback = orig_success
        litellm.failure_callback = orig_failure


class TestTracingHelpers:
    def test_disabled_by_default_and_children_noop_without_root(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        assert tracing_m._enabled() is False
        # With no root bound, child logging must be a silent no-op.
        tracing_m.log_llm_child("decide", [{"role": "user", "content": "x"}], None, 12)
        tracing_m.log_tool_child("t.id", {"a": 1}, ok=True, count=3, duration_ms=5)
        tracing_m.end_turn_trace(reply="r")

    def test_start_is_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        tracing_m.start_turn_trace("hello", "sess-1", "chat")
        assert tracing_m._CURRENT_ROOT.get() is None

    def test_content_flag_default_closed(self, monkeypatch):
        monkeypatch.delenv("FES_LANGSMITH_LOG_CONTENT", raising=False)
        assert tracing_m._log_content() is False
        monkeypatch.setenv("FES_LANGSMITH_LOG_CONTENT", "true")
        assert tracing_m._log_content() is True

    def test_sanitizer_redacts_only_data_bearing_messages(self):
        messages = [
            {"role": "system", "content": "You are the orchestrator..."},
            {"role": "user", "content": "which group does a@b.com belong to"},
            {"role": "assistant", "content": "PLAN:\n1. Get the user record"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "name": "get_user", "content": '{"GROUPS": ["SecretGroup"]}'},
            {"role": "assistant", "content": "a@b.com belongs to SecretGroup"},  # prior reply
        ]
        out = tracing_m._sanitized_messages(messages)
        text = str(out)
        # data-bearing parts are gone...
        assert "SecretGroup" not in text
        assert "hidden" in out[4]["content"] and out[4]["name"] == "get_user"
        assert "hidden" in out[5]["content"]
        # ...everything else survives verbatim
        assert out[0]["content"].startswith("You are the orchestrator")
        assert out[1]["content"] == "which group does a@b.com belong to"
        assert out[2]["content"].startswith("PLAN:")
        assert out[3].get("tool_calls")


class TestCallLlmWithTools:
    def test_call_llm_with_tools_is_plain_coroutine(self):
        import inspect

        # Must be a plain async def with no @traceable wrapper.
        # @traceable would expose tenant_config/migration_config tokens to LangSmith.
        fn = m.call_llm_with_tools
        assert inspect.iscoroutinefunction(fn), "call_llm_with_tools must be async"
        assert not hasattr(fn, "__wrapped__"), (
            "call_llm_with_tools must NOT be wrapped by @traceable — "
            "the decorator captures all inputs including auth tokens"
        )
