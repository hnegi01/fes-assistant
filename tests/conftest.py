"""
conftest.py — pytest session setup.

WHY THIS EXISTS:
  llm_agent.py executes LLM_CONFIG = _build_llm_config() at module import time.
  That function calls _require_env() for DATABRICKS_HOST / DATABRICKS_TOKEN /
  LLM_ENDPOINT (or the Azure equivalents). Without these env vars, any test that
  imports from backend.agent.llm_agent (or backend.api_server, which imports it)
  raises RuntimeError before a single test runs.

  This file sets fake/dummy values BEFORE any test module is collected so the
  import succeeds. No real LLM calls are made in these tests.
"""

import os

import pytest

os.environ.setdefault("LLM_PROVIDER", "databricks")
os.environ.setdefault("DATABRICKS_HOST", "https://fake.databricks.com")
os.environ.setdefault("DATABRICKS_TOKEN", "fake-ci-token")
os.environ.setdefault("LLM_ENDPOINT", "fake-endpoint")
os.environ.setdefault("PYSISENSE_MCP_HTTP_URL", "http://localhost:8002")


@pytest.fixture(autouse=True)
def _no_trace_writes(monkeypatch):
    """Never let tests append to the real logs/llm_traces.csv or llm_calls.csv.
    The trace writers run inside the real call_llm_with_tools that unit tests
    exercise; without this, every test run pollutes the production logs with
    fake-endpoint rows."""
    import backend.agent._config as cfg
    import backend.agent.llm_agent as agent

    monkeypatch.setattr(cfg, "_write_llm_trace", lambda *a, **k: None)
    monkeypatch.setattr(cfg, "write_llm_call", lambda *a, **k: None)
    monkeypatch.setattr(cfg, "write_tool_call", lambda *a, **k: None)
    # llm_agent imported _write_llm_trace by name, so patch that binding too.
    monkeypatch.setattr(agent, "_write_llm_trace", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(agent, "write_tool_call", lambda *a, **k: None, raising=False)
