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

os.environ.setdefault("LLM_PROVIDER", "databricks")
os.environ.setdefault("DATABRICKS_HOST", "https://fake.databricks.com")
os.environ.setdefault("DATABRICKS_TOKEN", "fake-ci-token")
os.environ.setdefault("LLM_ENDPOINT", "fake-endpoint")
os.environ.setdefault("PYSISENSE_MCP_HTTP_URL", "http://localhost:8002")
