"""FES_CSV_OBSERVABILITY gates the local CSV writers (default: off).

The real writer functions are captured at import time — the autouse
_no_trace_writes fixture replaces the module attributes afterwards, so these
references still point at the genuine implementations.
"""

import backend.agent._config as cfg
from backend.agent._config import (
    _csv_observability_enabled,
)
from backend.agent._config import (
    _write_llm_trace as real_write_llm_trace,
)
from backend.agent._config import (
    write_llm_call as real_write_llm_call,
)
from backend.agent._config import (
    write_tool_call as real_write_tool_call,
)


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("FES_CSV_OBSERVABILITY", raising=False)
    assert _csv_observability_enabled() is False
    monkeypatch.setenv("FES_CSV_OBSERVABILITY", "true")
    assert _csv_observability_enabled() is True


def test_writers_noop_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("FES_CSV_OBSERVABILITY", raising=False)
    monkeypatch.setattr(cfg, "LLM_TRACES_PATH", tmp_path / "llm_traces.csv")
    monkeypatch.setattr(cfg, "LLM_CALLS_PATH", tmp_path / "llm_calls.csv")
    monkeypatch.setattr(cfg, "TOOL_CALLS_PATH", tmp_path / "tool_calls.csv")

    real_write_llm_trace({"trace_id": "t1"})
    real_write_llm_call(call_type="plan", n_messages=1, n_tools=0, latency_ms=1)
    real_write_tool_call(tool_id="x.y", ok=True, count=1, latency_ms=1)

    assert not (tmp_path / "llm_traces.csv").exists()
    assert not (tmp_path / "llm_calls.csv").exists()
    assert not (tmp_path / "tool_calls.csv").exists()


def test_writers_write_when_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("FES_CSV_OBSERVABILITY", "true")
    monkeypatch.setattr(cfg, "LLM_TRACES_PATH", tmp_path / "llm_traces.csv")
    monkeypatch.setattr(cfg, "LLM_CALLS_PATH", tmp_path / "llm_calls.csv")
    monkeypatch.setattr(cfg, "TOOL_CALLS_PATH", tmp_path / "tool_calls.csv")

    real_write_llm_trace({"trace_id": "t1"})
    real_write_llm_call(call_type="plan", n_messages=1, n_tools=0, latency_ms=1)
    real_write_tool_call(tool_id="x.y", ok=True, count=1, latency_ms=1)

    assert (tmp_path / "llm_traces.csv").exists()
    assert (tmp_path / "llm_calls.csv").exists()
    assert (tmp_path / "tool_calls.csv").exists()
