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


def test_flag_default_on_and_explicit_off(monkeypatch):
    # Local CSVs default ON (2026-08-17): they feed model comparison + thumbs
    # feedback and never carry Sisense result data. false turns them off.
    monkeypatch.delenv("FES_CSV_OBSERVABILITY", raising=False)
    assert _csv_observability_enabled() is True
    monkeypatch.setenv("FES_CSV_OBSERVABILITY", "false")
    assert _csv_observability_enabled() is False
    monkeypatch.setenv("FES_CSV_OBSERVABILITY", "true")
    assert _csv_observability_enabled() is True


def test_writers_noop_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.setenv("FES_CSV_OBSERVABILITY", "false")
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


# ---------------------------------------------------------------------------
# Model-comparison columns (2026-08-17): every LLM-call row must record what
# the model was asked (step_text) and what it chose (tool_selected/tool_args)
# so cross-model accuracy comparisons can attribute a bad pick to its
# sub-question — including picks that never executed.
# ---------------------------------------------------------------------------
import csv  # noqa: E402

from backend.agent._routing import _extract_tool_choice  # noqa: E402


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_llm_call_row_carries_model_comparison_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("FES_CSV_OBSERVABILITY", "true")
    monkeypatch.setattr(cfg, "LLM_CALLS_PATH", tmp_path / "llm_calls.csv")

    real_write_llm_call(
        call_type="plan",
        n_messages=4,
        n_tools=10,
        latency_ms=12,
        step_text="Get jane@acme.com's record.",
        tool_selected="access_management.get_user",
        tool_args='{"user_email": "jane@acme.com"}',
    )

    (row,) = _read_rows(tmp_path / "llm_calls.csv")
    assert row["step_text"] == "Get jane@acme.com's record."
    assert row["tool_selected"] == "access_management.get_user"
    assert row["tool_args"] == '{"user_email": "jane@acme.com"}'
    # model comparison also needs the model identity on the same row
    assert row["model"]
    assert row["provider"]


def test_schema_change_rotates_old_file_not_breaks(tmp_path, monkeypatch):
    monkeypatch.setenv("FES_CSV_OBSERVABILITY", "true")
    path = tmp_path / "llm_calls.csv"
    monkeypatch.setattr(cfg, "LLM_CALLS_PATH", path)
    # simulate a file written under the previous column set
    path.write_text("timestamp,trace_id,user_message,call_type\n2026,t,q,plan\n")

    real_write_llm_call(call_type="plan", n_messages=1, n_tools=0, latency_ms=1)

    rows = _read_rows(path)
    assert len(rows) == 1  # fresh file, new schema
    assert "tool_selected" in rows[0]
    assert path.with_suffix(".csv.old").exists()  # old data preserved, not lost


def test_extract_tool_choice_single_and_multi_and_none():
    single = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "a.b", "arguments": '{"x": 1}'}},
                    ]
                }
            }
        ]
    }
    names, args = _extract_tool_choice(single)
    assert names == "a.b"
    assert args == '{"x": 1}'

    multi = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "m.groups", "arguments": "{}"}},
                        {"function": {"name": "m.users", "arguments": "{}"}},
                    ]
                }
            }
        ]
    }
    names, args = _extract_tool_choice(multi)
    assert names == "m.groups;m.users"
    assert args == "[{}, {}]"

    assert _extract_tool_choice({"choices": [{"message": {"content": "hi"}}]}) == ("", "")
    assert _extract_tool_choice({}) == ("", "")


def test_turn_usage_accumulates_and_pops(monkeypatch):
    """Per-turn token/cost totals: accumulated across calls, popped once by
    the API layer (the UI shows them after every answer)."""
    from backend.agent._config import add_turn_usage, pop_turn_usage, reset_current_turn, set_current_turn

    token = set_current_turn("trace-usage-1", "q")
    try:
        add_turn_usage(100, 20, 0.001)
        add_turn_usage(200, 30, 0.002)
        u = pop_turn_usage("trace-usage-1")
        assert u == {"tokens_in": 300, "tokens_out": 50, "cost": 0.003}
        # popped — second read returns zeros
        assert pop_turn_usage("trace-usage-1") == {"tokens_in": 0, "tokens_out": 0, "cost": 0.0}
    finally:
        reset_current_turn(token)


def test_turn_usage_unknown_pricing_reports_none_not_zero(monkeypatch):
    """A model missing from litellm's pricing map must surface as cost=None —
    0.0 would read as 'free' or 'broken math'. One unpriced call poisons the
    whole turn's total (a partial sum would silently understate)."""
    from backend.agent._config import add_turn_usage, pop_turn_usage, reset_current_turn, set_current_turn

    token = set_current_turn("trace-usage-2", "q")
    try:
        add_turn_usage(100, 20, 0.001)
        add_turn_usage(200, 30, None)  # unpriced call
        u = pop_turn_usage("trace-usage-2")
        assert u["tokens_in"] == 300  # tokens stay exact
        assert u["cost"] is None
    finally:
        reset_current_turn(token)


# ---------------------------------------------------------------------------
# Rotation (2026-08-18): the .log files roll nightly and keep 7 days; these
# CSVs are hand-rolled appends, so without a bound of their own they grow for
# the life of the instance on the shared host log directory.
# ---------------------------------------------------------------------------
def test_csv_rolls_once_it_passes_the_size_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("FES_CSV_OBSERVABILITY", "true")
    path = tmp_path / "llm_calls.csv"
    monkeypatch.setattr(cfg, "LLM_CALLS_PATH", path)
    # Cap above one header+row (~240B) so several rows accumulate before the roll.
    monkeypatch.setattr(cfg, "_CSV_MAX_BYTES", 1200)

    real_write_llm_call(call_type="plan", n_messages=1, n_tools=0, latency_ms=1)
    assert path.exists()
    while path.stat().st_size <= 1200:  # push it past the cap
        real_write_llm_call(call_type="plan", n_messages=1, n_tools=0, latency_ms=1)

    real_write_llm_call(call_type="plan", n_messages=1, n_tools=0, latency_ms=1)

    rolled = list(tmp_path.glob("llm_calls.*.csv"))
    assert rolled, "the oversized file should have been rolled aside"
    assert path.exists(), "a fresh current file must exist after the roll"
    # The fresh file starts over: header + only the row written after the roll.
    assert len(_read_rows(path)) == 1
    assert len(_read_rows(rolled[0])) > 1, "the accumulated rows went to the rolled file"


def test_rotation_keeps_only_the_newest_backups(tmp_path, monkeypatch):
    path = tmp_path / "llm_calls.csv"
    monkeypatch.setattr(cfg, "_CSV_MAX_BYTES", 10)
    monkeypatch.setattr(cfg, "_CSV_BACKUPS", 2)
    # more rolled files than the retention allows, oldest first by name
    for stamp in ("20260101-000000", "20260102-000000", "20260103-000000"):
        (tmp_path / f"llm_calls.{stamp}.csv").write_text("x")
    path.write_text("y" * 50)  # over the cap → triggers a roll + prune

    cfg._rotate_csv_if_large(path)

    rolled = sorted(p.name for p in tmp_path.glob("llm_calls.*.csv"))
    assert len(rolled) == 2, f"should keep exactly _CSV_BACKUPS files, got {rolled}"
    assert "llm_calls.20260101-000000.csv" not in rolled, "the oldest roll should be pruned"


def test_rotation_leaves_the_schema_change_file_alone(tmp_path, monkeypatch):
    """`.csv.old` comes from a column change, a different mechanism — the size
    roller must not count it as one of its backups or delete it."""
    monkeypatch.setattr(cfg, "_CSV_MAX_BYTES", 10)
    monkeypatch.setattr(cfg, "_CSV_BACKUPS", 1)
    path = tmp_path / "llm_calls.csv"
    schema_old = tmp_path / "llm_calls.csv.old"
    schema_old.write_text("old schema rows")
    path.write_text("y" * 50)

    cfg._rotate_csv_if_large(path)

    assert schema_old.exists() and schema_old.read_text() == "old schema rows"


def test_rotation_never_raises(tmp_path, monkeypatch):
    """Observability must not break a turn — a missing dir or unwritable path
    is swallowed."""
    monkeypatch.setattr(cfg, "_CSV_MAX_BYTES", 1)
    cfg._rotate_csv_if_large(tmp_path / "nope" / "missing.csv")  # must not raise
