"""Concurrent sessions must never see each other's turn outputs.

The LLM layer's LAST_* module globals are last-writer-wins across concurrently
running turns (each turn rebinds them at start and writes them after every
step), so they are debug/test aids ONLY. The authoritative copy is the
per-turn output store in _config, keyed by trace_id and resolved through the
same ContextVar the usage accumulator uses: fan-out child tasks inherit a copy
carrying the same trace_id, so their writes land in the right turn, and two
interleaved sessions can never cross-write.

These tests simulate the exact interleaving that leaked before the fix: turn B
starts (rebinding the globals) while turn A is mid-flight, both keep writing,
and each turn's popped output must contain only its own data.
"""

import asyncio

import backend.agent.llm_agent as m
from backend.agent._config import (
    begin_turn_output,
    current_turn_trace_id,
    pop_turn_output,
    set_current_turn,
    turn_output,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _fake_turn(tag: str, barrier: dict) -> dict:
    """A minimal turn: start (rebind globals + open store), write results with
    awaits in between (yield points where another session's turn interleaves),
    then collect the way runtime._run_turn_once does."""
    trace_id = f"trace-{tag}"
    set_current_turn(trace_id, f"question from {tag}")
    begin_turn_output(trace_id)
    # what call_llm_with_tools does at turn start (globals as debug aids)
    m.LAST_TOOL_RESULT = None
    m.LAST_STEP_RESULTS = []

    m._record_tool_result({"ok": True, "owner": tag, "step": 1})
    m._record_step(1, f"tool.{tag}.one", {"ok": True, "owner": tag})
    await asyncio.sleep(0)  # yield — the other turn runs (and rebinds globals)

    m._record_tool_result({"ok": True, "owner": tag, "step": 2})
    m._record_step(2, f"tool.{tag}.two", {"ok": True, "owner": tag})
    m._record_pending_clarification({"tool_id": f"tool.{tag}.two", "owner": tag})
    await asyncio.sleep(0)  # yield again before collecting

    # what runtime._run_turn_once does at turn end
    tid = current_turn_trace_id()
    out = pop_turn_output(tid)
    return {"trace_id": tid, **out}


def test_two_interleaved_turns_never_cross_write():
    async def _scenario():
        # gather() wraps each coroutine in its own task → own context copy,
        # exactly like runtime.run_turn_once's create_task per session.
        return await asyncio.gather(_fake_turn("alpha", {}), _fake_turn("beta", {}))

    turn_a, turn_b = run(_scenario())

    assert turn_a["trace_id"] == "trace-alpha"
    assert turn_b["trace_id"] == "trace-beta"
    for turn, tag in ((turn_a, "alpha"), (turn_b, "beta")):
        assert turn["tool_result"]["owner"] == tag
        assert len(turn["step_results"]) == 2, f"{tag} lost or gained steps: {turn['step_results']}"
        assert all(e["result"]["owner"] == tag for e in turn["step_results"])
        assert all(e["tool_id"].startswith(f"tool.{tag}.") for e in turn["step_results"])
        assert turn["pending_clarification"]["owner"] == tag


def test_fanout_child_task_records_into_the_parents_turn():
    """Fan-out branches run as child tasks: they inherit a COPY of the
    ContextVar carrying the same trace_id, and mutating the shared store entry
    propagates to the parent's snapshot."""

    async def _turn_with_fanout():
        set_current_turn("trace-fanout", "q")
        begin_turn_output("trace-fanout")
        m.LAST_STEP_RESULTS = []

        async def _branch(i: int):
            m._record_step(i, f"tool.branch{i}", {"ok": True, "branch": i})

        await asyncio.gather(_branch(1), _branch(2))
        return pop_turn_output(current_turn_trace_id())

    out = run(_turn_with_fanout())
    assert sorted(e["result"]["branch"] for e in out["step_results"]) == [1, 2]


def test_outside_any_turn_recorders_only_touch_the_globals():
    """Bare calls (unit tests, REPL) have no open turn slot — recorders must
    not crash, and must still write the debug globals."""
    set_current_turn("", "")  # no current turn
    m.LAST_STEP_RESULTS = []
    assert turn_output() is None
    m._record_tool_result({"ok": True})
    m._record_step(1, "tool.x", {"ok": True})
    assert m.LAST_TOOL_RESULT == {"ok": True}
    assert len(m.LAST_STEP_RESULTS) == 1


def test_pop_is_single_use_and_empty_safe():
    begin_turn_output("trace-pop")
    out1 = pop_turn_output("trace-pop")
    out2 = pop_turn_output("trace-pop")  # already popped → empty defaults
    assert out1["step_results"] == [] and out1["tool_result"] is None
    assert out2["step_results"] == [] and out2["pending_loop"] is None
    assert out1 is not out2  # defaults are fresh copies, not a shared object
