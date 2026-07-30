"""
Integration tests for the Step 7 scenarios not covered elsewhere, end to end.

Prompts here are the exact ones verified live against a real Sisense
environment on 2026-07-30 (see the Step 7 scenario matrix in .claude/docs/).
Coverage map across the integration suite:

  Scenario 1 (single missing arg → clarify → resume)  → test_clarification_flow.py
  Scenario 2 (multiple missing args → one ask → both) → HERE
  Scenario 3 (topic change on resume → fresh routing) → HERE
  Scenario 4 (attempt cap → terminal give-up)         → test_clarification_flow.py
  Scenario 5 (format error → hard block)              → test_validation_prompts.py
  Scenario 6 (mutation gate + English reason)         → HERE (reason assertion)
  Scenario 7 (clarify → then mutation gate)           → HERE
  Scenario 8 (off-topic short-circuit)                → test_validation_prompts.py
  Scenario 9 (approved mutation executes)             → test_clarification_flow.py
  Scenario 10 (mutation cancel)                       → frontend-only, no backend test

Like the other integration tests these need the full stack + real creds
(tests/integration/integration_config.yaml) and are skipped otherwise. They
assert on the FLOW (clarify vs gate vs execute), not on Sisense data, so they
hold on any instance. They are sensitive to planner behaviour — treat a
failure as a prompt/model-quality signal, not only a wiring bug.

    pytest tests/integration/test_step7_scenarios.py -v -m integration
"""

import uuid

import httpx
import pytest


def _turn(backend_url, tenant_config, session_id, message, history=None):
    """POST one turn the way the UI does: full accumulated history + new message.

    `history` is a mutable list maintained by the caller across turns; the new
    user message and the assistant reply are appended to it, mirroring
    st.session_state.messages in frontend/app.py.
    """
    messages = list(history or []) + [{"role": "user", "content": message}]
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": session_id,
            "messages": messages,
            "user_input": message,
            "mode": "chat",
            "tenant_config": tenant_config,
            "allow_summarization": False,
        },
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if history is not None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": body.get("reply") or ""})
    return body


def _looks_like_question(text: str) -> bool:
    t = (text or "").lower()
    return "?" in t or any(w in t for w in ("which", "what ", "please provide", "could you", "specify"))


@pytest.mark.integration
def test_multiple_missing_args_asked_together_then_resumed(backend_url, tenant_config):
    """Scenario 2: a request missing TWO required args gets ONE question asking
    for both; the answer supplying both resumes and executes.

    The resume uses env-specific names — a wrong-name SDK error still proves
    the flow (clarify → resume → execute) worked; only another question fails it.
    """
    session_id = f"integ-s2-{uuid.uuid4()}"
    history = []

    first = _turn(backend_url, tenant_config, session_id, "show me data from a table in a datamodel", history)
    reply1 = first.get("reply") or ""

    assert _looks_like_question(reply1), f"expected a clarifying question, got: {reply1!r}"
    assert first.get("tool_result") in (None, {}), "nothing should execute on the clarifying turn"
    # One question covering both blanks — it must mention both concepts.
    low = reply1.lower()
    assert "model" in low and "table" in low, f"expected both missing args in one question, got: {reply1!r}"

    second = _turn(backend_url, tenant_config, session_id, "datamodel AI_test, table Brand", history)
    reply2 = second.get("reply") or ""
    resumed = (second.get("tool_result") is not None) or (not _looks_like_question(reply2))
    assert resumed, f"expected the second turn to execute (even an SDK error), got: {reply2!r}"


@pytest.mark.integration
def test_topic_change_on_resume_routes_fresh(backend_url, tenant_config):
    """Scenario 3: answering a clarifying question with a NEW request drops the
    clarification and routes the new request normally."""
    session_id = f"integ-s3-{uuid.uuid4()}"
    history = []

    first = _turn(backend_url, tenant_config, session_id, "look up a single user by their email address", history)
    assert _looks_like_question(first.get("reply") or ""), "expected a clarifying question first"

    second = _turn(
        backend_url, tenant_config, session_id, "actually forget that, just show me all my dashboards", history
    )
    tool_result = second.get("tool_result") or {}

    # The new topic must execute — not re-ask for the email, not give up.
    assert tool_result, f"expected the topic change to execute a tool, got: {second.get('reply')!r}"
    assert "pending_confirmation" not in tool_result
    tool_id = tool_result.get("tool_id") or ""
    assert "dashboard" in tool_id, f"expected a dashboard tool after topic change, got: {tool_id!r}"


@pytest.mark.integration
def test_mutation_gate_has_english_reason(backend_url, tenant_config):
    """Scenario 6: a mutating request with all args present pauses at the gate,
    and pending_confirmation carries a plain-English explanation (reason)."""
    session_id = f"integ-s6-{uuid.uuid4()}"

    body = _turn(
        backend_url,
        tenant_config,
        session_id,
        "delete the user integ-test-nonexistent@example.com",
    )
    pc = (body.get("tool_result") or {}).get("pending_confirmation")
    assert pc is not None, f"expected pending_confirmation, got: {body!r}"

    reason = pc.get("reason") or ""
    assert reason, "pending_confirmation.reason must carry the English explanation"
    # It should be plain English naming the target, not internals.
    assert "integ-test-nonexistent@example.com" in reason, f"reason should name the target, got: {reason!r}"
    assert "tool" not in reason.lower() and "json" not in reason.lower(), f"reason leaks internals: {reason!r}"


@pytest.mark.integration
def test_clarify_then_mutation_gate(backend_url, tenant_config):
    """Scenario 7: a mutating request missing its required arg clarifies FIRST
    (no gate yet); supplying the arg then fires the mutation gate. Not approved —
    nothing executes."""
    session_id = f"integ-s7-{uuid.uuid4()}"
    history = []

    first = _turn(backend_url, tenant_config, session_id, "delete a user", history)
    reply1 = first.get("reply") or ""
    tool_result1 = first.get("tool_result") or {}

    assert _looks_like_question(reply1), f"expected a clarifying question, got: {reply1!r}"
    assert "pending_confirmation" not in tool_result1, "gate must not fire before the arg is known"

    second = _turn(backend_url, tenant_config, session_id, "integ-test-nonexistent@example.com", history)
    pc = (second.get("tool_result") or {}).get("pending_confirmation")
    assert pc is not None, f"expected the gate after the arg arrived, got: {second!r}"
    assert pc.get("tool_id", "").startswith("access_management."), f"unexpected tool: {pc.get('tool_id')!r}"
    assert pc.get("reason"), "gate must carry the English explanation"
