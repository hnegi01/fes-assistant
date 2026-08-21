"""
Integration test for the Step 7 clarification loop, end to end.

Exercises the real multi-turn behaviour against a running backend: a first turn
that omits a required identifier should pause and ask, and a second turn (same
session_id) supplying the answer should resume — proving the backend persisted
the clarification state across requests.

Like the other integration tests, this needs the full stack + real creds
(tests/integration/integration_config.yaml) and is skipped otherwise. It is
sensitive to planner behaviour (the planner must leave the missing identifier
empty rather than guessing it — the anti-hallucination guard), so treat a
failure here as a prompt/model-quality signal, not only a wiring bug.

    pytest tests/integration/test_clarification_flow.py -v -m integration
"""

import json
import uuid

import httpx
import pytest


def _turn(backend_url, tenant_config, session_id, message):
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": session_id,
            "messages": [{"role": "user", "content": message}],
            "user_input": message,
            "mode": "chat",
            "tenant_config": tenant_config,
            "allow_summarization": False,
        },
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _looks_like_question(text: str) -> bool:
    """A clarifying ask, in either shape: the LLM's free-form question, or the
    code-rendered template ("I need a bit more information to run this. ...
    needs: ...") introduced 2026-08-14 — it states the blanks without a '?'."""
    t = (text or "").lower()
    if "i need a bit more information" in t or "needs:" in t:
        return True
    return "?" in t or any(w in t for w in ("which", "what ", "please provide", "could you", "specify"))


@pytest.mark.integration
def test_clarify_then_resume(backend_url, tenant_config):
    """Turn 1 omits the user identifier → assistant asks; Turn 2 supplies it → resumes."""
    session_id = f"integ-clarify-{uuid.uuid4()}"

    # Turn 1: ask for a datamodel's columns WITHOUT naming the datamodel.
    # (Read-only, and it clarifies deterministically — unlike "look up a user
    # by their email address", which the model completes with an invented
    # placeholder address and executes; that behaviour has its own test below.)
    first = _turn(backend_url, tenant_config, session_id, "show me the columns of a datamodel")
    reply1 = first.get("reply") or ""

    # It should pause and ask, not execute a lookup with a guessed value.
    assert _looks_like_question(reply1), f"expected a clarifying question, got: {reply1!r}"
    assert first.get("tool_result") in (None, {}), "nothing should have executed on the clarifying turn"

    # Turn 2: same session, supply the identifier. The backend should resume the
    # pinned tool rather than ask again.
    second = _turn(backend_url, tenant_config, session_id, "AgricultureDataModel")
    reply2 = second.get("reply") or ""

    resumed = (second.get("tool_result") is not None) or (not _looks_like_question(reply2))
    assert resumed, f"expected the second turn to resume/execute, got: {reply2!r}"


@pytest.mark.integration
def test_approved_mutation_executes(backend_url, tenant_config):
    """Turn 1 gates a mutation; turn 2 with approved_keys passes the gate and executes.

    Assertion is on the FLOW, not on the Sisense outcome: after turn 2 there must be
    no pending_confirmation — something was attempted (success or SDK error both prove
    the gate was passed).
    """
    session_id = f"integ-mutation-approve-{uuid.uuid4()}"
    message = "delete the user integ-test-nonexistent@example.com"

    # Turn 1: mutation gate fires → pending_confirmation in tool_result.
    first = _turn(backend_url, tenant_config, session_id, message)
    tool_result1 = first.get("tool_result") or {}
    pc = tool_result1.get("pending_confirmation")
    assert pc is not None, (
        f"Expected pending_confirmation on mutating turn, got: {first!r}. "
        "If the planner picked a non-mutating tool the message may need adjustment."
    )

    # Build the approval key from what the backend returned.
    approved_keys = [[pc["tool_id"], json.dumps(pc["arguments"], sort_keys=True)]]

    # Turn 2: same session, same message, with approved_keys populated.
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": session_id,
            "messages": [{"role": "user", "content": message}],
            "user_input": message,
            "mode": "chat",
            "tenant_config": tenant_config,
            "allow_summarization": False,
            "approved_keys": approved_keys,
        },
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    second = resp.json()
    tool_result2 = second.get("tool_result") or {}

    # Gate passed — no another pending_confirmation on turn 2.
    assert "pending_confirmation" not in tool_result2, (
        "expected gate to pass on turn 2 with approved_keys, but got another pending_confirmation"
    )


@pytest.mark.integration
def test_clarification_gives_up_after_cap(backend_url, tenant_config):
    """Two non-answers in a row should end in a terminal 'what's required' message, not loop forever."""
    session_id = f"integ-clarify-cap-{uuid.uuid4()}"

    _turn(backend_url, tenant_config, session_id, "show me the columns of a datamodel")
    _turn(backend_url, tenant_config, session_id, "i'm not sure")
    third = _turn(backend_url, tenant_config, session_id, "still don't know")
    reply3 = (third.get("reply") or "").lower()

    # After the cap, it should state what it needs and stop, not keep asking.
    assert "required" in reply3 or "need" in reply3, f"expected a terminal requirements message, got: {reply3!r}"


@pytest.mark.integration
def test_descriptive_identifier_request_clarifies_instead_of_inventing(backend_url, tenant_config):
    """'look up a single user by their email address' names NO email — the only
    correct move is to ask for one.

    Was xfail until 2026-08-18: the tool-selection call filled the slot with
    an invented "user@example.com" and executed, reporting "user not found"
    for someone nobody asked about. PLANNING_SYSTEM_PROMPT forbids that exact
    string and the model produced it anyway, so the guard is code
    (_is_fabricated): a required value that looks invented AND that the user
    never typed counts as missing."""
    session_id = f"integ-placeholder-{uuid.uuid4()}"
    first = _turn(backend_url, tenant_config, session_id, "look up a single user by their email address")
    reply = first.get("reply") or ""
    steps = first.get("step_results") or []

    assert _looks_like_question(reply), f"expected a clarifying question, got: {reply!r}"
    # Nothing may run: the invented address must never reach Sisense.
    assert not steps, f"nothing should execute without an address, ran: {[s.get('tool_id') for s in steps]}"
    assert not (first.get("tool_result") or {}), "no tool payload on a clarifying turn"
    # The reply DOES contain an example.com address — the curated phrasing hint
    # ("For example, you could ask: ...") — and that is intended: it shows how to
    # ask, and is never presented as a value the user supplied.
    assert "for example" in reply.lower(), "the phrasing hint should still be offered"
