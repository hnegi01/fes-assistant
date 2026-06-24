"""
Integration tests for argument-validation behaviour, end to end.

These send the real prompts a user typed against a running backend and assert
on the observable reply. Unlike the unit tests in
tests/unit/test_arg_validation.py, these exercise the actual planner LLM —
so they require:

  - the full stack running (backend + MCP + a reachable Sisense instance)
  - real LLM credentials in .env
  - real Sisense credentials in tests/integration/integration_config.yaml

Credentials come from the shared fixtures in tests/integration/conftest.py
(backend_url, tenant_config). If integration_config.yaml is missing or still
holds placeholder values, these tests are skipped automatically.

They are NOT run by CI (slow, non-deterministic, need secrets). Run manually
before a release:

    pytest tests/integration/test_validation_prompts.py -v -m integration

Note: the SDK call never happens in either case (the request is blocked
before execution), so the assertions hold even against an empty Sisense
instance — but the planner LLM must still be reachable.
"""

import httpx
import pytest


def _turn(backend_url: str, tenant_config: dict, message: str) -> dict:
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": "integ-validation-session",
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


@pytest.mark.integration
def test_non_email_value_is_blocked(backend_url, tenant_config):
    """
    'show me the user profile for john' — the planner picks a user-lookup tool
    and tries user_email="john". format: email validation must block it; the
    reply should say it couldn't call the tool, and no real user data returns.
    """
    body = _turn(backend_url, tenant_config, "show me the user profile for john")
    reply = (body.get("reply") or "").lower()

    assert "couldn't call" in reply or "invalid or missing" in reply
    # Blocked before execution → no tool payload captured.
    assert body.get("tool_result") in (None, {}) or body["tool_result"].get("ok") is False


@pytest.mark.integration
def test_missing_identifier_is_not_hallucinated(backend_url, tenant_config):
    """
    'get datamodel datasecurity rule' — the user named NO datamodel. With the
    planner prompt guard, the model must NOT fabricate datamodel_name from the
    word 'datasecurity'. Expected: the required identifier is left missing, so
    the turn is blocked (and, once Step 7 lands, becomes a clarifying question).

    Regression target: the reply must NOT be a successful datasecurity result
    keyed off datamodel_name="datasecurity".
    """
    body = _turn(backend_url, tenant_config, "get datamodel datasecurity rule")
    reply = (body.get("reply") or "").lower()
    tool_result = body.get("tool_result") or {}

    # Must not have executed a lookup against a hallucinated datamodel name.
    args = (tool_result.get("arguments") or {}) if isinstance(tool_result, dict) else {}
    assert args.get("datamodel_name") != "datasecurity", "planner hallucinated 'datasecurity' as the datamodel name"

    # Acceptable outcomes: blocked now, or a clarifying ask after Step 7.
    blocked = "couldn't call" in reply or "invalid or missing" in reply
    asks = any(w in reply for w in ("which", "specify", "name of", "more specific"))
    assert blocked or asks, f"expected a block or a clarifying question, got: {reply!r}"
