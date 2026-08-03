"""
Full mutation lifecycle against a live Sisense environment — SELF-CONTAINED.

Safety rule: this test only ever mutates the asset IT CREATES. It creates a
throwaway user, verifies the gate→approve→execute flow on the create, then
deletes that same user through the same gated flow — leaving the environment
exactly as it found it. A finally-block attempts the delete even when an
assertion fails mid-way, so a red run cannot leak the test user.

Covers, end to end, on real infrastructure:
  - mutation gate fires with a plain-English explanation (human-in-the-loop)
  - approval (approved_keys) executes the exact gated tool
  - the same cycle for a destructive tool (delete), targeting only our asset

Cred-gated like all integration tests (skipped without integration_config.yaml).

    pytest tests/integration/test_mutation_lifecycle.py -v -m integration
"""

import json
import uuid

import httpx
import pytest

LIFECYCLE_EMAIL = "fes.integ.lifecycle@sisense-test.com"


def _turn(backend_url, tenant_config, session_id, message, approved_keys=None):
    payload = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": message}],
        "user_input": message,
        "mode": "chat",
        "tenant_config": tenant_config,
        "allow_summarization": True,
    }
    if approved_keys:
        payload["approved_keys"] = approved_keys
    resp = httpx.post(f"{backend_url}/agent/turn", json=payload, timeout=180)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _gate_then_approve(backend_url, tenant_config, session_id, message, expect_tool):
    """Run a mutating prompt: assert the gate fires (with an English reason),
    then approve and return the post-approval body."""
    first = _turn(backend_url, tenant_config, session_id, message)
    pc = (first.get("tool_result") or {}).get("pending_confirmation")
    assert pc is not None, f"expected the mutation gate for {expect_tool}, got: {first!r}"
    assert expect_tool in pc.get("tool_id", ""), f"unexpected gated tool: {pc.get('tool_id')!r}"
    assert pc.get("reason"), "gate must carry a plain-English explanation"
    # Only our throwaway asset may ever be targeted.
    assert LIFECYCLE_EMAIL in json.dumps(pc.get("arguments") or {}), (
        f"gated args do not target the test asset: {pc.get('arguments')!r}"
    )

    key = [pc["tool_id"], json.dumps(pc.get("arguments") or {}, sort_keys=True, ensure_ascii=False)]
    second = _turn(backend_url, tenant_config, session_id, message, approved_keys=[key])
    tool_result = second.get("tool_result") or {}
    assert "pending_confirmation" not in tool_result, "gate did not pass after approval"
    return second


def _force_delete(backend_url, tenant_config):
    """Best-effort cleanup: delete the lifecycle user via the gated flow.
    Swallows everything — used in finally so a failed test can't leak the asset."""
    try:
        sid = f"integ-lifecycle-cleanup-{uuid.uuid4()}"
        _gate_then_approve(backend_url, tenant_config, sid, f"delete the user {LIFECYCLE_EMAIL}", "delete_user")
    except Exception:
        pass


@pytest.mark.integration
def test_mutation_lifecycle_create_then_delete_own_asset(backend_url, tenant_config):
    create_msg = (
        f"create a new user with email {LIFECYCLE_EMAIL} first name Integ last name Lifecycle with the viewer role"
    )
    try:
        # --- create (gate → approve → execute) ---
        sid = f"integ-lifecycle-create-{uuid.uuid4()}"
        created = _gate_then_approve(backend_url, tenant_config, sid, create_msg, "create_user")
        create_results = created.get("step_results") or []
        created_ok = any(
            "create_user" in str(s.get("tool_id")) and (s.get("result") or {}).get("ok") for s in create_results
        ) or (created.get("tool_result") or {}).get("ok")
        assert created_ok, f"create_user did not succeed: {created!r}"

        # --- delete the SAME asset (gate → approve → execute) ---
        sid = f"integ-lifecycle-delete-{uuid.uuid4()}"
        deleted = _gate_then_approve(
            backend_url, tenant_config, sid, f"delete the user {LIFECYCLE_EMAIL}", "delete_user"
        )
        delete_results = deleted.get("step_results") or []
        deleted_ok = any(
            "delete_user" in str(s.get("tool_id")) and (s.get("result") or {}).get("ok") for s in delete_results
        ) or (deleted.get("tool_result") or {}).get("ok")
        assert deleted_ok, f"delete_user did not succeed: {deleted!r}"
    finally:
        # Never leak the throwaway user, even on a failed assertion above.
        _force_delete(backend_url, tenant_config)
