"""
Structural smoke tests for the FastAPI backend.

These tests spin up the FastAPI app in-process using Starlette's TestClient.
They verify that structural endpoints respond correctly without requiring a
real LLM, MCP server, or Sisense instance.

Note: conftest.py sets dummy env vars before this module is imported so that
llm_agent.LLM_CONFIG can be built without real credentials.
"""

from fastapi.testclient import TestClient

from backend.api_server import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tools_endpoint_returns_expected_shape():
    response = client.get("/tools")
    assert response.status_code == 200
    body = response.json()
    assert "tools" in body
    assert "registry" in body
    assert isinstance(body["tools"], list)
    assert isinstance(body["registry"], dict)


def test_cancel_endpoint_returns_cancelled_true():
    """POST /agent/cancel with a fake session_id must return {cancelled: true}."""
    response = client.post(
        "/agent/cancel",
        json={"session_id": "test-session-does-not-exist"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["cancelled"] is True
    assert body["session_id"] == "test-session-does-not-exist"


def test_cancel_endpoint_missing_body_returns_422():
    """POST /agent/cancel with no body must return 422 Unprocessable Entity."""
    response = client.post("/agent/cancel", json={})
    assert response.status_code == 422


def test_agent_turn_response_includes_trace_id(monkeypatch):
    """/agent/turn returns the turn's trace_id — the join key that lets UI
    feedback (thumbs) line up with the exact llm_calls.csv rows it judges.

    run_turn_once returns the per-turn SNAPSHOT dict (not a bare reply):
    the API layer must consume it instead of llm_agent module globals, which
    a concurrent session's turn may overwrite (cross-session leak otherwise)."""
    import backend.agent.llm_agent as llm_agent
    import backend.api_server as api

    async def _fake_run_turn_once(**kwargs):
        # Poison the module globals with ANOTHER session's turn — exactly what
        # a concurrent turn does between this turn finishing and the API layer
        # resuming. The response must carry the snapshot, not these.
        monkeypatch.setattr(llm_agent, "LAST_TRACE_ID", "other-sessions-trace", raising=False)
        monkeypatch.setattr(llm_agent, "LAST_TOOL_RESULT", {"leaked": "someone elses data"}, raising=False)
        monkeypatch.setattr(llm_agent, "LAST_STEP_RESULTS", [{"leaked": True}], raising=False)
        return {
            "reply": "done",
            "tool_result": None,
            "step_results": [],
            "trace_id": "trace-abc-123",
            "pending_clarification": None,
            "pending_loop": None,
        }

    monkeypatch.setattr(api, "run_turn_once", _fake_run_turn_once)

    response = client.post(
        "/agent/turn",
        json={
            "session_id": "test-trace-id",
            "messages": [{"role": "user", "content": "hi"}],
            "user_input": "hi",
            "mode": "chat",
            "tenant_config": {"domain": "https://x", "token": "t", "ssl": True},
            "allow_summarization": False,
        },
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "done"
    assert body["trace_id"] == "trace-abc-123"
    # cross-session leak guard: poisoned globals must NOT reach the response
    assert body["tool_result"] is None
    assert body["step_results"] == []
