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
