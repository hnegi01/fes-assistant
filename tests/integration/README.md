# Integration Tests

## Unit tests vs. integration tests

**Unit tests** (`tests/test_*.py`) run on every CI push. They are fast, use mocks and
fake credentials, and never touch a real LLM, MCP server, or Sisense instance. The
`conftest.py` at the repo root seeds all required env vars with dummy values so the
module-level imports succeed.

**Integration tests** (this folder) require the full stack to be running and valid
credentials in `.env`. They are **not** run by CI — run them manually before a release
or a major behaviour change.

## What an integration test looks like

An integration test sends a real user prompt to the backend and asserts on the
observable outcome — not on how the code works internally.

Example (not wired to CI):

```python
# tests/integration/test_e2e_prompt.py

import httpx, pytest

BACKEND = "http://localhost:8001"

@pytest.mark.integration
def test_list_dashboards_prompt():
    """A real 'list all dashboards' prompt must call a dashboard tool and return results."""
    resp = httpx.post(
        f"{BACKEND}/agent/turn",
        json={
            "session_id": "integ-test-session",
            "message": "show me all dashboards",
            "mode": "chat",
            "tenant_config": {
                "domain": "https://your-sisense-instance.example.com",
                "token": "your-real-token",
            },
        },
        timeout=60,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("tool_id", "").startswith("dashboard.")
    assert body.get("result") is not None
```

## How to run

```bash
# 1. Start all three services (see repo README for full instructions)
docker compose up --build

# 2. Run only integration tests (requires real .env with Sisense + LLM creds)
pytest tests/integration/ -v -m integration
```

## Why they are not in CI

- They require secrets that are not available in GitHub Actions runners.
- They are slow (real LLM calls, real Sisense API calls).
- They are non-deterministic — LLM responses vary between runs.

The intent is to run them manually before cutting a release branch.
