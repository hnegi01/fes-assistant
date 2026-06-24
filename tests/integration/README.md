# Integration Tests

## Unit tests vs. integration tests

**Unit tests** (`tests/test_*.py`) run on every CI push. They are fast, use mocks and
fake credentials, and never touch a real LLM, MCP server, or Sisense instance. The
`conftest.py` at the repo root seeds all required env vars with dummy values so the
module-level imports succeed.

**Integration tests** (this folder) require the full stack to be running and valid
credentials. They are **not** meant for CI — they skip automatically when no
credentials are configured, so a CI run that collects them stays green.

## Credentials — one place

All integration credentials live in **`tests/integration/integration_config.yaml`**
(gitignored). Copy the template and fill it in once:

```bash
cp tests/integration/integration_config.example.yaml \
   tests/integration/integration_config.yaml
# then edit it with a real Sisense domain + token
```

Tests pull credentials from shared fixtures in `conftest.py` — `backend_url`,
`tenant_config`, and `migration_config`. If `integration_config.yaml` is missing
or still holds the example placeholder values, every integration test is
**skipped** (not failed).

## What an integration test looks like

An integration test sends a real user prompt to the backend and asserts on the
observable outcome — not on how the code works internally. Credentials come in
through fixtures, never hardcoded:

```python
import httpx, pytest

@pytest.mark.integration
def test_list_dashboards_prompt(backend_url, tenant_config):
    """A real 'list all dashboards' prompt must call a dashboard tool and return results."""
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": "integ-test-session",
            "messages": [{"role": "user", "content": "show me all dashboards"}],
            "user_input": "show me all dashboards",
            "mode": "chat",
            "tenant_config": tenant_config,
        },
        timeout=60,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("tool_result") is not None
```

## How to run

```bash
# 1. Start all three services (see repo README for full instructions)
docker compose up --build

# 2. Configure credentials once (see "Credentials" above)

# 3. Run only integration tests
pytest tests/integration/ -v -m integration
```

## Why they are not in CI

- They require secrets that are not available in GitHub Actions runners.
- They are slow (real LLM calls, real Sisense API calls).
- They are non-deterministic — LLM responses vary between runs.

The intent is to run them manually before cutting a release branch.
