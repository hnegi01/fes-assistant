# Integration Tests

## Unit tests vs. integration tests

**Unit tests** (`tests/unit/test_*.py`) run on every CI push. They are fast, use mocks and
fake credentials, and never touch a real LLM, MCP server, or Sisense instance. The
shared `tests/conftest.py` seeds all required env vars with dummy values so the
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

# 3b. Run only the EVAL batteries (test_evals_planner.py,
#     test_evals_chat_mutations.py, test_evals_migration.py)
pytest tests/integration/ -v -m eval
```

## Evals vs integration tests

`test_evals_planner.py` is a **regression battery of real prompts that once
failed** — it asserts planner *strategy* (which tools were picked/avoided, what
the answer may claim), not wiring. The rule that keeps prompts from becoming
whack-a-mole: **prompts carry only generic strategy principles; every
scenario-specific failure becomes an eval case instead of a prompt rule.** Any
prompt change must pass the whole battery, so fixing one scenario can't silently
break another. Adding a case = appending one dict to `EVAL_CASES`.

## Why they are not in CI

- They require secrets that are not available in GitHub Actions runners.
- They are slow (real LLM calls, real Sisense API calls).
- They are non-deterministic — LLM responses vary between runs.

The intent is to run them manually before cutting a release branch. This is a
deliberate policy: LLM and Sisense secrets are never added to GitHub Actions.

## Reading a failure

Because a real LLM is in the loop, a single failure is not automatically a bug:

1. **A test fails once** → re-run just that test
   (`pytest tests/integration/<file>::<test> -m integration`).
   If it passes on retry, it was model non-determinism — fine.
2. **A test fails consistently** → something real changed: either code broke
   the flow, or the model's behaviour drifted and a prompt needs strengthening.
   Both are release blockers worth investigating.
