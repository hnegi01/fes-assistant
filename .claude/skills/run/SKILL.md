---
name: run
description: Launch and drive the FES Assistant stack (MCP :8002 → backend :8001 → UI :8501), send /agent/turn requests like the UI does, and run the unit / integration / eval test tiers. Use when starting the services, reproducing a prompt against the live agent, or running tests.
---

# Running & testing FES Assistant

Three processes over HTTP. Always start them **bottom-up** (MCP → backend → UI);
the backend proxies to MCP, the UI proxies to the backend.

```
UI (Streamlit :8501) → backend (FastAPI :8001) → MCP (Starlette :8002) → PySisense SDK → Sisense
```

Virtualenv: **`venv_pysisense_chatbot`**. Prefix every command with it
(`source venv_pysisense_chatbot/bin/activate` once per shell, or call
`venv_pysisense_chatbot/bin/<tool>` directly).

## Launch the stack

```bash
source venv_pysisense_chatbot/bin/activate

# 1. MCP server — MUST be --workers 1 (session/cancel state is in-process;
#    multiple workers would split Mcp-Session-Id across processes)
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8002 --workers 1 > logs/mcp_run.log 2>&1 &

# 2. Backend
uvicorn backend.api_server:app --host 0.0.0.0 --port 8001 > logs/backend_run.log 2>&1 &

# 3. UI (foreground, or drop the & )
streamlit run frontend/app.py --server.port 8501
```

Health checks (do this before driving anything):

```bash
curl -s http://localhost:8002/health   # {"ok": true, "tools": 119, ...}
curl -s http://localhost:8001/health   # {"status": "ok"}
```

Restart just the backend after editing agent code (MCP rarely needs a restart):

```bash
pkill -f "uvicorn backend.api_server"; sleep 1
uvicorn backend.api_server:app --host 0.0.0.0 --port 8001 > logs/backend_run.log 2>&1 &
sleep 3
```

Logs land in `./logs/` — `llm_agent.log` is the one to watch (plan text,
`Tool selected`, `Fan-out: running N`, `REPLAN`, `Clarification needed`,
dependency-gate skips). `llm_calls.csv` has one row per LLM call and `tool_calls.csv` one per tool
execution, grouped by turn id — written only when `FES_CSV_OBSERVABILITY=true`
(off by default; LangSmith tracing likewise needs `LANGSMITH_TRACING=true`).

## Drive the agent like the UI does

Reproduce a prompt against the **live agent** by POSTing `/agent/turn` — this is
the real path (UI → backend), not calling the SDK or MCP directly. Creds come
from `tests/integration/integration_config.yaml` (gitignored, holds a real
token). **`allow_summarization` defaults to `false` when omitted** — the agent
needs it `true` for adaptive chains and the critic, so set it explicitly.

```python
import uuid, json, requests as rq, yaml
cfg = yaml.safe_load(open("tests/integration/integration_config.yaml"))["tenant_config"]
TENANT = {"domain": cfg["domain"], "token": cfg["token"], "ssl": cfg.get("verify_ssl", True)}

def turn(msg, summ=True, approved_keys=None):
    body = {"session_id": f"drive-{uuid.uuid4()}", "messages": [{"role": "user", "content": msg}],
            "user_input": msg, "mode": "chat", "tenant_config": TENANT, "allow_summarization": summ}
    if approved_keys:
        body["approved_keys"] = approved_keys
    r = rq.post("http://localhost:8001/agent/turn", json=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"}, timeout=180).json()
    print("REPLY:", (r.get("reply") or "")[:400])
    for s in (r.get("step_results") or []):
        print(f"  step {s.get('step')} {s.get('tool_id')} ok={(s.get('result') or {}).get('ok')}")
    pc = (r.get("tool_result") or {}).get("pending_confirmation")
    if pc:
        print("GATE:", pc["tool_id"], "| reason:", pc.get("reason"))
    return r

turn("which group does user x@y.com belong to and show all its members")
```

**Approving a gated mutation** (two-phase): the first turn returns
`tool_result.pending_confirmation`; re-send the same prompt with
`approved_keys=[[tool_id, json.dumps(args, sort_keys=True)]]`.

## Test tiers

Markers are defined in `pyproject.toml`.

| Tier | Command | Needs |
|---|---|---|
| **Unit** | `pytest tests/unit -q` | nothing — mocked LLM/MCP, fast, always run these |
| **Integration** | `pytest tests/integration -m integration -v` | live stack + real creds |
| **Eval battery** | `pytest tests/integration/test_evals_planner.py -m eval -v` | live stack + real creds |

Integration/eval are **local-only, cost money, and are never wired into GitHub
Actions** (we never put LLM or Sisense secrets in CI — firm policy). Run them
explicitly when verifying agent behavior end to end.

- The **eval battery** (`test_evals_planner.py`) is the anti-whack-a-mole
  discipline: a prompt that once misbehaved becomes a data-driven `EVAL_CASES`
  entry, NOT a scenario-specific prompt rule. Add cases here; keep prompts
  generic. Harness supports `expect_tools_any` / `expect_tools_all` /
  `expect_min_steps` / `forbid_tools` / `expect_reply_any` / `forbid_reply` /
  per-case `allow_summarization`.
- LLM non-determinism means an integration test can flake on phrasing. Re-run
  the single test before treating a failure as a regression; if it passes in
  isolation it's variance, not a break.

## Testing mutations — the safety rule

**Only ever mutate an asset the test itself created.** Never delete/modify a
pre-existing user, group, or dashboard. The pattern (see
`tests/integration/test_mutation_lifecycle.py`): create a throwaway asset →
verify gate→approve→execute → delete *that same* asset via the same gated flow →
`finally:` force-delete so a failed run can't leak it. When driving mutations
manually, do the same: create your own `xyz@test` user first, then test delete
only against `xyz`.

## Gotchas

- **MCP is single-worker** by architecture — never add workers.
- **`allow_summarization` is a data-visibility switch**, not loop on/off. Off =
  result DATA never reaches the LLM (metadata only); adaptive chains stop
  gracefully (`BLOCKED`) and dependent plan steps are skipped up front.
- **Registry**: edit `config/tools.registry.with_examples.json` (the
  server-loaded one), not `tools.registry.json`. Tool descriptions are
  auto-generated from SDK docstrings — never hand-edit them.
- **Pre-commit**: ruff + ruff-format + commitizen. Use conventional-commit
  types (`feat`, `fix`, `docs`, `test`, `chore`, `refactor`); `feat(ui):` not
  `ui:`. ruff-format may reformat and fail the first commit — re-`git add` and
  commit again. End commit messages with the required `Co-Authored-By` trailer.
- **LLM config is import-time**: changing env vars needs a backend restart.
