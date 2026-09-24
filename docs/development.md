# Development — running, testing, rebuilding the registry

The development loop: run the three processes by hand, test at three tiers,
regenerate the tool registry after an SDK bump. For configuring and deploying a
running instance see [operations.md](operations.md).

---

## Prerequisites

- Python 3.11 (pinned `>=3.11,<3.12`)
- A Sisense Fusion deployment (or multiple, for migration use cases)
- Access to at least one LLM provider: Azure OpenAI, Databricks Model Serving,
  or HuggingFace Inference API
- (Optional but recommended) Docker + Docker Compose for containerized runs

---

## Running locally (without Docker)

Three processes over HTTP. Always start them **bottom-up** — the backend proxies
to MCP, the UI proxies to the backend.

**1) Environment.** Preferred — reproducible from the lock file:

```bash
uv sync                    # creates .venv from uv.lock (Python 3.11 pinned)
```

Or the classic flow:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**2) Create a `.env`** — see [`.env.example`](../.env.example).

**3) MCP server** (terminal 1) — **must be `--workers 1`**:

```bash
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8002 --workers 1
```

MCP Streamable HTTP sessions are stateful; running multiple workers breaks
session continuity unless you add sticky routing. This project relies on a single
worker and uses concurrency caps + streaming progress to stay responsive during
long migrations.

**4) Backend** (terminal 2):

```bash
uvicorn backend.api_server:app --host 0.0.0.0 --port 8001
```

**5) UI** (terminal 3):

```bash
streamlit run frontend/app.py
```

Streamlit prints a local URL (typically `http://localhost:8501`).

**Health checks** before driving anything:

```bash
curl -s http://localhost:8002/health   # {"ok": true, "tools": 110, ...}
curl -s http://localhost:8001/health   # {"status": "ok"}
```

**Restart just the backend** after editing agent code (MCP rarely needs it —
except after an allowlist edit, which it reads once at import):

```bash
pkill -f "uvicorn backend.api_server"; sleep 1
uvicorn backend.api_server:app --host 0.0.0.0 --port 8001 > logs/backend_run.log 2>&1 &
```

> The **`run` skill** (`.claude/skills/run/SKILL.md`) carries the full launch /
> drive / restart recipes, including a harness for POSTing `/agent/turn` exactly
> the way the UI does. Reproduce a prompt against the **live agent** that way —
> not by calling the SDK or MCP directly.

**Streamlit reload asymmetry:** page code in `frontend/app.py` hot-applies on
the next rerun; code that runs in worker threads needs a restart.

---

## Testing

Markers are defined in `pyproject.toml`. Three tiers:

| Tier | Command | Needs |
|---|---|---|
| **Unit** | `pytest tests/unit -q` | Nothing — mocked LLM/MCP, fast, runs in CI. Always run these |
| **Integration** | `pytest tests/integration -m integration -v` | The live 3-service stack + real credentials |
| **Eval batteries** | `pytest tests/integration -m eval -v` | Same — regression prompts asserting agent behaviour |

```bash
uv run pytest tests/unit -q                              # or just: pytest tests/unit -q
FES_AGENT_ENGINE=custom pytest tests/unit -q             # engine parity — both must pass
pytest tests/integration/test_evals_planner.py -m eval -v         # chat (read paths)
pytest tests/integration/test_evals_chat_mutations.py -m eval -v  # chat (mutation gating)
pytest tests/integration/test_evals_migration.py -m eval -v       # migration
```

**Integration and eval are local-only and never in GitHub Actions** — LLM and
Sisense secrets are deliberately kept out of CI (firm policy). To run them,
copy [`tests/integration/integration_config.example.yaml`](../tests/integration/integration_config.example.yaml)
to `tests/integration/integration_config.yaml` (gitignored) and fill in real
values; tests skip automatically when it is missing. See
[`tests/integration/README.md`](../tests/integration/README.md).

### Discipline

- **The eval battery is anti-whack-a-mole.** A prompt that once misbehaved
  becomes an `EVAL_CASES` entry, not a scenario-specific prompt rule. Cases
  are added **after** a fix is proven ("5/5 after"), pinning it — not while the
  behaviour is still broken, which would train everyone to ignore a red battery.
- **Three eval files, one harness per mode.** Chat cases assert on which tools
  *executed*; migration cases assert on which tool was *gated* — every migration
  tool mutates, so a migration turn stops at the approval dialog. Migration cases
  send no `approved_keys` (enforced in the file), so they never write to a real
  target. Don't merge the harnesses.
- **Mutation tests only ever mutate an asset they created** — create → gate →
  approve → delete that same asset → `finally:` force-delete. See
  `tests/integration/test_mutation_lifecycle.py`. Never touch a pre-existing asset.
- **Test subjects are discovered from the tenant, not hardcoded.** `eval_identities`
  reads a suitable user/group/model off the live environment; the config section
  is a preference, not truth. A deleted test user once faked four eval regressions.
- **LLM non-determinism:** re-run a single failing integration test before
  treating it as a regression. If it passes in isolation it's variance, not a break.
- **Both engines.** The unit suite is the parity harness for `FES_AGENT_ENGINE`;
  CI runs it under both flags.

---

## Running with Docker (local/dev)

Three Dockerfiles and a `docker-compose.yml`:

- [`Dockerfile.ui`](../Dockerfile.ui) – Streamlit UI
- [`Dockerfile.backend`](../Dockerfile.backend) – FastAPI backend
- [`Dockerfile.mcp`](../Dockerfile.mcp) – MCP tool server
- [`docker-compose.yml`](../docker-compose.yml) – runs all three together

Create a `.env` in the project root (same keys as the environment configuration;
not committed, not baked into images), then:

```bash
docker compose up --build --force-recreate
```

- UI: `http://localhost:8501`
- Backend docs: `http://localhost:8001/docs` (dev/local only)
- MCP health: `http://localhost:8002/health` (dev/local only)

In the production compose file only the Nginx port (80) is published — the
backend and MCP ports stay internal to the compose network.

Stop / hard reset:

```bash
docker compose down
docker compose down --rmi all --volumes --remove-orphans   # dev only — never on a shared host
docker builder prune -a -f
```

Images build from **`uv.lock`** (`uv sync --frozen`), so the lock is what ships —
see the pin-bump trap below.

---

## Tool registry generation

The MCP server uses a **tool registry JSON** that describes available tools,
parameters, descriptions, and examples. It is **generated from the PySisense
SDK** — never hand-edit it; the allowlist is where curation happens (see
[architecture.md → The tool registry and the curated surface](architecture.md#the-tool-registry-and-the-curated-surface)).

Two stages:

1. `config/tools.registry.json` — built directly from the SDK.
2. `config/tools.registry.with_examples.json` — enriched with curated examples;
   the one loaded at runtime. Also emits the 3-level routing tree `config/registry/`.

Scripts in [`scripts/`](../scripts/), run **as modules** (a plain
`python scripts/01_….py` fails on the package-relative imports):

```bash
python -m scripts.01_build_registry_from_sdk          # config/tools.registry.json
python -m scripts.02_add_llm_examples_to_registry     # with_examples.json + config/registry/ tree
python -m scripts.03_sync_examples_to_registry_tree --write   # examples-only sync (no SDK needed)
python -m scripts.04_generate_tool_allowlist          # audit allowlist drift after a rebuild
python -m scripts.04_generate_tool_allowlist --apply  # stage new tools (commented) / retire removed ones
```

- **01** introspects the SDK classes, parses docstrings, infers JSON Schemas,
  applies `SCHEMA_RULES`, tags tools, and writes the flat base registry.
- **02** reuses existing examples and calls the LLM only for tools that have none
  (needs LLM credentials in `.env`).
- **04 `--apply`** stages tools new in this SDK version **commented out** under a
  dated header, and moves tools the SDK removed to the DEPRECATED section.
  Uncommenting a line is the human review that exposes a tool.

At runtime, only the JSON files in `config/` are needed.

### Bumping the PySisense pin — the ritual

```
read the SDK changelog → bump pin → uv sync → scripts/01 → scripts/02
   → scripts/04 --apply → review the staged tools in config/allowed_tools.txt
   → pytest tests/unit (both engines) → integration + eval batteries
   → release the app (new FES version) → deploy
```

**Start with the changelog, not the pin.** The diff that matters is rarely the
new tools; it is the changed behaviour of tools already exposed. pysisense
2.3.0 fixed the scoping of a dashboard whose two copies name different
datasources — the bug that made `analyze_perspective_requirements` report four
dashboards with **zero** columns and raise no warning, which reads as "these
dashboards use nothing" rather than "I could not resolve their datasource".
Two exposed tools changed behaviour with no signature change, and nothing in
the rebuild would have told you.

**A changed RESPONSE SHAPE can need work in the skills, not just the
registry.** 2.3.0 started populating `join_path_choices`, which the skill's
`## Ask` had always rendered as "none found". Surfacing it took a new
`|join_paths` filter and a rewritten line. Read the changelog's "Changed"
section against `skills/*/SKILL.md` and the filters in
`skill_flow.py::render_text`, not only against the registry.

**When the guard test fails, that is the mechanism working.**
`TestSummaryOverrides` compares each `_SUMMARY_OVERRIDES` entry against the
INSTALLED docstring. A failure after a bump means the SDK changed that line —
usually because it adopted the replacement upstream. Delete the entry and
regenerate; do not re-record it to make the test pass. Both entries died this
way in 2.3.0.

**The ritual does not end at green tests.** The registry ships inside the
images, so a bump reaches production only through an app release. Bump the FES
version, merge to main, let CD publish, then deploy. Leaving it at "tests pass"
means local and production run different SDKs.

**Verify the bump in all five places before believing it:** `pyproject.toml`,
`uv.lock`, `requirements.txt`, `requirements-dev.txt`, and the venv.

**The trap:** `uv sync` may **silently refuse** a freshly-published version
(safe-chain suppresses releases below a minimum package age). Left alone,
`pyproject` says the new version while `uv.lock` stays on the old one — and the
Docker images build from the **lock**, so production ships the old SDK while
every local check passes. Use `uv sync --safe-chain-skip-minimum-package-age`,
then check the lock's `[[package]] name = "pysisense"` block directly.

**After the rebuild, diff every pre-existing tool's schema** against the
committed registry, not just the new ones. The generator's annotation→schema path
has lost information before (a union collapsing to `string`; an enum that only
existed in a docstring) and nothing pins those unless `SCHEMA_RULES` patches them.

**Drift guards** name what a bump broke: `TestSchemaRulesDrift` (every patched
method/param must still exist in the SDK), `test_option_enrichment.py` (every
`x-options-tool` must be real, allowlisted and read-only), `test_tool_examples.py`
(no example may invent a value its query omits, nor reference schema its query
never names), `test_internal_params.py`.

---

## Pre-commit

Ruff + ruff-format + commitizen. Conventional-commit types (`feat`, `fix`, `docs`,
`test`, `chore`, `refactor`); scope where useful — `feat(ui):` not `ui:`.
ruff-format may reformat and fail the first commit; re-`git add` and commit again.
