# Operations — configuration, deployment, logging

How to configure, deploy, upgrade and observe a running instance. For the
development loop (running the three processes by hand, tests, rebuilding the
registry) see [development.md](development.md).

---

## Environment configuration

This project keeps **LLM credentials and service configuration** in environment
variables. Sisense base URLs and tokens are entered directly into the Streamlit UI
and stored only in session state for the current browser session.

For local development you can use a `.env` file (see
[`.env.example`](../.env.example)). In Docker / production, set the same values
as real environment variables on each container (`--env-file`, docker-compose
`env_file:`, or sourcing [`config_prod.sh`](../config_prod.sh)).

> **[`.env.example`](../.env.example) is the authoritative, fully annotated
> list** — every variable with its default, allowed values, and effect in plain
> language. The table and notes below are the reference view.

### Reference table

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `databricks` | `azure`, `databricks`, or `huggingface` |
| `AZURE_OPENAI_ENDPOINT` | — | Azure endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | — (required) | Model deployment name — no default; the config build raises without it |
| `AZURE_OPENAI_API_KEY` | — | API key (or via AWS SM) |
| `AZURE_OPENAI_API_STYLE` | `v1` | `v1` (new) or `legacy` (deployment URL) |
| `FES_AZURE_OPENAI_SECRET_ID` | — | AWS Secrets Manager secret name |
| `AWS_REGION` | — | AWS region for Secrets Manager |
| `DATABRICKS_HOST` | — | Databricks workspace URL |
| `DATABRICKS_TOKEN` | — | Databricks PAT |
| `LLM_ENDPOINT` | — | Databricks serving endpoint name |
| `LLM_HTTP_TIMEOUT` | `60` | LLM HTTP call timeout (seconds) |
| `LLM_HTTP_MAX_RETRIES` / `LLM_HTTP_RETRY_BASE_DELAY` | `3` / `0.5` | LLM retry tuning: attempts, and base back-off delay in seconds |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per LLM call |
| `LLM_TEMPERATURE` | `0.2` | LLM temperature |
| `PYSISENSE_MCP_HTTP_URL` | `http://localhost:8002` | MCP server URL (from backend); the client connects to `/mcp/` under it |
| `PYSISENSE_MCP_HTTP_TIMEOUT` | `1800` | MCP HTTP call timeout — per-request and stream read. 30 min for migrations; set it above your longest migration |
| `FES_BACKEND_URL` | `http://localhost:8001` | Backend URL (from frontend) |
| `ALLOW_SUMMARIZATION` | `true` in code · **`false` in the shipped `.env.example`** | Backend-side hard cap on sending tool result **data** to the LLM (the loop still runs on metadata when off). Three layers decide the effective state: this cap, the per-turn `allow_summarization` field (treated as `false` when omitted), and the UI checkbox, which always starts OFF. "Disabled by default" in the docs refers to the shipped configuration and the checkbox, not the bare code default |
| `FES_ALLOW_SUMMARIZATION_TOGGLE` | `true` in code · `false` in the shipped `.env.example` | Whether the UI checkbox is shown; `false` also forces summarization off for every request |
| `FES_MAX_AGENT_STEPS` | `8` | Hard ceiling on tool-executing steps per agent turn (one SDK call each); cap → partial answer |
| `FES_CLARIFY_MAX_ATTEMPTS` | `2` | Max clarifying questions before the agent gives up and states what it needs |
| `FES_VERIFY_GOAL` | `true` | Independent goal checker (the critic): re-checks a "done" answer against the request before accepting it |
| `FES_VERIFY_MAX_RECHECKS` | `1` | How many times the goal checker may push the loop to run one more step |
| `FES_MAX_REPLANS` | `1` | How many times per turn the planner may revise the plan after a failed approach (0 = off) |
| `FES_MAX_PARALLEL_STEPS` | `3` | How many independent plan steps may execute concurrently (1 = off); mutations always sequential |
| `FES_MIGRATION_SINGLE_SHOT` | `true` | Migration turns plan every step in ONE call, one approval, execute in sequence (`migration_flow.py`). `false` routes migration through the reactive loop — a kill switch, not a mode |
| `FES_MIGRATION_COMPLETENESS_CHECK` | `false` | Opt-in second LLM call that checks a migration plan for omitted asset kinds (+1 re-plan if any). Off because the approval dialog's numbered step list is the human check; turn on for unattended/API use |
| `FES_AGENT_ENGINE` | `langgraph` | Turn harness: `langgraph` (StateGraph over shared helpers; default since 2026-08-15) or `custom` (the hand-rolled loop, kept as the dependency-free kill switch) |
| `FES_LANGSMITH_LOG_CONTENT` | `false` | Whether result data may appear in LangSmith traces (independent of summarization) — prompts shown, only data-bearing parts redacted; tool result payloads never go |
| `FES_CSV_OBSERVABILITY` | `true` | Whether local CSV observability files are written (llm_traces / llm_calls / tool_calls) — local-only; the rows feed model comparison + thumbs feedback. Mutations audit log is always on |
| `LANGSMITH_TRACING` | `false` | Master switch for the LangSmith trace tree (root `agent_turn` + llm/tool children) |
| `LANGSMITH_API_KEY` | — | LangSmith API key (must be in the same workspace as the project) |
| `LANGSMITH_PROJECT` | `default` | LangSmith project traces land in |
| `LLM_PLANNING_HISTORY_TURNS` | `5` | Prior conversation turns sent to the planner (0 = latest message only) |
| `FES_LOG_LEVEL` | `INFO` | Log level across all services, read at startup (so it cannot be raised retroactively). `DEBUG` writes full scrubbed tool payloads to `logs/`, which is customer data at rest when the deployment points at a real environment |
| `FES_UI_IDLE_TIMEOUT_HOURS` | `9` | Streamlit session idle timeout; when exceeded the UI clears `st.session_state` |
| `PYSISENSE_MAX_CONCURRENT_MIGRATIONS` | `1` | Max parallel migrations (single-worker friendly; reduces head-of-line blocking) |
| `PYSISENSE_MAX_CONCURRENT_READ_TOOLS` | `5` | Max parallel read-tool calls while migrations run |
| `PYSISENSE_ALLOW_MUTATIONS` | `true` | Mutation kill switch at the MCP dispatch boundary (`tools_core.py`): `false` blocks ALL mutating tools server-side, regardless of UI approvals or which client is calling |
| `PYSISENSE_SDK_DEBUG` | follows log level | Passed to `SisenseClient.from_connection(debug=...)`; `FES_LOG_LEVEL=DEBUG` → on when unset |
| `MCP_TOOL_NAME_MODE` | `claude` | `claude` (underscores) vs `canonical` (dots) for tool names |
| `PYSISENSE_REGISTRY_PATH` | `config/tools.registry.with_examples.json` | Path to tool registry |
| `FES_TOOL_ALLOWLIST` | `config/allowed_tools.txt` | Hand-edited curated tool surface — only listed tool_ids are exposed to the agent or the MCP server. Missing file = allow all (warns), never deny all |
| `FES_TOOL_EXAMPLES` | `1` | Few-shot examples per tool on the tool-**selection** call: 1 = the vetted `example[0]` (default), 0 = off (prompt byte-identical to pre-flag), 2–3 = uncurated siblings (don't, until curated). Users always see `example[0]` in dialogs/clarifications regardless. ~+35 tokens/tool |

### Who reads what

- **UI** (`frontend/app.py`): `FES_LOG_LEVEL`, `FES_BACKEND_URL`,
  `FES_UI_IDLE_TIMEOUT_HOURS`, `FES_ALLOW_SUMMARIZATION_TOGGLE`.
- **Backend** (`backend/api_server.py`, `backend/agent/`): everything `LLM_*`,
  `AZURE_*`, `DATABRICKS_*`, `ALLOW_SUMMARIZATION`, `FES_AGENT_ENGINE`,
  `FES_TOOL_EXAMPLES`, the `FES_MAX_*`/`FES_VERIFY_*` loop budgets, LangSmith and
  CSV switches.
- **Backend → MCP client** (`backend/agent/mcp_client.py`):
  `PYSISENSE_MCP_HTTP_URL`, `PYSISENSE_MCP_HTTP_TIMEOUT`.
- **MCP server** (`mcp_server/tools_core.py`, `mcp_server/server.py`):
  `PYSISENSE_REGISTRY_PATH`, `FES_TOOL_ALLOWLIST`, `PYSISENSE_SDK_DEBUG`, the
  concurrency caps, `PYSISENSE_ALLOW_MUTATIONS`.

**LLM config is built at import time** (`LLM_CONFIG = _build_llm_config()`).
Changing env vars after startup has no effect without a restart.

### Sisense configuration — entered in the UI, not in `.env`

- **Chat with deployment:** Sisense domain (base URL), API token, verify-SSL flag —
  or a username and password, which the UI exchanges for a token.
- **Migrate between deployments:** source domain + token (+ SSL flag), and the
  same for the target.

These are supplied via the Streamlit forms, used to build `SisenseClient`
instances inside the MCP tool server, and are not persisted. A bare domain like
`mycompany.sisense.com` is normalized to `https://`.

---

## Production deployment

The production shape is `docker-compose.prod.yml`: Nginx on the single published
port (`:80`), proxying to Streamlit; the backend and MCP server on an internal
network. Nothing in this repo terminates TLS — put a load balancer or proxy in
front. See [security.md](security.md) for the exposure model.

### What the host actually needs

The application code, the tool registry and the allowlist are all **baked into
the images** at build time. A deployment host consumes exactly two things from a
checkout:

```
./nginx/default.conf   →  /etc/nginx/conf.d/default.conf   (mounted)
./logs                 →  /app/logs                          (mounted)
```

plus a `.env` (never committed) and the compose file itself. Everything else in
the repo — `backend/`, `frontend/`, `tests/`, `scripts/`, `config/` — is unused
on the host; it lives in the images.

**Consequence: upgrading does not require `git pull`.** Pull the repo only when a
file the host *reads* changes — a new service, mount or `env_file` in the compose
file, or the nginx config. Never for application code, the registry, or the
allowlist.

### Upgrading to a new release

CD publishes three images to Docker Hub on every merge to `main`
(`hnegi01/fes-ui`, `fes-backend`, `fes-mcp`, tagged `:<version>`, `:latest` and
`:<sha>`), tags the commit, and cuts a GitHub Release. The version comes from
`pyproject.toml`. The images are published; **CD does not deploy them** — that
is a manual step on the host:

```bash
cd ~/fes-assistant

docker image prune -a -f              # first — a full pull has run the disk out before

FES_IMAGE_TAG=2.7.0 docker compose -f docker-compose.prod.yml pull
FES_IMAGE_TAG=2.7.0 docker compose -f docker-compose.prod.yml up -d
```

Pin `FES_IMAGE_TAG` explicitly. The compose file carries a default for it, but
the explicit tag is what makes the upgrade correct regardless of what the
checkout says.

**Do not add `--remove-orphans`.** Compose warns about containers it does not
manage (other stacks sharing the host, e.g. `fieldnotes`); that flag would delete
them.

Verify — the MCP server is not published to the host in prod, so query it from
inside the network:

```bash
docker exec fes-backend python -c "import urllib.request,json;print(json.load(urllib.request.urlopen('http://fes-mcp:8002/health'))['tools'])"
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
```

The tool count is the tell that the new version landed — compare it with the
count the release commit or PR mentions (a manual practice, not something CD
computes). Then in the UI, "What can I ask?" reports the exposed operation count.

### If you do pull the repo on the host

`nginx/default.conf` on a host commonly carries routes for other stacks sharing
the same Nginx (an OAuth/MCP block for a separate server, say). A `git pull` that
touches that file will refuse with "local changes would be overwritten" — git
fails safe, it never clobbers. Keep the host's copy:

```bash
cp nginx/default.conf /tmp/nginx-prod.conf
git checkout -- nginx/default.conf
git pull
cp /tmp/nginx-prod.conf nginx/default.conf
docker exec fes-nginx nginx -t          # confirm it still parses before anything restarts
```

Prefer `cp` over `git stash pop` here: both sides usually edit the same region,
and stash would hand you a conflict for no reason. Better still, commit the host's
routes so the repo stops misrepresenting production. **Never `git clean -fd`** on
a host — it deletes untracked directories, which is where other stacks live.

### Release ritual (for maintainers)

1. Bump `version` in `pyproject.toml` **and** the three `${FES_IMAGE_TAG:-…}`
   defaults in `docker-compose.prod.yml`, in the same release commit. (2.5.0
   bumped only pyproject; an `up -d` without the env var would have pulled the
   previous version and reported success.)
2. Push `dev`, open a PR to `main`, wait for CI (lint, unit, three Docker builds).
3. Merge. CD builds, pushes, tags and releases. Re-running CD on the same version
   is safe: the release job is guarded by `tag_exists == 'false'`.
4. Upgrade the host as above.

### Reverse proxies and SSE

- The shipped [`nginx/default.conf`](../nginx/default.conf) does **not** need
  `proxy_buffering off`: browser↔UI traffic is a Streamlit websocket, not SSE
  (the SSE hop is UI↔backend, inside the compose network, and never crosses
  Nginx). It does set a long `proxy_read_timeout` for the websocket.
- If you put a different proxy in front of the **backend** (port 8001) so that
  SSE does cross it, disable proxy buffering and raise idle timeouts for
  long-lived responses there.

### Secrets

Secrets like `AZURE_OPENAI_API_KEY` or `DATABRICKS_TOKEN` should be provided via
a secure channel (SSM Parameter Store, Secrets Manager, etc.), never baked into
images. An example non-secret env script is [`config_prod.sh`](../config_prod.sh).

### Single-worker constraints

- **The MCP server must run with 1 worker.** MCP Streamable HTTP sessions are
  stateful; multiple workers would route the same `Mcp-Session-Id` to different
  processes, breaking `initialize`/cancel state. Concurrency is handled via async
  + semaphores within the single worker.
- **The backend must stay single-worker too.** Session and approval state
  (`SESSION_POOL`, pending approvals, paused turns) are in-process.
- The MCP server reads the allowlist **once at import**; the backend re-reads it
  on mtime change. After an allowlist edit, restart the MCP container.

---

## Logging

Log files are written under `logs/` (git-ignored, and excluded from the images —
a deployment creates its own). Application logs rotate daily and keep 7 days; the
CSVs roll at 50 MB and keep 5 rolls; the two audit logs never rotate (audit ≠
observability). Nothing in `logs/` grows without bound.

Sensitive values such as tokens are scrubbed before being written.

The shipped default is `FES_LOG_LEVEL=INFO`, which records what ran — every tool
call, ok/failed, timings, the mutation audit — without writing Sisense result rows
to disk. `DEBUG` adds full (secret-scrubbed) tool payloads and prompts, which is
the right trade on a machine you control but means your Sisense data sits in
`logs/` for 7 days. Worth deciding before you need it: the level is read at
startup, so it cannot be raised *after* something odd happened.

### What each file is

| File | Written by | What's in it |
|---|---|---|
| `llm_agent.log` | agent loop | plans, tool selections, decisions — the main debugging read |
| `llm_routing.log` / `llm_registry.log` | routing / registry modules | tool-menu navigation; registry + allowlist loading |
| `backend_runtime.log` | session runtime | turns started/ended, session pool, cancellations |
| `backend_api.log` | FastAPI layer | request-level view of `/agent/turn` |
| `mcp_client.log` | backend↔MCP client | MCP session lifecycle, spec progress notifications |
| `server.log` | MCP transport | `tools/call`s and cancellations at the server door |
| `tools_core.log` | tool executor | SDK dispatch, credential routing, results (scrubbed) |
| `pysisense.log` | the PySisense SDK itself | the SDK's own logging (verbose at DEBUG) |
| `app.log` | Streamlit UI | frontend events |
| `mutations.log` / `server_mutations.log` | backend / MCP audit | every executed write, recorded at BOTH enforcement layers on purpose (the MCP one also catches non-backend callers); always on, never rotated |
| `llm_traces.csv` / `llm_calls.csv` / `tool_calls.csv` | observability | one row per turn / LLM call / tool run; on by default (`FES_CSV_OBSERVABILITY=false` turns them off) |
| `feedback.csv` | UI | one row per thumbs up/down a user gives an answer (verdict, optional comment, question, tools); joins the observability CSVs by `trace_id` |
| `*_run.log` (`mcp`/`backend`/`ui`) | your launcher's stdout redirect | uvicorn/streamlit process output; under Docker this is the container log instead |

### Observability destinations

Two destinations, each with its own switch. LangSmith (cloud) is **off by
default**; the local CSVs are **on by default**. The mutations audit log is
always on. Details of the trace tree and the redaction boundary are in
[architecture.md → Observability](architecture.md#observability--the-trace-tree).

| Destination | Switch | What you get |
|---|---|---|
| **LangSmith** (external cloud) | `LANGSMITH_TRACING` (+ `FES_LANGSMITH_LOG_CONTENT` for result data in traces) | Trace tree per turn: root `agent_turn` → llm children (planner/route/plan/decide/verify) + tool children (ok/rows/duration); Threads view groups a session; per-turn cost |
| **Local CSVs** (`logs/`) | `FES_CSV_OBSERVABILITY` | `llm_traces.csv` (per turn), `llm_calls.csv` (per LLM call), `tool_calls.csv` (per tool execution) — grouped by per-turn `trace_id`, no cloud required |
