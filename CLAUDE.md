# FES Assistant — Codebase Guide

## What This Is

A production-style AI assistant for Sisense administration. Users type natural language ("migrate all dashboards from staging to prod", "show me all users in the Sales group") and the agent selects the right PySisense SDK tool, executes it, and returns results or a summary.

Three separate processes communicate over HTTP:

```
Browser (Streamlit)
  └── POST /agent/turn ──▶ FastAPI Backend (port 8001)
                               └── POST /mcp/ (JSON-RPC) ──▶ MCP HTTP Server (port 8002)
                                                                   └── PySisense SDK ──▶ Sisense API
```

---

## Repository Layout

```
fes-assistant/
├── frontend/
│   └── app.py                        # Streamlit UI (1 300 lines)
├── backend/
│   ├── api_server.py                 # FastAPI routes (~487 lines)
│   ├── runtime.py                    # Session pool + cancellation (~580 lines)
│   └── agent/
│       ├── llm_agent.py              # Agentic loop: plan → execute → replan (+ _config/_prompts/_registry/_routing)
│       └── mcp_client.py             # JSON-RPC HTTP client for MCP server (~790 lines)
├── mcp_server/
│   ├── server.py                     # Starlette MCP HTTP server (~866 lines)
│   └── tools_core.py                 # Registry loading + SDK dispatch (~981 lines)
├── config/
│   └── tools.registry.with_examples.json   # Tool metadata (tool_id, schema, mutates, module)
├── docker-compose.yml                # Dev: 3 containers
├── docker-compose.prod.yml           # Prod: multi-worker backend + Nginx
├── Dockerfile.{backend,mcp,ui}
└── .env.example                      # All environment variables documented
```

---

## Key Concepts

### 1. Session Management

Each browser tab gets a UUID (`session_id`) generated once and stored in `st.session_state`. It is sent with every `/agent/turn` request.

The backend maintains `SESSION_POOL: Dict[str, SessionEntry]` — a long-lived `McpClient` per session. Clients are reused across turns and replaced only when:
- idle > 9 hours (`SESSION_IDLE_TIMEOUT`)
- Sisense connection config changed (domain/token)

### 2. Mode Selection

Two logical modes, selected in the UI sidebar:
- **Chat**: single Sisense deployment — non-migration tools only
- **Migration**: source + target deployments — migration tools only

Mode is determined by `_select_tools_for_mode(mode)` in `api_server.py`, which filters the tool registry by `meta.module == "migration"`.

### 3. The agentic loop — plan → execute → replan (llm_agent.py)

> This replaced the old fixed "plan → execute → summarize" three-step pipeline
> in Step 8. The whole turn is now one reactive loop, `_reactive_loop()` inside
> `call_llm_with_tools()`. See `AGENT_ARCHITECTURE.md` for the full diagram and
> the artifact for the visual.

Industry-standard **plan-and-execute** shape. Named parts (session-invented
terms in parentheses — kept out of docs, still live in some code identifiers):

- **Orchestrator** (`_make_plan` / `_replan`) — sees a compact **capability
  catalog** (every tool as `tool_id: one-line description`, NO schemas — safe
  because it writes prose steps, never emits tool calls). Drafts a
  dependency-ordered plan and tags steps that need an earlier step's result
  with `[needs-prior-result]`.
- **Executor** — one route→select→validate→execute pipeline per step. Routing
  is two-stage hierarchical (L1 package → L2 mixin → ~10 tools) so the
  tool-selecting LLM never sees all 119 tools/schemas (hallucination control).
- **Critic** (`_verify_goal_complete`) — an INDEPENDENT LLM call (maker/checker
  split) that re-reads the whole request against the results before a "done"
  ships. Summarization-ON only (judging goal completion needs the result data).

Per turn:
1. **Plan** — orchestrator drafts the ordered, dependency-tagged plan (shown in
   the UI).
2. **Fan-out** — independent (untagged) steps run **concurrently**
   (`_execute_branch` + `asyncio.gather`, width `FES_MAX_PARALLEL_STEPS`);
   results join into the shared transcript in plan order.
3. **Sequential tail** — dependent steps run one at a time (they need joined
   results); a decide call picks each next move.
4. **Verify** — per step: schema check + `ok` flag (code, no LLM). At "done":
   the critic (LLM).
5. **Recover** — the recovery ladder (see §8).

Multi-turn context: the planner receives the last `LLM_PLANNING_HISTORY_TURNS`
turns via `_build_planning_history()`, so follow-ups ("xyz datamodel" after a
clarifying question) resolve with prior context.

### 4. Mutation Approval Flow (two-phase, human-in-the-loop)

1. A mutating tool is selected → returns `pending_confirmation` dict inside `LAST_TOOL_RESULT` (with a plain-English `reason`)
2. UI stores this and renders an approval dialog
3. User approves → UI re-calls `/agent/turn` with `approved_keys` containing `(tool_id, args_json)`
4. Backend checks `approved_mutations` set → executes the tool

Mid-loop: if a mutation is reached partway through a multi-step turn, the loop
**pauses** (`LAST_PENDING_LOOP` → `SessionEntry.pending_loop`); the approval
turn resumes from the paused step instead of re-planning (Option A). In fan-out,
mutating branches never execute concurrently — they **defer to the sequential
loop** so the gate is handled one at a time. Mutations logged to
`logs/mutations.log` (audit trail).

### 5. Progress Streaming (SSE)

Used for long-running migrations. The flow:

```
UI sends Accept: text/event-stream
  ↓
api_server SSE generator creates asyncio.Queue
  ↓
runtime._progress_context() binds a ContextVar callback for this turn
  ↓
MCP client receives SSE notifications from MCP server mid-response
  ↓
mcp_client calls runtime.publish_progress(event)
  ↓
publish_progress() reads ContextVar → calls the queued SSE callback
  ↓
SSE generator yields frames to UI
  ↓
UI renders progress in sidebar (step name, status, count)
```

ContextVar isolation ensures concurrent sessions don't mix progress events.

### 6. Cancellation

Multi-layer best-effort:
1. UI detects disconnect / SSE drop → Backend's `_event_generator()` detects `request.is_disconnected()`
2. Backend calls `runtime.cancel_active_turn(session_id)`
3. Runtime calls `McpClient.cancel_session()` → `POST /mcp/cancel` with `Mcp-Session-Id` header (shielded from parent cancellation)
4. MCP server sets a cancel flag per session — tool's `emit()` callback checks flag each step → raises `CancelledError`
5. Back in backend: `asyncio.Task.cancel()` on the active turn task

### 7. Tool Registry

`config/tools.registry.with_examples.json` is a JSON array. Each entry:
```json
{
  "tool_id": "datamodel.get_all_datamodel",
  "module": "datamodel",          // "migration" | "datamodel" | "access" | ...
  "mutates": false,               // true = requires UI approval
  "description": "...",
  "parameters": { /* JSON Schema */ }
}
```

The backend uses mtime caching to avoid re-reading this file on every request.

### 8. The recovery ladder

Three recovery mechanisms at increasing altitude — each fires only when the
cheaper one below can't help. **Backtrack fixes the step, replan fixes the
strategy, the critic fixes completeness.**

| Mechanism | Granularity | Trigger | Changes | Who | Budget |
|---|---|---|---|---|---|
| **Backtrack** | one step | routing/selection miss (no tool) | same op, wider tool menu (whole package) | code | 1 retry/step |
| **Replan** | request's remaining plan (triggered by a step) | step result contradicts the plan, or dead end | new approach — orchestrator rewrites what's left | LLM | `FES_MAX_REPLANS` |
| **Critic INCOMPLETE** | whole request, at "done" | maker declared done but something's missing | pushes +1 step (never rewrites) | LLM | `FES_VERIFY_MAX_RECHECKS` |

There is no separate step-level replan (a step failure re-plans the *remaining*
request) and no standalone request-level replan (the critic only *adds* a step).

### 9. Summarization flag = data visibility, not loop on/off

`allow_summarization` is a **privacy kill-switch over result DATA**, enforced in
code (never LLM trust):

- **On** — result data reaches the LLM. Adaptive chains complete; the critic runs.
- **Off** — only metadata (`{tool, ok, count}`) reaches the LLM. Independent
  multi-step still works; **adaptive/dependent** steps are skipped up front
  (dependency gate — the value they need is unreadable) or stop gracefully
  (`BLOCKED`), and the reply names what was skipped. The critic is off.

API/UI default is `false` when the field is omitted — set it explicitly.

---

## File Reference

### `frontend/app.py`

**Key session_state keys:**
- `session_id` — UUID per tab, persisted in query params
- `messages` — full conversation history `[{role, content}]`
- `pending_confirmation` — pending mutating tool (shown as approval dialog)
- `approved_keys` — list of `(tool_id, args_json)` approved this turn
- `tenant_config` / `migration_config` — Sisense connection creds
- `allow_summarization` — checkbox state

**Key functions:**
- `_call_backend_sse()` — streams `/agent/turn` as SSE, parses events
- `_call_backend_json()` — regular POST, returns JSON
- `_render_tool_result()` — renders result payload as table/JSON in expander
- `_render_pending_confirmation()` — approval dialog for mutations
- `_render_migration_progress()` — sidebar progress for migrations

### `backend/api_server.py`

**Routes:**
- `GET /health` — liveness check
- `GET /tools` — returns tool list + registry metadata (UI uses for counts/display)
- `POST /agent/turn` — main entrypoint; negotiates SSE vs JSON via `Accept` header

**Key logic:**
- `_select_tools_for_mode(mode)` — filters registry by mode
- `_sse_pack(data, event)` — formats SSE frames
- SSE `_event_generator()` — drains asyncio.Queue with 10s keepalive; detects disconnect

### `backend/runtime.py`

**Globals:**
- `SESSION_POOL: Dict[str, SessionEntry]` — MCP client pool
- `SESSION_POOL_LOCK` — asyncio.Lock protecting pool structure
- `_ACTIVE_TURNS: Dict[str, Task]` — tracks one task per session for cancellation
- `_CURRENT_PROGRESS_CB` — ContextVar for per-turn progress callback

**Key functions:**
- `run_turn_once()` — public API: creates task, registers in `_ACTIVE_TURNS`, awaits it
- `_run_turn_once()` — internal: gets/creates MCP client, calls `call_llm_with_tools()`
- `_get_or_create_mcp_client()` — session pool logic (reuse / replace on timeout or config change)
- `cancel_active_turn()` — cancels both MCP server-side and local task
- `publish_progress()` — reads ContextVar, calls callback

### `backend/agent/llm_agent.py`

> Split across sub-modules: `_config.py` (env/logging/tracing), `_prompts.py`
> (all prompt constants), `_registry.py` (registry I/O + shrinkers), `_routing.py`
> (two-stage routing + raw LLM call). `llm_agent.py` orchestrates.

**Globals (module-level, read by the API layer via getattr):**
- `TOOL_REGISTRY: Dict[str, dict]` — loaded from JSON registry
- `LAST_TOOL_RESULT: Optional[dict]` — last tool result (single slot)
- `LAST_STEP_RESULTS: List[dict]` — every step's result this turn (UI shows all)
- `LAST_PENDING_CLARIFICATION` — set when the turn pauses to ask for a missing arg
- `LAST_PENDING_LOOP` — set when a multi-step turn pauses mid-loop for approval

**Key functions:**
- `call_llm_with_tools()` → `_reactive_loop()` — the plan→execute→replan loop
- `_make_plan()` / `_replan()` — orchestrator (capability-catalog planners)
- `_capability_catalog()` — one-liner catalog (name + description, no schemas)
- `_split_dependent_tail()` — partitions plan into independent vs `[needs-prior-result]`
- `_execute_branch()` — one fan-out branch (route→select→validate→execute)
- `_verify_goal_complete()` — the critic (independent goal checker)
- `call_llm_raw()` — raw LLM HTTP call, retry + per-call CSV trace (`label=`)
- `_fallback_direct_tool()` — keyword fallback if planning LLM call fails

**Prompts** (`_prompts.py`): `AGENT_PLAN_SYSTEM_PROMPT`,
`AGENT_REPLAN_SYSTEM_PROMPT`, `AGENT_DECIDE_SYSTEM_PROMPT` (+ `_NODATA` variant,
both with a `REPLAN:` verb), `VERIFY_GOAL_SYSTEM_PROMPT`,
`CLARIFY_QUESTION_SYSTEM_PROMPT`, routing/mode-context prompts. Prompts carry
**only generic strategy** — never scenario-specific rules (failures become eval
cases, not prompt patches). `SUMMARY_SYSTEM_PROMPT_*` and `PLANNING_SYSTEM_PROMPT`
are legacy (decide replaced summarize).

**LLM providers:** Azure OpenAI (with AWS Secrets Manager fallback) or Databricks Model Serving. Selected by `LLM_PROVIDER` env var. Config built once at import time into `LLM_CONFIG` frozen dataclass.

### `backend/agent/mcp_client.py`

Wraps `POST /mcp/` JSON-RPC. Key behaviors:
- On `connect()`: sends `initialize` RPC → gets back `Mcp-Session-Id` header (reused on all subsequent calls)
- On `invoke_tool()`: detects if response will be SSE (streaming tool) vs plain JSON
- For SSE responses: parses `event: notifications/message` frames → calls `publish_progress()`, waits for final `result` frame
- `cancel_session()`: `POST /mcp/cancel` with session id header

Credential injection: before each tool call, merges `tenant_config` (domain/token/ssl) or `migration_config` (source_*/target_*) into the tool arguments.

### `mcp_server/server.py`

Custom Starlette app implementing MCP Streamable HTTP transport (without the official `StreamableHTTPSessionManager` due to compatibility issues).

**Routes:**
- `GET /mcp` — SSE subscribe endpoint (for Claude Desktop probing)
- `POST /mcp` — JSON-RPC handler; for streaming tools returns SSE; for others returns JSON
- `POST /mcp/cancel` — sets per-session cancel flag
- `GET /health` — liveness

**Key state:**
- `_SESSION_CANCEL_FLAGS: Dict[str, asyncio.Event]` — one Event per session
- `_SESSION_CONTEXTS: Dict[str, contextvars.Context]` — context per session

### `mcp_server/tools_core.py`

Loads tool registry → builds SDK client from tool args → dispatches to PySisense SDK method.

- `STREAMING_TOOL_IDS` — set of tool IDs that emit progress notifications
- `_MIGRATION_SEMAPHORE` and `_READ_SEMAPHORE` — concurrency caps
- `emit()` callback passed to SDK migration functions — checks cancel flag, publishes progress via JSON-RPC notification frame

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `databricks` | `azure` or `databricks` |
| `AZURE_OPENAI_ENDPOINT` | — | Azure endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | Model deployment name |
| `AZURE_OPENAI_API_KEY` | — | API key (or via AWS SM) |
| `AZURE_OPENAI_API_STYLE` | `v1` | `v1` (new) or `legacy` (deployment URL) |
| `FES_AZURE_OPENAI_SECRET_ID` | — | AWS Secrets Manager secret name |
| `AWS_REGION` | — | AWS region for Secrets Manager |
| `DATABRICKS_HOST` | — | Databricks workspace URL |
| `DATABRICKS_TOKEN` | — | Databricks PAT |
| `LLM_ENDPOINT` | — | Databricks serving endpoint name |
| `LLM_HTTP_TIMEOUT` | `60` | LLM HTTP call timeout (seconds) |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per LLM call |
| `LLM_TEMPERATURE` | `0.2` | LLM temperature |
| `LLM_MAX_TOOLS` | `80` | Safety cap on tools sent to LLM |
| `PYSISENSE_MCP_HTTP_URL` | `http://localhost:8002` | MCP server URL (from backend) |
| `PYSISENSE_MCP_HTTP_TIMEOUT` | `1800` | MCP HTTP call timeout (30 min for migrations) |
| `FES_BACKEND_URL` | `http://localhost:8001` | Backend URL (from frontend) |
| `ALLOW_SUMMARIZATION` | `true` | Hard kill-switch for sending tool result **data** to LLM (loop still runs on metadata when off) |
| `FES_ALLOW_SUMMARIZATION_TOGGLE` | `true` | Whether UI checkbox is shown |
| `FES_MAX_AGENT_STEPS` | `8` | Hard ceiling on tool-executing steps per agent turn (one SDK call each); cap → partial answer |
| `FES_CLARIFY_MAX_ATTEMPTS` | `2` | Max clarifying questions before the agent gives up and states what it needs |
| `FES_VERIFY_GOAL` | `true` | Independent goal checker (verify #3): re-checks a "done" answer against the request before accepting it |
| `FES_VERIFY_MAX_RECHECKS` | `1` | How many times the goal checker may push the loop to run one more step |
| `FES_MAX_REPLANS` | `1` | How many times per turn the orchestrator may revise the plan after a failed approach (0 = off) |
| `FES_MAX_PARALLEL_STEPS` | `3` | How many independent plan steps may execute concurrently (1 = off); mutations always sequential |
| `FES_LOG_LEVEL` | `INFO` | Log level across all services |
| `FES_UI_IDLE_TIMEOUT_HOURS` | `9` | Streamlit session idle timeout |
| `PYSISENSE_MAX_CONCURRENT_MIGRATIONS` | `3` | Max parallel migrations |
| `PYSISENSE_MAX_CONCURRENT_READ_TOOLS` | `5` | Max parallel read-tool calls |
| `MCP_TOOL_NAME_MODE` | `claude` | `claude` (underscores) vs `canonical` (dots) for tool names |
| `PYSISENSE_USE_DEFAULT_TENANT` | `false` | Use env-defined default Sisense tenant |
| `PYSISENSE_REGISTRY_PATH` | `config/tools.registry.with_examples.json` | Path to tool registry |

---

## Local Development

```bash
# Install deps
pip install -r requirements.txt

# Set env vars
cp .env.example .env
# edit .env with your Sisense creds and LLM config

# Run MCP server
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8002 --workers 1

# Run backend
uvicorn backend.api_server:app --host 0.0.0.0 --port 8001

# Run UI
streamlit run frontend/app.py --server.port 8501
```

> Venv is `venv_pysisense_chatbot`. For the launch/drive/restart recipes, the
> manual `/agent/turn` harness, and gotchas, use the **`run` skill**
> (`.claude/skills/run/SKILL.md`).

## Testing

Markers in `pyproject.toml`. Three tiers:

```bash
# Unit — mocked, fast, no creds. Always run these.
pytest tests/unit -q

# Integration — needs the live stack + real creds. Local only.
pytest tests/integration -m integration -v

# Eval battery — planner-behaviour regression prompts. Live + creds.
pytest tests/integration/test_evals_planner.py -m eval -v
```

- **Integration/eval are local-only and never in GitHub Actions** — we never put
  LLM or Sisense secrets in CI (firm policy). Creds live in
  `tests/integration/integration_config.yaml` (gitignored, real token).
- **Eval battery = anti-whack-a-mole**: a prompt that once misbehaved becomes an
  `EVAL_CASES` entry, not a scenario-specific prompt rule.
- **Mutation tests only ever mutate an asset they created** (create → gate →
  approve → delete that same asset → `finally:` force-delete). See
  `tests/integration/test_mutation_lifecycle.py`. Never touch a pre-existing asset.
- LLM non-determinism → re-run a single failing integration test before calling
  it a regression.

## Docs

- `AGENT_ARCHITECTURE.md` — the living architecture doc (loop, recovery ladder,
  verify, LangGraph mapping) with Mermaid diagrams.
- `.claude/skills/run/SKILL.md` — launch/drive/test the stack.
- `.claude/PROGRESS.md` — build history / changelog.

## Docker

```bash
# Dev
docker compose up --build

# Prod (Nginx + multi-worker backend)
docker compose -f docker-compose.prod.yml up --build
```

Logs are written to `./logs/` (mounted into all containers).

---

## Architecture Constraints

- **MCP server must run with 1 worker**: Multiple workers would route the same `Mcp-Session-Id` to different processes, breaking `initialize`/cancel state. Concurrency is handled via async + semaphores within the single worker.
- **Session ID is browser-tab-scoped**: A single user with two tabs gets two separate MCP clients and separate conversation histories.
- **LAST_TOOL_RESULT is a module-level global**: This is safe in the current single-worker backend model but would break under multiple concurrent requests in a truly parallel (non-async) setup. The async event loop serializes access in practice.
- **LLM config is built at import time**: `LLM_CONFIG = _build_llm_config()` runs at module load. Changing env vars after startup has no effect without restart.
