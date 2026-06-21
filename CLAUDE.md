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
│       ├── llm_agent.py              # LLM orchestration: plan → execute → summarize (~1 133 lines)
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

### 3. Three-Step LLM Orchestration (llm_agent.py)

Every turn runs this pipeline inside `call_llm_with_tools()`:

```
Step 1 — Planning
  Messages sent to LLM: [system(PLANNING_SYSTEM_PROMPT), system(mode_context), latest_user_message]
  LLM returns: tool_call (tool_id + args) OR plain text if no tool matched

Step 2 — Execution (via MCP)
  If tool is mutating AND not in approved_mutations → return pending_confirmation (no execution)
  Otherwise → POST /mcp/ tools/call → get result → shrink result for LLM (_shrink_for_llm)

Step 3 — Summarization (optional)
  If ALLOW_SUMMARIZATION=false (env) → return "I ran the tool. Summarization disabled."
  Otherwise → second LLM call with tool result → natural language summary
```

**Critical gap**: Step 1 sends ONLY the `latest_user_message` to the LLM, not the conversation history. This means follow-up messages ("xyz datamodel" after "which datamodel?") arrive as standalone planning calls with no prior context.

### 4. Mutation Approval Flow (two-phase)

1. Planning picks a mutating tool → returns `pending_confirmation` dict inside `LAST_TOOL_RESULT`
2. UI stores this and renders an approval dialog
3. User approves → UI re-calls `/agent/turn` with `approved_keys` containing `(tool_id, args_json)`
4. Backend checks `approved_mutations` set → executes the tool

Mutations are logged separately to `logs/mutations.log` (audit trail).

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

**Globals:**
- `TOOL_REGISTRY: Dict[str, dict]` — loaded from JSON registry
- `LAST_TOOL_RESULT: Optional[dict]` — set after each tool call; API layer reads this

**Key functions:**
- `call_llm_with_tools()` — main orchestration (plan → execute → summarize)
- `call_llm_raw()` — raw LLM HTTP call with retry logic
- `load_tools_for_llm()` — loads registry → OpenAI tool format
- `_shrink_for_llm()` — generic payload shrinker before summarization
- `_fallback_direct_tool()` — keyword-based fallback if planning LLM call fails

**Prompts:**
- `PLANNING_SYSTEM_PROMPT` — instructs LLM to only select a tool + args, never summarize
- `SUMMARY_SYSTEM_PROMPT_CHAT/MIGRATION` — instructs LLM to summarize tool result

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
| `ALLOW_SUMMARIZATION` | `true` | Hard kill-switch for sending tool results to LLM |
| `FES_ALLOW_SUMMARIZATION_TOGGLE` | `true` | Whether UI checkbox is shown |
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
