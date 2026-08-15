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
│   ├── tools.registry.with_examples.json   # Tool metadata (tool_id, schema, mutates, module) — GENERATED from the SDK
│   ├── registry/                           # Same tools as a 3-level tree (index → package → mixin) for routing
│   └── allowed_tools.txt                   # HAND-EDITED curated surface: unlisted tool_ids are never exposed
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

**The turn's tool universe is scoped once, at entry.** `call_llm_with_tools`
re-filters the incoming `tools` by mode (`_tool_matches_mode`) and that list is
what the loop uses — `_select_tools_for_mode` falls back to returning *all*
tools when its filter comes up empty (a broken-registry safety valve), which
would otherwise put chat tools into a migration turn. Historically mode was a
rule each code path had to remember while the loop rebuilt its own menu from the
full 119-tool registry; two paths forgot (later-step routing, and the keyword
fallback's hardcoded tool IDs). Scope it once, then enforce at the execution
choke point — don't re-derive it per call site.

**Migration does not run the chat loop at all.** It has its own path
(`backend/agent/migration_flow.py`), shared by both engines:

```
chat       plan → execute → "what next?" → execute → "what next?" → …
migration  plan (ONE call, all 9 tools) → ONE approval → execute in order
```

The chat loop re-asks after every step because a step's RESULT can change the
plan ("get user X, then list others with that same role" cannot name step 2's
argument until step 1 runs). Nothing in migration works that way: no migration
tool consumes a value another produces, and there are no read tools, so the plan
is fully knowable from the request. A three-asset migration costs **one**
planning call instead of one plan + three selects + two decides.

- **Order is the planner's, from a principle — not a rank table.**
  `MIGRATION_PLANNING_CONTEXT_PROMPT` states *migrate what is referenced before
  what references it*, gives the reference directions to reason from (a user is
  assigned to groups; a dashboard queries a datamodel; shares are granted to
  both), and tells the model to judge each operation **by what it moves, not by
  its name**. A ranked list — in the prompt or in code — needs editing for every
  new migration tool and silently mis-ranks anything it doesn't recognise; a
  principle places a tool nobody has written yet. Wrong order fails QUIETLY
  (`migrate_users` preserves group assignments, so users migrated before their
  groups arrive without them), so the approval dialog lists the exact sequence,
  built in code from the calls that will run — never the model's summary.
- **One approval per request, not per step.** The steps are sequential, not
  dependent: nothing in step 1's result can change whether step 2 is wise, so
  asking per step repeats the same question with no new information. The dialog
  names every operation and its arguments — still explicit consent, gathered
  once. Keyed on the ordered step list (`PLAN_TOOL_ID` + `plan_arguments`), so
  editing or reordering a step re-gates; still single use.
- **Validated before it is proposed** — every planned call is schema-checked up
  front. Approving a plan whose third step cannot run wastes the approval, and
  nothing has been written at plan time.
- **Resume runs the approved plan, never replans** — re-asking the planner can
  produce a different plan from the one that was shown and agreed to. A
  per-step pause left over from the reactive loop (kill switch flipped
  mid-session) is dropped rather than resumed.
- **Stops on failure** and reports ran / failed / not-attempted. Migrating users
  into groups that failed to migrate leaves a half-configured target.
- Both engines share it. `FES_AGENT_ENGINE` models the chat loop's branching; a
  linear sequence has none. Kill switch: `FES_MIGRATION_SINGLE_SHOT=false`
  routes migration back through the reactive loop.

**How migration mode differs inside the loop** (all four enforced in code, both engines):

| | Chat | Migration |
|---|---|---|
| Routing | two-stage L1→L2 navigation | **bypassed** — `_navigate_for_step` hands over the turn's scoped tool list directly, on *every* step, not just the first. Routing exists to keep ~110 tools off the selection call; 9 already is a menu. The L1 index is not mode-aware, so walking it here can pick a chat tool, and chat tools get no credentials in migration mode (`_inject_credentials` sends non-migration tools down `_with_tenant`, and `tenant_config` is empty) |
| Fan-out | independent steps run concurrently | **off** — every migration tool mutates, and the gate is one-at-a-time |
| Planner catalog | non-migration tools | migration tools only (`_capability_catalog`) |
| Context prompt | `CHAT_PLANNING_CONTEXT_PROMPT` | `MIGRATION_PLANNING_CONTEXT_PROMPT`, which carries the **dependency order**: groups → users → datamodels → dashboards. A Sisense invariant (a user cannot join a group that does not exist; a dashboard cannot resolve against an unmigrated datamodel), not a scenario patch — which is why it is allowed in a prompt at all |

Migration mode has **no read tools**, so it cannot resolve a name to an ID
mid-plan. `migrate_dashboard_shares` needs concrete ID lists, so a request that
only names a dashboard dead-ends or clarifies. Adding read tools means also
deciding which environment they read from and plumbing credentials for it.

### 3. The agentic loop — plan → execute → replan (llm_agent.py)

> The whole turn is one reactive loop, `_reactive_loop()` inside
> `call_llm_with_tools()`. See `AGENT_ARCHITECTURE.md` for the full diagram.
> **Two interchangeable engines** run this same contract over the same helpers
> (`FES_AGENT_ENGINE`): `custom` (the loop) and `langgraph`
> (`backend/agent/graph_engine.py` — StateGraph, Send-API fan-out, no
> checkpointer/DB/files). The unit suite is the parity harness: run it with the
> flag flipped.

Industry-standard **plan-and-execute** shape. Named parts (session-invented
terms in parentheses — kept out of docs, still live in some code identifiers):

- **Planner** (`_make_plan` / `_replan`) — sees a compact **capability
  catalog** (every tool as `tool_id: one-line description`, NO schemas — safe
  because it writes prose steps, never emits tool calls). Drafts a
  dependency-ordered plan and tags steps that need an earlier step's result
  with `[needs-prior-result]`.
- **Orchestrator** — the loop (`_reactive_loop`) itself: reads the plan,
  dispatches each step, decides the next move, replans on failure. The planner
  is a step inside it, not the loop.
- **Executor** — one route→select→validate→execute pipeline per step. Routing
  is two-stage hierarchical (L1 package → L2 mixin → ~10 tools) so the
  tool-selecting LLM never sees all 119 tools/schemas (hallucination control).
- **Critic** (`_verify_goal_complete`) — an INDEPENDENT LLM call (maker/checker
  split) that re-reads the whole request against the results before a "done"
  ships. Summarization-ON only (judging goal completion needs the result data).

Per turn:
1. **Plan** — planner drafts the ordered, dependency-tagged plan (shown in
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
3. User approves → UI re-calls `/agent/turn` with `approved_keys` containing the ONE `(tool_id, args_json)` just approved
4. Backend **consumes** the approval (`_consume_approval`) → executes the tool

**Approvals are single use.** The key is `(tool_id, canonical-JSON args)`, and
every gate — sequential, fan-out branch, pending-loop resume, both engines —
goes through `_consume_approval()`, which discards the key as it authorises.
Asking for the identical operation again gates again, whether that repeat comes
later in the same turn (a decide `CONTINUE`, a critic push) or in a later turn.
A dialog that silently stops appearing is worse than no dialog: the user has
learned to expect it. Check-and-discard happens with no `await` between, so
concurrent fan-out branches cannot both claim one approval.

**The dialog discloses in code, not via the LLM** (`_approval_disclosure`), and
discloses only what the tool definition states: which optional settings the
schema declares, which this call left unset, and their enum values (`action`
(skip / overwrite / duplicate)) — the model picks those silently otherwise.
Appended on **every** path including the LLM-failure fallback template, because
a model free to write prose is free to omit the overwrite choice.

It says **nothing about scope or blast radius**. A tool definition does not
record whether an empty target list means "everything", "nothing", or a hard
error; that lives in SDK code the registry never sees, and it differs per tool.
An earlier attempt inferred it from a naming convention and produced a warning
that was confidently wrong ("this will run without a target list" for a call
that raises). When the definition cannot confirm it, say nothing: let the call
run and report the SDK's own error verbatim (`_describe_tool_result`).

Optional params surface in exactly two places, both modes: a clarification
question (inline) and this dialog (as a block). One selection function
(`_optional_specs`) feeds two renderers so the two cannot drift.

**Known gap — preconditions the registry does not carry.** Several SDK methods
`raise ValueError` on argument combinations that JSON Schema `required` cannot
describe: `migrate_dashboards` needs exactly one of `dashboard_names` /
`dashboard_ids` (given neither it raises — it does **not** migrate everything),
`change_ownership` only works with `migrate_share=True`, and
`migrate_dashboard_shares` rejects id lists of unequal length. Every selector is
optional in the generated schema, so such a call passes validation and fails
inside the SDK with the SDK's own error rather than a clarifying question.

Do **not** paper over this by hand-writing the rules into the registry or a code
table. The registry is generated from the SDK; invented entries are data no
rebuild reproduces and no reader can trace back to a source. If these should be
enforced, the constraints have to come from the SDK itself — a machine-readable
declaration on the methods, or generator logic that derives them — so a rebuild
keeps them true. Until then the agent must not assert what a call will do when
it cannot know.

Mid-loop: if a mutation is reached partway through a multi-step turn, the loop
**pauses** (`LAST_PENDING_LOOP` → `SessionEntry.pending_loop`); the approval
turn resumes from the paused step instead of re-planning (Option A). Two
mutations in one plan therefore produce two sequential dialogs, never one
combined. In fan-out, mutating branches never execute concurrently — they
**defer to the sequential loop** so the gate is handled one at a time. Mutations
logged to `logs/mutations.log` (audit trail).

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

**The registry is generated, the surface is curated.** `scripts/01_build_registry_from_sdk.py`
introspects the PySisense SDK, so a refresh can add methods that should never
reach a user. `config/allowed_tools.txt` is the hand-edited gate: **only tool_ids
listed there are exposed**, so new SDK methods stay invisible until someone adds
a line. Enforced in three places reading the same file — the agent registry +
planner catalog (`_registry.py::allowed_tool_ids`), the tool menu the selection
LLM sees (`_routing.py::_load_mixin_tools`), and `TOOLS_BY_ID` in
`mcp_server/tools_core.py` (the dispatch boundary — enforced independently so a
delisted tool is unreachable even from a non-backend MCP client). A **missing**
file means allow-all with a warning, never deny-all. The backend re-reads on
mtime change; the MCP server reads once at import. Audit drift after a rebuild
with `scripts/04_generate_tool_allowlist.py`.

**Few-shot examples — `example[0]` is dual-purpose.** Every tool carries
`user_query → arguments` examples. `example[0]` serves two consumers: it is
shown to **users** (approval dialogs and clarification questions render it via
`_example_hint` as *"For example, you could ask: …"* — always, no flag), and to
the **model** when `FES_TOOL_EXAMPLES` ≥ 1 appends examples to tool descriptions
on the **tool-selection call only** — the planner writes prose steps and never
emits arguments. `example[0]` is therefore curated to a double bar (pass
2026-08-14): an **imperative command**, never a question (it models what the
user should type next), and every value its arguments set — identities AND
numbers — is spoken in its query, which makes it teach *extraction* rather than
*invention*, reinforcing `PLANNING_SYSTEM_PROMPT`'s no-placeholder rule.
`examples[1..2]` are uncurated and question-phrased; they reach only the model
at flag 2–3 — don't raise past 1 without curating them the same way.
`tests/unit/test_tool_examples.py` fails if any property regresses; script 02
preserves existing examples on rebuild. A/B any flag change against the eval
battery — more examples is not automatically better.

**Internal params never reach the model.** `INTERNAL_PARAMS` (`_routing.py`) is
the set of signature params no caller can supply — currently `emit`, the SDK's
progress callback, which the MCP server injects itself and drops if a client
sends one. Because the registry is generated by introspecting the SDK, these
leak in on every rebuild, so they are removed at three levels: the generator
skips them (`scripts/01`), the shipped data carries none, and
`planner_schema()`/`_format_tool_examples()` strip them at the boundary anyway.
Showing the model a slot it cannot fill just invites it to invent a value —
and an example that demonstrates filling it beats any rule forbidding it.
Guarded by `tests/unit/test_internal_params.py`.

### 8. The recovery ladder

Three recovery mechanisms at increasing altitude — each fires only when the
cheaper one below can't help. **Backtrack fixes the step, replan fixes the
strategy, the critic fixes completeness.**

| Mechanism | Granularity | Trigger | Changes | Who | Budget |
|---|---|---|---|---|---|
| **Backtrack** | one step | routing/selection miss (no tool) | same op, wider tool menu (whole package) | code | 1 retry/step |
| **Replan** | request's remaining plan (triggered by a step) | step result contradicts the plan, or dead end | new approach — planner rewrites what's left | LLM | `FES_MAX_REPLANS` |
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

**Failure reasons are the one exception** (decided 2026-08-08). A failed step
contributes `error` on top of `{tool, ok, count}`; a successful one never does.
Without it the loop is blind exactly when it needs to think — a failed create left
the decide call with `ok: false` alone and it invented a cause. A recovery
reasoned from a guess is worse than one reasoned from the truth, and the
alternative (a code table mapping failures to approved labels) replaces the
agent's judgement with our enumeration of what can go wrong. Errors normally
restate what the user already typed, so they rarely add anything the model has not
seen; when they don't, that residual exposure is documented in README.md
("Security & data handling") rather than hidden.
`tests/unit/test_summarization_boundary.py` pins the scope — reason on failure,
never a payload on success.

### 10. Observability — two destinations, each with its own switch

Both **off by default**; enable what you need. Enforced in code (`_tracing.py`,
`_config.py`), never LLM trust. The mutations audit log is always on (audit ≠
observability).

| Destination | Switch | What you get |
|---|---|---|
| **LangSmith** (external cloud) | `LANGSMITH_TRACING` (+ `FES_LANGSMITH_LOG_CONTENT` for result data in traces) | Trace tree per turn: root `agent_turn` → llm children (planner/route/plan/decide/verify) + tool children (ok/rows/duration); Threads view groups a session; per-turn cost |
| **Local CSVs** (`logs/`) | `FES_CSV_OBSERVABILITY` | `llm_traces.csv` (per turn), `llm_calls.csv` (per LLM call), `tool_calls.csv` (per tool execution) — grouped by per-turn `trace_id`, no cloud required |

Hierarchy mapping (LangSmith ↔ app): Thread = UI session (`SESSION_POOL`),
Trace = one `/agent/turn` prompt (root run `agent_turn`), Runs = the loop's LLM
calls + tool executions. Details: `AGENT_ARCHITECTURE.md` → "Observability".

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
- `_make_plan()` / `_replan()` — the planner (drafts/revises the plan from the capability catalog)
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
cases, not prompt patches).

**LLM providers:** Azure OpenAI (with AWS Secrets Manager fallback) or Databricks Model Serving. Selected by `LLM_PROVIDER` env var. Config built once at import time into `LLM_CONFIG` frozen dataclass. All calls flow through one choke point (`call_llm_raw` → **LiteLLM SDK**) — a gateway-as-a-library in-process; no standalone gateway service (the LiteLLM Proxy would be a drop-in `api_base` change if centralized keys/budgets/rate-limits were ever needed).

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
| `FES_MAX_REPLANS` | `1` | How many times per turn the planner may revise the plan after a failed approach (0 = off) |
| `FES_MAX_PARALLEL_STEPS` | `3` | How many independent plan steps may execute concurrently (1 = off); mutations always sequential |
| `FES_MIGRATION_SINGLE_SHOT` | `true` | Migration turns plan every step in ONE call, sort by dependency in code, execute in sequence (`migration_flow.py`). `false` routes migration through the reactive loop — a kill switch, not a mode |
| `FES_MIGRATION_COMPLETENESS_CHECK` | `false` | Opt-in second LLM call that checks a migration plan for omitted asset kinds (+1 re-plan if any). Off because the approval dialog's numbered step list is the human check; turn on for unattended/API use |
| `FES_AGENT_ENGINE` | `langgraph` | Turn harness: `langgraph` (StateGraph over shared helpers; in-memory, no checkpointer — default since 2026-08-15, after carrying the full live M-battery) or `custom` (the hand-rolled loop, kept as the dependency-free kill switch until one langgraph upgrade + further live write-path use pass) |
| `FES_LANGSMITH_LOG_CONTENT` | `false` | Whether result data may appear in LangSmith traces (independent of summarization) — prompts shown, only data-bearing parts redacted; tool result payloads never go |
| `FES_CSV_OBSERVABILITY` | `false` | Whether local CSV observability files are written (llm_traces / llm_calls / tool_calls); mutations audit log is always on |
| `LANGSMITH_TRACING` | `false` | Master switch for the LangSmith trace tree (root agent_turn + llm/tool children) |
| `LANGSMITH_API_KEY` | — | LangSmith API key (must be in the same workspace as the project) |
| `LANGSMITH_PROJECT` | `default` | LangSmith project traces land in |
| `LLM_PLANNING_HISTORY_TURNS` | `5` | Prior conversation turns sent to the planner (0 = latest message only) |
| `FES_LOG_LEVEL` | `INFO` | Log level across all services |
| `FES_UI_IDLE_TIMEOUT_HOURS` | `9` | Streamlit session idle timeout |
| `PYSISENSE_MAX_CONCURRENT_MIGRATIONS` | `3` | Max parallel migrations |
| `PYSISENSE_MAX_CONCURRENT_READ_TOOLS` | `5` | Max parallel read-tool calls |
| `MCP_TOOL_NAME_MODE` | `claude` | `claude` (underscores) vs `canonical` (dots) for tool names |
| `PYSISENSE_REGISTRY_PATH` | `config/tools.registry.with_examples.json` | Path to tool registry |
| `FES_TOOL_ALLOWLIST` | `config/allowed_tools.txt` | Hand-edited curated tool surface — only listed tool_ids are exposed to the agent or the MCP server. Missing file = allow all (warns), never deny all |
| `FES_TOOL_EXAMPLES` | `0` | Few-shot examples per tool on the tool-**selection** call: 0 = none (prompt byte-identical to pre-flag), 1 = the vetted `example[0]` (local `.env` runs at 1 since 2026-08-14), 2–3 = uncurated siblings. Model-facing only — users always see `example[0]` in dialogs/clarifications regardless. ~+35 tokens/tool |

---

## Local Development

```bash
# Preferred: reproducible env from the lock file (Python 3.11 pinned)
uv sync            # creates .venv from uv.lock
uv run pytest tests/unit -q

# Or classic pip flow (ranges, not exact pins)
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

# Eval batteries — regression prompts, one file per mode. Live + creds.
pytest tests/integration -m eval -v                          # both
pytest tests/integration/test_evals_planner.py -m eval -v    # chat
pytest tests/integration/test_evals_migration.py -m eval -v  # migration
```

- **Integration/eval are local-only and never in GitHub Actions** — we never put
  LLM or Sisense secrets in CI (firm policy). Creds live in
  `tests/integration/integration_config.yaml` (gitignored, real token).
- **Eval battery = anti-whack-a-mole**: a prompt that once misbehaved becomes an
  `EVAL_CASES` entry, not a scenario-specific prompt rule.
- **One battery per mode, each with its own harness.** They assert on different
  evidence: chat cases on which tools *executed*, migration cases on which tool
  was *gated* — every migration tool mutates, so a migration turn stops at the
  approval dialog and executes nothing. Migration cases send no `approved_keys`
  (enforced by an assertion in the file, not per case), so they never write to a
  real target. Don't merge the two harnesses; sharing one bends the chat battery
  out of shape for evidence it never produces.
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
