# FES Assistant — Codebase Guide

## What This Is

A production-style AI assistant for Sisense. Token-scoped, not admin-only: the
agent can do exactly what the supplied Sisense token's permissions allow. Users
type natural language ("migrate all dashboards from staging to prod", "show me
all users in the Sales group") and the agent selects the right PySisense SDK
tool, executes it, and returns results or a summary.

Three separate processes communicate over HTTP:

```
Browser (Streamlit, port 8501)
  └── POST /agent/turn ──▶ FastAPI Backend (port 8001)
                               └── MCP Streamable HTTP · POST /mcp/ ──▶ MCP Server (port 8002)
                                                                            └── PySisense SDK ──▶ Sisense API
```

The deployed shape (EC2, containers, ports, external systems) is the Mermaid
diagram in `README.md`; this is just the hop order.

**This file is the operating rules.** The *why* behind every rule below lives
in `docs/architecture.md`; configuration, deployment and logging in
`docs/operations.md`; the security model in `docs/security.md`; the dev loop
and test tiers in `docs/development.md`. Read those before changing behaviour.

---

## Repository Layout

```
fes-assistant/
├── frontend/
│   └── app.py                        # Streamlit UI
├── backend/
│   ├── api_server.py                 # FastAPI routes
│   ├── runtime.py                    # Session pool + cancellation
│   └── agent/
│       ├── llm_agent.py              # Agentic loop: plan → execute → replan (+ _config/_prompts/_registry/_routing/_tracing)
│       ├── graph_engine.py           # Default engine: the same loop as a LangGraph StateGraph (FES_AGENT_ENGINE)
│       ├── migration_flow.py         # Migration mode: one plan, one approval, execute in order
│       └── mcp_client.py             # MCP client on the official SDK (ClientSession over Streamable HTTP)
├── mcp_server/
│   ├── server.py                     # MCP server on the official SDK (StreamableHTTPSessionManager + lowlevel Server)
│   └── tools_core.py                 # Registry loading + SDK dispatch; allowlist enforced here too
├── config/
│   ├── tools.registry.with_examples.json   # Tool metadata (tool_id, schema, mutates, module) — GENERATED from the SDK
│   ├── registry/                           # Same tools as a 3-level tree (index → package → mixin) for routing
│   └── allowed_tools.txt                   # HAND-EDITED curated surface: unlisted tool_ids are never exposed
├── skills/                           # Sisense-authored procedures the agent plans from — one dir per skill, SKILL.md
├── scripts/                          # Registry generation (01 build · 02 examples · 03 example sync · 04 allowlist audit)
├── docs/                             # architecture · security · operations · development · design/
├── tests/                            # unit (CI, both engines) · integration + eval batteries (live, local only)
├── docker-compose.yml                # Dev: 3 containers
├── docker-compose.prod.yml           # Prod: single-worker backend behind Nginx (MUST stay single-worker — in-process session/approval state)
├── Dockerfile.{backend,mcp,ui}
└── .env.example                      # All environment variables documented — the authoritative list
```

---

## Rules and invariants

Each is enforced in code; each has a longer explanation in `docs/architecture.md`
under the section named.

### Modes and the tool universe
- **Two modes**, selected in the UI: **Chat** (one deployment, non-migration
  tools) and **Migration** (source + target, migration tools only).
- **The turn's tool universe is scoped once, at entry**
  (`call_llm_with_tools` → `_tool_matches_mode`), then enforced at the execution
  choke point. Never re-derive mode per call site — two paths once forgot.
- **Migration does not run the chat loop.** `migration_flow.py`: ONE planning
  call over all migration tools → ONE approval listing the exact ordered
  sequence (built in code from the calls that will run) → execute in order →
  stop on first failure → report ran / failed / not-attempted. Order comes from a
  *principle* in `MIGRATION_PLAN_SYSTEM_PROMPT` (migrate what is referenced
  before what references it), never a rank table. Routing and fan-out are
  bypassed inside it. Resume runs the approved plan, never replans.
  → *Migration mode — the single-shot path*

### The agentic loop
- **Plan → execute → replan**, one SDK call per lap, in `_reactive_loop`. The
  **planner** sees a capability catalog (one-liners, no schemas) and writes
  prose steps tagged `[needs-prior-result]` where a value comes from an earlier
  result. Independent steps **fan out** concurrently; dependent ones run after
  the join. The **critic** (`_verify_goal_complete`) independently re-checks
  "done" — summarization-ON only.
- **Two interchangeable engines** over the same helpers (`FES_AGENT_ENGINE`:
  `langgraph` default, `custom` kill switch). The unit suite is the parity
  harness — run it under both.
- **The tool-selection call sees ~10 tools from one mixin, never the catalog.**
  Two-stage routing (L1 package → L2 mixin) does the narrowing; L1 shows each
  package's blurb, its module names, **and its tool names** — prose alone
  competes badly when two packages honestly describe the same word.
- **Recovery ladder:** backtrack (code, wider menu, 1/step) → replan (LLM,
  rewrites what's left, `FES_MAX_REPLANS`) → critic INCOMPLETE (LLM, +1 step,
  `FES_VERIFY_MAX_RECHECKS`). No step-level replan, no standalone request-level
  replan. → *The recovery ladder*
- **Every exit returns readable text.** `FES_MAX_AGENT_STEPS` is a hard ceiling.

### Mutations
- **Two-phase, human-in-the-loop.** A mutating tool returns
  `pending_confirmation`; the UI renders a dialog; approval comes back as
  `approved_keys` and is **consumed** by `_consume_approval` — single use,
  keyed on `(tool_id, canonical-JSON args)`. Asking again gates again. Every gate
  (sequential, fan-out branch, resume, both engines) goes through that one
  function. Mid-loop, the turn **pauses** (`pending_loop`) and resumes from the
  paused step — never re-plans.
- **The dialog discloses in code, not via the LLM** (`_approval_disclosure`):
  which optional settings the schema declares, which this call left unset, their
  enum values. It says **nothing about scope or blast radius** — the registry
  cannot know whether an empty target list means "everything" or "error", and
  a naming-convention guess was once confidently wrong. Let the call run and
  report the SDK's error verbatim.
- **Known gap:** SDK preconditions JSON Schema cannot express (`migrate_dashboards`
  needs exactly one of names/ids) pass validation and fail inside the SDK. Do
  **not** hand-write them into the registry — they must come from the SDK
  (`Literal`, declared constraints) so a rebuild keeps them true.
  → *Mutation approval*
- `PYSISENSE_ALLOW_MUTATIONS=false` blocks all writes at the MCP dispatch
  boundary, regardless of client. Mutations are audited to `logs/mutations.log`
  and `logs/server_mutations.log` — both layers, on purpose.

### Summarization = data visibility, not loop on/off
- `allow_summarization` is a **privacy kill-switch over result DATA**, enforced
  at one point (`_transcript_step` → `_metadata_record`). **Off:** only
  `{tool, ok, count}` reaches the LLM; dependent steps are skipped up front or
  stop with `BLOCKED`; the reply is built in code. **On:** shrunk result data
  reaches the LLM; chains complete; the critic runs. API/UI default is `false`
  when omitted — set it explicitly.
- **Failure reasons are the one exception:** a failed step contributes `error`
  on top of the metadata; a successful one never does. Don't trim or filter that
  string here — its aperture is the SDK's call. Pinned by
  `tests/unit/test_summarization_boundary.py`.
- **Result data on screen is always allowed** — the switch governs the LLM's
  view, not the user's. Tables render in both modes; `display_hints` is the
  screen-only channel for option names in clarifications.
  → *The summarization switch*, and `docs/security.md`

### The tool registry and the curated surface
- **The registry is GENERATED; the surface is CURATED.** Never hand-edit
  `config/tools.registry*.json` or `config/registry/`. `config/allowed_tools.txt`
  is the gate, enforced in **three** places reading the same file:
  `_registry.py::allowed_tool_ids` (agent + planner catalog),
  `_routing.py::_load_mixin_tools` (selection menu), and `TOOLS_BY_ID` in
  `mcp_server/tools_core.py` (dispatch — independent, so a delisted tool is
  unreachable even from a non-backend client). Missing file = allow-all with a
  warning, never deny-all. Backend re-reads on mtime; MCP reads once at import.
- **One home per kind of curated fact** — the litmus:
  - code applies it deterministically → `SCHEMA_RULES` in
    `scripts/01_build_registry_from_sdk.py` (enums, x-aliases, rich schemas,
    `x-options-tool` lookups, `x-followup` nudges). Guarded by
    `TestSchemaRulesDrift` and `test_option_enrichment.py`.
  - the model must reason from it → a scoped prompt in `_prompts.py`,
    **invariants only** — never rank tables, never scenario patches (failures
    become eval cases).
  - it is about a tool existing at all → the allowlist. Don't overload it.
  - SDK truths (enums, preconditions) should migrate **upstream** over time so a
    rebuild derives them.
- **`example[0]` is dual-purpose and curated to a double bar:** an imperative
  command (users see it as "you could ask…"), and every value its arguments set
  is spoken in its query — it teaches *extraction*, never *invention*.
  `tests/unit/test_tool_examples.py` fails on any regression, including invented
  schema references under any key name. `FES_TOOL_EXAMPLES` > 1 shows uncurated
  siblings to the model — don't, until curated.
- **Internal params never reach the model** (`INTERNAL_PARAMS`: `emit`).
  Stripped at the generator, the shipped data, and the boundary.
  → *The tool registry and the curated surface*

### Skills *(design: `docs/design/skills.md` — built through §13 step 3)*
- A skill is a Sisense-authored procedure under `skills/<name>/SKILL.md`.
  `_skills.py` loads them (mtime-cached, chat mode only; a file that fails to
  parse or names a tool outside the registry/allowlist is **excluded and logged
  at ERROR**). The planner always sees a one-line index; replying
  `SKILL: <name>` triggers ONE second pass with the body and the skill's tool
  schemas (`SKILL_PLAN_SYSTEM_PROMPT`) that returns a **typed plan** — JSON with
  tool ids, literal `args`, `args_from` references into earlier results,
  `for_each` loops and `when` conditions. With no skills on disk the planner
  prompt is byte-identical to before — `tests/unit/test_skills.py` pins that.
- **The hand-off rides the per-turn output slot** (`_take_skill_handoff`), not
  `_make_plan`'s return value, so the many tests that patch `_make_plan` keep
  working and both engines pick it up at the same point. `skill_flow.run`
  then owns the turn.
- **`skill_flow.py`** validates the plan before showing it (tools ⊆ the skill's
  `tools`, refs point backwards, literal args schema-valid, declared guardrails
  hold structurally), gates ONCE on `skill.plan` keyed to the canonical plan,
  and executes in code: references resolved at run time, loops expanded from
  live results, `when` evaluated, guardrails re-checked with values, every
  result checkpointed, and on the first failure the skill's declared
  **compensations** run in reverse (a copy-swap is skipped when the copy is
  deleted; a failed compensation stops the unwind loudly). Data never passes
  through the model, so a skill run is identical with summarization ON or OFF.
- Frontmatter `tools` is an allowlist within the allowlist — how "never change
  ownership" becomes impossible rather than discouraged. `requires_role` is a
  courtesy pre-check via `get_my_user`: a user below it gets the read steps and
  a handoff message, never a plan whose first write would fail. Skills are
  product content: never customer config, never a prompt patch.
- Resume runs exactly the approved plan and refuses if the skill's version
  changed underneath it. The response's `skill` field names the procedure
  ({name, version}) — set by `skill_flow`, never by the planner, so a skill
  that was named and then abandoned is not claimed. The UI renders the
  code-built outcome line and loop position per step. Still to build (design
  §13): live step expanders, a Stop button, durable approval.

### Transport and streaming
- **MCP server: 1 worker, always.** Session/cancel state is in-process.
- Both sides of the backend↔MCP hop are the **official MCP SDK**. Credentials
  are injected per call — never from env; missing = loud error.
- **Two kinds of progress event ride one SSE hop (backend → UI):**
  `agent_progress` published by the loop (planning / executing / deciding /
  verifying …), and MCP `notifications/message` re-published from tools that
  `emit()` (migration). The UI renders both from the same runtime queue.
- **Cancellation** is best-effort and layered: UI disconnect → backend
  `cancel_active_turn` → spec `notifications/cancelled` + `POST /mcp/cancel`
  fallback → per-session flag the tool's `emit()` checks → task cancel.
  → *The MCP transport*, *Progress streaming*

### Observability
- **LangSmith off by default** (`LANGSMITH_TRACING`); result data further gated
  by `FES_LANGSMITH_LOG_CONTENT`. **Local CSVs on by default**
  (`FES_CSV_OBSERVABILITY`), never carry result data. Mutation audit always on.
  → *Observability — the trace tree*

---

## File Reference

### `frontend/app.py`

**Key session_state keys:** `session_id` (UUID per tab), `messages`,
`pending_confirmation`, `approved_keys`, `tenant_config` / `migration_config`,
`allow_summarization`.

**Key functions:** `_call_backend_sse()` (streams `/agent/turn`),
`_call_backend_json()`, `render_tool_result()` (flattens nested results into
tables; raw JSON in an expander), `_render_agent_progress()` (plan, checklist,
status line), `_render_pending_confirmation()`, `_render_migration_progress()`.

Page code hot-applies per rerun; worker-thread code needs a restart. Theme is
pinned light in `.streamlit/config.toml`, which ships in the UI image; never set
`STREAMLIT_*` env in the Dockerfile — env silently overrides the file.

### `backend/api_server.py`

Routes: `GET /health`, `GET /tools`, `POST /agent/turn` (SSE vs JSON by `Accept`),
`POST /agent/cancel`. `_select_tools_for_mode(mode)` filters the registry; it
falls back to *all* tools on an empty filter (broken-registry valve) — which is
why the turn re-scopes at entry.

### `backend/runtime.py`

`SESSION_POOL` (one long-lived `McpClient` per session; replaced on idle > 9h or
credential change), `SESSION_POOL_LOCK`, `_ACTIVE_TURNS` (cancellation),
`_SESSION_PROGRESS_CBS` (session-keyed progress registry; `publish_progress_for`
is the primary path, ContextVar `publish_progress` the fallback).
`run_turn_once()` → `_run_turn_once()`; `cancel_active_turn()`.

### `backend/agent/llm_agent.py`

Split across `_config.py` (env/logging/tracing), `_prompts.py` (all prompt
constants), `_registry.py` (registry I/O, shrinkers, `_effective_ok`,
`allowed_tool_ids`), `_routing.py` (two-stage routing, `_reachable_*`,
`call_llm_raw`), `_tracing.py`. `llm_agent.py` orchestrates.

Globals read by the API layer: `TOOL_REGISTRY`, `LAST_TOOL_RESULT`,
`LAST_STEP_RESULTS`, `LAST_PENDING_CLARIFICATION`, `LAST_PENDING_LOOP` — unit-test
and debug aids; per-turn results are snapshotted inside the turn task.

Key functions: `call_llm_with_tools()` → `_reactive_loop()`; `_make_plan()` /
`_replan()`; `_capability_catalog()`; `_split_dependent_tail()`;
`_execute_branch()`; `_verify_goal_complete()`; `_consume_approval()`;
`_approval_disclosure()`; `call_llm_raw()` (one choke point → LiteLLM SDK;
retry + per-call CSV trace via `label=`); `_fallback_direct_tool()` (keyword
fallback if the planning call fails).

**Prompts** (`_prompts.py`): `AGENT_PLAN_SYSTEM_PROMPT`,
`AGENT_REPLAN_SYSTEM_PROMPT`, `AGENT_DECIDE_SYSTEM_PROMPT` (+ `_NODATA` variant,
both with a `REPLAN:` verb), `VERIFY_GOAL_SYSTEM_PROMPT`,
`CLARIFY_ANSWER_SYSTEM_PROMPT` (interprets the user's answer — the clarifying
question itself is rendered in code), `MIGRATION_PLAN_SYSTEM_PROMPT`,
`CHAT_PLANNING_CONTEXT_PROMPT` / `MIGRATION_PLANNING_CONTEXT_PROMPT`, and the
routing prompts. Prompts carry **only generic strategy** — never
scenario-specific rules (failures become eval cases, not prompt patches).

**LLM providers:** Azure OpenAI (with AWS Secrets Manager fallback) or
Databricks Model Serving, selected by `LLM_PROVIDER`.

**LLM config is import-time** (`LLM_CONFIG`); env changes need a restart.

### `backend/agent/mcp_client.py`

Official SDK `ClientSession` over `streamablehttp_client` (since 2026-08-16 —
the hand-rolled JSON-RPC/SSE parsing is gone). `connect()` enters the SDK
transport + session contexts (held open across turns in `SESSION_POOL`), runs
`initialize`, captures the `Mcp-Session-Id`. `invoke_tool()` injects
credentials + `fes_mcp_session_id` and calls `session.call_tool(...,
progress_callback=...)`. **Narration** (`notifications/message`) arrives via
`logging_callback` and is re-published to the runtime in the SAME envelope the
UI has always consumed; **spec progress** (`notifications/progress`) is
**logged only** — the display feed stays on the message channel, so there are
no duplicate lines. Cancellation, two paths to one server-side flag: spec
`notifications/cancelled` for every in-flight request id (primary), then
`POST /mcp/cancel` with the session header (ops fallback — works when the
session stream is wedged).

### `mcp_server/server.py` · `mcp_server/tools_core.py`

Lowlevel `Server` under `StreamableHTTPSessionManager` (since 2026-08-16 — the
hand-rolled transport it replaced existed because the SDK's session manager once
had compatibility issues that no longer reproduce, verified by a 5/5-pass probe
spike). Deliberately **not** FastMCP, which validates arguments against
signature-generated schemas and would reject the injected credentials. Streaming
tools publish on BOTH channels: `notifications/message` (full `emit()` payloads
— the run-log/UI contract) and spec `notifications/progress` when the caller
sent a `progressToken`. Spec `notifications/cancelled` is bridged to the
tools_core cancel flag — anyio cannot interrupt the SDK thread; the flag stops
it at the next `emit()` checkpoint. The backend passes its session id as the
`fes_mcp_session_id` argument (popped pre-dispatch) so both cancel paths key
the same flags. `tools_core.py` loads the
registry, applies the allowlist at dispatch, builds the SDK client from tool
args, dispatches, and normalises SDK failures (`_sdk_error_payload`: `ok: False`
is the primary test; the closed key set is the fallback). `STREAMING_TOOL_IDS`,
`_MIGRATION_SEM` / `_READ_SEM`, the `emit()` callback with cancel-flag checks.

---

## Local Development

```bash
uv sync && uv run pytest tests/unit -q
cp .env.example .env
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8002 --workers 1
uvicorn backend.api_server:app --host 0.0.0.0 --port 8001
streamlit run frontend/app.py --server.port 8501
```

Venv is `venv_pysisense_chatbot`. Use the **`run` skill**
(`.claude/skills/run/SKILL.md`) for launch/drive/restart recipes and the manual
`/agent/turn` harness. Full detail: `docs/development.md`.

## Testing

Three tiers (markers in `pyproject.toml`): **unit** (`pytest tests/unit -q`,
mocked, CI, run under both engines), **integration** (`-m integration`, live
stack + creds), **eval batteries** (`-m eval`). Integration/eval are
**local-only, never in CI** — no LLM or Sisense secrets in GitHub Actions.

- **Eval battery = anti-whack-a-mole.** A prompt that misbehaved becomes an
  `EVAL_CASES` entry, added *after* the fix is proven — never a prompt patch,
  never while still broken.
- **Chat cases assert on tools executed; migration cases on the tool gated.**
  Migration cases send no `approved_keys`. Don't merge the harnesses.
- **Mutation tests only touch assets they created.** Create → gate → approve →
  delete the same asset → `finally:` force-delete.
- **Fixtures are discovered from the tenant**, not hardcoded (`eval_identities`).
- Re-run a single failing integration test before calling it a regression.

Pre-commit: ruff + ruff-format + commitizen; conventional-commit types, scoped
(`feat(ui):`). Commits from Claude Code carry a `Co-Authored-By` trailer by convention.

## Docs

- `docs/architecture.md` — the living architecture doc: loop, recovery ladder,
  verify, approvals, migration, registry, MCP transport, LangGraph mapping.
- `docs/security.md` · `docs/operations.md` · `docs/development.md` · `docs/usage.md`
- `docs/design/` — proposals for features not yet built.
- `.claude/skills/run/SKILL.md` — launch/drive/test the stack.

## Architecture Constraints

- **MCP server: 1 worker.** Multiple workers split `Mcp-Session-Id` across
  processes. Concurrency is async + semaphores within one worker.
- **Backend: 1 worker.** Session and approval state are in-process.
- **Session ID is browser-tab-scoped.** Two tabs = two MCP clients, two histories.
- **Per-turn results are snapshotted inside the turn task** — no `await` between
  the loop returning and the snapshot, so concurrent sessions cannot swap
  results.
- **LLM config is built at import time.** Env changes need a restart.
- **The allowlist is the only place a tool is exposed or hidden.** Three
  enforcement points, one file.
