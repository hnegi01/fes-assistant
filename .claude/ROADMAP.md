# Roadmap — what's left

V2 is feature-complete: the agentic loop (plan → parallel fan-out → dependent
chains → recovery ladder → human-gated mutations → critic) is built, live-
verified, and documented (`AGENT_ARCHITECTURE.md`, `Execution_Flow.md`).

## MCP OAuth — OUT OF SCOPE for this repo (decision 2026-07-31)
This repo's MCP server stays **local and embedded in the FES stack**, by design:
it is highly customized for the agent (multi-tenant per-call credential
injection, session/cancel coupling, single-worker state) and is not meant to be
a public connector. OAuth + Claude connector live in the separate
**sisense-admin-mcp** project (standard-shaped, curated admin tools, auth modes
none→bearer→oauth; the product team's TypeScript MCP proxies to it). Brief:
https://claude.ai/code/artifact/83dd30e5-5e65-471f-8022-bfe018bb9fe5

What remains in THIS repo:

## 1. LangGraph refactor — DESIGNED, DEFERRED (revisit when ready)
Re-express the hand-rolled `_reactive_loop` as a LangGraph graph. The full
blueprint is ready in `AGENT_ARCHITECTURE.md` → "Mapping to LangGraph": nodes
`planner / router / agent / validator / tools / replanner / evaluator / respond`,
`interrupt()` for the mutation gate + clarification, checkpointer
(thread_id = session) for pause/resume, `Send` API for fan-out. Structural
re-expression, not behavioral — thin nodes calling the existing helpers.
Estimated ~2–4 sessions incl. full re-verification (unit + eval + integration +
UI smoke). Caveat noted in the doc: keep our `_tracing.py` (LangGraph's native
tracing would bypass the privacy redaction). Adoption strategy: engine flag
(`FES_AGENT_ENGINE=custom|langgraph`) — both engines share the same helpers,
parallel-run until parity, nothing thrown away either way. Checkpointer:
in-memory (= today's behavior) or a SQLite file — no database required.

## 2. Langfuse tracing backend — OPTIONAL, LATER
Add a Langfuse implementation of our `_tracing.py` abstraction behind
`FES_TRACING_BACKEND=langsmith|langfuse` (redaction carries over untouched).
Preferred route: a project on the product team's existing Langfuse instance
(no new infra/account); fallback Langfuse Cloud. Self-hosting rejected — it
requires Postgres+ClickHouse+Redis, conflicting with the no-database stance.

---

Everything else (agentic loop, fan-out, replan, critic, eval battery, tests,
docs) is done.
