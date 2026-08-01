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

## 1. LangGraph engine — BUILT (2026-07-31), parallel-run in progress
`backend/agent/graph_engine.py` implements the blueprint: StateGraph with
nodes planner / branch+join (Send-API fan-out) / first_select / next_select /
validator / gate / tools / decide(replanner+evaluator), thin wrappers over the
SAME llm_agent helpers, selected by `FES_AGENT_ENGINE=custom|langgraph`
(default custom). No checkpointer/DB/files — pauses END the run and persist via
SessionEntry exactly like the custom loop. Parity: all 150 unit tests pass
under BOTH engines (helpers accessed via module attributes, so the same mocks
exercise both). LIVE GATES PASSED under langgraph (2026-07-31, refreshed token): eval battery
6/6, integration 17/17 (incl. mutation lifecycle + clarify-resume), SSE progress
verified with concurrent fan-out phase interleaving. Remaining: a short
parallel-run observation window (user drives the UI with
FES_AGENT_ENGINE=langgraph in .env), then flip the repo default and retire
_reactive_loop + the flag (decision already made: single engine, LangGraph).

## 2. Langfuse tracing backend — OPTIONAL, LATER
Add a Langfuse implementation of our `_tracing.py` abstraction behind
`FES_TRACING_BACKEND=langsmith|langfuse` (redaction carries over untouched).
Preferred route: a project on the product team's existing Langfuse instance
(no new infra/account); fallback Langfuse Cloud. Self-hosting rejected — it
requires Postgres+ClickHouse+Redis, conflicting with the no-database stance.

---

Everything else (agentic loop, fan-out, replan, critic, eval battery, tests,
docs) is done.
