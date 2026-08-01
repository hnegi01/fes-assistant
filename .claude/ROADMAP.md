# Roadmap — what's left

V2 is feature-complete: the agentic loop (plan → parallel fan-out → dependent
chains → recovery ladder → human-gated mutations → critic) is built, live-
verified, and documented (`AGENT_ARCHITECTURE.md`, `Execution_Flow.md`).
The LangGraph engine is BUILT (2026-07-31) and passed all live gates: 150/150
unit tests under both engines, eval battery 6/6, integration 17/17 (incl.
mutation lifecycle + clarify-resume), SSE progress with concurrent fan-out.

MCP OAuth is NOT part of this repo (decision 2026-07-31, seeded 2026-08-01):
this repo's MCP server stays local and embedded in the FES stack (multi-tenant
per-call credential injection, session/cancel coupling, single-worker state).
The standard-shaped, OAuth-capable MCP server is its own project:
**`~/Desktop/Sisense/fes_mcp`** (FastMCP, registry-driven tool factory, auth
ladder none→bearer→oauth; see its `KICKOFF_PROMPT.md`).

What remains in THIS repo:

## 1. V2 UI testing — including LangGraph → then retire the custom engine
Drive the full V2 feature set from the Streamlit UI with
`FES_AGENT_ENGINE=langgraph` in `.env` (already set): multi-step plans,
fan-out, clarification pause/resume, mutation approval gate, summarization
on/off, cancellation, SSE progress. If everything holds up, flip the repo
default to langgraph and **remove `_reactive_loop` + the `FES_AGENT_ENGINE`
flag** — decision already made: single engine, LangGraph.

## 2. Migration mode — full test pass (untouched since V1)
Migration mode has had no dedicated testing since V1; everything since
(agentic loop, fan-out, gates, critic, LangGraph) was validated in Chat mode.
Needed: end-to-end runs in Migration mode against real source + target
deployments — tool selection from the migration-only registry slice, dual
credential injection (source_*/target_*), SSE progress for long migrations,
cancellation mid-migration, and the mutation approval gate on migration tools.

## Backlog (optional, later)
- **Langfuse tracing backend** — a Langfuse implementation of `_tracing.py`
  behind `FES_TRACING_BACKEND=langsmith|langfuse` (redaction carries over).
  Preferred route: a project on the product team's existing Langfuse instance;
  fallback Langfuse Cloud. Self-hosting rejected (needs Postgres+ClickHouse+
  Redis — conflicts with the no-database stance).

---

Everything else (agentic loop, fan-out, replan, critic, eval battery, tests,
docs) is done.
