# Roadmap — what's left

V2 is feature-complete: the agentic loop (plan → parallel fan-out → dependent
chains → recovery ladder → human-gated mutations → critic) is built, live-
verified, and documented (`AGENT_ARCHITECTURE.md`, `Execution_Flow.md`).

Two items remain before we call the version done:

## 1. MCP OAuth + Claude connector — NEXT
Add OAuth to the MCP server so it can be registered as a Claude Desktop / web
connector and used directly by third-party clients (not only through the FES
backend). Relates to the standalone `sisense-admin-mcp` project (customer-facing
MCP, proxy architecture).

## 2. LangGraph refactor — DESIGNED, DEFERRED (decision 2026-07-31: after OAuth)
Re-express the hand-rolled `_reactive_loop` as a LangGraph graph. The full
blueprint is ready in `AGENT_ARCHITECTURE.md` → "Mapping to LangGraph": nodes
`planner / router / agent / validator / tools / replanner / evaluator / respond`,
`interrupt()` for the mutation gate + clarification, checkpointer
(thread_id = session) for pause/resume, `Send` API for fan-out. Structural
re-expression, not behavioral — thin nodes calling the existing helpers.
Estimated ~2–4 sessions incl. full re-verification (unit + eval + integration +
UI smoke). Caveat noted in the doc: keep our `_tracing.py` (LangGraph's native
tracing would bypass the privacy redaction).

---

Everything else (agentic loop, fan-out, replan, critic, eval battery, tests,
docs) is done.
