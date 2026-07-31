# Roadmap — what's left

V2 is feature-complete: the agentic loop (plan → parallel fan-out → dependent
chains → recovery ladder → human-gated mutations → critic) is built, live-
verified, and documented (`AGENT_ARCHITECTURE.md`, `Execution_Flow.md`).

Two items remain before we call the version done:

## 1. MCP OAuth + Claude connector
Add OAuth to the MCP server so it can be registered as a Claude Desktop / web
connector and used directly by third-party clients (not only through the FES
backend). Relates to the standalone `sisense-admin-mcp` project (customer-facing
MCP, proxy architecture).

## 2. LangGraph refactor
Re-express the hand-rolled `_reactive_loop` as named LangGraph nodes
(plan / execute / decide / approval / fan-out) with explicit edges. The current
loop already maps cleanly onto this — see the "LangGraph mapping" table in
`AGENT_ARCHITECTURE.md`. Structural change, not behavioral.

---

Everything else (agentic loop, fan-out, replan, critic, eval battery, tests,
docs) is done.
