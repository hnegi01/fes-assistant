# FES Assistant — Agent Architecture

> **Status: WORK IN PROGRESS.** Living document describing how the agent
> actually works, written to explain it to others once V2 is complete. Reflects
> the codebase as of Step 8 (agentic loop). Sections marked _(planned)_ are not
> built yet.

---

## The one big idea

There is **no single "smart" LLM call** that reads your message and orchestrates
everything. Instead, a turn is a **pipeline of narrow LLM calls**, each with one
small job, wired together by ordinary Python code (`call_llm_with_tools` in
`backend/agent/llm_agent.py`). The intelligence is in how the code composes many
dumb-but-reliable calls — not in any one call being clever.

Each LLM call sees only what it needs:

| Call | Sees | Decides |
|---|---|---|
| **Decompose** | your message | what is the *first* single operation? |
| **Route L1** | one sub-task | which of ~12 packages? |
| **Route L2** | one sub-task | which mixin within that package? |
| **Plan** | one sub-task + ~10 tools | pick exactly ONE tool + its arguments |
| **Decide** | your message + results so far | done (answer) or continue (next op)? |

Crucially, the **planner never sees the whole request and never sees more than
~10 tools.** It cannot "choose to call two tools across packages" because it
never sees two packages at once. Multi-part requests are handled by the
_decompose_ and _decide_ calls, one layer above the planner — not by the planner
itself.

Why build it this way? Because the full tool catalog is ~119 tools. Showing all
of them to one call produces hallucinated tool names and noisy plans. The
routing hierarchy narrows the menu to ~10 relevant tools before the planner
ever picks — and that narrowing is what makes each pick reliable.

---

## The processes (unchanged from V1)

```
Browser (Streamlit :8501)
  └── POST /agent/turn ─▶ FastAPI backend (:8001)
                            └── POST /mcp/ (JSON-RPC) ─▶ MCP server (:8002)
                                                          └── PySisense SDK ─▶ Sisense API
```

The agent logic lives entirely in the backend. The MCP server is a thin,
generic tool-executor over the PySisense SDK; it has no notion of the loop.

---

## The agentic loop (Step 8)

One user turn can now chain multiple tool executions. The shape:

```
decompose ── first sub-task
   │
   ▼
route ─▶ plan ─▶ execute ─▶ DECIDE ──┬── "answer"   ─▶ final reply, done
   ▲                                 │
   └──────────── "CONTINUE: <next>" ─┘   (loop, max FES_MAX_AGENT_STEPS = 8)
```

- **DISCOVER / PLAN** = route + plan (pick one tool for one sub-task)
- **EXECUTE** = call the tool via MCP
- **VERIFY** = the decide call — reads the results and judges completion
- **ITERATE** = a `CONTINUE:` reply feeds the next sub-task back through routing

Stop conditions (every one returns something readable — never a silent stop):
- decide returns a plain answer → goal met
- `FES_MAX_AGENT_STEPS` reached → partial answer + what still remains
- a continued step needs info the user never gave → *overreach*, stop and
  answer from what was gathered (`_finalize_from_transcript`)
- a mutating tool → pause for approval, resume next turn (see below)

### Worked example — a compound request

**User:** _"show me all datamodels and also show me all the user groups"_

| # | LLM call | Given | Result |
|---|---|---|---|
| 1 | Decompose | full message | _"List all datamodels"_ |
| 2 | Route L1 | _"List all datamodels"_ | `datamodel` |
| 3 | Route L2 | _"List all datamodels"_ | core |
| 4 | Plan | sub-task + ~10 datamodel tools | `datamodel.get_all_datamodel` |
| — | _execute_ | | 400 datamodels |
| 5 | Decide | full message + result | _"CONTINUE: list all user groups"_ |
| 6 | Route L1 | _"list all user groups"_ | `access_management` |
| 7 | Route L2 | _"list all user groups"_ | groups |
| 8 | Plan | sub-task + group tools | `access_management.users_per_group_all` |
| — | _execute_ | | 35 groups |
| 9 | Decide | full message + BOTH results | final answer combining both |

The "break it into two" decision is split across **call 1** (pull out the first
task) and **call 5** (notice the second task is still undone). The planner
(calls 4, 8) only ever picks one tool from a small menu and never sees the
compound request.

### Two kinds of multi-step

- **Static / independent** — _"show all datamodels AND all groups."_ Both parts
  are knowable from the request alone; neither needs the other's result.
- **Adaptive / dependent** — _"get the username for user_id xyz, then find which
  datamodels that user owns."_ Step 2 needs step 1's *output* to form its
  request. The decide call must *read* step 1's result to plan step 2.

This distinction drives everything about the summarization switch below.

---

## The summarization switch

`ALLOW_SUMMARIZATION` is a **hard privacy kill-switch: when false, tool results
are never sent to the LLM.** It is not merely "skip the prose summary."

The decide/verify call *requires* reading results. So:

| | Summarization ON | Summarization OFF |
|---|---|---|
| Tool results → LLM | yes | **never** |
| Decide / verify call | yes | impossible |
| Multi-step loop | yes | no — single shot |
| Adaptive chains | yes | **no** (need result content) |
| Static compound | yes | _(see "planned" below)_ |
| Final answer | LLM prose | local raw description |

**Current behaviour with summarization OFF (as built):** the turn routes the
full message, plans one tool, executes it, returns the raw result — a single
shot. Decompose does not run, so a compound request runs only its first-matched
tool. This is a known gap; the target design below fixes it.

### Target design — one reactive planner, flag controls visibility _(planned)_

Do NOT branch the orchestration on the summarization flag. Keep **one reactive
planner call** — `next_step(goal, history, summ_flag)` — that runs first and on
every step, always deciding "the next single operation, or done." The flag does
not switch loop-vs-no-loop; it only controls **how much of each result the
planner sees in `history`:**

| | Summarization ON | Summarization OFF |
|---|---|---|
| `history` contains | full tool results | **action metadata only** — which tool ran + ok/fail, never the data |
| Sequence independent tasks | yes | yes (it knows what's done from the metadata) |
| Pass a value between steps (adaptive) | yes (reads the id/name) | no — the value isn't in `history`, so it stops and explains |
| Completion check / verify | yes | yes (metadata is enough to know a step ran) |
| Final answer | LLM prose | local raw description of the results |

Why this is the right shape:

- **The first LLM call is the planner.** On step 1 `history` is empty; the
  planner decides the first operation. This replaces the separate decompose
  call — decompose-first was just this planner with an empty history (see the
  `next_step` merge under "What's next").
- **It is all reactive.** The same call re-decides each step from goal +
  history, so it naturally handles unknown-shape tasks ("restart every failed
  datamodel") that up-front planning cannot express.
- **Data enforcement is in code, not the LLM.** The code builds `history` with
  full results (summ on) or metadata only (summ off). The planner physically
  never receives data it shouldn't. It is *told* the flag so it can explain the
  limit — but the guarantee is that the data was never put in the messages, not
  that the model was asked not to look. Never trust the model to enforce a
  privacy boundary.
- **Adaptive degrades gracefully.** With summ off, the planner sees "step 1 ran
  get_user (ok)" but not the id it returned; when the next step needs that id
  there is nothing to fill it with, so it stops: _"I listed X and Y, but finding
  john's datamodels needs the id from a prior step, which I can't read with
  summarization off."_

Irreducible floor: **truly reactive tasks that branch on result *content*
(unknown iteration count driven by the data) are only possible with summ on** —
not a design choice, but because reacting to data requires seeing data. Ordering
and sequencing (text reasoning) are mode-independent; only value-passing needs
the data.

Tradeoff: fully reactive = one LLM call per step (vs one up-front plan for N
steps). Worth it — the flexibility to handle unknown-shape tasks is the point of
an autonomous agent.

---

## Mutation approval (carried from Step 7, extended in Step 8)

A mutating tool never executes without explicit approval, **even mid-loop**.

- The loop pauses at the gate, saves its state (`pending_loop` in
  `SessionEntry`: transcript + step count + the exact tool & args).
- The turn returns a plain-English explanation of what will change.
- On the next turn, if the user approves that exact tool+args, it executes
  **directly** (no re-plan — deterministic) and the loop **resumes from where it
  paused** and continues (Option A).
- A second mutation later in the same loop pauses again — approval is always
  per-operation.

---

## Clarification (Step 7)

If step-1 planning leaves a required argument missing (the user never provided
it), the turn pauses and asks, then resumes next turn with the answer. The
planner is shown a schema with `required` stripped (`planner_schema`) so it
omits values it doesn't have rather than hallucinating placeholders; validation
against the *real* schema then routes genuinely-missing fields into the
clarification loop.

Mid-loop, a continued step that needs unprovided info is treated as decide
*overreach* (see above) — the loop stops and answers, rather than asking a
confusing question about something the user never mentioned.

---

## Progress streaming (Step 8)

Each loop phase emits an `agent_progress` SSE event
(`deciding | planning | executing | completed`, with step / max_steps /
tool_id). Chat mode streams these to the UI, which renders a live step checklist
plus a current-phase status line. (Migration mode keeps its SDK-emitted progress
log.) With summarization off there is no loop, so only step 1's execution shows.

---

## Hand-rolled building blocks (and their LangGraph future)

Everything today is **hand-rolled** — plain Python control flow inside
`backend/agent/llm_agent.py`, no agent framework. This was deliberate: build the
machinery by hand first, understand what each piece is *for*, then adopt
LangGraph in Step 10 knowing exactly which primitive replaces which hand-rolled
part. This section is the glossary + the mapping to fill in after Step 10.

### The building blocks, in our own terms

| Term | What it does | Where it lives now |
|---|---|---|
| **Decompose** | splits a compound request; returns the first sub-task | `_decompose_first_step` |
| **Route** | narrows 119 tools → one package → one mixin (~10 tools) | `_navigate_to_tools` (`_routing.py`) |
| **Plan** | given one sub-task + ~10 tools, pick ONE tool + args | planning `call_llm_raw(..., tools=...)` |
| **Execute** | run the chosen tool via MCP → Sisense | `mcp_client.invoke_tool` |
| **Decide** | given goal + results so far, answer or `CONTINUE:` | `_agent_continuation_loop` (decide call) |
| **Loop control** | while-loop tying decide→route→plan→execute together | `_agent_continuation_loop` (`while True`) |
| **Pause/resume state** | externalized state so a turn can stop and continue next turn | `SessionEntry.pending_loop` / `.pending_clarification` |
| **Mutation gate** | stop before a destructive tool, wait for approval | `REQUIRE_MUTATION_CONFIRM` check + `pending_confirmation` |
| **Clarification** | stop and ask when a required arg is missing | `_generate_clarification_question` + `LAST_PENDING_CLARIFICATION` |
| **Graceful stop** | every exit returns readable text, never a silent halt | `_finalize_from_transcript`, `_loop_partial_message` |
| **Progress** | emit a step event each phase | `_emit_agent_progress` |

### Mapping to LangGraph _(to confirm after Step 10)_

The expected correspondence — the table you point at when someone asks _"how did
you use LangGraph here?"_ Verify/adjust each row once Step 10 is built.

| Our hand-rolled piece | Expected LangGraph primitive |
|---|---|
| `_agent_continuation_loop` `while` loop | the **graph** itself (nodes + edges) |
| Route / Plan / Execute / Decide | individual **nodes** |
| Decompose + Decide (see merge below) | one `next_step` **node** |
| "`CONTINUE:` → loop back, else → answer" | a **conditional edge** |
| `pending_loop` / `pending_clarification` in `SessionEntry` | **checkpointer** (persistent state / `thread_id`) |
| Mutation gate → return, resume next turn | **interrupt** (human-in-the-loop) |
| The dict passed between steps (goal, transcript, results) | the graph **state** object |
| Graceful-stop terminal returns | **END** node / terminal edges |
| _(new in Step 10)_ plan → replan | a `plan` node + a **replan edge** on divergence |
| _(new in Step 10)_ cross-package parallel | **fan-out / fan-in** (parallel branches) |

The point of the left column: none of these are LangGraph inventions — they are
real problems any agent hits (how to loop, pause, resume, stop safely, keep a
human in the loop). We solved each by hand first; LangGraph just gives each a
named, reusable primitive. That is the honest answer to "why LangGraph" — not
"it makes agents," but "it replaces this hand-rolled plumbing with tested
primitives, and makes each node independently testable."

---

## What's next _(planned)_

- **`next_step` merge.** Today decompose-first and decide are two prompts asking
  nearly the same question — "given the goal and progress so far, what's next?"
  decompose-first is just decide with an empty history; they are separate only
  because of build order (decide came first in Step 8, decompose-first was
  bolted on to fix a compound-request routing bug). Merge them into one reactive
  `next_step(goal, history, summ_flag)` call that runs every step including the
  first. Not a performance fix — pure clarity: one prompt, one path, step 1 stops
  being special. Best done at Step 10 (LangGraph), where the loop is torn into
  named nodes anyway and `next_step` becomes one node — the duplication dissolves
  for free, and each node is independently testable, so the refactor is safer
  there than churning the stable hot path now.
- **Summarization-off path.** Implement the "one reactive planner, flag controls
  visibility" design above: build `history` with full results (summ on) or
  action metadata only (summ off). Fixes today's single-shot gap without
  branching the orchestration.
- **Step 9 — MCP OAuth + Claude connector.** Bearer/OAuth on the MCP server so
  it can be a hosted, standalone connector. (See the separate
  `sisense-admin-mcp` brief.)
- **Step 10 — LangGraph.** Refactor the hand-rolled loop into named graph nodes
  (plan / execute / decide / approval); fold in the `next_step` merge. This is
  also where **cross-package parallel fan-out** lands — decompose into
  independent sub-goals, run their route→plan→execute pipelines concurrently,
  join the results. That parallelism is the feature that graduates this from a
  single-agent loop into a genuine multi-agent system.

---

## Design rules that must not break

| Rule | Why |
|---|---|
| Planner sees ~10 tools from one package, picks one | Avoids hallucination / noise from the 119-tool catalog |
| Mutation gate never bypassed, even mid-loop | Safety — no destructive action without per-op approval |
| Result **data** never reaches the LLM when summarization is off (action metadata may) | Privacy kill-switch is absolute; enforced in code by what `history` contains, never by asking the model not to look |
| `MAX_AGENT_STEPS` is a hard ceiling | Runaway-loop backstop; caps cost per turn |
| Every loop exit returns readable text | The user never sees a silent dead stop |
