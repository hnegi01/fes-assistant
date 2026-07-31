# FES Assistant — Agent Architecture

> Living document describing how the agent actually works. Reflects the current
> V2 codebase: the agentic loop — plan → parallel fan-out → dependent chains →
> recovery ladder → human-gated mutations → critic. Sections marked _(planned)_
> — the LangGraph refactor — are not built yet; see `.claude/ROADMAP.md` for
> what remains.

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
| **Plan (planner)** | your message + capability catalog (one-liners) | the ordered plan; its first operation starts the loop |
| **Route L1** | one sub-task | which of ~12 packages? |
| **Route L2** | one sub-task | which mixin within that package? |
| **Plan** | one sub-task + ~10 tools | pick exactly ONE tool + its arguments |
| **Decide** | your message + results so far | done (answer) or continue (next op)? |

Crucially, the **planner never sees the whole request and never sees more than
~10 tools.** It cannot "choose to call two tools across packages" because it
never sees two packages at once. Multi-part requests are handled by the
_planner_ (plan/replan) and _decide_ calls, one layer above the tool-selection step — not
by the planner itself.

Why build it this way? Because the full tool catalog is ~119 tools. Showing all
of them to one call produces hallucinated tool names and noisy plans. The
routing hierarchy narrows the menu to ~10 relevant tools before the planner
ever picks — and that narrowing is what makes each pick reliable.

---

## The processes

```
Browser (Streamlit :8501)
  └── POST /agent/turn ─▶ FastAPI backend (:8001)
                            └── POST /mcp/ (JSON-RPC) ─▶ MCP server (:8002)
                                                          └── PySisense SDK ─▶ Sisense API
```

The agent logic lives entirely in the backend. The MCP server is a thin,
generic tool-executor over the PySisense SDK; it has no notion of the loop.

**LLM access — gateway-as-a-library.** Every LLM call goes through one choke
point (`call_llm_raw` in `_routing.py`) using the **LiteLLM SDK** in-process:
unified API over Azure OpenAI / Databricks (env-switched), retries, param
dropping. There is deliberately **no standalone LLM gateway service** — keys
live with the backend. If centralized governance were ever needed (shared keys,
budgets, org-wide rate limits), the LiteLLM Proxy speaks the same interface, so
the migration is pointing `api_base` at it — a config change at one choke point.

---

## The full flow — orchestrator view (high level)

Standard plan-and-execute shape: a **planner** drafts and repairs the plan; an
**orchestrator** — the loop — runs it, dispatching **executors** one step each
(independent steps in parallel); and a **critic** independently evaluates the
goal before the answer ships.

```mermaid
flowchart TD
    U(["🧑 User prompt"]) --> ORC

    ORC["PLANNER (LLM)<br/>drafts the plan from the tool catalog<br/>orders steps · tags data dependencies"]
    ORC --> S1["Step 1 · executor<br/>route → select → verify → run"]
    ORC --> S2["Step 2 · executor"]
    ORC --> S3["Step 3 · executor"]

    S1 --> J[("JOIN<br/>results, plan order")]
    S2 --> J
    S3 --> J

    J --> DEP["Dependent step(s) · executor<br/>sequential — use joined results<br/>(mutations pause here for approval)"]
    DEP --> CR{{"CRITIC (independent LLM)<br/>whole request achieved?"}}
    CR -->|"INCOMPLETE · +1 step"| DEP
    CR -->|"COMPLETE"| ANS(["💬 Reply"])
    DEP -. "a step failed → REPLAN" .-> ORC

    classDef term fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef orch fill:#f4eefe,stroke:#7c3aed,color:#4c1d95;
    classDef exec fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;
    class ANS term
    class ORC,CR orch
    class S1,S2,S3,DEP exec
```

- **Planner** = `_make_plan` / `_replan` (sees the tool catalog as
  one-liners, writes prose steps, never calls tools)
- **Orchestrator** = the loop (`_reactive_loop`) that reads the plan and runs it
  — dispatch → execute → decide → replan/next; the planner is a step inside it
- **Executor** = one route→select→validate→execute pipeline per step
  (`_execute_branch` when parallel, the loop body when sequential)
- **Critic** = `_verify_goal_complete` — separate context, adversarial prompt
- Independent steps fan out concurrently (`FES_MAX_PARALLEL_STEPS`); dependent
  steps run after the join; mutations always pause for human approval

Each executor step runs through the reactive loop machinery detailed next.

---

## The agentic loop

One user turn can chain multiple tool executions. It is a single loop
(`_reactive_loop`) wrapped in **plan → execute → replan**: on the first pass a
planner call drafts the full plan (request + a compact capability catalog —
tool one-liners, no schemas) and the plan is shown to the user; every later pass
a decide call picks the next move. When a step's outcome shows the approach
cannot work (failed / found nothing / wrong kind of data), decide says
`REPLAN:` and the planner revises the plan with the catalog — a retry that
CHANGES approach, budgeted by `FES_MAX_REPLANS`. Each lap executes **exactly
one** SDK call.

```mermaid
flowchart TD
    U(["🧑 User message"]) --> WN

    WN{{"WHAT'S NEXT?<br/>step 0 · PLAN (planner + catalog)<br/>step N · decide · REPLAN on failure"}}
    WN -->|"REPLAN: approach failed"| RP["REPLAN<br/>planner + catalog<br/>revise the plan"]
    RP -->|"new approach"| WN
    RP -->|"GIVEUP"| ANS
    WN -->|"DONE · final answer"| ANS(["💬 Reply to user"])
    WN -->|"BLOCKED · summ-off adaptive"| ANS
    WN -->|"CONTINUE: next op"| RT

    RT["ROUTE<br/>119 tools → 1 package → ~10"] --> PL
    PL["PLAN<br/>pick ONE tool + args"] --> VAL

    VAL{"args valid?"}
    VAL -->|"missing required · step 0"| ASK(["❓ Ask the user"])
    VAL -->|"wrong format"| BLK(["⛔ Hard block"])
    VAL -->|"ok"| GT

    GT{"mutating &<br/>not approved?"}
    GT -->|"yes"| APR(["⏸️ Pause · await approval"])
    GT -->|"no"| EX

    EX["EXECUTE<br/>one SDK call via MCP"] --> HI[("append result<br/>to history")]
    HI -. loop back .-> WN

    classDef term fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef pause fill:#fef9c3,stroke:#ca8a04,color:#713f12;
    classDef exec fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;
    classDef dec fill:#f1f5f9,stroke:#475569,color:#0f172a;
    class ANS term
    class ASK,BLK,APR pause
    class EX exec
    class WN,VAL,GT dec
    class RP dec
```

Reading it: the diamond **WHAT'S NEXT?** is the only thing that changes by step —
the planner's plan on the first lap, decide after (with `REPLAN:` routing back
through the planner when an approach fails). Green = the turn ends with an answer;
yellow = it pauses or blocks (and resumes/explains); blue = the one SDK call this
lap. The dashed edge is the loop: append the result, ask "what's next?" again.

- **DISCOVER / PLAN** = the planner's plan, then route + tool-pick per step
- **EXECUTE** = call the tool via MCP — one SDK call per lap
- **VERIFY** = the decide call (+ the independent goal checker at "done")
- **ITERATE** = `CONTINUE:` feeds the next sub-task back through routing;
  `REPLAN:` sends the failure back through the planner for a new approach

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
| 1 | Plan (planner) | full message + capability catalog | _"1. List all datamodels 2. List all user groups"_ — step 1 starts |
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

The "break it into two" decision lives in **call 1** (the planner's plan)
and is re-checked by **call 5** (decide notices what's still undone — and can
say `REPLAN:` if a step's outcome shows the approach failed). The planner
(calls 4, 8) only ever picks one tool from a small menu and never sees the
compound request.

### Two kinds of multi-step

- **Static / independent** — _"show all datamodels AND all groups."_ Both parts
  are knowable from the request alone; neither needs the other's result. Works
  in **both** summarization modes.
- **Adaptive / dependent** — _"get the username for user_id xyz, then find which
  datamodels that user owns."_ Step 2 needs step 1's *output* to form its
  request. The decide call must *read* step 1's result — so adaptive chains
  complete only with summarization **on**; with it off they stop gracefully
  (`BLOCKED`).

This distinction drives the summarization switch below.

---

## Anatomy of a turn — every call, with inputs and outputs

### High level — the calls in order

A turn is **one loop** (`_reactive_loop`), the same body every step:

```
loop, until done:
    WHAT'S NEXT?          step 0 → PLAN: planner + catalog → ordered plan (shown)
                          step N → DECIDE: CONTINUE · REPLAN · DONE · (off) BLOCKED
    ROUTE L1              which package? (of ~12)
    ROUTE L2              which mixin? (→ ~10 tools)
    PLAN                  pick ONE tool + args from those ~10
    (validate · gate)     code, no LLM — schema check, mutation approval
    EXECUTE               run the tool via MCP → result → back to WHAT'S NEXT?
```

Each box is one LLM call except *validate/gate* (code) and *execute* (MCP). The
only thing that differs by step is **WHAT'S NEXT?** — on the first pass the
planner drafts the plan (request + capability catalog) and its first
operation starts the loop; on later passes the decide call reads goal + history
and ends the turn (its answer becomes the reply), emits `CONTINUE:` (next
sub-task into ROUTE), or emits `REPLAN:` (the planner revises the plan with
the catalog + failure evidence). There is no separate step-1 code path.

### Detailed — an adaptive request, call by call

**User:** _"get the details of user gowtham@sisense.com, then list all other
users who have that same role"_ — adaptive: step 2 needs the *role* that step 1
returns.

For each call: the **prompt logic** (what it's told to do, not the full text),
its **input**, and its **output**.

**1 · PLAN (planner)** — _logic:_ draft the ordered plan using the capability
catalog (tool one-liners, no schemas); dependency order, not sentence order;
never promote a descriptive word into an object name.
- in: the full user message
- out: `"Get the details of user gowtham@sisense.com"`
  _(the prerequisite — you can't list by role until you have the role)_

**2 · ROUTE L1** — _logic:_ reply with the one package that fits, or `none`.
- in: `"Get the details of user gowtham@sisense.com"`
- out: `access_management`

**3 · ROUTE L2** — _logic:_ pick the mixin within that package.
- in: same sub-task
- out: `users` → loads ~10 user tools

**4 · PLAN** — _logic:_ pick ONE tool; fill only args the user actually gave;
never use a placeholder for a missing one.
- in: sub-task + the ~10 user tools
- out: tool_call `access_management.get_user(user_email="gowtham@sisense.com")`

**· validate · gate** — code: args pass the schema; tool is not mutating → no
approval needed.

**5 · EXECUTE** — MCP → SDK → Sisense.
- in: `get_user`, `{user_email: "gowtham@sisense.com"}`
- out: `{ok: true, result: {firstName: "Gowtham", role: "sysAdmin", groups: [...]}}`

**6 · DECIDE (after step 1)** — _logic (data mode):_ given the goal + results so
far, reply `CONTINUE: <next op>` or the final answer; do prerequisites first;
counting/filtering is your job, not a CONTINUE.
- in: goal + history containing step-1's **full result** (role visible)
- out: `"CONTINUE: list all users whose role is sysAdmin"`
  _(it read `role: sysAdmin` out of the result — this is the adaptive hand-off)_

**7 · ROUTE + PLAN + EXECUTE (step 2)** — as above for the new sub-task.
- plan out: `access_management.get_users_with_role_names_and_group_names()`
- execute out: `{ok: true, result: [ ...all users with roles... ]}`

**8 · DECIDE (after step 2)** — everything asked for is present.
- in: goal + both results
- out: final answer — _"Gowtham is a sysAdmin; no other users share that role."_

### The same request with summarization OFF

Only call 6 changes — and it changes everything downstream:

**6 · DECIDE (nodata mode)** — _logic:_ you see only which tools ran + ok/count,
never the data; reply `CONTINUE` for an undone independent op, `BLOCKED` if the
next op needs a value you can't see, or `DONE`.
- in: goal + history = `{tool: get_user, ok: true, count: 1}` — **no role**
- out: `"BLOCKED: need gowtham's role from step 1, which I can't see"`

→ the loop stops, keeps step 1's result, and replies: _"I got the user's
details, but listing others with the same role needs a value I can't read with
summarization off — turn it on to continue,"_ then renders step 1 locally. The
role data never reached the LLM.

Contrast a **non-adaptive** request (_"list all datamodels and all groups"_):
call 6's nodata decide sees `{tool: get_all_datamodel, ok, count: 400}`, which
is enough to know that part is done, so it replies `CONTINUE: list all user
groups` and the loop runs both — no block, because neither step needs the
other's data.

---

## The summarization switch _(as built)_

`ALLOW_SUMMARIZATION` is a **hard privacy control: when false, tool result *data*
is never sent to the LLM.** It is NOT "loop vs no loop" and NOT "skip the prose
summary." The reactive loop runs in **both** modes — the flag only changes **how
much of each result the planner sees:**

| | Summarization ON | Summarization OFF |
|---|---|---|
| `history` the planner sees | full tool results | **action metadata only** — `{tool, ok, count, error}`, never the data |
| Multi-step (independent tasks) | yes | **yes** (knows what's done from metadata) |
| Adaptive (pass a value between steps) | yes (reads the id/name) | no — value isn't in history → `BLOCKED`, stop + explain |
| Final answer | LLM prose | local raw description (rendered in-process, no LLM) |

**Where the boundary is enforced:** one place in code — `_transcript_step`
builds each step's history entry with the full result (`_shrink_for_llm`) when
summ is on, or `_metadata_record` (`{tool, ok, count}`) when off. The planner
*physically never receives* data it shouldn't. It is *told* the mode (via the
`AGENT_DECIDE_NODATA` prompt) so it can explain the limit — but the guarantee is
that data was never put in the messages, **never that the model was asked not to
look.** Never trust the model to enforce a privacy boundary.

**Adaptive degrades gracefully — and cheaply.** Two layers:

1. **Plan-time dependency gate (code).** The planner tags plan steps that
   need a value from an earlier step's RESULT (`[needs-prior-result]`) — pure
   text reasoning, privacy-safe. With summ off, code splits the plan there:
   the independent prefix runs, the dependent tail is **skipped up front** (no
   doomed call, no wasted replan) and named in the reply with "turn
   summarization on to run it".
2. **Runtime safety net (decide).** Anything the plan missed still hits the
   `AGENT_DECIDE_NODATA` `BLOCKED:` reply — the loop stops and says _"the rest
   needs a value from an earlier step I can't see with summarization off"_ and
   renders what it got, locally.

**Irreducible floor:** tasks that branch on result *content* (unknown iteration
count driven by the data — "restart every failed datamodel") are only possible
with summ on. Not a design choice — reacting to data requires seeing data.
Ordering/sequencing is text reasoning (mode-independent); only value-passing
needs the data.

### The four flows

Non-adaptive compound — _"show all datamodels AND all groups"_:

```
                    plan (planner) → "1. list datamodels 2. list groups"
SUMM ON:   route→plan→exec(datamodels)  ─┐ history=full data
           decide: CONTINUE list groups  │
           route→plan→exec(groups)       │
           decide: DONE → LLM prose combining both
SUMM OFF:  route→plan→exec(datamodels)  ─┐ history=metadata {ok,count}
           decide(nodata): CONTINUE      │
           route→plan→exec(groups)       │
           decide(nodata): DONE → local render "399 datamodels… 35 groups…"
```
Both run the two steps. Only the *final answer* and *what the planner saw*
differ. (Verified live.)

Adaptive — _"get user X's details, then list others with that same role"_:

```
SUMM ON:   exec(get_user) → history has role=sysAdmin
           decide: CONTINUE "list users with role sysAdmin"   ← reads the role
           exec(get_users_with_role) → decide: DONE → prose answer
SUMM OFF:  exec(get_user) → history = {ok, count} (NO role)
           decide(nodata): BLOCKED "need the role from step 1, can't see it"
           → stop + explain + local render of step 1
```
Same first step; summ-on passes the value and finishes, summ-off blocks. (Verified live.)

### Why this shape

- **First LLM decision is a planner.** Step 1 and every later step are the same
  "what's next?" question, run by the **one** `_reactive_loop` — step 1 is not a
  special path. On the first pass the planner (`_make_plan`) drafts the plan;
  on later passes the decide call reads goal + history, with `REPLAN:` routing
  back through the planner when an approach fails. (The old
  `_decompose_first_step` + separate continuation loop were unified, then
  upgraded to plan→replan.)
- **All reactive** — re-decide each step from goal + history — so unknown-shape
  tasks work (summ on).
- Tradeoff: one LLM call per step (vs one up-front plan). Worth it — flexibility
  is the point of an autonomous agent.

---

## Verify — how the loop checks its work

There is no single VERIFY box; checking happens at three points, at two levels
of trust.

| # | Checks | When | Who | Trust |
|---|---|---|---|---|
| 1 | the **call** is well-formed | before execute | `jsonschema.validate` (`args valid?`) | objective — code |
| 2 | the **step** succeeded | after execute | the result's `ok` flag (`_metadata_record`) | objective — code |
| 3 | the **whole request** is done | at "done" | the decide call (maker), then an independent **critic** — an LLM call | judgment — LLM |

**Per-step (1 and 2) is deterministic code** — a schema passes or it doesn't; `ok`
is true or false. No LLM, nothing to second-guess.

**Goal completion (3) is judgment, so it gets a maker/checker pair:**

- **Maker** = the decide call (`WHAT'S NEXT?`). When it stops emitting `CONTINUE`
  it believes the request is satisfied.
- **Critic (the checker)** = an independent call (`_verify_goal_complete`,
  `VERIFY_GOAL_SYSTEM_PROMPT`) that re-reads the *whole* request against the
  results, prompted adversarially to find what was missed. It sees only goal +
  results (not the maker's reasoning), so it doesn't inherit the maker's
  rationalisations. `INCOMPLETE: <op>` pushes the loop one more step; `COMPLETE`
  accepts the answer.

Why a checker only on #3: the maker is biased toward declaring victory and can't
catch its own "stopped too early." A second, differently-prompted pass catches
that. #1 and #2 need no checker — code doesn't misjudge schemas or booleans.

**Summarization-on only.** Judging whether the goal was actually *achieved* means
reading the results — so the checker runs only when summarization is on. With it
off the checker would see just metadata (which the decide call already checked),
adding an LLM call for no real depth; there it's skipped and the decide call's
`DONE` stands.

Guards: `FES_VERIFY_GOAL` toggles it; `FES_VERIFY_MAX_RECHECKS` (default 1) bounds
overrides; the step cap still applies; any checker failure defaults to *complete*
so it can never block a good answer. Cost: one extra LLM call per turn (summ-on).

Ceiling: the checker is still an LLM judging completion — better than the maker
alone, not infallible. And "the whole request" means the current turn's prompt;
there is no standing cross-turn goal.

---

## The recovery ladder

Three recovery mechanisms at increasing altitude — each fires only when the
cheaper one below it can't help. **Backtrack fixes the step, replan fixes the
strategy, the critic fixes completeness.**

| Mechanism | Granularity | Trigger | What it changes | Who | Budget |
|---|---|---|---|---|---|
| **Backtrack** | within one step | routing/planning miss (no tool picked) | same op, **wider tool menu** (whole package instead of ~10) | code | 1 retry per step |
| **Replan** | triggered by a step, revises the **request's remaining plan** | a step's result contradicts the plan (failed / found nothing / wrong data), or a dead end | **new approach** — the planner + catalog rewrites what's left | LLM (planner) | `FES_MAX_REPLANS` per turn |
| **Critic INCOMPLETE** | whole request, at "done" | the maker declared done but something is missing | pushes **+1 step** (a missing op) — never rewrites the plan | LLM (critic) | `FES_VERIFY_MAX_RECHECKS` |

Two clarifications that prevent a common misreading:

- There is **no separate step-level replan**: when a step fails, the
  planner re-plans everything *remaining* for the request (catalog +
  failure evidence) — one mechanism, one budget, three trigger points
  (decide's `REPLAN:` verb, routing dead-end, planning dead-end).
- There is **no standalone request-level replan** that fires without a step
  failure. The closest thing is the critic — but it only *adds* a missing
  step; it never rewrites the plan.

---

## Mutation approval

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

## Clarification

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

## Progress streaming

Each loop phase emits an `agent_progress` SSE event
(`deciding | planning | executing | completed`, with step / max_steps /
tool_id). Chat mode streams these to the UI, which renders a live step checklist
plus a current-phase status line. (Migration mode keeps its SDK-emitted progress
log.) The loop runs in both summarization modes, so multi-step progress shows
either way.

---

## Observability — the trace tree

Everything the agent does is reported to **LangSmith** as a proper trace tree
(`backend/agent/_tracing.py`, via the LangSmith SDK — LiteLLM's bundled
callback is deliberately not used: it drops custom metadata and posts every
call as an isolated root run). Standard distributed-tracing vocabulary
(trace / span / root / child / thread), and the hierarchy mirrors both the app's
session model and the agent architecture exactly:

```
Project   fes_agent                        — the application
└─ Thread    one chat session              — thread_id = the UI session_id (SESSION_POOL)
   └─ Trace     one user prompt (a "turn") — root run `agent_turn` (run_type: chain)
      └─ Runs      the work                — llm children (planner/route/plan/decide/verify)
                                             + tool children (each MCP execution: ok, rows, duration)
```

- **Tool executions are first-class spans** — tool_id, scrubbed args, ok, row
  count, duration. "Why was this turn slow?" is almost always a tool span.
- **Cost per turn**: each llm child carries `ls_provider`/`ls_model_name`, so
  LangSmith prices every call from token usage and the root aggregates the
  whole turn (something per-call logging can never show).
- **Depth semantics**: our tree is deliberately flat (root=0, children=1) —
  one agent, many hands. Depth measures *nesting of work units*, not agents; a
  level-3 MAS worker would appear as a depth-1 chain with its own depth-2
  children. Reading the tree IS reading the architecture.
- Lifecycle: root opened/closed in `runtime._run_turn_once` (all exit paths);
  llm children attach from `call_llm_raw`; tool children from
  `_invoke_tool_traced`. Best-effort everywhere — tracing never breaks a turn.

### The observability privacy boundary

LangSmith is an **external cloud — a separate trust boundary from the LLM
provider**, so it gets its own switch, independent of the summarization flag:
`FES_LANGSMITH_LOG_CONTENT` (default **false**). The redaction is surgical, not
blanket — prompts stay readable; only the parts that carry Sisense data are
replaced with `[… hidden · N chars]` placeholders:

| Content | flag=false | flag=true |
|---|---|---|
| System prompts, user text, plan text, route outputs, tokens/cost | shown | shown |
| Tool results embedded in decide/verify prompts (`role: tool`) | hidden | shown |
| **Result-derived text** (adaptive value-passing — decide/replan lift values like a group name into later step texts; taint-tracked via `mark_tainted`) | hidden | shown |
| Outputs of answer-producing calls (decide/finalize/verify/replan) + final reply | length-only | shown |
| Tool spans' result payloads | **never** | **never** |

### Local CSV twins (no cloud required — opt-in via `FES_CSV_OBSERVABILITY`)

Both observability destinations are **off by default**, each with its own
switch: `LANGSMITH_TRACING` for the cloud tree, `FES_CSV_OBSERVABILITY` for the
local files. The mutations audit log is always on (audit, not observability).

| File | One row per | Grouped by |
|---|---|---|
| `logs/llm_calls.csv` | LLM call (type, tokens, latency, ok) | `trace_id` (one per turn) |
| `logs/tool_calls.csv` | MCP tool execution (tool_id, ok, count, latency) | `trace_id` |
| `logs/llm_traces.csv` | whole turn (outcome, steps, tool) | — |

---

## Hand-rolled building blocks (and their LangGraph future)

Everything is **hand-rolled** — plain Python control flow inside
`backend/agent/llm_agent.py`, no agent framework. This is deliberate: the
machinery is built by hand so each piece's purpose is explicit, which is also
what makes the planned LangGraph refactor a mechanical mapping rather than a
redesign. This section is the glossary + that mapping.

### The building blocks, in our own terms

| Term | What it does | Where it lives now |
|---|---|---|
| **Plan (planner)** | request + capability catalog (tool one-liners, NO schemas) → ordered plan; step 1 seeds the loop, plan shown in UI | `_make_plan` (in `_reactive_loop` when `steps==0`) |
| **Replan (planner)** | failed approach + catalog → revised plan for remaining work, or GIVEUP; budget `FES_MAX_REPLANS` | `_replan` / `_attempt_replan`; decide's `REPLAN:` verb + routing/planning dead-ends trigger it |
| **Fan-out (level 1+2)** | independent (untagged) plan steps execute concurrently — per-branch route→plan→validate→execute, joined in plan order; mutations/missing-args/dead-ends defer to the sequential loop; width `FES_MAX_PARALLEL_STEPS` | `_execute_branch` + `asyncio.gather` in `_reactive_loop` |
| **Route** | narrows 119 tools → one package → one mixin (~10 tools) | `_navigate_to_tools` (`_routing.py`) |
| **Plan** | given one sub-task + ~10 tools, pick ONE tool + args | planning `call_llm_raw(..., tools=...)` |
| **Execute** | run the chosen tool via MCP → Sisense | `mcp_client.invoke_tool` |
| **Decide** | given goal + history, answer / `CONTINUE:` / (summ off) `BLOCKED:`/`DONE` | `_reactive_loop` (decide call; prompt is data or nodata) |
| **History visibility** | full result (summ on) vs metadata only (summ off) — the privacy boundary | `_transcript_step` / `_metadata_record` |
| **Loop control** | ONE while-loop: decide→route→plan→execute for step 1 through N; both modes | `_reactive_loop` (`while True`) |
| **Pause/resume state** | externalized state so a turn can stop and continue next turn | `SessionEntry.pending_loop` / `.pending_clarification` |
| **Mutation gate** | stop before a destructive tool, wait for approval | `REQUIRE_MUTATION_CONFIRM` check + `pending_confirmation` |
| **Clarification** | stop and ask when a required arg is missing | `_generate_clarification_question` + `LAST_PENDING_CLARIFICATION` |
| **Graceful stop** | every exit returns readable text, never a silent halt | `_finalize_from_transcript`, `_loop_partial_message` |
| **Progress** | emit a step event each phase | `_emit_agent_progress` |

### Mapping to LangGraph _(designed, deferred — MCP OAuth first)_

Decision (2026-07-31): the migration is **designed but deferred** until after
MCP OAuth. The graph below is the ready-made blueprint — a structural
re-expression of `_reactive_loop`, NOT a behavior change. Node names follow the
**Plan-and-Execute / ReAct** conventions.

**The node graph:**

| Node | Kind | Absorbs (current code) |
|---|---|---|
| `planner` | LLM | `_make_plan`, capability catalog, dependency tags, faithfulness guard, `_split_dependent_tail` |
| `router` | LLM×2 | `_navigate_to_tools` (L1/L2) + backtrack widening |
| `agent` | LLM | tool-selection call (~10 schemas) + `planner_schema` |
| `validator` | code | jsonschema validate, `_missing_required_fields`, hard block |
| `tools` | code/IO | `_invoke_tool_traced` → MCP invoke |
| `replanner` | LLM | decide (CONTINUE/DONE/REPLAN/BLOCKED) + `_replan` + budget |
| `evaluator` | LLM | `_verify_goal_complete` (the critic) + recheck budget |
| `respond` | LLM/code | `_finalize_from_transcript`, `_describe_results_local`, `_done` |

```
planner ─Send(independent steps)→ router → agent → validator ─missing→ interrupt: clarify
                                                        │ok
                                                   mutating? ─yes→ interrupt: approval
                                                        │no
                                                      tools → replanner ─CONTINUE→ router
                                                                 │REPLAN→ planner
                                                                 │DONE→ evaluator ─INCOMPLETE→ router
                                                                              │COMPLETE→ respond
```

**Framework primitives replacing custom machinery:**

| Our hand-rolled piece | LangGraph primitive |
|---|---|
| `_reactive_loop` `while` loop + its if/elif control flow | the **graph** (nodes + **conditional edges**) |
| `pending_loop` / `pending_clarification` in `SessionEntry` | **checkpointer** (persistent, `thread_id` = session; survives restarts) |
| Mutation gate + clarification pause → return, resume next turn | **interrupt()** (human-in-the-loop) |
| Transcript / results / budgets / `LAST_*` globals | the graph **state** (TypedDict + reducers) |
| Fan-out (`_execute_branch` + `asyncio.gather`, plan-order join) | **Send API** + a state **reducer** |
| Graceful-stop terminal returns | **END** node / terminal edges |

**Unchanged in a migration:** `_prompts.py`, the privacy boundary
(`_transcript_step`/`_metadata_record`), registry + routing data, `mcp_client`,
the eval battery. **Caveat:** LangGraph's native LangSmith tracing ships full
state and would bypass our redaction — keep `_tracing.py` as the reporter.

The point of the left column: none of these are LangGraph inventions — they are
real problems any agent hits (how to loop, pause, resume, stop safely, keep a
human in the loop). We solved each by hand first; LangGraph just gives each a
named, reusable primitive. That is the honest answer to "why LangGraph" — not
"it makes agents," but "it replaces this hand-rolled plumbing with tested
primitives, and makes each node independently testable."

---

## What's next _(planned)_

- **MCP OAuth + Claude connector.** Bearer/OAuth on the MCP server so it can be
  a hosted, standalone connector. (See the separate `sisense-admin-mcp` brief.)
- **LangGraph.** Refactor the hand-rolled loop into named graph nodes
  (plan / execute / decide / approval / fan-out).
- **Level-3 workers (true MAS)** — each independent sub-goal running its OWN
  reactive loop (own transcript, own decide/replan, own budget) and reporting a
  summary back to the orchestrator. Fan-out today (built — see below) is level
  1+2: concurrent single-shot pipelines joining one shared transcript — one
  agent doing parallel I/O, not multiple agents. The graduation test: count the
  conversations, not the LLM calls.

---

## Design rules that must not break

| Rule | Why |
|---|---|
| Planner sees ~10 tools from one package, picks one | Avoids hallucination / noise from the 119-tool catalog |
| Mutation gate never bypassed, even mid-loop | Safety — no destructive action without per-op approval |
| Result **data** never reaches the LLM when summarization is off (action metadata may) | Privacy kill-switch is absolute; enforced in code by what `history` contains, never by asking the model not to look |
| `MAX_AGENT_STEPS` is a hard ceiling | Runaway-loop backstop; caps cost per turn |
| Every loop exit returns readable text | The user never sees a silent dead stop |
