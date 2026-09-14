# Skills — design

**Status:** draft for review · **Branch:** `design/skills` · **Author:** FES · **Date:** 2026-09-13

---

## 1. What this is

A **skill** is a Sisense-authored procedure the agent can plan from: a document that
says what to achieve, in what order, why that order, what to never do, and how to
undo it. The agent reads it and produces a concrete plan for *this* request; code
validates that plan, gathers one approval, and executes it step by step with
checkpoints and compensation.

The motivating case is **"make this data model AI-ready"** — analyze what its
dashboards need, build a perspective from that, move the dashboards onto it, verify
every widget still answers, clean up. Fifteen or so tool calls, five mutation kinds,
a loop over dashboards, a conditional (swap an original only if its copy validated),
and cleanup that must run even on failure.

That workflow cannot be done well by either of the two things the agent has today:

| | reactive loop | hardcoded flow (`migration_flow.py`) |
|---|---|---|
| Fifteen steps | capped at `FES_MAX_AGENT_STEPS=8` | fine |
| Step N needs step N-1's result | only with summarization ON; production default is OFF | fine |
| A loop over dashboards | the planner writes a flat list | fine |
| A conditional | cannot express | fine |
| Five mutations × N dashboards | one approval dialog **each** | one approval |
| Cleanup on failure | a model may declare done early | guaranteed |
| Adding the *next* workflow | free | a new Python module per workflow |

The loop is flexible and unreliable at this length; the flow is reliable and turns
the agent into a router to scripts. Skills take the first row of that table from the
loop and every other row from the flow.

**Consumer.** This is a Sisense product feature. The primary user is any Sisense user
at any customer, token-scoped — a viewer, a designer, an admin. Secondary users are
Sisense support and FES engineers running a review with their own token. That fixes
several decisions below: skills are Sisense-authored content, not customer
configuration; there is no privilege escalation of any kind; and the read-only
side — two reads the agent already does — is the part most users will ever run.

---

## 2. Principles

1. **The LLM decides *what* from the skill. Code makes sure *only that* happens.**
   The model is in the planning path — it reads the skill and live state and emits
   a typed plan — and in the narration path. It is never in the execution path
   deciding step 9 from step 8's result.
2. **Skill teaches, runtime guarantees.** A skill can *ask* for anything. The two or
   three constraints where a mistake is expensive are also *enforced* in code,
   regardless of what the plan says.
3. **Run exactly what was approved.** Execution takes the plan artifact, never the
   request. Replanning after approval is forbidden — the plan the human read is the
   plan that runs.
4. **Token-scoped, no escalation, ever.** The signed-in user's Sisense permissions
   are the boundary. Ownership changes are not a step a skill may plan. A dashboard
   the user cannot publish is reported with its owner's name, not worked around.
5. **Data flows between steps in code, not through the model.** A skill run works
   identically with summarization ON and OFF, because step N's result reaches step
   N+1 by reference resolution in the runtime, not by the LLM reading it.
6. **Skills are product content.** Versioned in the repo, shipped in the image,
   held to the same bar as tool descriptions, guarded by tests. A customer never
   writes one.

---

## 3. What a skill is

One **directory per skill** under `skills/` at the repo root, following the Agent
Skills format (the layout this repo already uses for its Claude Code skills at
`.claude/skills/run/SKILL.md`):

```
skills/
  optimize-datamodel-for-ai-assistant/
    SKILL.md          ← frontmatter + procedure (required)
    scripts/          ← optional: deterministic helpers a step may invoke
    references/       ← optional: longer material loaded only on demand
```

Root, not `config/`: a skill is something the agent *knows how to do* — product
content, closer to code than to settings — and it deserves a top-level home the
way `backend/` and `mcp_server/` do. Not `.claude/`: those are dev-tooling skills
for working *on* the repo; these ship *in* it.

`SKILL.md` is YAML frontmatter + a markdown body.

```markdown
---
name: optimize-datamodel-for-ai-assistant
description: Optimize a data model for Sisense AI Assistant — build a perspective
  containing only what its dashboards actually use, move those dashboards onto
  it, and verify every widget still answers.
version: 1
requires_role: dataDesigner
tools:
  - datamodel.analyze_perspective_requirements
  - datamodel.create_perspective
  - datamodel.deploy_datamodel
  - dashboard.get_dashboards_by_datasource
  - dashboard.duplicate_dashboard
  - dashboard.replace_datasource
  - dashboard.validate_dashboard_queries
  - dashboard.delete_dashboard
guardrails:
  - id: validate-before-swap
    rule: replace_datasource on a dashboard NOT created by this run requires a
      prior validate_dashboard_queries on its stage copy with zero failed widgets
---

## When this applies
…

## Procedure
1. …

## Never
- …

## On failure
- …

## Report
- …
```

**Frontmatter is the machine-readable contract; the body is for the planner.**

| key | purpose | enforced by |
|---|---|---|
| `name` | stable id; appears in plan artifacts, audit log, traces | — |
| `description` | one line, shown in the skill index the planner always sees | — |
| `version` | bumped on any body change; recorded in every plan artifact | — |
| `requires_role` | minimum Sisense role for the write steps | pre-plan role check (§6) |
| `tools` | the **only** tool ids a plan from this skill may contain | plan validation (§7); drift guard (§10) |
| `guardrails` | constraints the runtime enforces regardless of the plan | plan validation + runtime (§7, §8) |

The `tools` list is a per-skill **allowlist within the allowlist** — a plan step
naming any other tool is rejected before it is shown. This is how "never change
ownership" becomes impossible rather than discouraged: `change_dashboard_owner` is
not in the list.

**Body sections** are conventional, not parsed. The planner reads prose. The
convention exists so every skill answers the same five questions in the same order,
and so review is possible.

- **When this applies** — the intent, in the user's vocabulary.
- **Procedure** — numbered steps. Each names its purpose *and* its tool, and says
  what the step's output is for. Reasons over instructions: *"use analyze's
  `perspectives` output as-is — it already includes join columns the dashboards
  never reference directly"* teaches something the planner can apply when the
  situation varies; *"call create_perspective with X"* does not.
- **Never** — what no plan from this skill does. Restated in `tools` and
  `guardrails` where code can hold it.
- **On failure** — compensation per step kind, and what to report.
- **Report** — what the final reply must contain.

**Naming tools in the body: yes.** It removes routing ambiguity entirely (the
"columns" misroute of 2026-09-04 cannot happen to a step that names its tool), it
makes the selector's job trivial, and it is falsifiable — a name can be checked
against the registry; "the columns tool" cannot. Always pair the name with the
purpose so a rename leaves the intent intact for the drift guard to point at.

---

## 4. Where skills live and how they load

- **Path:** `skills/<name>/SKILL.md`. `FES_SKILLS_DIR` overrides the root.
- **Shipped in the image**: `COPY skills ./skills` in `Dockerfile.backend`, beside
  `config/`. (The UI image gets it too if the capability browser lists skills.)
- **Loaded** by `_registry.py` next to the registry: parsed once, **mtime-cached**,
  re-read when a file changes — so an edit takes effect without a restart, the same
  contract the allowlist has.
- **Parse failures are loud.** A skill with malformed frontmatter, an unknown key,
  a tool not in the registry, or a tool not allowlisted is **excluded and logged
  at ERROR** — not shipped broken, not silently dropped. The drift guard (§10)
  catches the same conditions in CI so they never reach a deployment.
- **Missing directory = no skills.** The agent behaves exactly as today. This is
  the kill switch in its simplest form; `FES_SKILLS_ENABLED=false` is the explicit
  one.

---

## 5. Skill selection — progressive disclosure

Two stages, the same shape as hierarchical tool routing (L1 package one-liners →
L3 full schemas for one mixin) — and the same shape the Agent Skills format is
built around: frontmatter is the index, the body loads on match, `references/`
loads only when a step needs it.

**Stage 1 — the index is always in the planner's context.** Every loaded skill as
`name: description`, one line each, appended to the capability catalog under a
`Procedures available:` heading. Ten skills is ~200 tokens.

**Stage 2 — the body loads on match.** The planner's output format gains one
optional first line: `SKILL: <name>`. When present, `_make_plan` re-runs once with
the skill body prepended to the planning prompt and emits the typed plan (§6). When
absent, the turn proceeds exactly as today.

Why not a separate router call: it is another LLM call per turn for every turn,
and the planner already reads the request against a catalog — one more section of
that catalog is the cheapest possible signal. Why not keyword matching: the same
reason the L1 router is a model and not a regex.

**Chat mode only.** Migration mode has its own single-shot planner and no read
tools; skills do not apply there. (Whether migration itself becomes a skill later
is an open question, §12.)

---

## 6. Planning from a skill — the typed plan

A skill run's planner output is not a prose list. It is a **plan artifact**:

```json
{
  "skill": "optimize-datamodel-for-ai-assistant",
  "skill_version": 1,
  "request": "make Sample ECommerce AI ready",
  "steps": [
    {"id": 1, "tool": "datamodel.analyze_perspective_requirements",
     "args": {"datamodel": "Sample ECommerce"}},
    {"id": 2, "tool": "datamodel.create_perspective",
     "args": {"datamodel": "Sample ECommerce", "name": "Sample ECommerce_AI"},
     "args_from": {"tables": "steps[1].result.perspectives.tables"}},
    {"id": 3, "tool": "datamodel.deploy_datamodel",
     "args": {"datamodel_name": "Sample ECommerce_AI"}},
    {"id": 4, "tool": "dashboard.get_dashboards_by_datasource",
     "args": {"datamodel": "Sample ECommerce"}},
    {"id": 5, "for_each": "steps[4].result[*]", "as": "dash", "steps": [
        {"id": "5a", "tool": "dashboard.duplicate_dashboard",
         "args_from": {"dashboard": "dash.oid"}},
        {"id": "5b", "tool": "dashboard.replace_datasource",
         "args": {"datasource": "Sample ECommerce_AI"},
         "args_from": {"dashboard": "steps[5a].result.oid"}},
        {"id": "5c", "tool": "dashboard.validate_dashboard_queries",
         "args": {"datasource": "Sample ECommerce_AI"},
         "args_from": {"dashboard": "steps[5a].result.oid"}},
        {"id": "5d", "tool": "dashboard.replace_datasource",
         "args": {"datasource": "Sample ECommerce_AI"},
         "args_from": {"dashboard": "dash.oid"},
         "when": "steps[5c].result.failed == 0"},
        {"id": "5e", "tool": "dashboard.delete_dashboard",
         "args_from": {"dashboard_id": "steps[5a].result.oid",
                       "title": "steps[5a].result.title"}}
    ]}
  ]
}
```

Three step shapes cover the workflow and, as far as we can see, any procedure worth
writing down:

| shape | meaning | resolved |
|---|---|---|
| **literal** | `args` known at plan time (the user named the model) | at planning |
| **derived** | `args_from` maps a param to a path into an earlier step's result | at runtime, in code |
| **for-each** | run a sub-sequence once per item of an earlier result | at runtime, in code |

Plus `when`, a boolean over earlier results, for the conditional. The path language
is deliberately tiny: `steps[<id>].result`, dotted keys, `[*]`, and the loop
variable. If a skill needs more than that, the skill is doing something the runtime
should do.

**This is the part that makes summarization irrelevant to a skill run.** The model
never sees step 1's result; it wrote *a reference to it*. The runtime resolves the
reference. That is why a fifteen-step skill works with the privacy switch off, where
the reactive loop's dependent chains block.

**Permission check happens here, before anything is shown.** `requires_role` is
compared against the signed-in user's role (`access_management.get_my_user`, one
cheap read). A viewer asking to run `optimize-datamodel-for-ai-assistant` gets: *"This needs a data designer.
I can run the readiness check for you and give you the plan to hand to one."* — the
report (§11) instead of a plan whose first write would fail. Token-scoping still
enforces; this is the courtesy that turns a dead end into a handoff.

The plan artifact is **persisted** (`logs/plans/<plan_id>.json`) before it is shown.
It is the audit record of what was proposed, and what approval binds to.

---

## 7. Validation — before the plan is shown

Every plan goes through, in order:

1. **Tool set.** Every `tool` ∈ skill `tools` ∩ global allowlist. Anything else →
   reject with the offending step named.
2. **Schema.** Every literal `args` validates against the tool's JSON Schema. For
   `args_from`, the *referenced path* must point at a step that precedes it (or the
   loop variable), and the target param must exist in the schema. Type is checked
   at runtime when the value exists.
3. **Guardrails.** Each declared guardrail is a predicate over the plan structure.
   `validate-before-swap`: any `replace_datasource` whose `dashboard` is not the
   output of a `duplicate_dashboard` in this plan must be guarded by a `when` that
   references a `validate_dashboard_queries` on that dashboard's stage copy. Plans
   that fail a guardrail are rejected, and the reason is shown to the user in plain
   words — not silently repaired.
4. **Budget.** Step count after loop expansion is bounded (`FES_SKILL_MAX_STEPS`,
   default 60) — a loop over 400 dashboards is a job, not a turn, and should say so.

A plan that fails validation is **not** retried by re-asking the planner blind. The
failure is fed back once (`_replan` with the validation error), then the turn ends
with the error explained. Two bad plans in a row is a skill bug, not a user problem.

---

## 8. Approval — one dialog, keyed on the artifact

Generalizes what `migration_flow.py` does with `PLAN_TOOL_ID = "migration.plan"`:

- Gate key: `("skill.plan", canonical-JSON of the plan artifact)`. Editing or
  reordering any step re-gates. Still single use via `_consume_approval`.
- The dialog lists **every step that mutates**, humanised, in execution order, with
  the arguments that are known and, for derived ones, *what they come from*:
  *"Swap 'Sales Overview' (and up to 3 other dashboards found in step 4) to
  'Sample ECommerce_AI' — only those whose copy validated clean."* Built in code
  from the plan, never from the model's summary.
- Reads are listed collapsed, so the reader sees the shape of the run without
  wading through it.
- Where the plan deviates from the skill's own suggested defaults (a name the user
  chose, a step the planner added), the dialog says so.

One approval, not one per mutation. The steps are sequential and the plan is
complete before it runs; asking per step would repeat the same question with no new
information, and fifty clicks over ten dashboards would train the user to stop
reading — which is the outcome the gate exists to prevent.

**Durable approval** (later, §13): today approval must arrive in the next turn, over
HTTP. A plan should survive the user closing the tab and approving in an hour. That
means persisting pending plans keyed on session + plan id and resuming from disk.
Not in the first cut.

---

## 9. Execution — the runtime

`backend/agent/skill_runtime.py`. Takes an approved plan artifact. Never calls the
planner.

- **Sequential**, in plan order. Loops expand at runtime from the referenced result.
- **Resolve** each step's `args_from` and `when` against the results so far. A
  reference that does not resolve (empty result, missing key) is a **step failure
  with a clear reason**, not a guess.
- **Checkpoint after every step**: `logs/plans/<plan_id>.state.json` records
  results and status per step. This is what resume reads.
- **Idempotency**: before a create-shaped step, check for the thing it would create
  (perspective by name; stage copy by its `_perspective_stage` title) and **adopt**
  it if present rather than making a second one. The SDK's deterministic stage
  suffix exists for exactly this.
- **Stop on first failure.** Then run **compensations in reverse** for completed
  steps that declare one:

  | step | compensation |
  |---|---|
  | `create_perspective` | `delete_perspective` |
  | `deploy_datamodel` | none — a built perspective is harmless |
  | `duplicate_dashboard` | `delete_dashboard` (the copy) |
  | `replace_datasource` on a copy | covered by deleting the copy |
  | `replace_datasource` on an original | `replace_datasource` back to `previous_datasource` (the SDK returns it) |

  A compensation that itself fails is reported loudly and stops the unwind; it is
  never retried silently.
- **Report** ran / failed (with the SDK's own error, verbatim) / not attempted /
  compensated — the same table migration reports, plus the skill's `Report` items
  (here: which dashboards now need their owner to publish, by name).
- **Guardrails are re-checked at runtime** with actual values, not only at plan
  time: `5d` will not execute if `5c`'s `failed` is non-zero even if a plan somehow
  claimed otherwise. Plan-time validation catches structure; runtime enforcement
  catches data.

**Cancellation** reuses `runtime.cancel_active_turn`: the runtime checks the
session's cancel flag between steps and treats cancel as a failure at that point
(compensations run). The UI's Stop button (§11) wires to it.

---

## 10. Testing

Three layers, matching the three things that can be wrong.

**The skill file (unit, `tests/unit/test_skills.py`):**
- Frontmatter parses; only known keys; `version` is an int; `requires_role` is a
  real Sisense role.
- **Drift guard**: every id in `tools` exists in the shipped registry *and* the
  allowlist. Every tool id mentioned in the body also does. A rename or delisting
  fails CI instead of breaking a customer six months later. Same pattern as
  `TestSchemaRulesDrift` and `test_option_enrichment`.
- Every guardrail id is one the validator implements.
- No compensation references a tool outside `tools`.
- **Allowlist comments agree with the registry.** Every `# [write]` marker in
  `config/allowed_tools.txt` must match that tool's `mutates` flag, and every
  unmarked exposed tool must be `mutates: false`. Script 04 writes those
  comments when it *stages* a tool and never refreshes them, so a later
  generator fix leaves a stale marker behind — found 2026-09-13, when the two
  read-only perspective tools still read `[write]` in the file after the
  `mutates` fix. Nothing enforces from the comment, but it is what a human
  reads when deciding whether to uncomment a line. Five lines; belongs beside
  the other drift guards, not in a skill test — the skill just exposed it.

**The runtime (unit, mocked MCP):** reference resolution, loop expansion, `when`,
stop-on-failure ordering, compensation ordering, idempotent adopt, checkpoint and
resume from a mid-run state file, cancellation between steps.

**The plan (eval, `tests/integration/test_evals_skills.py`):** the contract of a
skill is *request → plan*. Cases assert the step sequence, that `requires_role`
blocks a viewer with the handoff message, that a plan naming a tool outside the
skill's set is rejected, and that the guardrail rejects a plan missing the
validation gate. Same harness discipline as the planner battery, no approvals
sent, nothing written.

---

## 11. What the user sees

Chat mode already streams `agent_progress` events over SSE: the plan, a checklist
that ticks per step, a status line per phase. A skill run publishes the same events
and adds three things:

1. **An outcome line per step**, code-built from metadata, safe in both summarization
   modes: `✅ Found 4 dashboards on Sample ECommerce` rather than
   `✅ Step 4: dashboard.get_dashboards_by_datasource`. The human label comes from the
   registry description's first line; the count from `{ok, count}`.
2. **Loop sub-progress**: `Dashboard 2 of 4 — validating…`.
3. **Step result expanders as each step lands**, not at the end. Result *data* on
   screen is allowed in both modes — the privacy switch governs the LLM's view, not
   the user's — so nothing here waits for the run to finish.

Plus a **Stop** button in chat mode, wired to the existing cancel path. A
fifteen-step run without a way to stop it is not a product.

**The read side is the same skill's other branch — not a new tool, and not
free.** "Is Sample ECommerce ready for AI Assistant?" resolves to two
independent reads — `analyze_perspective_requirements` and
`get_dashboards_by_datasource`, both keyed only on the model name — which the
runtime fans out concurrently in both summarization modes. The *mechanism*
exists today. The *knowledge* does not: nothing in the tool catalog mentions
"AI Assistant", so without the skill the planner has no reason to connect that
phrase to those two tools. The skill's `When this applies` section is what
makes the connection, for both phrasings.

So one skill, two branches:

| asked | plan | gate |
|---|---|---|
| *"is X ready for AI Assistant?"* | analyze → find dashboards → report | none — nothing mutates |
| *"optimize X for AI Assistant"* | the full procedure (§6) | one approval |

The read branch is the write branch's first steps plus a `Report`, and the
skill body tells the planner to stop there when the user only asked *whether*.
A composite `check_ai_readiness` tool was considered and rejected: it would have
to be hand-added to a registry that is otherwise generated from the SDK, and it
would duplicate two tools that already exist.

This is also the viewer's path: a user without the role for the write branch
gets the read branch and the handoff message (§6), not a plan that would fail
on its first write.

---

## 12. Open questions

- **Should migration become a skill?** It fits the model (plan once, approve once,
  execute in order) and would delete `migration_flow.py`. But migration is a mode
  with its own credentials, and skills are chat-mode. Decide after the first skill
  has run in production, not before.
- **Plan-time vs runtime resolution of names.** `create_perspective` needs a name.
  The skill suggests `<model>_AI`; the user may have said one. Where the planner
  invents a name it must be visible in the dialog as invented. Is that enough, or
  should names always be asked?
- **Loop budget UX.** Sixty steps is the cap; what does the user see at step 40 of
  a 60-step run? Probably fine with sub-progress. Confirm on a real large model.
- **Multi-run hygiene.** Two people run `optimize-datamodel-for-ai-assistant` on the same model a day apart.
  Idempotent adopt handles the perspective; does it handle a half-finished stage
  copy from a run that was cancelled? Adopt-and-continue vs delete-and-redo.

---

## 13. Build order

Each step is shippable and useful on its own. Nothing below depends on a later step.

1. **Skill loading + index + drift guard** — files parse, the index appears in
   the planner context, CI catches a stale tool id. With this alone the READ
   branch works end to end through the existing reactive loop (two independent
   reads, no mutation, no runtime needed) — the feature most users can run, and
   proof the planner reads skills correctly. Add an eval case pinning the
   two-tool plan for "is <model> ready for AI Assistant?".
2. **Typed plan + validation + one approval** — the planner emits the artifact;
   validation, guardrails, role check; the dialog. Generalizes `PLAN_TOOL_ID`.
   Needed by the WRITE branch only.
3. **The runtime** — resolution, loops, `when`, checkpoints, idempotent adopt,
   compensation, report, cancel. The write branch becomes fully runnable.
4. **Progress UX** — outcome lines, loop sub-progress, expanders as they land, Stop.
5. **Durable approval** — persisted pending plans; approve an hour later.

Step 1 ships the read branch on its own. Steps 2–3 are the platform the write
branch needs; 4–5 polish it.

---

## Appendix A — the first skill

The skill lives at **`skills/optimize-datamodel-for-ai-assistant/SKILL.md`** — the real file, not a copy
here, so this document and the skill cannot drift apart. Read it alongside §3:
the frontmatter is the machine contract (`tools`, `guardrails`, `requires_role`),
the body is what the planner reads.

---

## Appendix B — how this relates to what exists

| existing | relationship |
|---|---|
| `migration_flow.py` | the precedent for plan-once / approve-once / execute-in-order; skills generalize its `PLAN_TOOL_ID` approval path. Migration itself stays as-is for now (§12). |
| `_capability_catalog` | gains the `Procedures available:` index (§5). |
| `_make_plan` / `_replan` | gain the `SKILL:` first line and a second pass with the body (§5); emit the typed plan when a skill is active (§6). |
| `_consume_approval` | unchanged; the plan artifact is just another `(tool_id, args)` key (§8). |
| `agent_progress` events | unchanged shape; new phases `skill_step_done` (outcome line) and `skill_loop` (sub-progress) (§11). |
| `SCHEMA_RULES` / `allowed_tools.txt` | untouched. A skill's `tools` list is a subset *within* the allowlist, never an addition to it. |
| `x-followup` | the natural place to offer `optimize-datamodel-for-ai-assistant` after the readiness reads run. |
| Prompts | untouched. Prompts carry generic strategy; a skill is loaded per intent, versioned and tested — which is exactly what makes a domain procedure legitimate to write down. |
