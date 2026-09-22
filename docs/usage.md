# Using the app

Two modes in the UI, selected in the sidebar. Everything runs with the Sisense
token you connect with — you see exactly what that token can see, and every
change asks before it runs.

---

## Chat with deployment

Connect to a single Sisense deployment and talk to an agent that can inspect and
operate on that environment.

- Select **Chat with deployment**.
- Enter the Sisense domain and either an API token or a username and password
  (the UI exchanges it for a token), plus your SSL preference.
- Click **Connect**.

Example questions:

- "Which columns are unused in fes_assistant and Sample ECommerce?"
- "Which columns are used in the dashboard Usage - Users?"
- "Which group does jane.doe@acme.com belong to, and show all members of that group."
- "Show me all users in the 'Analysts' group."
- "Create a datamodel called FES_Demo" — it asks for what it's missing.

For write operations (create / update / delete) you will see a confirmation
step before execution. **"What can I ask?"** in the sidebar lists every
operation available in the current mode, and which ones change something.

### What you see during a turn

The plan up front, a checklist that ticks as each step lands, and a status line
for the current phase — planning, running a tool, checking progress, rethinking
after a failure, double-checking the result. Then the answer, with each step's
result in a collapsible expander (tables, with CSV / JSON / TXT export) and the
full detail in the **Run log**.

### The summarization toggle

A checkbox in the sidebar, **off by default**. It decides whether data returned
from Sisense may be sent to the LLM:

- **Off** — the model sees only that a step ran and how many rows came back.
  Answers are built in code and shown on screen; the data never leaves for the
  model. Multi-step requests still run. A step that needs a *value* from an
  earlier result ("which group is X in, then list its members") stops honestly
  and tells you why, instead of guessing.
- **On** — results reach the model, so it can chain dependent steps, verify its
  own work, and answer in prose.

It is per turn, never remembered. Full detail in [security.md](security.md).

### Approvals

Nothing that writes runs without a dialog naming the operation and its exact
arguments, plus any optional settings the operation offers that you didn't set.
Approvals are **single use** — asking for the same thing again asks again. In
chat mode each mutating call is gated individually; a multi-step turn pauses at
the mutation and resumes after you approve. The chat input freezes while a
dialog is pending, so a new message can't race an unanswered approval.

### Clarifications

When a required detail is missing — "create a datamodel called X" without saying
which connection — the agent asks, shows an example of how you could phrase it,
and where it can, tells you how many valid options exist (with a few example
names shown under the reply). Answer in the next message and it continues from
where it stopped. Change the subject instead and the pending question is dropped.

---

## Migrate between deployments

Connect **source** and **target** Sisense environments and move assets between
them.

- Switch to **Migrate between deployments**.
- Fill in source and target (domain + token + SSL) and connect both.

Example requests:

- "Migrate the Sales Team group and the user jane@acme.com."
- "Migrate all datamodels, overwriting existing ones."
- "Migrate the groups, users and dashboards to the target environment."

The whole request is planned in one shot and shown as a **single numbered
approval dialog**, ordered by dependency (groups → users → datamodels →
dashboards) regardless of the order you said them in. Nothing runs until you
approve; a failed step stops the run; the final summary is built from the SDK's
own counters (succeeded / failed / not attempted).

Long migrations stream progress live — milestones in the sidebar, full detail in
the Run log — and **Stop** cancels the run mid-flight.

---

## UI conveniences

- Per-answer caption showing the turn's token usage and cost.
- Thumbs up/down feedback on every answer, written to `logs/feedback.csv`
  (joins the observability CSVs by `trace_id`).
- Export buttons on tool results — CSV, JSON, or TXT.
- Nested results are flattened into tables; the raw JSON stays one click away.
- Domain normalization on connect — `mycompany.sisense.com` defaults to `https://`.
- Sessions idle out after `FES_UI_IDLE_TIMEOUT_HOURS` (default 9).
