# Security & data handling

What leaves your infrastructure, what reaches the LLM, and what to do about it.
Everything here is enforced in code, not by instructing the model — a prompt can
be ignored; these cannot.

---

## Where things run and what they can see

**Deployment & execution control**

- **Local SDK usage (PySisense):** all processing logic runs locally on your
  machine or server. No data is transmitted to Sisense Field Engineering.
- **Self-hosted components (FES Assistant / MCP server):** designed for
  deployment within your own environment (on-prem or VPC). You maintain complete
  control over infrastructure, security configuration, access controls, and logs.

**Data & LLM handling**

- **LLM feature status:** the summarization feature is **disabled by default**.
- **Data transmission:** when summarization is enabled, responses retrieved via
  the Sisense SDK may be sent to your chosen LLM provider for processing.
- **Customer responsibility:** you are responsible for selecting an LLM provider
  that meets your organization's data privacy and security requirements.
- **Optional observability (LangSmith):** tracing is **disabled by default**
  (`LANGSMITH_TRACING=false`). If you enable it, trace metadata is sent to
  LangSmith (a third-party SaaS by LangChain, Inc.) under **your own** LangSmith
  account/API key. Tool result payloads are never sent; prompt/response content is
  additionally gated by `FES_LANGSMITH_LOG_CONTENT` (default `false`). Local CSV
  logging (`FES_CSV_OBSERVABILITY`, on by default) stays on your machine and never
  contains Sisense result data.

**Network shape** — two things the deployment is deliberately honest about:

- **Nothing in this repo terminates TLS.** Nginx listens on plain `:80`, so HTTPS
  has to come from a load balancer or proxy you put in front of it.
- **Only the UI is ever exposed.** Nginx publishes the single host port and
  proxies to Streamlit; the backend and the MCP server sit on an internal Docker
  network and are unreachable from outside the instance. The MCP server is an
  internal component of this application, not a public endpoint — see
  [`mcp_server/README.md`](../mcp_server/README.md) for how it differs from a
  generic MCP server and what connecting to it directly would require.

Running locally with `docker-compose.yml` differs in two ways: there is no Nginx
(the browser hits Streamlit on `:8501` directly), and the backend and MCP ports
bind to `127.0.0.1` only, since neither service authenticates its callers.

---

## The summarization switch — exactly what the LLM sees

The summarization switch decides whether **data returned from Sisense** may be
sent to your LLM provider. It is enforced in code at a single point
(`_transcript_step` → `_metadata_record` in `backend/agent/llm_agent.py`), not by
instructing the model.

### Defaults and control

| | |
|---|---|
| Default | **OFF.** `ALLOW_SUMMARIZATION` is a backend-side hard cap — when `false`, no request can send result data to the LLM, regardless of the UI checkbox. The checkbox itself always starts OFF, and the API treats a missing `allow_summarization` field as `false` |
| Per request | Every `/agent/turn` call carries its own `allow_summarization`. Two users, or two turns by the same user, can differ |
| User control | A checkbox in the UI sidebar, sent with each turn. Hiding it with `FES_ALLOW_SUMMARIZATION_TOGGLE=false` also forces summarization off for every request |
| Scope | One turn. It is never remembered or inferred, and no model output can change it |

### Summarization OFF

Per executed step, the model's history receives **only**:

```json
{"tool": "access_management.get_user", "ok": true, "count": 12}
```

`count` appears only for list results. **No rows, no field values, no payload** —
for a successful call the model never learns anything Sisense returned, only that
something was returned and how many.

The model still sees, as it must to function at all:

- **your request**, verbatim — it cannot pick a tool or fill arguments otherwise.
  This includes anything you re-type from results shown on your screen: the
  switch governs what the *application* forwards from tool results, never what
  *you* choose to say
- **prior turns** of the conversation (`LLM_PLANNING_HISTORY_TURNS`, default 5)
- **tool names, descriptions and parameter schemas** for the ~10 tools routing selected
- **the arguments it proposes**, which derive from your words, not from results
- **the failure reason when a step fails** — see below

What the loop gives up in this mode:

- **Adaptive chains are refused, not attempted.** A step needing a value from an
  earlier result (`[needs-prior-result]`) is skipped up front, or the turn stops
  with `BLOCKED` and says so. It never guesses the value. (Example: "get user X,
  then list all users with X's role" — step 2 needs the role from step 1's
  result, which the model can't see, so the turn stops after step 1 and says
  why.)
- **The critic is off.** Judging whether a goal was met requires reading results.
- **Answers are rendered locally.** The final reply is built in code from the raw
  results (`_describe_results_local`) — the data goes to your screen, not to the model.

What still works in this mode — the switch is a data boundary, not a feature
kill:

- **Independent multi-step turns** ("list all datamodels AND all groups") run
  every step: knowing a step is done needs only `{ok, count}`, not the data.
- **Replanning after a failed step** still works — the decide call reasons from
  the metadata plus the failure reason (the one exception below).
- **Mutations, approvals, clarifications** are unchanged — the gate and the
  clarification questions are built in code either way.
- **Option lookups in clarifying questions** work identically in both modes.
  The question text carries the count of existing values (e.g. "I found 914
  existing options") and offers to list them as a follow-up turn; a few example
  names appear **below** the reply via a display-only field the UI never adds
  to the message text. Message text is what rides back to the model in history
  on later turns — so the names stay on your screen in every mode, and reach
  the model only if you type one (which is your input, always visible to it).
  A count is the same metadata the model already sees.
- **Migration turns lose nothing**: the plan is knowable from the request alone
  (no migration tool needs another's result), and the final summary is built in
  code from the SDK's own counters in **both** modes.

### The one exception: failure reasons

When a step fails, its `error` string is included:

```json
{"tool": "access_management.create_user", "ok": false,
 "error": "username/email already exists"}
```

**Why.** Without it the agent is blind exactly when it needs to think. In
practice: a create failed, the decide call saw `ok: false` and nothing else, and
it *invented* a cause — "ensure the email is not already in use." It happened to
be right. A recovery reasoned from a guess is worse than one reasoned from the
truth, and the alternative (a code table translating failures into approved
labels) replaces the agent's judgement with our own list of what can go wrong.

**The residual exposure, stated plainly.** An error usually restates what you
already typed — "username/email already exists" for the address *you* supplied —
so it rarely carries anything the model has not seen in your request. Not never,
and the size of "not never" is set by the SDK, not by us.

The message is built by PySisense (`utils._extract_error_message`, the single
place it constructs failure dicts). For a Sisense response it recognises, you get
that server's own sentence plus the status — *"Access denied (HTTP 403)"*. For a
response shape it does **not** recognise, it falls back to passing the body
through: an unfamiliar JSON object stringified, or raw non-JSON text, truncated
at 300 characters. That fallback is deliberate and correct — inventing a
friendlier message would mean discarding the only account of a failure nobody
anticipated — but it means the channel can carry whatever Sisense chose to put in
an error body, including a value you never typed: a row from a failing query, a
name from a list the tool had fetched.

Two bounds apply. Credential-shaped values are redacted upstream by the SDK
before the message is built, so tokens and passwords do not travel this path at
all. And the passthrough is capped at 300 characters — the untruncated body goes
only to `logs/pysisense.log` on your own disk.

If your threat model cannot accept the remainder, run with summarization off
**and** treat the error channel as in-scope for review; the behaviour is one
function (`_metadata_record`) and `tests/unit/test_summarization_boundary.py`
pins it.

### Summarization ON

Tool results are sent to your LLM provider, shrunk first
(`_shrink_for_llm`: caps on list length, object keys, depth, string length and
total size — a size guard, not a privacy one). Assume **any field of any record a
tool returned may reach your provider**. In exchange the agent can complete
adaptive chains, verify its own work with the critic, and write answers in prose.

---

## Everything else, regardless of the switch

- **Credentials are never sent.** Domain, token and SSL settings are stripped
  from arguments before any LLM call and scrubbed from audit logs.
- **Mutations require explicit approval.** Nothing that writes runs without a
  dialog naming the operation and its arguments. Approvals are **single use** — the
  same request again asks again. Every execution is recorded in `logs/mutations.log`.
- **Cloud observability is opt-in** — see *Optional observability (LangSmith)*
  at the top of this page. Local CSV logs carry request text + call metadata,
  never Sisense result data.
- **What lands on the host disk, and for how long.** Everything written stays
  in `logs/` on the machine running the stack — nothing is shipped in the
  images or the repo. At the default `FES_LOG_LEVEL=INFO` those files record
  *what happened* (tool, ok/failed, timing, the mutation audit) but not the
  rows Sisense returned; raising it to `DEBUG` adds full payloads, which then
  sit there for the 7-day retention. Application logs rotate daily and keep 7
  days, the observability CSVs roll at 50 MB keeping 5 rolls, and the audit
  logs are kept deliberately — so the directory is bounded, not unbounded,
  without an ops cron. See [operations.md → Logging](operations.md#logging).
- **These controls live in the backend**, which is the only thing that talks to
  the MCP server in a deployed instance — the server itself publishes no port
  and is not an entry point.
- **Token-scoped, not admin-only.** The agent does exactly what the supplied
  Sisense token's permissions allow — a viewer gets viewer capabilities, an admin
  gets admin. The Sisense API enforces that scoping on every call; the assistant
  never widens what you are allowed to do.

---

## Recommended usage guidelines

- **Environment:** use the tool primarily in sandbox or non-production environments.
- **Access:** utilize a dedicated Sisense service account with limited privileges.
- **Validation:** thoroughly review and validate the tool's behavior before any
  broader adoption within your organization.

## Deployment best practices

While this is an experimental tool, we recommend the following "security first"
approach for your deployment:

- **Authentication:** deploy the UI and backend behind your organization's SSO, a
  VPN, or a secure reverse proxy (e.g. Nginx with auth).
- **Credential management:** use a dedicated, limited-privilege Sisense service
  account and ensure your LLM API keys are stored securely (environment secrets,
  never hard-coded).
- **Network isolation:** implement network-level restrictions (firewalls / VPC
  rules) so only trusted internal hosts can reach the backend and MCP server
  endpoints.
- **Service scoping:** publish only the UI. The backend and MCP server
  authenticate no one, so they belong on an internal network — which is what
  `docker-compose.prod.yml` does. If you ever need to reach them from another
  machine, tunnel (SSH, Tailscale) rather than publishing the port.
