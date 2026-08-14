# 🤖 FES Assistant: Your Agentic Sisense Co-pilot

## ⚠️ Experimental Project Notice

### Community-Contributed Tool from Sisense Field Engineering

This project is an experimental tool developed by Sisense Field Engineering to facilitate customer learning and exploration of Sisense capabilities. While maintained by Field Engineering, it is shared "as-is" to encourage feedback and experimentation.

Important Disclaimer: This tool is not part of the core Sisense product release lifecycle and does not undergo the same validation, support, or certification processes as generally available (GA) Sisense features. It is intended to complement, not replace, officially supported Sisense features.

---

## Technical & Security Considerations

### Deployment & Execution Control
- Local SDK Usage (PySisense): All processing logic runs locally on your machine or server. No data is transmitted to Sisense Field Engineering.
- Self-hosted Components (FES Assistant / MCP Server): These components are designed for deployment within your own environment (on-prem or VPC). You maintain complete control over infrastructure, security configuration, access controls, and logs.

### Data & LLM Handling
- LLM Feature Status: The FES Assistant summarization feature is disabled by default.
- Data Transmission: When the summarization feature is enabled, responses retrieved via the Sisense SDK may be sent to your chosen Large Language Model (LLM) provider for processing.
- Third-Party Clients: When using the MCP Server with third-party clients (e.g., IDE agents or desktop assistants like Claude Desktop), data retrieved from Sisense is passed directly to the client’s LLM.
- Customer Responsibility: Customers are responsible for selecting an LLM provider that meets their organization’s data privacy and security requirements.
- Optional Observability (LangSmith): Tracing is **disabled by default** (`LANGSMITH_TRACING=false`). If you enable it, trace metadata is sent to LangSmith (a third-party SaaS by LangChain, Inc.) under **your own** LangSmith account/API key. Tool result payloads are never sent; prompt/response content is additionally gated by `FES_LANGSMITH_LOG_CONTENT` (default `false`). Local CSV logging (`FES_CSV_OBSERVABILITY`) is likewise opt-in and stays on your machine.

---

## Security & data handling — exactly what the LLM sees

The summarization switch decides whether **data returned from Sisense** may be
sent to your LLM provider. It is enforced in code at a single point
(`_transcript_step` → `_metadata_record` in `backend/agent/llm_agent.py`), not by
instructing the model — a prompt can be ignored, this cannot.

### Defaults and control

| | |
|---|---|
| Default | **OFF.** `ALLOW_SUMMARIZATION=true` in `.env` only sets the UI checkbox's starting position; the API treats a missing `allow_summarization` field as `false` |
| Per request | Every `/agent/turn` call carries its own `allow_summarization`. Two users, or two turns by the same user, can differ |
| User control | A checkbox in the UI sidebar, sent with each turn. Hide it with `FES_ALLOW_SUMMARIZATION_TOGGLE=false` to force whatever the caller sends |
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

- **your request**, verbatim — it cannot pick a tool or fill arguments otherwise
- **prior turns** of the conversation (`LLM_PLANNING_HISTORY_TURNS`, default 5)
- **tool names, descriptions and parameter schemas** for the ~10 tools routing selected
- **the arguments it proposes**, which derive from your words, not from results
- **the failure reason when a step fails** — see below

What the loop gives up in this mode:

- **Adaptive chains are refused, not attempted.** A step needing a value from an
  earlier result (`[needs-prior-result]`) is skipped up front, or the turn stops
  with `BLOCKED` and says so. It never guesses the value.
- **The critic is off.** Judging whether a goal was met requires reading results.
- **Answers are rendered locally.** The final reply is built in code from the raw
  results (`_describe_results_local`) — the data goes to your screen, not to the model.

### The one exception: failure reasons

When a step fails, its `error` string is included:

```json
{"tool": "access_management.create_user", "ok": false,
 "error": "username/email already exists"}
```

**Why.** Without it the agent is blind exactly when it needs to think. Observed
2026-08-08: a create failed, the decide call saw `ok: false` and nothing else, and
it *invented* a cause — "ensure the email is not already in use." It happened to
be right. A recovery reasoned from a guess is worse than one reasoned from the
truth, and the alternative (a code table translating failures into approved
labels) replaces the agent's judgement with our own list of what can go wrong.

**The residual exposure, stated plainly.** An error usually restates what you
already typed — "username/email already exists" for the address *you* supplied —
so it rarely carries anything the model has not seen in your request. Not never:
an error raised deeper in the stack can quote a value you did not supply, such as
a row from a failing query or a name from a list the tool had fetched. If your
threat model cannot accept that, run with summarization off **and** treat the
error channel as in-scope for review; the behaviour is one function
(`_metadata_record`) and `tests/unit/test_summarization_boundary.py` pins it.

### Summarization ON

Tool results are sent to your LLM provider, shrunk first
(`_shrink_for_llm`: caps on list length, object keys, depth, string length and
total size — a size guard, not a privacy one). Assume **any field of any record a
tool returned may reach your provider**. In exchange the agent can complete
adaptive chains, verify its own work with the critic, and write answers in prose.

### Everything else, regardless of the switch

- **Credentials are never sent.** Domain, token and SSL settings are stripped
  from arguments before any LLM call and scrubbed from audit logs.
- **Mutations require explicit approval.** Nothing that writes runs without a
  dialog naming the operation and its arguments. Approvals are **single use** — the
  same request again asks again. Every execution is recorded in `logs/mutations.log`.
- **Observability is opt-in.** `LANGSMITH_TRACING=false` and
  `FES_CSV_OBSERVABILITY=false` by default. With LangSmith on, tool result
  payloads are never sent and prompt content is further gated by
  `FES_LANGSMITH_LOG_CONTENT`. CSV logs stay on your machine.
- **Third-party MCP clients bypass all of this.** Point Claude Desktop or an IDE
  agent at the MCP server and results go straight to that client's model — the
  summarization switch lives in this backend and is not in that path.

---

## Recommended Usage Guidelines
- Environment: Use the tool primarily in sandbox or non-production environments.
- Access: Utilize a dedicated Sisense service account with limited privileges.
- Validation: Thoroughly review and validate the tool's behavior before any broader adoption within your organization.

---

## About FES Assistant

FES Assistant is an MCP-powered, agentic toolkit for Sisense environment operations. It helps you automate governance checks, migrations, and day-to-day admin workflows using natural language, so you can orchestrate tasks without writing one-off API scripts.

---

## 🚀 Why every Sisense user needs a Co-pilot:

* **📈 For Dashboard Designers:** Instantly find dashboards, audit your own widgets, and get environment well-checks without digging through menus.
* **🏗️ For Data Designers:** A specialized co-pilot for model optimization—find unused fields, audit M2M relationships, and build data models via natural language chat.
* **🛡️ For Admins:** An automation engine for environment/tenant migrations, bulk governance, and platform-wide orchestration-as-code.

---

![FES Assistant architecture](images/FES_ASSISTANT_AD.jpeg)

*For the full end-to-end execution flow (including SSE streaming, progress propagation, and the mutation approval loop), see [`Execution_Flow.md`](./Execution_Flow.md).*

---

## Key Agentic Capabilities

* **Multi-Step Planning & Self-Correction:** The agent breaks a request into steps, runs independent ones in parallel, chains dependent ones, and **replans** when an approach fails — verifying it actually met your goal before it answers.
* **Autonomous Infrastructure Audits:** Ask the agent to find Many-to-Many relationships, unused datamodel fields, or orphaned assets across your entire environment.
* **Zero-Touch Migrations:** Execute complex cross-tenant moves for dashboards and datamodels with built-in safety loops and confirmation steps.
* **Protocol-First Integration:** Operates as a **Streamable HTTP MCP Server**, allowing you to use this UI or plug Sisense "tools" directly into external agents like Claude Desktop.
* **Real-Time Progress Visibility:** Built with **Server-Sent Events (SSE)** to provide live streaming updates (V2) for long-running migrations and bulk tasks.
* **Privacy-First Logic:** Includes a manual **Summarization Toggle** to ensure raw data responses stay within your infrastructure when required.

---

## 🏗️ Architecture & Flow

The FES Assistant is built as a modular stack to ensure you can use the MCP server independently if desired:

- **The Cockpit:** A **Streamlit UI** (`frontend/app.py`) for mission control.
- **The Brain:** A **Backend API + Agent Layer** (`backend/api_server.py`, `backend/agent/`) running an **agentic loop** — a *planner* drafts the steps, an *orchestrator* (the loop) runs them (independent steps in parallel), and a *critic* verifies the goal before answering. Failed approaches trigger a replan. See [`AGENT_ARCHITECTURE.md`](./AGENT_ARCHITECTURE.md).
- **The Bridge:** An **MCP Streamable HTTP Server** (`mcp_server/server.py`) that translates AI intent into [PySisense](https://github.com/sisense/pysisense) SDK actions.

MCP Server docs: [Meta-Management MCP Server](mcp_server/README.md)

---

## Quick links

- [`docker-compose.yml`](./docker-compose.yml)
- [`docker-compose.prod.yml`](./docker-compose.prod.yml)
- [`Dockerfile.mcp`](./Dockerfile.mcp)
- [`Dockerfile.backend`](./Dockerfile.backend)
- [`Dockerfile.ui`](./Dockerfile.ui)
- [`.env.example`](./.env.example)
- [`config_prod.sh`](./config_prod.sh)

- [`frontend/app.py`](./frontend/app.py)
- [`backend/api_server.py`](./backend/api_server.py)
- [`backend/runtime.py`](./backend/runtime.py)
- [`backend/agent/llm_agent.py`](./backend/agent/llm_agent.py)
- [`backend/agent/mcp_client.py`](./backend/agent/mcp_client.py)

- [`mcp_server/server.py`](./mcp_server/server.py)
- [`mcp_server/tools_core.py`](./mcp_server/tools_core.py)

- [`Execution_Flow.md`](./Execution_Flow.md)
- [`refresh_registry.sh`](./refresh_registry.sh)

---

## Features

- **Two main modes in the UI**
  - **Chat with deployment**
    - Connect to a single Sisense deployment and talk to an agent that can inspect and operate on that environment.
  - **Migrate between deployments**
    - Connect **source** and **target** Sisense environments and use migration tools to move assets.

- **SSE progress streaming (V2)**
  - The UI streams agent turns and shows live progress updates.
  - Progress is captured into a per-run “run log” and rendered under assistant responses.
  - Works especially well for long migrations and bulk operations.

- **MCP-powered tools over PySisense**
  - PySisense SDK methods are wrapped as MCP tools and registered via a **tool registry JSON**.
  - Tools cover areas like access management, datamodels, dashboards, migration, and well-checks.

- **Two LLM backends (configurable) — via LiteLLM as an in-process gateway**
  - Switch between **Azure OpenAI** and **Databricks Model Serving** by changing environment variables.
  - All LLM traffic goes through one choke point (`call_llm_raw` → the **LiteLLM SDK**), which acts as a
    "gateway-as-a-library": unified API across providers, retries, provider-specific param handling —
    embedded in the backend process, with **no separate gateway service** deployed.
  - If centralized governance is ever needed (shared keys, per-team budgets, org-wide rate limits,
    cross-model fallback), the LiteLLM Proxy speaks the same interface — the single choke point means
    pointing `api_base` at a gateway is a config change, not a refactor.

- **Safety via confirmation loops**
  - For **create / modify / delete / migration**-style operations, the agent uses a **confirmation loop**:
    - The agent explains what it plans to do (which assets, which environments, what changes).
    - The UI shows this plan to the user.
    - The action is only executed after explicit confirmation.

- **Optional “no summarization” privacy mode**
  - You can disable sending tool results back to the LLM via an environment variable and (optionally) a UI toggle.
  - In that mode, tools still run, but the assistant only returns lightweight status messages.

---

## Architecture

High-level flow:

1. User interacts with **Streamlit** in `frontend/app.py`.
2. The UI calls the **backend API** (`backend/api_server.py`) over HTTP (for example `/health`, `/tools`, `/agent/turn`).
3. The backend:
   - Manages **per-session MCP clients** and state in `backend/runtime.py`.
   - Runs the **agentic loop** in `backend/agent/llm_agent.py` — planner (plan/replan), executors (route + tool selection, parallel fan-out), critic (goal verification), and mutation approvals. See [`AGENT_ARCHITECTURE.md`](./AGENT_ARCHITECTURE.md) / [`Execution_Flow.md`](./Execution_Flow.md).
   - Uses `backend/agent/mcp_client.py` to call the MCP server (JSON-RPC over Streamable HTTP).
   - Streams progress to the UI over SSE when the UI requests it.
4. The **MCP Streamable HTTP server** (`mcp_server/server.py`):
   - Exposes `/health`.
   - Exposes an MCP endpoint `/mcp/` implementing MCP **Streamable HTTP** (JSON-RPC).
   - For streaming-capable tool calls, responds with **SSE** containing:
     - JSON-RPC notifications (progress), then
     - a final JSON-RPC response message with the matching request id.
   - Uses `mcp_server/tools_core.py` to map tool IDs to PySisense SDK calls.
   - Reads the tool registry JSON from `config/`.
5. PySisense uses Sisense REST APIs to talk to your Sisense deployments.

### Folder structure

```text
Root/
  backend/
    agent/
      __init__.py
      llm_agent.py        # Agentic loop (orchestrator): planner (plan/replan), executors + fan-out, critic, approvals
      _config.py / _prompts.py / _registry.py / _routing.py  # loop sub-modules (env, prompts, registry I/O, routing)
      mcp_client.py       # MCP Streamable HTTP client (JSON-RPC over POST /mcp/, supports SSE tool progress)
    __init__.py
    runtime.py            # Session pool, long-lived McpClient per UI session, progress bridging
    api_server.py         # FastAPI backend (JSON + SSE on /agent/turn; exposes /health and /tools)

  config/
    tools.registry.json                 # Base tool registry generated from the SDK
    tools.registry.with_examples.json   # Registry enriched with LLM examples

  frontend/
    app.py               # Streamlit UI (SSE client for backend /agent/turn)

  images/
    FES_ASSISTANT_AD.png
    ui1.png
    ui2.png

  logs/                  # Runtime logs (rotated; not committed)

  mcp_server/
    server.py            # MCP Streamable HTTP server (/mcp/ JSON-RPC, /health; SSE for streaming tools/call)
    tools_core.py        # Registry loading, SDK client construction, tool dispatch, emit/progress integration

  scripts/
    __init__.py
    01_build_registry_from_sdk.py       # Introspects PySisense SDK and builds tools.registry.json
    02_add_llm_examples_to_registry.py  # Uses an LLM to add examples; writes tools.registry.with_examples.json
    README.md                           # Notes for the scripts

  .env.example
  .gitignore
  .dockerignore
  LICENSE
  README.md
  Execution_Flow.md
  refresh_registry.sh
  requirements.txt

  # Docker-related files
  Dockerfile.backend        # Image for backend FastAPI service
  Dockerfile.ui             # Image for Streamlit UI
  Dockerfile.mcp            # Image for MCP Streamable HTTP server
  docker-compose.yml        # Local/dev docker-compose (uses .env)
  docker-compose.prod.yml   # Example production compose (uses real env vars)
  config_prod.sh            # Example script to export prod env vars (no secrets)
```

---

## Prerequisites

- Python 3.11+
- A Sisense Fusion deployment (or multiple, for migration use cases)
- Access to at least one LLM provider:
  - Azure OpenAI, or
  - Databricks Model Serving
- (Optional but recommended) Docker + Docker Compose for containerized runs

---

## Environment configuration

This project keeps **LLM credentials and service configuration** in environment variables.  
Sisense base URLs and tokens are entered directly into the Streamlit UI and stored only in session state for the current browser session.

For local development you can use a `.env` file (see [`.env.example`](./.env.example)).  
In Docker / production, you should set the same values as real environment variables on each container (for example via `--env-file`, `docker-compose` `env_file:`, or sourcing `config_prod.sh`).

### 1) UI (Streamlit) configuration

Read by `frontend/app.py`:

- `FES_LOG_LEVEL`  
  Log level for the UI process: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

- `FES_BACKEND_URL`  
  URL of the backend FastAPI server that the UI calls for each turn.  
  Example: `http://localhost:8001`

- `FES_UI_IDLE_TIMEOUT_HOURS`  
  Idle timeout in hours for a Streamlit session. When exceeded, the UI clears `st.session_state`.

- `FES_ALLOW_SUMMARIZATION_TOGGLE`  
  Controls whether the “Allow summarization” checkbox is enabled in the UI.  
  - `true`  → user can toggle per session  
  - `false` → checkbox is disabled and always off

### 2) Backend (FastAPI) configuration

Read by `backend/api_server.py` and `backend/agent/llm_agent.py`:

- `FES_LOG_LEVEL`  
  Same as UI; controls backend logging.

- `ALLOW_SUMMARIZATION`  
  Backend hard kill switch for sending tool results (Sisense data) to the LLM.  
  - `true`  → allowed (subject to UI toggle)  
  - `false` → never sent to the LLM

- `LLM_PROVIDER`  
  Which LLM backend to use: `azure` or `databricks`.

Azure OpenAI (when `LLM_PROVIDER=azure`):

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_STYLE` (usually `v1`)

Databricks (when `LLM_PROVIDER=databricks`):

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `LLM_ENDPOINT`

Optional LLM retry tuning:

- `LLM_HTTP_MAX_RETRIES`
- `LLM_HTTP_RETRY_BASE_DELAY`

### 3) Backend → MCP client configuration (SSE-aware)

Read by `backend/agent/mcp_client.py`:

- `PYSISENSE_MCP_HTTP_URL`  
  Base URL for the MCP Streamable HTTP server. The client calls `/mcp/` under this base URL.

- `PYSISENSE_MCP_HTTP_TIMEOUT`  
  Default timeout (seconds) for MCP calls.  
  Note: streaming tool calls remove the read timeout (unbounded) so long-running migrations can stream progress.

- `MCP_HTTP_MAX_RETRIES` and `MCP_HTTP_RETRY_BASE_DELAY`  
  Retry tuning for MCP calls. Recommended to keep retries low for long-running / non-idempotent tools.

Optional SSE behavior:

- `MCP_AUTO_SUBSCRIBE`  
  If `true`, the MCP client starts an optional long-lived `GET /mcp/` SSE subscription on connect.  
  This is useful for servers that emit progress on the GET stream instead of (or in addition to) the POST response.

- `MCP_STREAMING_TOOL_IDS`  
  Comma-separated list of tool ids treated as “streaming-sensitive” (long-running).  
  For these tools, the client removes read timeouts and expects SSE responses.

### 4) MCP tool server / PySisense configuration

Read by `mcp_server/tools_core.py` and `mcp_server/server.py`:

- `PYSISENSE_REGISTRY_PATH`  
  Path to the tools registry JSON.  
  Default: `config/tools.registry.with_examples.json`

- `ALLOW_MODULES`  
  Optional comma-separated list of modules to expose.  
  Example: `ALLOW_MODULES=access,datamodel`

- `PYSISENSE_SDK_DEBUG`  
  Optional flag passed down to `SisenseClient.from_connection(debug=...)`.  
  Set to `true` or `false`. Recommended: unset (or `false`) for normal use.

#### MCP tool naming (Claude compatibility)

- `MCP_TOOL_NAME_MODE`  
  Claude Desktop rejects tool names that contain `.` during tools/list discovery.  
  - `claude` → publish underscore tool names (recommended)  
  - `canonical` → publish dotted tool ids (legacy)

The server will still accept both underscore and dotted names on tool calls.

#### Concurrency caps (single-worker friendly)

These reduce head-of-line blocking with long-running migrations while keeping the MCP server at a single worker:

- `PYSISENSE_MAX_CONCURRENT_MIGRATIONS` (default: `1`)  
  Max number of migrations allowed to run concurrently.

- `PYSISENSE_MAX_CONCURRENT_READ_TOOLS` (default: `5`)  
  Max number of short/read tools allowed to run concurrently while migrations run.

### 5) Sisense configuration (entered in the UI, not in `.env`)

In **Chat with deployment** mode:
- Sisense domain (base URL)
- API token
- Verify SSL flag

In **Migrate between deployments** mode:
- Source domain + source API token (+ source SSL flag)
- Target domain + target API token (+ target SSL flag)

These credentials are supplied via the Streamlit forms, used to build `SisenseClient` instances inside the MCP tool server, and are not persisted.

---

## Tool registry generation

The MCP server uses a **tool registry JSON** that describes available tools, parameters, descriptions, and examples.

There are two stages:

1. `config/tools.registry.json` – built directly from the PySisense SDK.
2. `config/tools.registry.with_examples.json` – the same registry but enriched with examples.

Scripts in [`scripts/`](./scripts/) are responsible for this:

1. [`01_build_registry_from_sdk.py`](./scripts/01_build_registry_from_sdk.py)  
   Introspects the PySisense SDK classes, parses docstrings, infers JSON Schemas for parameters, tags tools, and writes `config/tools.registry.json`.

2. [`02_add_llm_examples_to_registry.py`](./scripts/02_add_llm_examples_to_registry.py)  
   Reads `config/tools.registry.json`, uses an LLM to generate examples per tool, and writes `config/tools.registry.with_examples.json`.

[`refresh_registry.sh`](./refresh_registry.sh) is a convenience wrapper to rebuild both registries.

At runtime, only the JSON files in `config/` are needed.

---

## Running locally (without Docker)

This is a simple three-process dev setup.

1) Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Create a `.env` (see [`.env.example`](./.env.example))

4) Start the MCP Streamable HTTP server

In terminal 1:

```bash
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8002 --workers 1
```

Why `--workers 1`:
- MCP Streamable HTTP sessions are stateful, and running multiple workers can break session continuity unless you add sticky routing.
- This project relies on a single worker and uses concurrency caps + streaming progress to stay responsive during long migrations.

5) Start the backend API

In terminal 2:

```bash
uvicorn backend.api_server:app --host 0.0.0.0 --port 8001
```

6) Start the Streamlit UI

In terminal 3:

```bash
streamlit run frontend/app.py
```

7) Open the UI

Streamlit will print a local URL (typically `http://localhost:8501`).

---

## Claude Desktop Integration (MCP Remote)

You can connect Claude Desktop directly to the PySisense MCP HTTP server using `mcp-remote`.

### 1) Start the MCP server

Make sure the MCP server is running:

```bash
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8002 --workers 1
```

To sanity-check the MCP server before wiring Claude Desktop, run:
```bash
npx -y @modelcontextprotocol/inspector
```

This launches the Inspector and opens a browser UI. The Inspector generates a local session URL

### 2) Configure Claude Desktop

Steps:
1. Open Claude Desktop.
2. Go to Settings.
3. Under **Developer**, select **Edit Config**. This opens `claude_desktop_config.json`.
4. Add the following configuration:

```json
{
  "mcpServers": {
    "my-local-server": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8002/mcp/"]
    }
  }
}
```

Restart Claude Desktop after saving the file.

### 3) Avoid exposing Sisense credentials in Claude

To avoid putting your Sisense domain/token inside Claude, set them as environment variables on the machine where the MCP server is running (for example in that machine’s `.env`).

The MCP server’s tools_core.py includes optional default-tenant fallback logic that fills in the Sisense domain and token from environment variables when the client omits them, before invoking the underlying SDK method.

Add these env vars on the MCP server host:

```bash
PYSISENSE_USE_DEFAULT_TENANT=true
PYSISENSE_DEFAULT_DOMAIN="https://your-sisense-domain"
PYSISENSE_DEFAULT_TOKEN="your-api-token"
PYSISENSE_DEFAULT_SSL=false
```

Important: You do not need to tell Claude to pass empty domain/token fields. If default-tenant fallback is enabled on the MCP server, the server can fill them in from its environment.

Note on SSE in Claude Desktop:
- The MCP server supports SSE progress streaming.
- Some MCP clients may not yet render progress events in the main UI (they may only appear in logs). This does not affect the Streamlit UI path, which fully supports SSE progress end-to-end.

---

## Running with Docker (local/dev)

The repo includes three Dockerfiles and a `docker-compose.yml` for local development:

- [`Dockerfile.ui`](./Dockerfile.ui) – Streamlit UI
- [`Dockerfile.backend`](./Dockerfile.backend) – FastAPI backend
- [`Dockerfile.mcp`](./Dockerfile.mcp) – MCP tool server
- [`docker-compose.yml`](./docker-compose.yml) – runs all three together

### 1) Create a `.env` for local Docker

Create a `.env` in the project root with your LLM and service configuration (same keys as in the Environment configuration section).  
This file is not committed to git and is not baked into images.

### 2) Build and start the stack

From the project root:

```bash
docker compose up --build --force-recreate
```

Then open:

- UI: `http://localhost:8501`
- Backend docs: `http://localhost:8001/docs`
- MCP health: `http://localhost:8002/health`

### Useful Docker commands

Stop the stack:

```bash
docker compose down
```

Hard reset (remove containers, images, volumes, and build cache):

```bash
docker compose down --rmi all --volumes --remove-orphans
docker builder prune -a -f
```

---

## Production-style deployment (example)

For production you typically:

- Push built images to a registry (Docker Hub, ECR, etc.)
- Use a separate compose file (for example [`docker-compose.prod.yml`](./docker-compose.prod.yml))
- Set environment variables on the host or via your orchestrator

An example non-secret env script is included: [`config_prod.sh`](./config_prod.sh).

Secrets like `AZURE_OPENAI_API_KEY` or `DATABRICKS_TOKEN` should be provided via a secure channel (SSM Parameter Store, Secrets Manager, etc.).

SSE notes for production reverse proxies:
- Ensure your proxy/load balancers support SSE and do not buffer responses.
- Common requirements include disabling proxy buffering and increasing idle timeouts for long-lived responses.

---

## Using the app

### 1) Chat with deployment

- Select **Chat with deployment**.
- Enter Sisense domain, API token, and SSL preference.
- Click **Connect**.

Example questions:

- “List all dashboards.”
- “Show all users in the ‘Analysts’ group.”
- “Find all fields that are not used in datamodel XYZ.”

For write operations (create/update/delete), you will see a confirmation step before execution.

### 2) Migrate between deployments

- Switch to **Migrate between deployments**.
- Fill in Source and Target Sisense environments (domain + token + SSL).
- Connect both.

Example requests:

- “Migrate this dashboard from source to target.”
- “Migrate all datamodels, overwriting existing ones.”
- “Migrate these three dashboards and duplicate them on target.”

Screenshots:

![FES Assistant UI](images/ui1.png)

![FES Assistant UI](images/ui2.png)

---

## Logging

- Log files are written under `logs/` (git-ignored).
- Sensitive values such as tokens are scrubbed before being written to logs where possible.
- For production, set log levels to `INFO` or `WARNING` instead of `DEBUG`.

---

## 🔒 Security & Deployment Best Practices

While this is an experimental tool, we recommend the following "Security First" approach for your deployment:

* **Authentication:** Deploy the UI and Backend behind your organization’s SSO, a VPN, or a Secure Reverse Proxy (e.g., Nginx with Auth).
* **Credential Management:** Use a dedicated, limited-privilege Sisense Service Account and ensure your LLM API keys are stored securely (e.g., via environment secrets, not hard-coded).
* **Network Isolation:** Implement network-level restrictions (firewalls/VPC rules) so only trusted internal hosts can reach the Backend and MCP server endpoints.
* **Service Scoping:** If using Claude Desktop via mcp-remote, ensure you utilize a secure tunneling method (e.g., SSH Tunnel, Tailscale) if the server is not on your local machine.

---

## ⚖️ Community Disclaimer & Liability Shield

**Important: Field-Developed Ecosystem Extension**

These tools are community-contributed projects developed by Sisense Field Engineering. They are **not** official Sisense product features and do not fall under standard Sisense SLAs, Support, or Security Certifications.

* **Local Library Execution (PySisense SDK):** As a Python package installed via PyPI, all logic executes locally on your workstation or server. No data is ever transmitted to Sisense Field Engineering.
* **Self-Hosted Applications (MCP Server & FES Assistant):** These are designed to be deployed within your own private network or VPC. You maintain full ownership of the hosting environment, logs, and security configurations.
* **LLM Data Exposure & Summarization:**
    * **FES Assistant:** Features a manual **Summarization Toggle**. By default, the LLM only sees your prompt and tool definitions to determine intent. Optionally, when enabled, the raw response from the SDK (which may contain metadata or specific tool-level data) is sent to the LLM to generate a natural language summary.
    * **MCP Server:** When used with third-party clients (e.g., Claude Desktop, IDE Agents), all data retrieved via the SDK is passed directly to the host client's LLM to generate a response.
* **Responsibility:** By using these tools, the customer/user acknowledges that Sisense metadata and API responses will be processed by their chosen LLM provider. Customers are **solely responsible** for ensuring their LLM provider (OpenAI, Anthropic, Databricks Foundation Models API, etc.) meets their organization’s data privacy and security standards.
* **Liability & Risk:** These tools are provided **"as-is"** for experimental purposes. Sisense and its employees are not liable for security vulnerabilities, third-party LLM data exposure, or environment disruptions.
* **Non-Production Recommendation:** We strongly recommend testing these tools in a sandbox environment and using them with a dedicated, limited-privilege Sisense Service Account.

---
## DEMO

https://github.com/user-attachments/assets/1ef44ff2-21c4-4be9-8a3f-8761f0641d6e

---

## Related project

- [PySisense](https://github.com/sisense/pysisense) – the unofficial Python SDK for Sisense Fusion APIs. This project uses PySisense for Sisense-side actions and leverages its docs/examples to build the MCP tool registry.

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.
