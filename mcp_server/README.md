# Sisense Meta-Management MCP Server

## ⚠️ Experimental Project Notice

### Community-Contributed Tool from Sisense Field Engineering

This project is an experimental tool developed by Sisense Field Engineering to facilitate customer learning and exploration of Sisense capabilities. While maintained by Field Engineering, it is shared "as-is" to encourage feedback and experimentation.

Important Disclaimer: This tool is not part of the core Sisense product release lifecycle and does not undergo the same validation, support, or certification processes as generally available (GA) Sisense features. It is intended to complement, not replace, officially supported Sisense features.

---

## Technical and Security Considerations

### Deployment and Execution Control
- Local SDK Usage (PySisense): All processing logic runs locally on your machine or server. No data is transmitted to Sisense Field Engineering.
- Self-hosted Components (FES Assistant / MCP Server): These components are designed for deployment within your own environment (on-prem or VPC). You maintain complete control over infrastructure, security configuration, access controls, and logs.

### Data and LLM Handling
- LLM Feature Status: The FES Assistant summarization feature is disabled by default.
- Data Transmission: When the summarization feature is enabled, responses retrieved via the Sisense SDK may be sent to your chosen Large Language Model (LLM) provider for processing.
- Third-Party Clients: When using the MCP Server with third-party clients (e.g., IDE agents or desktop assistants like Claude Desktop), data retrieved from Sisense is passed directly to the client’s LLM.
- Customer Responsibility: Customers are responsible for selecting an LLM provider that meets their organization’s data privacy and security requirements.

---

## Recommended Usage Guidelines

To ensure secure and effective use of this experimental tool:
- Environment: Use the tool primarily in sandbox or non-production environments.
- Access: Utilize a dedicated Sisense service account with limited privileges.
- Validation: Thoroughly review and validate the tool's behavior before any broader adoption within your organization.

---

## Overview

The Sisense Meta-Management MCP Server is a Streamable HTTP MCP server that exposes Sisense environment operations as AI-ready tools.

Under the hood:
- The MCP server (`mcp_server/server.py`) is built on the **official MCP Python SDK**: the SDK's `StreamableHTTPSessionManager` hosts a lowlevel `Server` at `/mcp/`.
- Tool execution is registry-driven and dispatched through `mcp_server/tools_core.py`.
- Each MCP tool ultimately calls a PySisense SDK method, which performs the actual Sisense API operations.

This server is designed for environment operations (governance, migrations, lifecycle tasks, well-checks), not chart-building or analytics Q&A.

---

## How this differs from a generic MCP server

Any MCP client can connect to this server, and the spec's own mechanisms are
used wherever they exist (Streamable HTTP transport, `notifications/progress`,
`notifications/cancelled`). Three things are deliberate design choices that a
generic, public-facing MCP server would do differently — all three exist
because this server's primary client is the FES Assistant's own agent rather
than an end user with a personal login:

1. **Credentials travel with every tool call, not with the session.**
   A generic MCP server authenticates the *user* once (typically OAuth) and
   scopes every call to that identity. This server is **multi-tenant per
   call**: each `tools/call` carries the Sisense `domain`/`token` (or
   `source_*`/`target_*` for migration) as arguments, so one server instance
   can serve calls against any Sisense deployment — two browser tabs can talk
   to two different environments through the same server. The registry's tool
   schemas are augmented at `tools/list` time to declare these as required,
   and a call without them fails loudly. There is no server-side default
   environment.

2. **An out-of-band cancel endpoint exists alongside spec cancellation.**
   Spec `notifications/cancelled` targets one request by id from inside the
   MCP conversation. But real cancellations here can originate *outside* it —
   a Stop click in the UI, a browser disconnect — where only the session is
   known and several requests may be in flight. `POST /mcp/cancel` flags the
   whole session in one shot. It is an operational fallback, not a
   replacement: the spec path is still the primary mechanism.

3. **The tool surface is curated server-side, independent of any client.**
   The registry is generated from the PySisense SDK, so a rebuild can surface
   methods that should never reach a user. `config/allowed_tools.txt` is the
   hand-edited gate, enforced at the dispatch boundary — a tool not listed
   there is invisible in `tools/list` and unreachable via `tools/call`, no
   matter what the client asks for.

Two further choices are implementation-driven rather than client-driven:

4. **Built on the lowlevel `Server`, not FastMCP.** FastMCP validates each
   call's arguments against schemas generated from the tool function's
   signature — which would reject the injected credential fields
   (`domain`/`token`, `source_*`/`target_*`) that ride on every call. The
   lowlevel `Server` leaves argument handling to `tools_core.py`, which knows
   about them.

5. **Cancellation is a flag bridge, not task cancellation.** anyio
   cancellation cannot interrupt the SDK worker thread running a PySisense
   call, so both cancel paths (spec `notifications/cancelled` and
   `POST /mcp/cancel`) set a per-session flag that the tool's `emit()`
   callback observes at each progress checkpoint — the run stops at the next
   checkpoint, not mid-API-call.

One operational constraint follows from the session model: the server must run
as a **single worker** (`--workers 1`). Session and cancellation state are
in-process, and multiple workers would route the same MCP session to different
processes. Concurrency is handled inside the one worker via async +
semaphores.

A spec-faithful, public-facing Sisense MCP (OAuth identity, per-user scoping,
any client) is the separate `sisense-admin-mcp` project; this server is
optimized for being the FES Assistant's execution layer while remaining a real
MCP server any client can probe.

---

## High-level architecture

- MCP Endpoint:
  - `/mcp/` is the official SDK's Streamable HTTP transport.
  - Long-running tools stream spec `notifications/progress` (with `progressToken`), plus human-readable narration as `notifications/message` log frames tied to the request.
  - Cancellation follows the spec (`notifications/cancelled` per in-flight request); a small `POST /mcp/cancel` endpoint remains as an operational fallback that flags a whole session.

- Tool dispatch:
  - The server reads a tool registry JSON and exposes tools via MCP discovery (`tools/list`) — but only tool_ids listed in the hand-edited allowlist (`config/allowed_tools.txt`).
  - On `tools/call`, `tools_core.py` validates inputs, constructs the Sisense client(s), and invokes the mapped PySisense method.

- Registry-driven tools:
  - Tools are not manually coded one-by-one in the MCP server.
  - The registry is generated from the PySisense SDK, keeping tool definitions aligned with SDK methods as they evolve.

---

## Configuration (MCP tool server / PySisense)

These settings are read by `mcp_server/tools_core.py` and `mcp_server/server.py`.

### Tool registry path
- `PYSISENSE_REGISTRY_PATH`
  - Path to the tools registry JSON.
  - Default: `config/tools.registry.with_examples.json`

### Curated tool surface
- `FES_TOOL_ALLOWLIST`
  - Path to the hand-edited allowlist of tool_ids (default: `config/allowed_tools.txt`).
  - Only listed tools are exposed via `tools/list` or reachable via `tools/call`.
  - A missing file means allow-all with a warning, never deny-all.

### SDK debug flag
- `PYSISENSE_SDK_DEBUG`
  - Optional flag passed down to `SisenseClient.from_connection(debug=...)`.
  - Set to `true` or `false`; when unset it follows the log level (`FES_LOG_LEVEL=DEBUG` → on).

---

## MCP tool naming (Claude compatibility)

Some MCP clients may reject tool names containing `.` during `tools/list` discovery.

- `MCP_TOOL_NAME_MODE`
  - `claude` -> publish underscore tool names (recommended)
  - `canonical` -> publish dotted tool ids

Note: The server will still accept both underscore and dotted names on tool calls.

---

## Concurrency caps (single-worker friendly)

These caps reduce head-of-line blocking when long-running migrations run on a single server process:

- `PYSISENSE_MAX_CONCURRENT_MIGRATIONS` (default: `1`)
  - Max number of migrations allowed to run concurrently.

- `PYSISENSE_MAX_CONCURRENT_READ_TOOLS` (default: `5`)
  - Max number of short/read tools allowed to run concurrently while migrations run.

---

## Sisense credentials (how they are provided)

In the full FES Assistant application, Sisense credentials are entered in the UI and used to construct `SisenseClient` instances inside the MCP tool server. These credentials are not persisted.

- Chat with deployment mode:
  - Sisense domain (base URL)
  - API token
  - Verify SSL flag

- Migrate between deployments mode:
  - Source domain + source API token (+ source SSL flag)
  - Target domain + target API token (+ target SSL flag)

If you connect the MCP server directly to a third-party MCP client, every tool
call must carry its Sisense connection details as tool arguments. There is
deliberately **no server-side env fallback**: missing credentials fail loudly
rather than silently running against whatever environment the server's env
last pointed at.

---

## Tool registry generation

The MCP server uses a tool registry JSON that describes available tools, parameters, descriptions, and examples.

Two stages:
1. `config/tools.registry.json` is built directly from the PySisense SDK.
2. `config/tools.registry.with_examples.json` is the same registry enriched with examples.

Scripts responsible for generation (run as modules — `python -m scripts.01_build_registry_from_sdk` etc.; see `scripts/README.md`):
- `scripts/01_build_registry_from_sdk.py`
  - Introspects the PySisense SDK classes, parses docstrings, infers JSON Schemas for parameters, tags tools, and writes `config/tools.registry.json` (the flat base registry only).

- `scripts/02_add_llm_examples_to_registry.py`
  - Reads `config/tools.registry.json`, uses an LLM to generate examples per tool, and writes `config/tools.registry.with_examples.json` plus the routing tree under `config/registry/`.

At runtime, only the JSON files in `config/` are needed.

---

## Important: the MCP server requires access to the config registry

The MCP server loads the registry file from the `config/` directory in the main repo.

If you deploy the MCP server separately (for example as its own container/image), ensure that the `config/` directory (or at minimum the registry JSON file) is included and accessible at runtime.

Common approaches:
- Copy the `config/` folder into the MCP server image during build, or
- Mount the `config/` folder as a runtime volume, or
- Set `PYSISENSE_REGISTRY_PATH` to the absolute path where the registry is available.

---

## Available MCP Tools (Tool Catalog)

_Generated from the shipped registry filtered by `config/allowed_tools.txt` (119 tools). Regenerate after a registry rebuild._

### Access Management (19 tools)

Read / Inspect (13):
- `get_all_dashboard_shares`: Retrieve all dashboard shares, including user and group details for each shared dashboard.
- `get_datamodel_columns`: Retrieve columns from a DataModel by collecting them from its datasets and tables.
- `get_group`: Retrieve group details by name.
- `get_my_user`: Retrieve the currently logged-in user for the API token.
- `get_roles`: Retrieve all Sisense roles.
- `get_unused_columns`: Identify unused columns in a DataModel by comparing all columns against dashboard usage.
- `get_unused_columns_bulk`: Run unused-column analysis for one or more data models and return a.
- `get_user`: Retrieve a user's details by email address, expanding group and role information.
- `get_user_with_role_and_group_names`: Retrieve a single user by email/username with role and group details.
- `get_users_all`: Retrieve all users with group and role information.
- `get_users_with_role_names_and_group_names`: Retrieve all users enriched with role names and group names.
- `users_per_group`: Retrieve all users within a specific group by name.
- `users_per_group_all`: Retrieve all groups mapped to the users belonging to them.

Write / Mutating (6):
- `change_folder_and_dashboard_ownership`: Change the ownership of folders and optionally dashboards.
- `change_user_password`: Change a user's password.
- `create_schedule_build`: Create a schedule build for a DataModel.
- `create_user`: Create a new user in Sisense.
- `delete_user`: Delete a user by their email (username).
- `update_user`: Update an existing Sisense user identified by their email address.

### Blox (3 tools)

Read / Inspect (1):
- `get_blox_actions`: Retrieve all custom Blox actions from the Sisense instance.

Write / Mutating (2):
- `delete_blox_action`: Delete a custom Blox action from the Sisense instance.
- `save_blox_action`: Save a custom Blox action on the Sisense instance.

### Custom Code (8 tools)

Read / Inspect (3):
- `export_notebook`: Export a notebook definition.
- `get_notebooks`: Retrieve notebooks from Sisense.
- `list_notebook_folder_contents`: List contents of a custom-code notebook folder.

Write / Mutating (5):
- `create_notebook`: Create a new notebook.
- `delete_notebook`: Delete a notebook by ID.
- `rename_notebook_file`: Rename or update a notebook resource file.
- `rename_notebook_folder`: Rename a custom-code notebook folder.
- `update_notebook`: Update an existing notebook.

### Dashboard (19 tools)

Read / Inspect (13):
- `can_be_owned`: Check whether a dashboard can be owned by the current user.
- `export_dashboard`: Export a dashboard definition using the Sisense admin export endpoint.
- `get_all_dashboards`: Retrieve all dashboards from the Sisense server.
- `get_dashboard_by_id`: Retrieve a specific dashboard by its ID.
- `get_dashboard_by_name`: Retrieve a specific dashboard by its name.
- `get_dashboard_columns`: Retrieve columns referenced by a dashboard, including widget and filter columns.
- `get_dashboard_script`: Build a formatted dashboard script helper object.
- `get_dashboard_share`: Retrieve share details (users and groups) for a dashboard by title.
- `get_dashboard_shares_v1`: Retrieve share details for a dashboard using the v1 shares endpoint.
- `get_dashboard_widgets`: Retrieve widget definitions from an admin export of the dashboard.
- `get_dashboards`: Retrieve dashboards visible to the authenticated user.
- `get_widget_script`: Build a formatted widget script helper object.
- `resolve_dashboard_reference`: Resolve a dashboard reference (ID or name) to a concrete dashboard ID and title.

Write / Mutating (6):
- `add_dashboard_script`: Add or overwrite a script on a dashboard.
- `add_dashboard_shares`: Add or update shares for a dashboard for the given users and groups.
- `add_widget_script`: Add or overwrite a script for a specific widget within a dashboard.
- `move_dashboard_to_folder`: Move a dashboard into a folder.
- `publish_dashboard`: Publish (republish) a dashboard.
- `rename_dashboard`: Rename a dashboard.

### Data Model (28 tools)

Read / Inspect (15):
- `describe_datamodel`: Retrieve data model structure in a flat, row-based format.
- `describe_datamodel_raw`: Retrieve detailed information about a specific data model.
- `get_all_datamodel`: Retrieve metadata for all data models using an internal API.
- `get_connection`: Retrieve connections matching a name.
- `get_connections`: Retrieve all connections.
- `get_data`: Retrieve data from a specific table in a data model.
- `get_datamodel`: Retrieve a data model by its title.
- `get_datamodel_shares`: Retrieve all share entries (users and groups) for a given data model.
- `get_datasecurity`: Retrieve datasecurity table and column entries for a given data model.
- `get_datasecurity_detail`: Retrieve detailed datasecurity rules for a data model, including share-level visibility.
- `get_elasticubes`: List all ElastiCubes using the legacy v1 endpoint.
- `get_model_schema`: Retrieve the schema of a data model, including tables and columns.
- `get_row_count`: Retrieve the row count for each table in a specific data model.
- `get_table_schema`: Retrieve the schema of a table within a connection's data source.
- `resolve_datamodel_reference`: Resolve a data model reference (ID or title) to a concrete data model ID and title.

Write / Mutating (13):
- `add_datamodel_shares`: Add share entries (users and groups) to a data model.
- `create_connections`: Create a new connection using the provided payload.
- `create_datamodel`: Create a new data model in Sisense.
- `create_dataset`: Create a new dataset in the specified data model.
- `create_table`: Create a new table in the specified data model.
- `delete_datamodel`: Delete a data model by title and server using the GraphQL ECM endpoint.
- `deploy_datamodel`: Deploy (build or publish) the specified data model based on its type.
- `generate_connections_payload`: Generate a connection payload for a given data source type.
- `load_datamodel`: Look up a data model's OID by title using the GraphQL ECM endpoint.
- `set_live_datasecurity_add_many`: Add multiple datasecurity rules to a LIVE datamodel.
- `setup_datamodel`: Set up a data model end to end using an existing connection.
- `update_connection`: Update an existing connection.
- `update_datasecurity`: Replace datasecurity rules on an EXTRACT (Elasticube) datamodel.

### Encryption (2 tools)

Write / Mutating (2):
- `decrypt`: Decrypt a value using the Sisense encryption service.
- `encrypt`: Encrypt a value using the Sisense encryption service.

### Folder (8 tools)

Read / Inspect (5):
- `get_all_folders`: Retrieve the full folder tree.
- `get_folder_ancestors`: Retrieve folders by a caller-supplied structure value.
- `get_folder_id`: Retrieve a single folder by OID.
- `get_folders`: Retrieve folders using a configurable ``structure`` query parameter.
- `get_navver`: Retrieve the Sisense navigation tree (navver payload).

Write / Mutating (3):
- `create_folder`: Create a new Sisense folder.
- `delete_folder`: Delete a folder by OID.
- `update_folder`: Update an existing folder.

### Metadata (5 tools)

Read / Inspect (3):
- `get_datasource_dimensions`: Retrieve saved filter dimensions for a datasource.
- `get_datasource_measures`: Retrieve saved formula measures for a datasource.
- `get_datasources`: Retrieve all datasources visible to the authenticated user.

Write / Mutating (2):
- `add_datasource_measure`: Create a saved formula measure in Sisense metadata.
- `post_metadata_query`: Execute a metadata query against Sisense.

### Migration (9 tools)

Write / Mutating (9):
- `migrate_all_dashboards`: Migrates all dashboards from the source to the target environment in batches.
- `migrate_all_datamodels`: Migrates all data models from the source environment to the target environment in batches.
- `migrate_all_groups`: Migrate groups from the source environment to the target environment using the bulk endpoint.
- `migrate_all_users`: Migrate all eligible users from the source environment to the target environment using the bulk endpoint.
- `migrate_dashboard_shares`: Migrate shares for specific dashboards from the source to the target environment.
- `migrate_dashboards`: Migrate dashboards from the source to the target environment using Sisense bulk import.
- `migrate_datamodels`: Migrates specific data models from the source environment to the target environment.
- `migrate_groups`: Migrate specific groups from the source environment to the target environment.
- `migrate_users`: Migrate specific users from the source environment to the target environment.

### Plugins (8 tools)

Read / Inspect (2):
- `get_all_plugins`: Retrieve all plugins installed on the Sisense instance.
- `get_plugin`: Get a single plugin by its name or folder name.

Write / Mutating (6):
- `disable_plugin`: Disable a single plugin by name or folder name.
- `disable_plugins`: Disable one or more plugins by name or folder name.
- `enable_plugin`: Enable a single plugin by name or folder name.
- `enable_plugins`: Enable one or more plugins by name or folder name.
- `restore_snapshot`: Restore plugin states to exactly match a previously saved snapshot.
- `save_snapshot`: Capture the current plugin enable/disable state as a snapshot.

### Queries (2 tools)

Write / Mutating (2):
- `elasticube_run_jaql_query`: Run a JAQL query against a datasource (elasticube).
- `elasticubes_run_jaql_csv`: Run a JAQL query and return CSV output.

### WellCheck (8 tools)

Read / Inspect (8):
- `check_dashboard_structure`: Analyze the structure of one or more dashboards.
- `check_dashboard_widget_counts`: Compute widget counts for one or more dashboards.
- `check_datamodel_custom_tables`: Inspect custom tables in one or more data models and flag the use of UNION.
- `check_datamodel_import_queries`: Inspect tables in one or more data models for import queries.
- `check_datamodel_island_tables`: Identify island tables (tables with no relationships) in one or more data models.
- `check_datamodel_m2m_relationships`: Check for potential many-to-many (M2M) relationships between tables.
- `check_datamodel_rls_datatypes`: Inspect row-level security (RLS) rules for one or more data models and.
- `check_pivot_widget_fields`: Analyze pivot widgets on one or more dashboards and report those with many fields.

---

## Support and contributing

This is an experimental, community-contributed project maintained by Sisense Field Engineering and provided "as-is."

- Do not open a GSS ticket (this is not a GA Sisense feature).
- For usage questions or help getting started, contact your Customer Success Manager (CSM), who will route feedback to the Field Engineering team.
- For bugs and improvements, use GitHub Issues or submit a Pull Request.
- For feature requests, open a GitHub Issue with details.