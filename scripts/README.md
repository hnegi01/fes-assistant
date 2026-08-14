# Scripts

Registry pipeline, in order. Steps 1–2 regenerate tool metadata from the SDK;
steps 3–4 keep the derived artifacts and the curated surface in sync with it.

1. `01_build_registry_from_sdk.py`  
   Introspects the pysisense SDK (AccessManagement, Dashboard, DataModel, Migration, WellCheck) and writes `config/tools.registry.json`, plus the 3-level tree under `config/registry/` via `build_registry_hierarchical()`.

2. `02_add_llm_examples_to_registry.py`  
   Reads `config/tools.registry.json`, uses an LLM to generate example `user_query → arguments` pairs per tool, and writes `config/tools.registry.with_examples.json`.

   These examples ARE sent to the tool-selection LLM when `FES_TOOL_EXAMPLES` is 1–3 (default 0 = off). **`example[0]` is dual-purpose** (curation pass 2026-08-14): it is also shown to *users* — in mutation approval dialogs and clarification questions, regardless of the flag — as "how you could phrase this". It is therefore held to a stricter bar than its siblings: an **imperative command** (never a question), where **every argument value it sets is spoken in the query** (identity values AND numbers) — that is what teaches "take values from the request, never invent them", matching the rule `PLANNING_SYSTEM_PROMPT` states in prose. `tests/unit/test_tool_examples.py` enforces all of it against the shipped registry. `examples[1..2]` are uncurated, question-phrased, and reach only the model at flag 2–3 — do not raise the flag past 1 until they get the same pass. The script preserves existing examples on rebuild (LLM generation runs only for tools that have none), so the curated data survives `refresh_registry.sh`.

3. `03_sync_examples_to_registry_tree.py`  
   Copies the `examples` field from the flat registry into `config/registry/{package}/{mixin}.json` by tool_id. Step 1 already does this as part of a full rebuild; this script covers the examples-only case (e.g. a curation pass over `user_query` text) without needing the SDK installed.

   ```bash
   python scripts/03_sync_examples_to_registry_tree.py           # dry run
   python scripts/03_sync_examples_to_registry_tree.py --write
   ```

4. `04_generate_tool_allowlist.py`  
   Audits (or initialises) `config/allowed_tools.txt` — the hand-edited list of tool_ids allowed to reach the agent and the MCP server. Run the audit after every registry rebuild: new SDK methods stay hidden until someone adds a line, which is the point, but you need to know they appeared.

   ```bash
   python scripts/04_generate_tool_allowlist.py            # drift audit
   python scripts/04_generate_tool_allowlist.py --init     # first-time creation
   ```

## After a registry rebuild

```bash
python scripts/01_build_registry_from_sdk.py
python scripts/02_add_llm_examples_to_registry.py
python scripts/03_sync_examples_to_registry_tree.py --write
python scripts/04_generate_tool_allowlist.py     # review new/stale tools
pytest tests/unit -q
```
