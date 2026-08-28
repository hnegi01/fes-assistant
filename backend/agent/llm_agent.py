"""
backend/agent/llm_agent.py

Main orchestration for the FES Assistant agent turn.

What lives here:
  - TOOL_REGISTRY and LAST_TOOL_RESULT — mutable globals read by the API layer
    (must stay on this module; api_server.py reads them via getattr(llm_agent, ...))
  - load_tools_for_llm() — loads registry JSON and populates TOOL_REGISTRY
  - _get_module_tools() — groups tools by module (uses TOOL_REGISTRY directly)
  - _infer_mode_from_tools() — detects chat vs migration mode
  - _approval_key() — stable key for mutation approval matching
  - call_llm_with_tools() — the main plan → execute → summarize pipeline

What is imported from sub-modules and re-exported for backward compatibility:
  - llm_config: logging, env helpers, LLM provider config, observability
  - llm_tools: registry I/O, payload shrinkers, result description
  - llm_routing: prompts, routing, planning history, raw LLM call, fallback

Dependency order (no circular imports):
  llm_config ← llm_tools ← llm_routing ← llm_agent
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema

# --- _config ---
from ._config import (
    ALLOW_MUTATING_TOOLS,
    ALLOW_SUMMARIZATION,
    CLARIFY_MAX_ATTEMPTS,
    LLM_CONFIG,
    LLM_PLANNING_HISTORY_TURNS,
    LLM_PROVIDER,
    MAX_AGENT_STEPS,
    MAX_PARALLEL_STEPS,
    MAX_REPLANS,
    MIGRATION_COMPLETENESS_CHECK,  # noqa: F401 — late-bound as A.MIGRATION_COMPLETENESS_CHECK (migration_flow); tests monkeypatch it here
    MIGRATION_SINGLE_SHOT,
    REQUIRE_MUTATION_CONFIRM,
    VERIFY_GOAL,
    VERIFY_MAX_RECHECKS,
    _log_json,
    _scrub_secrets,
    _write_llm_trace,
    audit_logger,
    begin_turn_output,
    current_turn_user_corpus,
    logger,
    set_current_turn,
    turn_output,
    write_tool_call,
)

# --- _prompts ---
from ._prompts import (
    AGENT_DECIDE_NODATA_SYSTEM_PROMPT,
    AGENT_DECIDE_SYSTEM_PROMPT,
    AGENT_PLAN_SYSTEM_PROMPT,
    AGENT_REPLAN_SYSTEM_PROMPT,
    CHAT_PLANNING_CONTEXT_PROMPT,
    CLARIFY_ANSWER_SYSTEM_PROMPT,
    MIGRATION_COMPLETENESS_SYSTEM_PROMPT,  # noqa: F401 — late-bound as A.MIGRATION_COMPLETENESS_SYSTEM_PROMPT (migration_flow)
    MIGRATION_PLAN_SYSTEM_PROMPT,  # noqa: F401 — late-bound as A.MIGRATION_PLAN_SYSTEM_PROMPT (migration_flow)
    MIGRATION_PLANNING_CONTEXT_PROMPT,
    MUTATION_EXPLAIN_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    VERIFY_GOAL_SYSTEM_PROMPT,
)

# --- _registry ---
from ._registry import (
    _describe_tool_result,
    _effective_ok,
    _load_registry_rows,
    _payload_failure_reason,
    _safe_json_loads,  # re-exported: test_smoke.py imports from llm_agent
    _shrink_for_llm,  # re-exported: test_smoke.py imports from llm_agent
)

# --- _routing ---
from ._routing import (
    _build_planning_history,  # re-exported: test_planning_history.py calls m._build_planning_history
    _extract_latest_user_message,
    _fallback_direct_tool,
    _load_all_package_tools,
    _navigate_to_tools,
    _pick_tool_calls_from_llm_response,
    call_llm_raw,
)
from ._routing import (
    planner_schema as _planner_schema,
)
from ._tracing import log_tool_child, mark_tainted
from .mcp_client import McpClient

# -----------------------------------------------------------------------------
# Mutable globals — DEBUG/UNIT-TEST AIDS ONLY. They are last-writer-wins across
# concurrently running sessions, so nothing outside the turn may read them for
# real data; the authoritative per-turn copy lives in _config's trace-id-keyed
# output store, written through the _record_* helpers below and popped by
# runtime._run_turn_once. Unit tests (single turn at a time) may still assert
# on these; production code paths must not.
# -----------------------------------------------------------------------------
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
LAST_TOOL_RESULT: Optional[Dict[str, Any]] = None

# Every tool result of the current turn, in order: [{step, tool_id, result}].
# LAST_TOOL_RESULT is only the LAST one (a single slot); this keeps the whole
# chain so the UI can show each step's output instead of just the final table.
LAST_STEP_RESULTS: List[Dict[str, Any]] = []

# Set when a turn pauses to ask the user for a missing required argument.
# Kept separate from LAST_TOOL_RESULT so the API layer does not surface it
# as a tool payload — the clarifying question is delivered as the plain reply.
LAST_PENDING_CLARIFICATION: Optional[Dict[str, Any]] = None

# Step 8: set when the agentic loop pauses mid-turn for a mutation approval.
# The approval turn resumes the loop from the paused step instead of
# re-planning from scratch (Option A semantics).
# Shape: {transcript, steps_executed, tool_id, arguments}.
LAST_PENDING_LOOP: Optional[Dict[str, Any]] = None

# The turn's trace_id (the same id that groups this turn's rows in
# llm_calls.csv / llm_traces.csv and its LangSmith runs). Returned to the UI so
# user feedback (thumbs up/down) can be joined back to the exact calls and
# tool picks it judges. Set at turn start.
LAST_TRACE_ID: Optional[str] = None

# CLARIFY_MAX_ATTEMPTS imported from _config (env: FES_CLARIFY_MAX_ATTEMPTS, default 2)


# -----------------------------------------------------------------------------
# Per-turn output recorders — the ONLY sanctioned way to record a turn's
# outputs. Each writes the module global (test/debug aid) AND the per-turn
# store keyed by the turn's trace_id, which is what the runtime returns to the
# API layer. The store lookup resolves through the same ContextVar the usage
# accumulator uses, so fan-out branches and engine-internal tasks (which
# inherit a copy of the ContextVar carrying the same trace_id) all land in the
# turn they belong to — never in a concurrently running session's turn.
# -----------------------------------------------------------------------------
def _record_tool_result(result: Optional[Dict[str, Any]]) -> None:
    global LAST_TOOL_RESULT
    LAST_TOOL_RESULT = result
    _out = turn_output()
    if _out is not None:
        _out["tool_result"] = result


def _record_step(step: Any, tool_id: str, result: Dict[str, Any]) -> None:
    entry = {"step": step, "tool_id": tool_id, "result": result}
    LAST_STEP_RESULTS.append(entry)
    _out = turn_output()
    if _out is not None:
        _out["step_results"].append(entry)


def _record_pending_clarification(value: Optional[Dict[str, Any]]) -> None:
    global LAST_PENDING_CLARIFICATION
    LAST_PENDING_CLARIFICATION = value
    _out = turn_output()
    if _out is not None:
        _out["pending_clarification"] = value


def _record_pending_loop(value: Optional[Dict[str, Any]]) -> None:
    global LAST_PENDING_LOOP
    LAST_PENDING_LOOP = value
    _out = turn_output()
    if _out is not None:
        _out["pending_loop"] = value


# -----------------------------------------------------------------------------
# Registry → OpenAI-style tool definitions
# -----------------------------------------------------------------------------
def load_tools_for_llm() -> List[Dict[str, Any]]:
    """Load tools from the registry and convert them to OpenAI-style tool definitions."""
    global TOOL_REGISTRY

    rows = _load_registry_rows()
    if not rows:
        TOOL_REGISTRY = {}
        logger.warning("Registry empty; no tools available to LLM.")
        return []

    registry_by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        tid = row.get("tool_id")
        if tid:
            registry_by_id[tid] = row

    TOOL_REGISTRY = registry_by_id
    logger.info("TOOL_REGISTRY populated with %d tools", len(TOOL_REGISTRY))

    tools: List[Dict[str, Any]] = []
    skipped_mutating: List[str] = []

    for tid, meta in registry_by_id.items():
        mutates = bool(meta.get("mutates", False))
        if mutates and not ALLOW_MUTATING_TOOLS:
            skipped_mutating.append(tid)
            continue

        params = meta.get("parameters") or {}
        desc = meta.get("description") or ""
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tid,
                    "description": desc,
                    "parameters": _planner_schema(params),
                },
            }
        )

    if skipped_mutating:
        logger.info("Mutating tools hidden (ALLOW_MUTATING_TOOLS=False): %s", skipped_mutating)

    logger.info("Tools loaded from registry: %d", len(tools))
    return tools


# -----------------------------------------------------------------------------
# Helpers that depend on TOOL_REGISTRY (must stay on this module)
# -----------------------------------------------------------------------------
def _get_module_tools(tools: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group tools by their registry module. Returns {module_name: [tools]}."""
    by_module: Dict[str, List[Dict[str, Any]]] = {}
    for tool in tools:
        fn = tool.get("function") or {}
        name = fn.get("name", "")
        meta = TOOL_REGISTRY.get(name) or {}
        module = meta.get("module") or "unknown"
        by_module.setdefault(module, []).append(tool)
    return by_module


def _infer_mode_from_tools(tools: List[Dict[str, Any]]) -> str:
    """Infer mode ("chat" or "migration") based on registry metadata."""
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        meta = TOOL_REGISTRY.get(name) or {}
        if meta.get("module") == "migration":
            return "migration"
    return "chat"


def _approval_key(tool_id: str, args: Dict[str, Any]) -> Tuple[str, str]:
    """Stable key for UI approval matching."""
    return tool_id, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def _consume_approval(approved: Set[Tuple[str, str]], tool_id: str, args: Dict[str, Any]) -> bool:
    """Check-and-consume one mutation approval. Returns True if this execution is authorised.

    Approvals are SINGLE USE. An approval authorises exactly one execution of
    exactly these arguments; the same operation requested again — later in the
    turn, or later in the session — gates again. A confirmation dialog that
    silently stops appearing is worse than none, because the user has learned to
    expect it, and a destructive op (delete, cross-environment migrate) repeated
    silently is precisely the failure the gate exists to prevent.

    There is no await between the membership test and the discard, so concurrent
    fan-out branches cannot both claim the same approval.
    """
    key = _approval_key(tool_id, args)
    if key in approved:
        approved.discard(key)
        return True
    return False


_CREDENTIAL_FIELDS: frozenset = frozenset(
    {
        "domain",
        "token",
        "ssl",
        "source_domain",
        "source_token",
        "source_ssl",
        "target_domain",
        "target_token",
        "target_ssl",
    }
)


def _optional_arg_hint(tool_id: str, used_args: Dict[str, Any], tool_meta: Dict[str, Any]) -> str:
    schema = tool_meta.get("parameters") or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    unused_optional = [k for k in props if k not in required and k not in _CREDENTIAL_FIELDS and k not in used_args]
    if not unused_optional or len(unused_optional) > 3:
        return ""
    params = ", ".join(f"`{k}`" for k in unused_optional)
    return f"\n\nOptional filters you could also specify: {params}."


# -----------------------------------------------------------------------------
# Approval-dialog disclosure — only what the tool definition actually states
# -----------------------------------------------------------------------------
# The dialog reports FACTS the registry can back: which optional settings the
# schema declares, which of them this call leaves unset, and their allowed
# values. It does not predict what the operation will DO with them.
#
# Anything about scope or blast radius is deliberately absent. A tool definition
# does not say whether an empty target list means "everything", "nothing", or a
# hard error — that lives in SDK code the registry never sees, and it differs
# per tool (migrate_dashboards raises; migrate_all_dashboards is unbounded by
# design). Guessing from a naming convention produced a warning that was
# confidently wrong. When the definition cannot confirm it, say nothing: let the
# call run and report the SDK's own error verbatim (_describe_tool_result).
def _is_filled(value: Any) -> bool:
    """A value the user actually supplied — not absent, blank, or an empty list."""
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple, set)):
        return bool(value) and not (isinstance(value, str) and not value.strip())
    return True


# -----------------------------------------------------------------------------
# Fabricated values — the third way a required argument can be "not given".
#
# _is_filled catches the first two (absent, or blank/empty), but a model that
# has nothing to put in a slot often invents something well-formed instead:
# "look up a single user by their email address" names NO address, and the
# tool-selection call answers with user_email="user@example.com" — non-empty,
# schema-valid (format: email), and therefore invisible to every emptiness
# check. PLANNING_SYSTEM_PROMPT forbids that exact string by name and the
# model does it anyway ~4 runs in 5, which is why this lives in code.
#
# Two conditions, both required, because either alone is wrong:
#   - it matches a pattern no real Sisense object plausibly carries, AND
#   - the user never typed it this conversation.
# Pattern alone would reject genuine data (a tenant really can hold a user at
# example.com, and the integration suite deletes nobody@fes-test.invalid — a
# reserved TLD, typed deliberately). The corpus check alone would reject
# values a later step legitimately took from an earlier step's RESULT, which
# is how every adaptive chain works.
#
# Names that ARE real here are deliberately absent from the list: this tenant
# holds datamodels called "test", "123" and "1".
# -----------------------------------------------------------------------------
_PLACEHOLDER_EXACT = frozenset(
    {
        "string",
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "not provided",
        "not specified",
        "placeholder",
        "example",
        "foo",
        "bar",
        "tbd",
        "todo",
        "yourname",
        "your-value",
    }
)
# RFC 2606 reserves these precisely so they can never resolve to anything real.
_PLACEHOLDER_EMAIL_DOMAINS = ("example.com", "example.net", "example.org")
_PLACEHOLDER_TLDS = (".example", ".invalid", ".localhost", ".test")
_PLACEHOLDER_MARKERS = ("<", "{{")  # <your-email>, {{user_email}}


def _is_fabricated(value: Any, corpus: str) -> bool:
    """A value that looks invented AND that the user never typed.

    No corpus (outside a turn) → no judgement: we cannot check what was said,
    and guessing would be worse than the miss.
    """
    if not corpus:
        return False
    if isinstance(value, (list, tuple)):
        items = [v for v in value if isinstance(v, str)]
        return bool(items) and len(items) == len(value) and all(_is_fabricated(v, corpus) for v in items)
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v or v.lower() in corpus.lower():
        return False  # the user said it — nothing else matters
    low = v.lower()
    return (
        low in _PLACEHOLDER_EXACT
        or any(low.endswith("@" + d) or low.endswith("." + d) for d in _PLACEHOLDER_EMAIL_DOMAINS)
        or any(low.endswith(t) for t in _PLACEHOLDER_TLDS)
        or any(m in v for m in _PLACEHOLDER_MARKERS)
    )


def _example_hint(meta: Dict[str, Any]) -> str:
    """The curated example's user_query as a phrasing demo — how to ask in
    plain English — never a claim about what the current call will do. Only
    example[0] is held to the curation bar (test_tool_examples.py), so only it
    is ever shown to a user."""
    examples = meta.get("examples") or []
    query = (examples[0].get("user_query") or "").strip() if examples else ""
    return f'*For example, you could ask: "{query}"*' if query else ""


def _options_note(
    tool_id: str,
    meta: Dict[str, Any],
    args: Dict[str, Any],
    with_call_to_action: bool = True,
    heading: Optional[str] = None,
) -> str:
    """Unset optional settings, with their allowed values where the schema
    declares an enum. Runs before an irreversible action, so it lists more than a
    clarification question does — a migration's `action` choice is skip vs
    overwrite, and the model picks it silently.

    Target-selecting params (`dashboard_ids`, `dashboard_names`) are included
    like any other: the schema marks them optional and this call left them
    unset, which is a fact. What the operation will do about that is not stated
    anywhere we can read, so it is not claimed.

    `heading` lets a multi-step plan dialog attribute the block to its step;
    the default suits a single-tool dialog.
    """
    specs = _optional_specs(meta.get("parameters") or {}, args or {}, _OPTIONALS_IN_APPROVAL)
    if not specs:
        return ""
    note = (heading or "**Optional settings, not set**") + "\n" + _optionals_block(specs)
    hint = _example_hint(meta)
    if hint:
        note += f"\n\n{hint}"
    if not with_call_to_action:
        # Inside a multi-step plan the dialog has already said how to approve;
        # repeating it per tool puts two "Approve to…" sentences on screen.
        return note
    return (
        note + "\n\nApprove to run as described, or cancel and ask again including any of the optional settings above."
    )


def _approval_disclosure(
    tool_id: str,
    meta: Dict[str, Any],
    args: Dict[str, Any],
    with_call_to_action: bool = True,
    heading: Optional[str] = None,
) -> str:
    """Deterministic notes appended to the LLM's plain-English explanation."""
    note = _options_note(tool_id, meta, args, with_call_to_action, heading=heading)
    return f"\n\n{note}" if note else ""


# -----------------------------------------------------------------------------
# Clarification loop (Step 7) — ask for missing required args, resume next turn
# -----------------------------------------------------------------------------
def _prop_at(props: Dict[str, Any], field: str) -> Dict[str, Any]:
    """Resolve a possibly-dotted field name (`user_data.email`) to its property
    schema. Dotted names come from _missing_required_fields walking into object
    params; plain names resolve exactly as before."""
    node: Dict[str, Any] = {"properties": props}
    for part in field.split("."):
        inner = node.get("properties")
        if not isinstance(inner, dict) or part not in inner:
            return {}
        node = inner[part] if isinstance(inner[part], dict) else {}
    return node


def _missing_required_fields(args: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Required schema fields the user did not actually provide, excluding
    injected credential fields.

    Three ways a value can be absent, and the model uses all three:
      - the key is omitted;
      - the key carries `""`, null, or `[]` (live 2026-08-14:
        migrate_dashboard_shares gated with empty ID lists) — `_is_filled`,
        the same emptiness rule the optionals use;
      - the key carries an INVENTED value (live 2026-08-18:
        user_email="user@example.com" for a request that named no address)
        — `_is_fabricated`, judged against what the user actually typed.

    Object params get the same treatment ONE level deep: an SDK `dict` param
    (create_user's user_data) keeps its requirements in the property's inner
    `required` list, where the flat walk never looked — so "create user
    himanshu negi" carried neither email nor role, passed `required`
    (user_data existed), and the gated call was doomed before approval (live
    2026-08-27). A parent whose inner requireds are all missing reports the
    dotted children (`user_data.email`), never itself: the children are what
    the user must actually supply, and the question renderer / option lookup
    resolve dotted names via _prop_at.

    Fields that carry `x-options-tool` get one rule more: their valid values
    are a deployment-specific option set (roles, connections), so a value the
    user never actually said can only be the model's guess — "create user
    himanshu negi" was sometimes gated with role="viewer", a value too
    plausible for the placeholder patterns to brand (live 2026-08-27). Judged
    against the corpus with punctuation/spacing stripped, so "Data Designer"
    in the user's words matches a canonical `dataDesigner`. Same abstention as
    _is_fabricated: no corpus (outside a turn) → no judgement.
    """
    required = schema.get("required") or []
    props = schema.get("properties") or {}
    corpus = current_turn_user_corpus()

    def _guessed_option(value: Any, prop: Dict[str, Any]) -> bool:
        if not corpus or not prop.get("x-options-tool") or not isinstance(value, str):
            return False
        squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
        return bool(squash(value)) and squash(value) not in squash(corpus)

    def _absent(value: Any, prop: Dict[str, Any]) -> bool:
        return not _is_filled(value) or _is_fabricated(value, corpus) or _guessed_option(value, prop)

    missing: List[str] = []
    for f in required:
        if f in _CREDENTIAL_FIELDS:
            continue
        prop = props.get(f) if isinstance(props.get(f), dict) else {}
        inner_required = prop.get("required")
        inner_props = prop.get("properties")
        if inner_required and isinstance(inner_props, dict):
            # Object param with declared inner requirements: judge the children.
            value = args.get(f)
            inner_args = value if isinstance(value, dict) else {}
            missing.extend(
                f"{f}.{sub}"
                for sub in inner_required
                if _absent(inner_args.get(sub), inner_props.get(sub) if isinstance(inner_props.get(sub), dict) else {})
            )
        elif _absent(args.get(f), prop):
            missing.append(f)
    return missing


def _validate_tool_args(schema: Dict[str, Any], args: Dict[str, Any]) -> None:
    """The one validation entry point for tool arguments.

    Validates against the generated schema and nothing else. Some SDK methods
    enforce argument combinations the schema cannot express (`migrate_dashboards`
    raises unless exactly one of dashboard_names/dashboard_ids is given), but
    those constraints are not present in the registry, and the registry is
    generated — hand-written rules here would be invented data that a rebuild
    silently drops. Such a call reaches the SDK and its own error surfaces.

    One rule on top of the schema: a required field that is present but EMPTY
    (`[]`, `""`, null) fails here too. JSON Schema `required` only demands the
    key exists, so a model that emits `{"source_dashboard_ids": []}` instead of
    omitting the key sails past `required` — seen live 2026-08-18, gating
    migrate_dashboard_shares with blank id lists when the same prompt clarifies
    whenever the model omits the keys instead. Raising ValidationError sends
    every caller down its existing except-path, where _missing_required_fields
    (same _is_filled rule) names the field and triggers clarification.
    """
    jsonschema.validate(instance=args, schema=schema, format_checker=jsonschema.FormatChecker())
    empty_required = _missing_required_fields(args, schema)
    if empty_required:
        raise jsonschema.ValidationError(f"required field(s) empty: {', '.join(empty_required)}")


# Optional params are surfaced in exactly two places, in BOTH modes:
#   1. a clarification question — "I need X; you can also give me Y"
#   2. a mutation approval dialog — "approve, or cancel and re-ask with Y"
# Both draw from the same selection and render the same way, so they read as one
# feature rather than two that drifted. The approval list is longer because it
# precedes an irreversible action, and cancelling to re-ask is cheap.
_OPTIONALS_IN_QUESTION = 3
_OPTIONALS_IN_APPROVAL = 6


def _clean_desc(text: str) -> str:
    """Strip reStructuredText ``literal`` markup carried over from SDK docstrings."""
    return re.sub(r"``([^`]+)``", r"'\1'", str(text or "")).strip()


def _curated_optionals(
    schema: Dict[str, Any], filled_args: Dict[str, Any], limit: int = _OPTIONALS_IN_QUESTION
) -> List[str]:
    """Up to `limit` optional, non-credential, currently-unfilled param NAMES."""
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    optionals = [
        k for k in props if k not in required and k not in _CREDENTIAL_FIELDS and not _is_filled(filled_args.get(k))
    ]
    return optionals[:limit]


def _param_summary(prop: Dict[str, Any]) -> str:
    """One line a user can act on: the description's first sentence, plus the
    SDK's own `Default: …` sentence when the description states one elsewhere.
    Defaults usually live in sentence two ("Whether to republish dashboards
    after migration. Default: False."), and in an approval dialog the default
    IS the decision — it is what happens if the user approves without it.
    """
    text = _clean_desc(prop.get("description"))
    if not text:
        return ""
    first = text.split(". ")[0].rstrip(".")
    if "default" not in first.lower():
        m = re.search(r"Default:\s*[^.\n]*", text, flags=re.IGNORECASE)
        if m:
            return f"{first}. {m.group(0).rstrip('.')}"
    return first


def _optional_specs(
    schema: Dict[str, Any], filled_args: Dict[str, Any], limit: int
) -> List[Tuple[str, Optional[List[Any]], str]]:
    """The single source of truth for 'which optional params to offer': each is
    its NAME, its allowed values when the schema declares an enum, and a
    one-line summary from the schema's own description.

    The name is what the user has to say back to us. The two surfaces below
    render this same list differently — inline inside a question, as a block in
    a dialog — but they must never disagree about WHICH params they name.
    """
    props = schema.get("properties") or {}
    return [
        (name, (props.get(name) or {}).get("enum"), _param_summary(props.get(name) or {}))
        for name in _curated_optionals(schema, filled_args, limit)
    ]


def _optionals_inline(specs: List[Tuple[str, Optional[List[Any]], str]]) -> str:
    """Comma-run for use mid-sentence (clarification questions). Summaries are
    deliberately dropped: a question is one sentence, not a settings manual."""
    return ", ".join(f"`{n}` ({' / '.join(str(e) for e in enum)})" if enum else f"`{n}`" for n, enum, _ in specs)


def _optionals_block(specs: List[Tuple[str, Optional[List[Any]], str]]) -> str:
    """Markdown list for use as its own section (approval dialogs). `st.info()`
    renders markdown in both dialogs, and an enum buried in a comma-run is an
    enum nobody reads. Each line carries the schema's own one-line summary so
    the user can judge a setting without leaving the dialog."""
    lines = []
    for n, enum, summary in specs:
        parts = [f"`{n}`"]
        if enum:
            parts.append(" / ".join(str(e) for e in enum))
        if summary:
            parts.append(summary)
        lines.append("- " + " — ".join(parts))
    return "\n".join(lines)


def _tool_def_for(tool_id: str) -> Optional[Dict[str, Any]]:
    """Build a single OpenAI-style tool definition from the registry, for resume planning."""
    meta = TOOL_REGISTRY.get(tool_id)
    if not meta:
        return None
    return {
        "type": "function",
        "function": {
            "name": tool_id,
            "description": meta.get("description") or "",
            "parameters": _planner_schema(meta.get("parameters") or {}),
        },
    }


_OPTIONS_EXAMPLES_SHOWN = 3  # example values named inline (summ-ON questions only)


async def _fetch_param_options(
    meta: Dict[str, Any],
    missing_fields: List[str],
    mcp_client: Optional[McpClient],
    mode: str,
) -> Dict[str, Tuple[List[str], int, str]]:
    """Live choices for missing params whose schema names a lookup tool.

    A param property may carry `x-options-tool` (curated in SCHEMA_RULES): the
    tool_id of a READ tool whose rows are the param's valid values. CODE runs
    it — the model never sees the key (strip_internal_params) and never decides
    to look anything up. The question TEXT always gets the count (metadata —
    {tool, ok, count} is what the model sees anyway) and an offer to list the
    full set as its own turn; a few example NAMES go inline only when the
    caller says values may (summarization on): the question enters chat history
    and later LLM prompts, so names inside it ARE result data reaching the
    model — the count is not.

    Returns {field: (names, row_count, note)} for fields whose lookup produced
    rows. Every failure path degrades to the plain question: unknown tool,
    mutating tool (never run a write the user didn't ask for), wrong mode,
    error, or an empty result — log and skip.
    """
    out: Dict[str, Tuple[List[str], int, str]] = {}
    if mcp_client is None:
        return out
    props = (meta.get("parameters") or {}).get("properties") or {}
    for field in missing_fields:
        prop = _prop_at(props, field)
        options_tool = prop.get("x-options-tool")
        if not options_tool:
            continue
        opt_meta = TOOL_REGISTRY.get(options_tool)
        if opt_meta is None or bool(opt_meta.get("mutates")):
            logger.warning(
                "x-options-tool %r for %s.%s is %s — skipping the option lookup.",
                options_tool,
                meta.get("tool_id"),
                field,
                "not in the registry" if opt_meta is None else "a mutating tool",
            )
            continue
        try:
            result = await _invoke_tool_traced(mcp_client, options_tool, {}, mode)
        except Exception as exc:  # noqa: BLE001 — a failed lookup must not block the question
            logger.warning("Option lookup %s failed (%s); asking without it.", options_tool, exc)
            continue
        rows = result.get("result") if isinstance(result, dict) else None
        if not _effective_ok(result) or not isinstance(rows, list) or not rows:
            continue
        names: List[str] = []
        for row in rows:
            if len(names) >= _OPTIONS_EXAMPLES_SHOWN:
                break
            if isinstance(row, dict):
                name = row.get("name") or row.get("title")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
            elif isinstance(row, str) and row.strip():
                names.append(row.strip())
        out[field] = (names, len(rows), str(prop.get("x-options-note") or ""))
    return out


async def _generate_clarification_question(
    tool_id: str,
    meta: Dict[str, Any],
    missing_fields: List[str],
    filled_args: Dict[str, Any],
    trace_id: Optional[str],
    mcp_client: Optional[McpClient] = None,
    mode: str = "chat",
) -> str:
    """The clarifying question, rendered in code — structured like the approval
    dialog, and zero LLM calls.

    This used to be an LLM call whose prompt demanded ONE question covering
    every missing item, so a tool with two required fields plus an optional got
    a cramped run-on paragraph (seen live on migrate_dashboard_shares,
    2026-08-10). Format is structure, and structure is code: a lead line, one
    bullet per missing item from the schema's own descriptions, optionals on
    their own line. Deterministic, testable, and one less call per clarify.

    Kept async and under the old name so call sites and test monkeypatches
    across both engines are untouched.
    """
    schema = meta.get("parameters") or {}
    props = schema.get("properties") or {}

    def _desc(field: str) -> str:
        # First sentence only: some schema descriptions are whole docstrings
        # (user_data enumerates every supported key), and a bullet is a
        # question, not a manual page. Dotted fields (user_data.email) resolve
        # into the object param's inner properties.
        text = _clean_desc(_prop_at(props, field).get("description"))
        if not text:
            return field.rsplit(".", 1)[-1].replace("_", " ")
        return text.split(". ")[0].rstrip(".")

    # Live option lookups for params that declare one (x-options-tool), run by
    # code — see _fetch_param_options. The question TEXT gets only the count
    # and a list-on-request offer, in EVERY summarization mode: this text is an
    # assistant message, and assistant message content re-enters LLM prompts
    # via planning history and clarify-resume — a count is metadata, names are
    # result data. The names themselves go to `display_hints`: a screen-only
    # channel the UI renders under the reply and never puts in a message's
    # content. No client (migration mode, re-asks, bare unit tests) → plain
    # question, byte-identical to before.
    options = await _fetch_param_options(meta, missing_fields, mcp_client, mode)

    # No tool-description header (it read as stiff, UI feedback 2026-08-20) —
    # a plain lead, then one bullet per missing item. The bullets stay clean;
    # option availability gets its own paragraph after the example hint.
    lines = ["I need a bit more information to run this:", ""]
    for f in missing_fields:
        lines.append(f"- {_desc(f)}")
    inline = _optionals_inline(_optional_specs(schema, filled_args, _OPTIONALS_IN_QUESTION))
    if inline:
        lines += ["", f"Optionally, you can also include: {inline}."]
    # The registry's curated example — example[0]'s query names every value its
    # arguments use (pinned by test_tool_examples), which makes it a phrasing
    # template the user can copy with their own values. Worth its lines when a
    # question needs shape, not just names: "paired positionally" says little;
    # "from source dashboards A, B to target dashboards X, Y" shows it.
    # Same renderer as the approval dialog so the two surfaces cannot drift.
    hint = _example_hint(meta)
    if hint:
        lines += ["", hint]
    for f in missing_fields:
        if f in options:
            names, count, note = options[f]
            label = f.rsplit(".", 1)[-1].replace("_", " ")
            plural = "s" if count != 1 else ""
            lines += [
                "",
                f"I found {count} existing option{plural} for the {label}. "
                f"Let me know if you'd like the full list first.{' ' + note if note else ''}",
            ]
            if names:
                shown = ", ".join(f"`{n}`" for n in names)
                _record_display_hint(f"A few existing options for the {label}: {shown} (of {count} total).")
    return "\n".join(lines)


def _clarification_giveup_message(tool_id: str, meta: Dict[str, Any], missing_fields: List[str]) -> str:
    """Terminal message after the clarification attempt cap is exhausted."""
    props = (meta.get("parameters") or {}).get("properties") or {}
    fields = "; ".join((props.get(f) or {}).get("description") or f for f in missing_fields)
    return (
        "I still don't have everything I need to do that. "
        f"The required information is: {fields}. "
        "Please send a new request with those details included."
    )


async def _generate_mutation_explanation(
    tool_id: str,
    meta: Dict[str, Any],
    args: Dict[str, Any],
    trace_id: Optional[str],
) -> str:
    """Plain-English description of what a mutating tool will do, for the approval
    dialog. One LLM call; falls back to a generic-but-safe template on failure.
    Credential fields are stripped before the args reach the LLM.

    The scope/options disclosure is appended in code on every path — an approver
    must not depend on the LLM having chosen to mention that no targets were
    named, and the fallback template needs it just as much as the LLM answer."""
    disclosure = _approval_disclosure(tool_id, meta, args)
    safe_args = {k: v for k, v in (args or {}).items() if k not in _CREDENTIAL_FIELDS}
    user_msg = (
        f"Operation purpose: {meta.get('description', '')}\n"
        f"It will run with these details: {json.dumps(safe_args, ensure_ascii=False)}"
    )
    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": MUTATION_EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
            trace_id=trace_id,
            label="mutation_explain",
        )
        content, _ = _pick_tool_calls_from_llm_response(data)
        if content and content.strip():
            return content.strip() + disclosure
    except Exception as exc:  # noqa: BLE001 — any LLM failure falls back to the template
        logger.warning("Mutation explanation generation failed (%s); using template.", exc)

    purpose = (meta.get("description") or "").rstrip(".")
    base = purpose or "This will modify your Sisense deployment"
    return f"{base}. Review the details below before approving." + disclosure


# -----------------------------------------------------------------------------
# Agentic loop (Step 8) — decide → route → plan → execute, until done or capped
# -----------------------------------------------------------------------------
async def _navigate_for_step(
    step_message: Dict[str, Any],
    mode: str,
    trace_id: Optional[str],
    mode_tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], str, str, int]:
    """Pick the tool menu for one loop step.

    Routing exists to keep ~110 chat tools away from the tool-selection call.
    Migration mode has 9 — that IS a menu — so there is nothing to narrow and
    no tree to walk. It uses the turn's already-scoped tool list directly: the
    same 9 definitions the API filtered from the UI's mode radio, byte for byte,
    without re-reading four JSON files to rebuild them.

    Skipping the walk also closes a hole. The L1 index is not mode-aware, so
    routing in migration mode could land on a chat package — and chat tools get
    no credentials there (the client injects source_*/target_* for the migration
    module and tenant_config for everything else, and tenant_config is empty in
    migration mode), so the call would fail at the SDK boundary with something
    unhelpful.
    """
    if mode == "migration":
        return list(mode_tools or _load_all_package_tools("migration")), "migration", "all", 0
    return await _navigate_to_tools(step_message, [], trace_id)


def _capability_catalog(mode: str) -> str:
    """One line per tool — `tool_id: first line of description (takes: …)` —
    for the planner (plan/replan). NO schemas: the planner writes prose steps,
    it never emits tool calls, so the compact full catalog is safe where showing
    119 schemas to the CALLING tool-selection step would not be. Mode-filtered the same way
    the registry is (migration tools only in migration mode).

    The `(takes: …)` hint lists the tool's REQUIRED param names, straight from
    the generated registry. Without it the planner plans blind to which
    identifier an operation accepts, and hedges: with dashboard listings in
    history it reliably planned "get the dashboard by name, THEN its columns
    [needs-prior-result]" for a tool that takes the NAME — an invented
    id-dependency that the summ-off gate then blocked (live 2026-08-27). Data
    over instruction: a rebuild keeps the hint true for tools nobody has
    written yet, where a prompt rule commands blind trust. Credential and
    internal params never appear (they are stripped from the registry's
    schemas before this reads them)."""
    lines: List[str] = []
    for tid in sorted(TOOL_REGISTRY):
        meta = TOOL_REGISTRY[tid]
        is_migration = meta.get("module") == "migration"
        if (mode == "migration") != is_migration:
            continue
        desc = (meta.get("description") or "").strip().splitlines()
        required = (meta.get("parameters") or {}).get("required") or []
        hint = f" (takes: {', '.join(required)})" if required else ""
        lines.append(f"- {tid}: {desc[0] if desc else ''}{hint}")
    return "\n".join(lines)


def _parse_plan_lines(text: str) -> List[str]:
    """Extract the numbered steps from a planner reply."""
    import re

    steps = []
    for ln in (text or "").splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", ln)
        if m and m.group(1).strip():
            steps.append(m.group(1).strip())
    return steps


def _planner_text_worth_surfacing(op_text: str, user_text: str) -> bool:
    """Should a routing-unclear step be answered with the planner's own text?

    When the planner disobeys its output format and writes a message TO THE
    USER instead of a plan (seen live 2026-08-20: a stale clarify exchange in
    history made it role-play the assistant — "I need more details… Could you
    provide these?"), that prose becomes the step text, routing rightly says
    unclear, and the canned "I didn't quite understand" would discard a reply
    that was actually useful. Surface the planner's text instead — but only
    when ALL of:

    - the step text is planner-AUTHORED (differs from the user's message;
      unclear routing on the user's own words is genuine gibberish, where the
      canned message is right), and
    - it reads like a message to the user (a question mark, or a conversational
      opening) — the other way planner text fails routing is a mangled
      IMPERATIVE ("Provision the abc schema entity"), which would read as
      nonsense if echoed back as a reply.

    Every guard failing keeps the canned fallback, so this can only upgrade
    dead ends, never downgrade working paths.
    """
    op = (op_text or "").strip()
    if not op or op.lower() == (user_text or "").strip().lower():
        return False
    if "?" in op:
        return True
    conversational_openings = ("i need", "i can", "i'm ", "i am ", "could you", "please provide", "to proceed")
    return op.lower().startswith(conversational_openings)


_DEP_MARKER = "[needs-prior-result]"


def _split_dependent_tail(plan_steps: List[str]) -> Tuple[List[str], List[str]]:
    """For summarization-OFF turns: PARTITION the plan into runnable vs skipped.

    Steps tagged [needs-prior-result] need values from earlier RESULTS, which
    the LLM cannot see with summarization off — executing them would only
    produce doomed calls, so they are skipped. Untagged steps run regardless of
    position (independent steps are order-free by definition, so an untagged
    step after a tagged one still runs). Detection is the planner's (text
    reasoning at plan time); this enforcement is code. The marker on step 1 is
    ignored (nothing precedes it). Markers are stripped from both halves."""

    def _clean(st: str) -> str:
        return st.replace(_DEP_MARKER, "").replace(_DEP_MARKER.upper(), "").strip()

    runnable: List[str] = []
    skipped: List[str] = []
    for i, st in enumerate(plan_steps):
        if i > 0 and _DEP_MARKER in st.lower():
            skipped.append(_clean(st))
        else:
            runnable.append(_clean(st))
    return runnable, skipped


async def _make_plan(user_text: str, mode: str, history: List[Dict[str, Any]], trace_id: str) -> List[str]:
    """The upfront planner call: request + capability catalog → ordered plan
    (a list of one-operation instructions). Falls back to [user_text] on any
    failure — planning must never block a turn. Privacy-safe in both summ modes:
    it reads only the request text and the catalog, never tool results."""
    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": AGENT_PLAN_SYSTEM_PROMPT},
                {"role": "system", "content": f"Operation catalog:\n{_capability_catalog(mode)}"},
                *history,
                {"role": "user", "content": user_text},
            ],
            tools=None,
            trace_id=trace_id,
            label="planner",
        )
        text, _ = _pick_tool_calls_from_llm_response(data)
        steps = _parse_plan_lines(text or "")
        if steps:
            return steps
        # A bare unnumbered one-liner still counts as a single-step plan.
        text = (text or "").strip()
        return [text] if text else [user_text]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plan call failed (%s); using the raw message as a single step.", exc)
        return [user_text]


async def _replan(
    user_text: str,
    mode: str,
    transcript: List[Dict[str, Any]],
    reason: str,
    trace_id: str,
) -> Tuple[List[str], str]:
    """The recovery planner (replan): request + what ran (with outcomes) + why the
    executor gave up + the catalog → a REVISED plan for the remaining work, or
    ("GIVEUP", <user-facing sentence>) when no alternative exists. Returns
    (steps, giveup_message) — one of the two is empty."""
    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": AGENT_REPLAN_SYSTEM_PROMPT},
                {"role": "system", "content": f"Operation catalog:\n{_capability_catalog(mode)}"},
                {"role": "user", "content": user_text},
                *transcript,
                {"role": "user", "content": f"The executor gave up on the current plan because: {reason}"},
            ],
            tools=None,
            trace_id=trace_id,
            label="replan",
        )
        text, _ = _pick_tool_calls_from_llm_response(data)
        text = (text or "").strip()
        if text.upper().startswith("GIVEUP"):
            msg = text.split(":", 1)[1].strip() if ":" in text else ""
            return [], msg or "I couldn't find another way to do this with the available operations."
        steps = _parse_plan_lines(text)
        return (steps, "") if steps else ([], "I couldn't find another way to do this with the available operations.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Replan call failed (%s).", exc)
        return [], "I couldn't work out an alternative approach."


def _metadata_record(tool_id: str, result: Any) -> Dict[str, Any]:
    """The privacy-safe view of a tool result for summarization-OFF turns: what
    ran and whether it worked — never the data itself. This is what goes into the
    LLM's history when data must not reach the model.

    `{tool, ok, count}`, plus `error` when the step FAILED — a deliberate, narrow
    exception, decided 2026-08-08.

    Result data stays withheld; the failure reason does not. Without it the loop
    is blind exactly when it needs to think: a failed create left the decide call
    with `ok: false` and nothing else, and it invented a cause ("ensure the email
    is not already in use") that happened to be right. A recovery reasoned from a
    guess is worse than one reasoned from the truth, and the alternative — a code
    table classifying failures into safe labels — replaces the agent's judgement
    with our own enumeration of what can go wrong.

    The exception is narrow because an error normally restates what the user
    already told us ("username/email already exists" for the address THEY typed),
    so it rarely carries anything the model has not already seen in the request.
    Not never, though: an error raised deeper down can quote a value the user
    never supplied — a row in a failing query, a name from a list the tool
    fetched. That residual exposure is accepted and documented in README.md
    ("Security & data handling"), not hidden.

    Successful results are unaffected: no payload, no rows, no field values.
    """
    ok = _effective_ok(result)
    rec: Dict[str, Any] = {"tool": tool_id, "ok": ok}
    if isinstance(result, dict):
        payload = result.get("result")
        if isinstance(payload, list):
            rec["count"] = len(payload)
        if not ok:
            # Wrapper error when the call itself failed; the SDK's own report
            # when the call ran but its payload says failed. Same narrow
            # reason-on-failure exception either way.
            rec["error"] = result.get("error") or _payload_failure_reason(payload) or None
    return rec


def _tool_matches_mode(tool_id: str, mode: str) -> bool:
    """Is this tool reachable in this mode at all?

    Migration mode may use ONLY the migration module, and chat mode may not
    touch it. Mode was previously enforced by every call site remembering to
    filter — the catalog filtered, routing filtered, the API filtered — which
    means one missed check anywhere puts an unreachable tool back in play. It
    already did: the routing bypass covered step 1 only, so step 2 of a
    multi-asset migration could select a chat tool.
    """
    is_migration_tool = (TOOL_REGISTRY.get(tool_id) or {}).get("module") == "migration"
    return is_migration_tool == (mode == "migration")


async def _invoke_tool_traced(
    mcp_client: McpClient, tool_id: str, args: Dict[str, Any], mode: str = "chat"
) -> Dict[str, Any]:
    """Execute one MCP tool call with observability: a `tool` child run in the
    LangSmith trace and a row in tool_calls.csv. Both carry METADATA ONLY
    (tool_id, scrubbed args, ok, count, duration) — result payloads never leave
    for either destination. Errors from the tool propagate unchanged.

    This is the one place every execution path converges, so the mode boundary
    is enforced here rather than trusted upstream. A violation is a bug in the
    caller, not user error: refuse it, log it loudly, and hand the loop an
    ordinary failed result so the turn degrades instead of dying — and so the
    tool never goes out with the wrong credentials attached (chat tools get
    `tenant_config`, which is empty in migration mode).
    """
    if not _tool_matches_mode(tool_id, mode):
        logger.error(
            "Mode violation: %s is not reachable in %s mode — refusing to execute.",
            tool_id,
            mode,
        )
        return {
            "ok": False,
            "tool_id": tool_id,
            "error": f"{tool_id} is not available in {mode} mode.",
        }
    meta = TOOL_REGISTRY.get(tool_id) or {}
    t0 = time.perf_counter()
    try:
        result = await mcp_client.invoke_tool(tool_id, args)
    except Exception as exc:
        ms = int((time.perf_counter() - t0) * 1000)
        write_tool_call(
            tool_id=tool_id, ok=False, count=None, latency_ms=ms, mutates=bool(meta.get("mutates")), error=str(exc)
        )
        log_tool_child(tool_id, args, ok=False, count=None, duration_ms=ms, error=str(exc))
        raise
    ms = int((time.perf_counter() - t0) * 1000)
    rec = _metadata_record(tool_id, result)
    write_tool_call(
        tool_id=tool_id,
        ok=rec.get("ok"),
        count=rec.get("count"),
        latency_ms=ms,
        mutates=bool(meta.get("mutates")),
        error=str(rec.get("error") or ""),
    )
    log_tool_child(tool_id, args, ok=rec.get("ok"), count=rec.get("count"), duration_ms=ms, error=rec.get("error"))
    _record_followup_hint(tool_id, meta, args, result)
    return result


def _record_followup_hint(tool_id: str, meta: Dict[str, Any], args: Dict[str, Any], result: Any) -> None:
    """Queue the tool's follow-up nudge for the end of the turn's reply.

    A tool may carry `x-followup` (curated in SCHEMA_RULES): a consequence the
    user must act on themselves after a successful run — e.g. a schema change
    isn't queryable until the model is deployed. Rendered IN CODE into the final
    reply (`_done` / `_done_reply`), as a suggestion only: the follow-up is a
    mutation the user didn't ask for, so it is never executed, only phrased as
    the exact request they could send next (ask_template formatted from this
    call's own arguments — values the user already supplied).
    """
    followup = meta.get("x-followup")
    if not isinstance(followup, dict) or not _effective_ok(result):
        return
    try:
        ask = str(followup.get("ask_template") or "").format(**args)
    except (KeyError, IndexError):
        logger.warning("x-followup template for %s references an argument the call didn't have; skipping.", tool_id)
        return
    note = str(followup.get("note") or "").strip()
    hint = f'ℹ️ {note} To do that now, you could ask: *"{ask}"*' if ask else (f"ℹ️ {note}" if note else "")
    if not hint:
        return
    out = turn_output()
    if out is not None:
        out.setdefault("followup_hints", []).append(hint)


def _followup_tail() -> str:
    """The turn's queued follow-up nudges, deduplicated, as a reply suffix."""
    out = turn_output()
    hints = list(dict.fromkeys((out or {}).get("followup_hints") or []))
    return ("\n\n" + "\n".join(hints)) if hints else ""


def _record_display_hint(text: str) -> None:
    """Queue a SCREEN-ONLY line for the UI to render under this turn's reply.

    Display hints never enter the reply text or any message content — the only
    thing history carries back into LLM prompts — so they may hold tool result
    data (e.g. connection names) in every summarization mode: data flows tool →
    code → the user's screen, and the model sees it only if the user types it.
    """
    out = turn_output()
    if out is not None and text:
        out.setdefault("display_hints", []).append(text)


def _describe_results_local(raw_results: List[Tuple[str, Any]]) -> str:
    """Render collected results locally (no LLM) for summarization-OFF final
    answers — the raw data never leaves the process."""
    if not raw_results:
        return "No results."
    return "\n\n".join(_describe_tool_result(tid, res) for tid, res in raw_results)


def _transcript_step(call: Dict[str, Any], tool_id: str, result: Any, summ_on: bool) -> List[Dict[str, Any]]:
    """The two messages (assistant tool_call + tool result) appended to the
    LLM-visible history for one executed step. Content is the full result when
    summarization is on, metadata only when off — this is the single point where
    the privacy boundary is enforced in code."""
    content = _shrink_for_llm(result) if summ_on else _metadata_record(tool_id, result)
    return [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call_id": call.get("id"),
            "name": tool_id,
            "content": json.dumps(content, ensure_ascii=False, default=str),
        },
    ]


async def _verify_goal_complete(
    latest_user_message: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    turn_trace_id: str,
) -> Tuple[bool, str]:
    """Independent goal check (verify #3): a separate adversarial LLM call decides
    whether the whole request is actually done. Returns (complete, missing_op).

    This is the *checker* half of maker/checker — the decide call is the maker.
    Scoped to goal completion only; per-step verify (schema, ok flag) is
    deterministic code and needs no second opinion. Sees the same transcript the
    decide call saw, so summarization-off keeps its metadata-only privacy. Any
    failure defaults to 'complete' — the checker must never block a good answer."""
    messages = [
        {"role": "system", "content": VERIFY_GOAL_SYSTEM_PROMPT},
        latest_user_message,
        *transcript,
    ]
    try:
        data = await call_llm_raw(messages, tools=None, trace_id=turn_trace_id, label="verify")
        text, _ = _pick_tool_calls_from_llm_response(data)
        text = (text or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Goal checker failed (%s); accepting the answer.", exc)
        return True, ""
    if text.upper().startswith("INCOMPLETE"):
        missing = text.split(":", 1)[1].strip() if ":" in text else ""
        return (False, missing) if missing else (True, "")
    return True, ""


async def _emit_agent_progress(event: Dict[str, Any]) -> None:
    """Publish a loop progress event to the turn's SSE stream (best-effort)."""
    try:
        # Lazy import: runtime imports this module, so a top-level import would be circular.
        from backend import runtime as runtime_mod

        await runtime_mod.publish_progress({"type": "agent_progress", **event})
    except Exception:  # noqa: BLE001 — progress is cosmetic, never break the turn
        logger.debug("agent progress emit failed", exc_info=True)


def _loop_partial_message(steps_executed: int, remains: str, reason: str) -> str:
    """Terminal message when the loop stops before the goal is complete."""
    return (
        f"I completed {steps_executed} step(s) but stopped before finishing ({reason}). "
        f"Still to do: {remains} "
        "The results so far are shown above — send a follow-up message to continue."
    )


async def _finalize_from_transcript(
    *,
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    raw_results: List[Tuple[str, Any]],
    summ_on: bool,
    turn_trace_id: str,
) -> str:
    """Force a final answer from the results gathered so far — used when the loop
    must stop (e.g. a continued step overreached into a tool needing info the user
    never gave).

    Summarization off: render the raw results locally — data must not reach the
    LLM. Summarization on: one LLM call to summarise; fall back to a local
    description on failure."""
    if not summ_on:
        return _describe_results_local(raw_results)
    messages = [
        {"role": "system", "content": AGENT_DECIDE_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "The turn is ending now. Answer the user based only on the results already gathered. "
            "Do NOT reply CONTINUE.",
        },
        *history,
        latest_user_message,
        *transcript,
    ]
    try:
        data = await call_llm_raw(messages, tools=None, trace_id=turn_trace_id, label="finalize")
        text, _ = _pick_tool_calls_from_llm_response(data)
        text = (text or "").strip()
        # Strip a stray CONTINUE if the model ignores the instruction.
        if text and not text.upper().startswith("CONTINUE:"):
            return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Loop finalize call failed (%s); describing results locally.", exc)
    return _describe_results_local(raw_results)


async def _answer_clarify_question(pc_meta: Dict[str, Any], user_text: str, trace_id: Optional[str]) -> str:
    """Answer a user's question ABOUT a pending clarification from the tool's own
    definition (description + parameter docs + defaults). One LLM call, schema
    only — no result data, so it is safe in both summarization modes. Returns ""
    on any failure: the structured re-ask below stands on its own."""
    if not (user_text or "").strip():
        return ""
    definition = {
        "purpose": (pc_meta.get("description") or "").strip(),
        "parameters": pc_meta.get("parameters") or {},
    }
    try:
        data = await call_llm_raw(
            [
                {"role": "system", "content": CLARIFY_ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Operation definition:\n"
                    + json.dumps(_scrub_secrets(definition), ensure_ascii=False, indent=1)
                    + f"\n\nThe user's question:\n{user_text}",
                },
            ],
            tools=None,
            trace_id=trace_id,
            label="clarify_answer",
        )
        content, _ = _pick_tool_calls_from_llm_response(data)
        return (content or "").strip()
    except Exception as exc:  # noqa: BLE001 — an unanswered question must not block the re-ask
        logger.warning("Clarify-question answer failed (%s); re-asking without it.", exc)
        return ""


async def _reask_clarification_or_giveup(
    resume_clar: Dict[str, Any], turn_trace_id: str, trace: Dict[str, Any], user_text: str = ""
) -> str:
    """Declined-clarification resume whose answer had no clear intent either — it
    was a non-answer ("I'm not sure") or a question about the operation, not a
    topic change. Answer their question if they asked one (from the tool's own
    definition), then re-ask (counting the attempt) or give up at the cap. Sets
    LAST_PENDING_CLARIFICATION when re-asking."""
    global LAST_PENDING_CLARIFICATION
    pc_tool_id = resume_clar.get("tool_id") or ""
    pc_meta = TOOL_REGISTRY.get(pc_tool_id) or {}
    pc_missing = resume_clar.get("missing_fields") or []
    pc_filled = resume_clar.get("filled_args") or {}
    answer = await _answer_clarify_question(pc_meta, user_text, turn_trace_id)
    prefix = f"{answer}\n\n" if answer else ""
    attempts = int(resume_clar.get("attempts", 1)) + 1
    if attempts > CLARIFY_MAX_ATTEMPTS:
        logger.info(
            "Clarification cap (%d) reached for %s after non-answer; giving up.", CLARIFY_MAX_ATTEMPTS, pc_tool_id
        )
        trace["outcome"] = "clarification_exhausted"
        _write_llm_trace(trace)
        return prefix + _clarification_giveup_message(pc_tool_id, pc_meta, pc_missing)
    question = await _generate_clarification_question(pc_tool_id, pc_meta, pc_missing, pc_filled, turn_trace_id)
    question = prefix + question
    _record_pending_clarification(
        {
            "tool_id": pc_tool_id,
            "missing_fields": pc_missing,
            "filled_args": pc_filled,
            "attempts": attempts,
            "question": question,
        }
    )
    logger.info("Clarification re-asked after non-answer: tool=%s attempt=%d", pc_tool_id, attempts)
    trace["outcome"] = "awaiting_clarification"
    _write_llm_trace(trace)
    return question


# The ORCHESTRATOR: reads the planner's blueprint and manages live execution —
# dispatch a step, run it, decide the next move, replan on failure. The planner
# (_make_plan) drafts; this loop orchestrates.
async def _run_loop_engine(**kwargs: Any) -> str:
    """Turn dispatch.

    Migration mode takes its own path (`migration_flow`): plan once, order in
    code, execute in sequence. It is shared by both engines — FES_AGENT_ENGINE
    exists to model the chat loop's branching, and a linear sequence has none.

    Chat mode picks a harness, same contract either way:
      FES_AGENT_ENGINE=langgraph (default) → graph_engine.run_graph_loop (LangGraph
      FES_AGENT_ENGINE=custom             → the hand-rolled `_reactive_loop` —
      kept as the dependency-free kill switch until the retirement criterion is
      met (one langgraph upgrade + more live write-path use; decided 2026-08-15)
      StateGraph over the SAME helpers — thin nodes, no checkpointer/DB/files).
    Read dynamically so tests can flip engines per run without reimport.
    """
    # Only migration_flow understands a whole-plan pause; the loop re-derives.
    pending_plan = kwargs.pop("pending_plan", None)

    if kwargs.get("mode") == "migration" and MIGRATION_SINGLE_SHOT:
        from . import migration_flow  # lazy: avoids circular import at module load

        return await migration_flow.run(pending_plan=pending_plan, **kwargs)

    if os.getenv("FES_AGENT_ENGINE", "langgraph").strip().lower() != "custom":
        from . import graph_engine  # lazy: avoids circular import at module load

        return await graph_engine.run_graph_loop(**kwargs)
    return await _reactive_loop(**kwargs)


async def _reactive_loop(
    *,
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    planning_context: str,
    mode: str,
    passed_tools: List[Dict[str, Any]],
    user_text: str,
    mcp_client: McpClient,
    approved_mutations: Set[Tuple[str, str]],
    summ_on: bool,
    turn_trace_id: str,
    trace: Dict[str, Any],
    transcript: Optional[List[Dict[str, Any]]] = None,
    raw_results: Optional[List[Tuple[str, Any]]] = None,
    steps_executed: int = 0,
    seed_call: Optional[Dict[str, Any]] = None,
    clarify_attempts_base: int = 0,
    resume_clarification: Optional[Dict[str, Any]] = None,
) -> str:
    """
    The single reactive loop for an entire turn — step 1 is not special.

    One iteration = decide-what's-next → route → plan → validate → gate → execute.
    The "what's next" differs only by where we are:
      - step 0, fresh          → decompose the request to its first sub-task
      - step 0, clarify-resolved → use the pinned tool call directly (`seed_call`)
      - step > 0               → the decide call reads goal + history

    Runs in BOTH summarization modes: the flag only controls what the decide call
    and tool-selection step see of each result (full data vs action metadata) — enforced by
    `_transcript_step`. Final answer is LLM prose (summ on) or a local render of
    `raw_results` (summ off, so data never reaches the model).

    Step-0-only exits (a real conversation is possible): clarification (ask the
    user), unclear-intent short-circuit, planning-failure fallback. Later-step
    exits stop-and-summarise instead (`_finalize_from_transcript`). Every exit
    returns readable text — never a silent stop.
    """
    global LAST_TOOL_RESULT, LAST_PENDING_CLARIFICATION, LAST_PENDING_LOOP, LAST_STEP_RESULTS

    transcript = transcript if transcript is not None else []
    raw_results = raw_results if raw_results is not None else []
    first_tool_hint: Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]] = None
    pending_seed = seed_call  # consumed on the first iteration only
    checker_overrides = 0  # times the goal checker has pushed a "done" back into the loop
    replans_used = 0  # times the planner revised the plan this turn
    next_op_override: Optional[str] = None  # set by a replan; consumed instead of a decide call
    blocked_tail: List[str] = []  # summ-off: plan steps skipped because they need prior-result values

    def _done(answer: str) -> str:
        trace["outcome"] = "ok"
        trace["summarization_used"] = summ_on
        trace["agent_steps"] = steps_executed
        _write_llm_trace(trace)
        hint = ""
        if steps_executed == 1 and first_tool_hint:
            _ht, _ha, _hm = first_tool_hint
            hint = _optional_arg_hint(_ht, _ha, _hm)
        tail_note = ""
        if blocked_tail:
            skipped = "; ".join(blocked_tail)
            tail_note = (
                "\n\n⏭️ Skipped (needs a value from an earlier result, which I can't read "
                f"with summarization off): {skipped}. Turn summarization on to run it."
            )
        return answer + hint + tail_note + _followup_tail()

    async def _finalize() -> str:
        return await _finalize_from_transcript(
            latest_user_message=latest_user_message,
            history=history,
            transcript=transcript,
            raw_results=raw_results,
            summ_on=summ_on,
            turn_trace_id=turn_trace_id,
        )

    async def _attempt_replan(reason: str) -> Tuple[Optional[str], str]:
        """Ask the planner for a revised plan after the current approach failed.
        Returns (next_op, giveup_msg): next_op None = no viable alternative
        (budget spent, or the planner gave up)."""
        nonlocal replans_used
        if replans_used >= MAX_REPLANS:
            return None, ""
        replans_used += 1
        trace["replans"] = replans_used
        await _emit_agent_progress({"phase": "replanning", "step": steps_executed, "max_steps": MAX_AGENT_STEPS})
        new_steps, giveup = await _replan(user_text, mode, transcript, reason, turn_trace_id)
        if not new_steps:
            return None, giveup
        plan_text = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(new_steps))
        if summ_on:
            # Replan reasons over the data-bearing transcript — its steps and the
            # failure reason may quote result values.
            for _st in new_steps:
                mark_tainted(_st)
            mark_tainted(reason)
        transcript.append({"role": "assistant", "content": f"REVISED PLAN (after: {reason}):\n{plan_text}"})
        await _emit_agent_progress(
            {"phase": "replanned", "step": steps_executed, "max_steps": MAX_AGENT_STEPS, "plan": plan_text}
        )
        logger.info(
            "Replanned (%d/%d) after: %s -> next: %s", replans_used, MAX_REPLANS, reason[:120], new_steps[0][:120]
        )
        return new_steps[0], ""

    async def _execute_branch(op_text: str, branch_step: int) -> Dict[str, Any]:
        """One independent plan step run concurrently (fan-out): its own
        route→plan→validate→execute pipeline, blind to sibling branches; the
        caller joins results into the shared transcript in plan order. NOT a
        sub-agent — no loop, no own memory. Anything that needs one-at-a-time
        handling (mutation gate, missing-arg clarification, dead ends) is
        DEFERRED back to the sequential loop."""
        step_message = {"role": "user", "content": op_text}
        try:
            nav_tools, nav_pkg, _nm, _ms = await _navigate_to_tools(step_message, [], turn_trace_id)
            if (not nav_tools) and nav_pkg and nav_pkg != "__unclear__":
                nav_tools = _load_all_package_tools(nav_pkg)
            if not nav_tools:
                return {"status": "deferred", "step": branch_step, "op": op_text, "why": "no route"}

            planning_messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "system", "content": planning_context},
                latest_user_message,
                step_message,
            ]
            plan_data = await call_llm_raw(planning_messages, tools=nav_tools, trace_id=turn_trace_id, label="plan")
            _bc, bcalls = _pick_tool_calls_from_llm_response(plan_data)
            if not bcalls:
                return {"status": "deferred", "step": branch_step, "op": op_text, "why": "no tool"}

            bcall = bcalls[0]
            bfn = bcall.get("function") or {}
            btool_id = str(bfn.get("name") or "")
            bargs = _safe_json_loads(bfn.get("arguments", "{}"), default={})
            if not isinstance(bargs, dict):
                bargs = {}
            bmeta = TOOL_REGISTRY.get(btool_id) or {}

            bschema = bmeta.get("parameters")
            if bschema:
                try:
                    _validate_tool_args(bschema, bargs)
                except jsonschema.ValidationError:
                    missing = _missing_required_fields(bargs, bschema)
                    if missing:
                        # `missing` already excludes invented values — a
                        # placeholder must never be echoed back as "you gave me this".
                        filled = {k: v for k, v in bargs.items() if _is_filled(v) and k not in missing}
                        return {
                            "status": "missing",
                            "step": branch_step,
                            "op": op_text,
                            "tool_id": btool_id,
                            "meta": bmeta,
                            "missing": missing,
                            "filled": filled,
                        }
                    return {"status": "deferred", "step": branch_step, "op": op_text, "why": "bad args"}

            if bool(bmeta.get("mutates")) and REQUIRE_MUTATION_CONFIRM:
                if not _consume_approval(approved_mutations, btool_id, bargs):
                    # Gating needs one-at-a-time UX — the sequential loop handles it.
                    return {"status": "deferred", "step": branch_step, "op": op_text, "why": "mutation gate"}
            if bool(bmeta.get("mutates")):
                audit_logger.info(
                    "EXECUTING mutation tool=%s args=%s",
                    btool_id,
                    json.dumps(_scrub_secrets(bargs), ensure_ascii=False),
                )

            await _emit_agent_progress(
                {"phase": "executing", "step": branch_step, "max_steps": MAX_AGENT_STEPS, "tool_id": btool_id}
            )
            bresult = await _invoke_tool_traced(mcp_client, btool_id, bargs, mode)
            await _emit_agent_progress(
                {
                    "phase": "completed",
                    "step": branch_step,
                    "max_steps": MAX_AGENT_STEPS,
                    "tool_id": btool_id,
                    "ok": _effective_ok(bresult),
                }
            )
            return {
                "status": "executed",
                "step": branch_step,
                "op": op_text,
                "tool_id": btool_id,
                "args": bargs,
                "meta": bmeta,
                "call": bcall,
                "result": bresult,
            }
        except Exception as exc:  # noqa: BLE001 — a failed branch defers, never kills the turn
            logger.warning("Fan-out branch %d failed (%s); deferring to sequential loop.", branch_step, exc)
            return {"status": "deferred", "step": branch_step, "op": op_text, "why": "error"}

    while True:
        is_first = steps_executed == 0
        step_number = steps_executed + 1
        calls: List[Dict[str, Any]] = []
        remains = ""

        # ============================================================ what's next
        if is_first and pending_seed is not None:
            # Clarification resolved on a prior turn → the pinned tool is already
            # planned with the user's answer; skip decompose/route/plan.
            calls = [pending_seed]
            pending_seed = None

        elif is_first:
            # Fresh turn: the planner drafts the full plan (request + capability
            # catalog, no schemas), the loop executes its first operation. The plan
            # is stashed in the transcript so decide/verify follow it, and emitted
            # to the UI for transparency.
            await _emit_agent_progress({"phase": "planning", "step": 1, "max_steps": MAX_AGENT_STEPS})
            _raw_plan = await _make_plan(user_text, mode, history, turn_trace_id)
            independent_steps, dependent_steps = _split_dependent_tail(_raw_plan)
            if len(independent_steps) + len(dependent_steps) == 1 and not history:
                # Faithfulness guard (code, not prompt): for a fresh single-step
                # request the user's own message IS the step — the planner's
                # paraphrase tends to echo the catalog description of whichever
                # operation matches, and the step text feeds routing + tool
                # selection downstream. With history present, keep the
                # planner's line: it may resolve references ("its members")
                # that the raw text alone cannot.
                independent_steps, dependent_steps = [user_text], []
            if summ_on:
                # Dependent steps run too — sequentially, after the results they
                # need exist. Order: independents first (fan-out set), then tail.
                plan_steps = independent_steps + dependent_steps
                blocked_tail = []
            else:
                # Dependency gate: dependent steps need values the model can't
                # read with summarization off — skip them (named in the reply).
                plan_steps = independent_steps
                blocked_tail = dependent_steps
                if blocked_tail:
                    logger.info("Summ-off dependency gate: skipping %d dependent step(s).", len(blocked_tail))
            if len(plan_steps) > 1:
                _plan_text = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(plan_steps))
                transcript.append({"role": "assistant", "content": f"PLAN:\n{_plan_text}"})
                await _emit_agent_progress(
                    {"phase": "planned", "step": 1, "max_steps": MAX_AGENT_STEPS, "plan": _plan_text}
                )

            # ------------------------------------------ parallel fan-out (level 1+2)
            # Independent steps need only the user's message, so they can run
            # concurrently — each branch routes/plans/executes on its own, results
            # join here in plan order. Mutations / missing args / dead ends defer
            # to the sequential loop below. Downstream concurrency is bounded by
            # the MCP server's read-tool semaphore.
            _fan = independent_steps[:MAX_PARALLEL_STEPS] if MAX_PARALLEL_STEPS > 1 else []
            if mode != "migration" and len(_fan) >= 2:
                logger.info("Fan-out: running %d independent steps concurrently.", len(_fan))
                _branches = await asyncio.gather(*[_execute_branch(op, i + 1) for i, op in enumerate(_fan)])
                _clarify_branch = None
                for _br in _branches:
                    if _br["status"] == "executed":
                        _record_tool_result(_br["result"])
                        _record_step(_br["step"], _br["tool_id"], _br["result"])
                        raw_results.append((_br["tool_id"], _br["result"]))
                        transcript.extend(_transcript_step(_br["call"], _br["tool_id"], _br["result"], summ_on))
                        if steps_executed == 0:
                            first_tool_hint = (_br["tool_id"], _br["args"], _br["meta"])
                        steps_executed += 1
                        trace["tool_selected"] = trace["tool_selected"] or _br["tool_id"]
                    elif _br["status"] == "missing" and _clarify_branch is None:
                        _clarify_branch = _br
                if _clarify_branch is not None:
                    # A fanned step needs a value the user never gave → ask, like a
                    # fresh-turn clarification (executed siblings' results are kept).
                    question = await _generate_clarification_question(
                        _clarify_branch["tool_id"],
                        _clarify_branch["meta"],
                        _clarify_branch["missing"],
                        _clarify_branch["filled"],
                        turn_trace_id,
                        mcp_client=mcp_client,
                        mode=mode,
                    )
                    _record_pending_clarification(
                        {
                            "tool_id": _clarify_branch["tool_id"],
                            "missing_fields": _clarify_branch["missing"],
                            "filled_args": _clarify_branch["filled"],
                            "attempts": clarify_attempts_base + 1,
                            "question": question,
                        }
                    )
                    trace["outcome"] = "awaiting_clarification"
                    trace["agent_steps"] = steps_executed
                    _write_llm_trace(trace)
                    return question
                if steps_executed > 0:
                    # Joined — hand control to the decide loop for anything left
                    # (deferred branches, dependent tail, goal check).
                    continue
                # All branches deferred → fall through to the sequential path.

            first_op = plan_steps[0] if plan_steps else user_text
            step_message = {"role": "user", "content": first_op if (first_op or "").strip() else user_text}

            if mode == "migration":
                # The 9 tools, already scoped for this turn — no tree walk.
                nav_tools, nav_pkg, nav_mixin, _ = await _navigate_for_step(
                    step_message, mode, turn_trace_id, passed_tools
                )
            else:
                nav_tools, nav_pkg, nav_mixin, _routing_ms = await _navigate_to_tools(
                    step_message, history, turn_trace_id
                )
                if nav_pkg == "__unclear__":
                    if resume_clarification:
                        # Declined clarification + no fresh intent = a non-answer.
                        return await _reask_clarification_or_giveup(
                            resume_clarification, turn_trace_id, trace, user_text
                        )
                    if _planner_text_worth_surfacing(first_op, user_text):
                        # The planner wrote a message to the user instead of a
                        # plan — its text IS the reply; the canned line would
                        # discard it (live 2026-08-20).
                        trace["outcome"] = "planner_text_reply"
                        _write_llm_trace(trace)
                        return first_op.strip()
                    trace["outcome"] = "unclear_intent"
                    _write_llm_trace(trace)
                    return (
                        "I didn't quite understand that. What would you like me to help with? "
                        "For example: 'show all users', 'list dashboards', or 'get all datamodels'."
                    )
                if not nav_tools:
                    nav_tools = passed_tools

            _trace_pkg = f"{nav_pkg}/{nav_mixin}" if nav_mixin else nav_pkg
            trace["routing_module"] = _trace_pkg
            # The request itself goes alongside the step text, as it does in the
            # fan-out branch, the steps>0 path and both graph-engine nodes — this
            # was the one selection site missing it.
            #
            # `history` is prior TURNS, not the current message, so without this
            # the call saw only the planner's sentence. Anything the planner left
            # out of that sentence was then unrecoverable: a request for "a user
            # with the viewer role" planned as "1. Create a user with the email
            # X / 2. Assign the viewer role" produced a create with no role at
            # all, and the SDK rejected it. The value still existed in the user's
            # own words; we simply were not showing them.
            #
            # Skipped when the step text IS the request (the single-step
            # faithfulness guard sets them equal) — no point sending it twice.
            _same = (step_message.get("content") or "").strip() == (latest_user_message.get("content") or "").strip()
            planning_messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "system", "content": planning_context},
                *history,
                *([] if _same else [latest_user_message]),
                step_message,
            ]
            try:
                plan_data = await call_llm_raw(planning_messages, tools=nav_tools, trace_id=turn_trace_id, label="plan")
                content, calls = _pick_tool_calls_from_llm_response(plan_data)
            except Exception as exc:  # noqa: BLE001 — planning failure → keyword fallback
                logger.warning("Planning LLM call failed (%s). Using fallback direct tool.", exc)
                trace["outcome"] = "fallback"
                summary, result = await _fallback_direct_tool(user_text, mcp_client, mode)
                _record_tool_result(result)
                _record_step(1, result.get("tool_id", "fallback"), result)
                _write_llm_trace(trace)
                return summary
            if not calls:
                # Tool-selection chose to answer in natural language (no tool fits).
                trace["outcome"] = "no_tool"
                _write_llm_trace(trace)
                return content or ""

        elif next_op_override is not None:
            # A replan already chose the next operation — skip the decide call.
            remains = next_op_override
            next_op_override = None

        else:
            # ---------------------------------------------------------- decide
            await _emit_agent_progress({"phase": "deciding", "step": steps_executed, "max_steps": MAX_AGENT_STEPS})
            decide_prompt = AGENT_DECIDE_SYSTEM_PROMPT if summ_on else AGENT_DECIDE_NODATA_SYSTEM_PROMPT
            decide_messages = [{"role": "system", "content": decide_prompt}, *history, latest_user_message, *transcript]
            try:
                decide_data = await call_llm_raw(decide_messages, tools=None, trace_id=turn_trace_id, label="decide")
                decide_text, _ = _pick_tool_calls_from_llm_response(decide_data)
                decide_text = (decide_text or "").strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent decide call failed (%s); rendering results locally.", exc)
                trace["outcome"] = "decide_failed"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return _describe_results_local(raw_results)

            # Hedge handling: an action line anywhere wins over surrounding prose.
            dlines = [ln.strip() for ln in decide_text.splitlines() if ln.strip()]
            continue_line = next((ln for ln in dlines if ln.upper().startswith("CONTINUE:")), None)
            replan_line = next((ln for ln in dlines if ln.upper().startswith("REPLAN:")), None)
            blocked_line = next((ln for ln in dlines if ln.upper().startswith("BLOCKED:")), None)

            if continue_line is not None:
                remains = continue_line.split(":", 1)[1].strip()
                if summ_on:
                    # Adaptive value-passing: this text may embed values lifted
                    # from results → redact it in LangSmith traces (content off).
                    mark_tainted(remains)
                logger.info("Agent loop step %d done; continuing: %s", steps_executed, remains[:200])
            elif replan_line is not None:
                # The last step's outcome contradicts the plan → planner revises
                # with the capability catalog (a retry that CHANGES approach).
                reason = replan_line.split(":", 1)[1].strip()
                op, giveup = await _attempt_replan(reason)
                if op is None:
                    trace["outcome"] = "replan_giveup"
                    trace["agent_steps"] = steps_executed
                    _write_llm_trace(trace)
                    prefix = f"{giveup}\n\n" if giveup else ""
                    return prefix + await _finalize()
                remains = op
            elif (not summ_on) and blocked_line is not None:
                reason = blocked_line.split(":", 1)[1].strip()
                trace["outcome"] = "loop_blocked_no_data"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return (
                    "I did the parts I can without reading returned data, but the rest needs a value "
                    f"from an earlier step that I can't see with summarization off ({reason}). "
                    "Turn summarization on to let me continue.\n\n" + _describe_results_local(raw_results)
                )
            else:
                # VERIFY #3 (goal): the decide call (maker) thinks it's done. An
                # independent checker re-reads the whole request against the
                # results and can push one more step if something was missed.
                # Summarization-on only: judging whether the goal was actually
                # ACHIEVED needs the result data. With summarization off the
                # checker would see only metadata — which the decide call already
                # checked — so it adds a call for no real depth; skip it.
                answer = decide_text if summ_on else _describe_results_local(raw_results)
                if summ_on and VERIFY_GOAL and checker_overrides < VERIFY_MAX_RECHECKS:
                    await _emit_agent_progress(
                        {"phase": "verifying", "step": steps_executed, "max_steps": MAX_AGENT_STEPS}
                    )
                    complete, missing = await _verify_goal_complete(latest_user_message, transcript, turn_trace_id)
                    if not complete and missing:
                        checker_overrides += 1
                        trace["goal_rechecks"] = checker_overrides
                        logger.info("Goal checker: INCOMPLETE → continuing with: %s", missing[:160])
                        remains = missing
                    else:
                        return _done(answer)
                else:
                    return _done(answer)

        # ------------------------------------------------ route + plan (steps > 0)
        # Runs for both a decide CONTINUE and a replan-injected op (`calls` is
        # already set on the first step / clarification-seed paths).
        if not calls:
            if steps_executed >= MAX_AGENT_STEPS:
                trace["outcome"] = "step_cap"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return _loop_partial_message(steps_executed, remains, "per-turn step limit reached")

            await _emit_agent_progress({"phase": "planning", "step": step_number, "max_steps": MAX_AGENT_STEPS})
            step_message = {"role": "user", "content": remains}
            nav_tools, nav_pkg, nav_mixin, _ms = await _navigate_for_step(
                step_message, mode, turn_trace_id, passed_tools
            )
            if (not nav_tools) and nav_pkg and nav_pkg != "__unclear__":
                nav_tools = _load_all_package_tools(nav_pkg)
            if not nav_tools:
                # No drawer fits this op — let the planner rephrase/reroute once.
                op, giveup = await _attempt_replan(f"no matching operation found for: {remains}")
                if op:
                    next_op_override = op
                    continue
                trace["outcome"] = "loop_routing_dead_end"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return (f"{giveup}\n\n" if giveup else "") + await _finalize()

            planning_messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "system", "content": planning_context},
                latest_user_message,
                *transcript,
                step_message,
            ]
            try:
                plan_data = await call_llm_raw(planning_messages, tools=nav_tools, trace_id=turn_trace_id, label="plan")
                _content, calls = _pick_tool_calls_from_llm_response(plan_data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent loop planning failed (%s).", exc)
                calls = []
            if not calls and nav_pkg and nav_pkg != "__unclear__":
                full = _load_all_package_tools(nav_pkg)
                if full and len(full) != len(nav_tools):
                    logger.info("Agent loop backtrack: retrying step %d with all %s tools", step_number, nav_pkg)
                    try:
                        plan_data = await call_llm_raw(
                            planning_messages, tools=full, trace_id=turn_trace_id, label="plan"
                        )
                        _content, calls = _pick_tool_calls_from_llm_response(plan_data)
                    except Exception:  # noqa: BLE001
                        calls = []
            if not calls:
                # The tool-selection call couldn't pick a tool for this op — planner retry (replan).
                op, giveup = await _attempt_replan(f"could not pick an operation for: {remains}")
                if op:
                    next_op_override = op
                    continue
                trace["outcome"] = "loop_planning_dead_end"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return (f"{giveup}\n\n" if giveup else "") + await _finalize()

        # ============================================ one tool for this iteration
        call = calls[0]
        fn = call.get("function") or {}
        tool_id = str(fn.get("name") or "")
        if not tool_id:
            trace["outcome"] = "no_execution"
            _write_llm_trace(trace)
            return ""
        args = _safe_json_loads(fn.get("arguments", "{}"), default={})
        if not isinstance(args, dict):
            args = {}
        meta = TOOL_REGISTRY.get(tool_id) or {}
        trace["tool_selected"] = tool_id

        # -------------------------------------------------------------- validate
        tool_schema = meta.get("parameters")
        if tool_schema:
            try:
                _validate_tool_args(tool_schema, args)
            except jsonschema.ValidationError as _ve:
                missing = _missing_required_fields(args, tool_schema)
                if missing and is_first:
                    # First step missing a required arg the user never gave → ask.
                    attempts = clarify_attempts_base + 1
                    if attempts > CLARIFY_MAX_ATTEMPTS:
                        logger.info("Clarification cap (%d) reached for %s; giving up.", CLARIFY_MAX_ATTEMPTS, tool_id)
                        trace["outcome"] = "clarification_exhausted"
                        _write_llm_trace(trace)
                        return _clarification_giveup_message(tool_id, meta, missing)
                    filled = {k: v for k, v in args.items() if _is_filled(v) and k not in missing}
                    question = await _generate_clarification_question(
                        tool_id,
                        meta,
                        missing,
                        filled,
                        turn_trace_id,
                        mcp_client=mcp_client,
                        mode=mode,
                    )
                    _record_pending_clarification(
                        {
                            "tool_id": tool_id,
                            "missing_fields": missing,
                            "filled_args": filled,
                            "attempts": attempts,
                            "question": question,
                        }
                    )
                    logger.info("Clarification needed: tool=%s missing=%s attempt=%d", tool_id, missing, attempts)
                    trace["outcome"] = "awaiting_clarification"
                    _write_llm_trace(trace)
                    return question
                if missing:
                    # Mid-loop: the decide call overreached into a detail the user
                    # didn't ask for → stop and answer from what we have.
                    logger.info(
                        "Agent loop overreach at step %d: %s needs %s; finalizing.", step_number, tool_id, missing
                    )
                    trace["outcome"] = "loop_overreach_finalized"
                    trace["agent_steps"] = steps_executed
                    _write_llm_trace(trace)
                    return await _finalize()
                # Value present but wrong (format/type/enum) → hard block.
                logger.error("Tool %s arg validation failed: %s", tool_id, _ve.message)
                if is_first:
                    trace["outcome"] = "validation_failed"
                    _write_llm_trace(trace)
                    return (
                        f"I couldn't call `{tool_id}` — a required argument is invalid or missing: "
                        f"{_ve.message}. Please provide more details."
                    )
                trace["outcome"] = "loop_validation_failed"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return _loop_partial_message(steps_executed, remains, f"invalid argument for {tool_id}: {_ve.message}")

        logger.info("Tool selected: %s (mutates=%s)", tool_id, bool(meta.get("mutates")))
        _log_json("Tool args (from tool selection)", args)

        # ------------------------------------------------------------------ gate
        if bool(meta.get("mutates")) and REQUIRE_MUTATION_CONFIRM:
            if not _consume_approval(approved_mutations, tool_id, args):
                explanation = await _generate_mutation_explanation(tool_id, meta, args, turn_trace_id)
                _record_tool_result(
                    {
                        "ok": False,
                        "pending_confirmation": {"tool_id": tool_id, "arguments": args, "reason": explanation},
                    }
                )
                _record_pending_loop(
                    {
                        "transcript": transcript,
                        "raw_results": raw_results,
                        "steps_executed": steps_executed,
                        "tool_id": tool_id,
                        "arguments": args,
                    }
                )
                logger.info("Agent loop paused for mutation approval at step %d: %s", step_number, tool_id)
                trace["outcome"] = "loop_pending_mutation" if not is_first else "pending_mutation"
                trace["agent_steps"] = steps_executed
                _write_llm_trace(trace)
                return explanation

        if bool(meta.get("mutates")):
            audit_logger.info(
                "EXECUTING mutation tool=%s args=%s",
                tool_id,
                json.dumps(_scrub_secrets(args), ensure_ascii=False),
            )

        # --------------------------------------------------------------- execute
        await _emit_agent_progress(
            {"phase": "executing", "step": step_number, "max_steps": MAX_AGENT_STEPS, "tool_id": tool_id}
        )
        result = await _invoke_tool_traced(mcp_client, tool_id, args, mode)
        _record_tool_result(result)
        _record_step(step_number, tool_id, result)
        raw_results.append((tool_id, result))
        transcript.extend(_transcript_step(call, tool_id, result, summ_on))
        if is_first:
            first_tool_hint = (tool_id, args, meta)
        steps_executed += 1
        await _emit_agent_progress(
            {
                "phase": "completed",
                "step": step_number,
                "max_steps": MAX_AGENT_STEPS,
                "tool_id": tool_id,
                "ok": _effective_ok(result),
            }
        )


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
async def call_llm_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    mcp_client: McpClient,
    approved_mutations: Optional[Set[Tuple[str, str]]] = None,
    allow_summarization: Optional[bool] = None,
    pending_clarification: Optional[Dict[str, Any]] = None,
    pending_loop: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Run a single agent turn: planning -> tool execution -> optional summarization.

    Parameters
    ----------
    messages:
        Full UI conversation history (user + assistant turns).
    tools:
        OpenAI-style tool definitions (already filtered by the API layer per mode).
    mcp_client:
        Connected MCP client for calling tools.
    approved_mutations:
        Set of approval keys allowing a mutating tool to execute this turn.
    allow_summarization:
        Per-turn override from the API layer.
        Global env ALLOW_SUMMARIZATION is still a hard cap.
    pending_clarification:
        Carried-over clarification state from the previous turn (Step 7). When set,
        the router is skipped and the tool-selection call re-runs constrained to the pinned tool
        to merge the user's answer. Shape: {tool_id, missing_fields, filled_args, attempts}.
    pending_loop:
        Carried-over agentic-loop state from a turn that paused mid-loop for a
        mutation approval (Step 8). When set AND this turn approves that exact
        tool+args, the gated tool executes directly and the loop resumes from the
        paused step. Shape: {transcript, steps_executed, tool_id, arguments}.
    """
    global LAST_TOOL_RESULT, LAST_PENDING_CLARIFICATION, LAST_PENDING_LOOP, LAST_STEP_RESULTS, LAST_TRACE_ID

    approved_mutations = approved_mutations or set()

    # Global is a hard cap; per-turn can only further restrict.
    if allow_summarization is None:
        allow_summarization_flag = ALLOW_SUMMARIZATION
    else:
        allow_summarization_flag = ALLOW_SUMMARIZATION and bool(allow_summarization)

    LAST_TOOL_RESULT = None
    LAST_STEP_RESULTS = []
    # Cleared each turn; set again only if this turn pauses for clarification.
    LAST_PENDING_CLARIFICATION = None
    # Cleared each turn; set again only if this turn pauses mid-loop for approval.
    LAST_PENDING_LOOP = None

    latest_user_message = _extract_latest_user_message(messages)
    user_text = str(latest_user_message.get("content", ""))

    mode = _infer_mode_from_tools(tools)
    planning_context = MIGRATION_PLANNING_CONTEXT_PROMPT if mode == "migration" else CHAT_PLANNING_CONTEXT_PROMPT

    # Scope the turn's tool universe ONCE, here, rather than asking every code
    # path downstream to remember the mode. `tools` already arrives filtered by
    # the UI's mode radio → _select_tools_for_mode, but that function falls back
    # to returning ALL tools if its filter comes up empty (a broken-registry
    # safety valve), which in migration mode would put chat tools back in play.
    # Re-filter so what the loop holds is authoritative, not advisory.
    tools = [t for t in (tools or []) if _tool_matches_mode(((t.get("function") or {}).get("name") or ""), mode)]

    # One UUID per agent turn — groups planning + summarization LLM calls into a
    # single LangSmith trace. Contains no credentials or customer data.
    turn_trace_id = str(uuid.uuid4())
    LAST_TRACE_ID = turn_trace_id

    # Stamp every LLM call this turn makes with this id + the user message, for
    # the per-call log (llm_calls.csv). Task-isolated, so no reset needed.
    # The corpus is every user message of the conversation: the record of what
    # the person actually typed, which is how _is_fabricated tells a value they
    # supplied from one the model invented. Whole conversation, not just this
    # message — an identifier given two turns ago is still theirs.
    _user_corpus = "\n".join(str(m.get("content") or "") for m in (messages or []) if (m.get("role") or "") == "user")
    set_current_turn(turn_trace_id, user_text, _user_corpus)
    # Open this turn's race-free output slot (keyed by trace_id); every
    # _record_* call lands here, and runtime._run_turn_once pops it at the end.
    begin_turn_output(turn_trace_id)

    _trace: Dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "trace_id": turn_trace_id,
        "mode": mode,
        "user_message": user_text[:500],
        "model": LLM_CONFIG.model,
        "provider": LLM_PROVIDER,
        "tools_available": len(tools),
        "routing_module": "",
        "routing_latency_ms": 0,
        "tool_selected": "",
        "outcome": "unknown",
        "planning_tokens_in": 0,
        "planning_tokens_out": 0,
        "planning_latency_ms": 0,
        "summary_tokens_in": 0,
        "summary_tokens_out": 0,
        "summary_latency_ms": 0,
        "summarization_used": False,
    }

    logger.info(
        "call_llm_with_tools start: mode=%s tools=%d approvals=%d allow_summarization=%s trace_id=%s",
        mode,
        len(tools),
        len(approved_mutations),
        allow_summarization_flag,
        turn_trace_id,
    )

    # -------------------------------------------------------------------------
    # 1) Build planning history (last-N-turns context)
    # -------------------------------------------------------------------------
    _history = _build_planning_history(messages, latest_user_message, LLM_PLANNING_HISTORY_TURNS)
    logger.debug("Planning history: %d prior messages (max turns=%d)", len(_history), LLM_PLANNING_HISTORY_TURNS)

    # -------------------------------------------------------------------------
    # 1a-loop) Agentic loop resume (Step 8): a previous turn paused mid-loop for
    # a mutation approval. If this turn approves that exact tool+args, execute it
    # directly (no re-plan — deterministic) and hand back to the continuation
    # loop with the saved transcript. Any other input drops the paused loop and
    # processes normally (typing something else = implicit cancel).
    # -------------------------------------------------------------------------
    # A migration pause under single-shot is a whole PLAN, not one tool —
    # migration_flow owns matching the approval and running the sequence.
    if mode == "migration" and MIGRATION_SINGLE_SHOT and pending_loop and pending_loop.get("plan") is None:
        # A per-step pause left over from the reactive loop (kill switch flipped
        # mid-session, or an older deploy). Executing it here and then handing on
        # would make the flow replan and re-propose work that just ran, so drop
        # it and treat this as a fresh request.
        logger.info("Dropping a per-step migration pause; single-shot plans and approves as a whole.")
        pending_loop = None

    if pending_loop and pending_loop.get("plan") is not None:
        return await _run_loop_engine(
            latest_user_message=latest_user_message,
            history=_history,
            planning_context=planning_context,
            mode=mode,
            passed_tools=tools,
            user_text=user_text,
            mcp_client=mcp_client,
            approved_mutations=approved_mutations,
            summ_on=allow_summarization_flag,
            turn_trace_id=turn_trace_id,
            trace=_trace,
            transcript=list(pending_loop.get("transcript") or []),
            raw_results=list(pending_loop.get("raw_results") or []),
            steps_executed=int(pending_loop.get("steps_executed", 0)),
            pending_plan=pending_loop,
        )

    if pending_loop:
        _pl_tool_id = str(pending_loop.get("tool_id") or "")
        _pl_args = pending_loop.get("arguments") or {}
        if _pl_tool_id in TOOL_REGISTRY and _consume_approval(approved_mutations, _pl_tool_id, _pl_args):
            _pl_meta = TOOL_REGISTRY.get(_pl_tool_id) or {}
            audit_logger.info(
                "EXECUTING mutation (loop resume) tool=%s args=%s",
                _pl_tool_id,
                json.dumps(_scrub_secrets(_pl_args), ensure_ascii=False),
            )
            _pl_step = int(pending_loop.get("steps_executed", 0)) + 1
            await _emit_agent_progress(
                {"phase": "executing", "step": _pl_step, "max_steps": MAX_AGENT_STEPS, "tool_id": _pl_tool_id}
            )
            result = await _invoke_tool_traced(mcp_client, _pl_tool_id, _pl_args, mode)
            _record_tool_result(result)
            _record_step(_pl_step, _pl_tool_id, result)
            await _emit_agent_progress(
                {
                    "phase": "completed",
                    "step": _pl_step,
                    "max_steps": MAX_AGENT_STEPS,
                    "tool_id": _pl_tool_id,
                    "ok": _effective_ok(result),
                }
            )

            _resume_call = {
                "id": f"resume-{turn_trace_id[:8]}",
                "type": "function",
                "function": {"name": _pl_tool_id, "arguments": json.dumps(_pl_args, ensure_ascii=False)},
            }
            _resume_transcript = list(pending_loop.get("transcript") or [])
            _resume_transcript.extend(_transcript_step(_resume_call, _pl_tool_id, result, allow_summarization_flag))
            _resume_raw = list(pending_loop.get("raw_results") or [])
            _resume_raw.append((_pl_tool_id, result))
            _trace["routing_module"] = "loop_resume"
            _trace["tool_selected"] = _pl_tool_id
            return await _run_loop_engine(
                latest_user_message=latest_user_message,
                history=_history,
                planning_context=planning_context,
                mode=mode,
                passed_tools=tools,
                user_text=user_text,
                mcp_client=mcp_client,
                approved_mutations=approved_mutations,
                summ_on=allow_summarization_flag,
                turn_trace_id=turn_trace_id,
                trace=_trace,
                transcript=_resume_transcript,
                raw_results=_resume_raw,
                steps_executed=_pl_step,
            )
        logger.info("Dropping paused agent loop (no matching approval this turn).")

    # -------------------------------------------------------------------------
    # 1b) Clarification resume (Step 7): skip routing, re-plan the pinned tool.
    # On the resume turn the latest user message is the answer to a prior
    # clarifying question, so routing on it alone would be unreliable. Instead we
    # re-run the tool-selection call constrained to the one tool we were resolving, with
    # tool_choice="auto" so a decline (the answer wasn't really an answer →
    # topic change) cleanly falls back to fresh routing.
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 1b) Clarification resume (Step 7): the latest user message is the answer to
    # a prior clarifying question. Re-plan the pinned tool constrained to it
    # (tool_choice="auto" so a decline = topic change). A resolved re-plan becomes
    # the loop's first step (seed_call); a decline enters the loop fresh, carrying
    # the pending state so a non-answer ("I'm not sure") re-asks instead of routing.
    # -------------------------------------------------------------------------
    seed_call: Optional[Dict[str, Any]] = None
    clarify_attempts_base = 0
    resume_clarification: Optional[Dict[str, Any]] = None

    if pending_clarification:
        _pc_tool_id = pending_clarification.get("tool_id")
        _pc_def = _tool_def_for(_pc_tool_id) if _pc_tool_id else None
        if _pc_def is None:
            logger.warning("Resume: pending tool %s not in registry — dropping clarification.", _pc_tool_id)
        else:
            clarify_attempts_base = int(pending_clarification.get("attempts", 1))
            # Anchor the re-plan with the stored clarifying question if the client
            # didn't echo it in history (the UI does; bare API clients may not).
            _pc_question = pending_clarification.get("question") or ""
            _needs_q = _pc_question and not any(
                m.get("role") == "assistant" and m.get("content") == _pc_question for m in _history
            )
            _resume_messages = [
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "system", "content": planning_context},
                *_history,
                *([{"role": "assistant", "content": _pc_question}] if _needs_q else []),
                latest_user_message,
            ]
            logger.info("Resume: re-planning pinned tool %s (attempt base=%d).", _pc_tool_id, clarify_attempts_base)
            _tool_calls: List[Dict[str, Any]] = []
            try:
                _rdata = await call_llm_raw(
                    _resume_messages, tools=[_pc_def], trace_id=turn_trace_id, tool_choice="auto", label="plan_resume"
                )
                _pc_content, _tool_calls = _pick_tool_calls_from_llm_response(_rdata)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Resume planning failed (%s); falling back to fresh routing.", exc)
                _tool_calls = []
            if _tool_calls:
                seed_call = _tool_calls[0]
                _trace["routing_module"] = "resume"
            else:
                # Declined the pinned tool → either a topic change (enter fresh)
                # or a non-answer ("i'm not sure"). Decide WHICH here, before the
                # planner runs, by routing the user's words ALONE.
                #
                # The planner sees conversation history, so it happily turns a
                # non-answer into a concrete step: "i'm not sure", following a
                # delete-user clarification, was planned as "List all users" and
                # executed (live 2026-08-18) — the clarification died silently
                # and the attempt cap never fired. Downstream `__unclear__`
                # checks can't catch that: by then the message has been replaced
                # by the planner's confident sentence. History is exactly what
                # makes anything routable, so this test gets none: a real topic
                # change stands on its own words; a non-answer does not.
                logger.info("Resume: tool selection declined %s — testing for fresh intent.", _pc_tool_id)
                _, _fresh_pkg, _, _ = await _navigate_to_tools(latest_user_message, [], turn_trace_id)
                if _fresh_pkg == "__unclear__":
                    return await _reask_clarification_or_giveup(pending_clarification, turn_trace_id, _trace, user_text)
                resume_clarification = pending_clarification
                clarify_attempts_base = 0

    # -------------------------------------------------------------------------
    # 2) The single reactive loop — handles step 1 through N (both summ modes).
    # -------------------------------------------------------------------------
    return await _run_loop_engine(
        latest_user_message=latest_user_message,
        history=_history,
        planning_context=planning_context,
        mode=mode,
        passed_tools=tools,
        user_text=user_text,
        mcp_client=mcp_client,
        approved_mutations=approved_mutations,
        summ_on=allow_summarization_flag,
        turn_trace_id=turn_trace_id,
        trace=_trace,
        seed_call=seed_call,
        clarify_attempts_base=clarify_attempts_base,
        resume_clarification=resume_clarification,
    )
