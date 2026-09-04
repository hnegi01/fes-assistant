"""
backend/agent/_routing.py

Two-stage module routing, conversation history, and raw LLM call.

What lives here:
  - System prompts re-exported from _prompts.py (edit prompts there, not here)
  - _MODULE_DESCRIPTIONS — module name → one-liner for the routing prompt
  - _build_planning_history() — last-N-turns context extraction
  - _parse_module_from_response() — extract module name from LLM response
  - _route_to_module() — stage-1 routing LLM call
  - _pick_tool_calls_from_llm_response() — parse tool_calls from OpenAI response
  - _extract_latest_user_message() — find the latest user message in history
  - call_llm_raw() — single LiteLLM call with retry and tracing
  - _fallback_direct_tool() — keyword-based fallback when planning fails
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import litellm

from ._config import (
    ALLOW_MUTATING_TOOLS,
    LLM_CONFIG,
    MAX_LLM_HTTP_RETRIES,
    ROOT_DIR,
    TOOL_EXAMPLES_COUNT,
    _log_json,
    _make_module_logger,
    _scrub_secrets,
    add_turn_usage,
    write_llm_call,
)
from ._prompts import (
    ROUTING_SYSTEM_PROMPT,
)
from ._registry import _load_registry_rows, allowed_tool_ids
from ._tracing import log_llm_child
from .mcp_client import McpClient

logger = _make_module_logger("backend.agent.llm_routing", "llm_routing.log")

REGISTRY_DIR: Path = ROOT_DIR / "config" / "registry"


# -----------------------------------------------------------------------------
# Conversation history
# -----------------------------------------------------------------------------
def _build_planning_history(
    messages: List[Dict[str, Any]],
    latest_user_message: Dict[str, Any],
    n_turns: int,
) -> List[Dict[str, Any]]:
    """
    Extract the last n_turns of conversation history for the planning call.

    Rules:
    - Skips the latest_user_message (it is appended separately by the caller).
    - Assistant messages are stripped to their text content only — tool result
      payloads are excluded so they don't bloat the planning prompt.
    - Empty assistant messages (e.g. pending-confirmation turns) are skipped.
    - Returns at most n_turns * 2 messages (n_turns user + n_turns assistant).
    """
    history: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        if msg is latest_user_message:
            continue
        if role == "assistant":
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            history.append({"role": "assistant", "content": content})
        else:
            history.append({"role": "user", "content": msg.get("content", "")})

    if n_turns <= 0:
        return []
    return history[-(n_turns * 2) :]


# -----------------------------------------------------------------------------
# Two-stage routing helpers
# -----------------------------------------------------------------------------
def _parse_module_from_response(content: str, modules: Dict[str, str]) -> Optional[str]:
    """
    Extract a module name from an LLM routing response.

    Tries exact match first, then substring match. Returns None if no
    known module name is found in the response.
    """
    if not content:
        return None
    content_lower = content.strip().lower()
    if content_lower in modules:
        return content_lower
    for name in modules:
        if name in content_lower:
            return name
    return None


async def _route_to_module(
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    modules: Dict[str, str],
    trace_id: Optional[str],
) -> Tuple[Optional[str], int]:
    """
    Stage 1 of two-stage routing: ask the LLM which module best fits the request.

    Returns (module_name, latency_ms). module_name is None on any failure so
    the caller can fall back to the full tool list.
    """
    module_list = "\n".join(f"- {name}: {desc}" for name, desc in sorted(modules.items()))
    routing_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ROUTING_SYSTEM_PROMPT.format(module_list=module_list)},
        *history,
        latest_user_message,
    ]
    t0 = time.perf_counter()
    try:
        data = await call_llm_raw(routing_messages, tools=None, trace_id=trace_id, label="route")
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("Routing LLM call failed (%s). Falling back to full tool list.", exc)
        return None, latency_ms
    latency_ms = int((time.perf_counter() - t0) * 1000)

    content, _ = _pick_tool_calls_from_llm_response(data)
    raw = (content or "").strip().lower()
    if raw == "none":
        logger.info("Router signalled no Sisense intent for this message.")
        return "__unclear__", latency_ms
    chosen = _parse_module_from_response(content or "", modules)
    if not chosen:
        logger.warning("Router returned unrecognised response %r. Falling back.", raw[:80])
    return chosen, latency_ms


# -----------------------------------------------------------------------------
# 3-level hierarchical navigation
# -----------------------------------------------------------------------------


def _load_registry_index() -> Dict[str, Any]:
    """Load config/registry/index.json — Level 1 package descriptions."""
    path = REGISTRY_DIR / "index.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load registry index: %s", exc)
        return {}


def _load_package_index(package: str) -> Dict[str, Any]:
    """Load config/registry/{package}/index.json — Level 2 mixin descriptions."""
    path = REGISTRY_DIR / package / "index.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load package index for %s: %s", package, exc)
        return {}


# Parameters that exist on the SDK signature but can never come from a user.
# `emit` is the progress callback the MCP server injects itself for streaming
# tools — a client can only ever supply nonsense for it, and tools_core drops
# whatever arrives. The registry is generated by introspecting the SDK, so these
# leak in on every rebuild; strip them at the boundary rather than trusting the
# data. The MCP server does the same to its advertised list (server.py).
INTERNAL_PARAMS: frozenset = frozenset({"emit"})

# Hard provider limit: OpenAI rejects a tools array longer than this.
MAX_TOOLS_PER_CALL: int = 128

# The one package that is mode-exclusive. Mirrors llm_agent._tool_matches_mode,
# which reaches the same verdict from a tool's `module` — same rule, two levels:
# that one keeps a mis-selected tool from executing, this one keeps the router
# from being offered it in the first place.
MIGRATION_PACKAGE: str = "migration"


def strip_internal_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop INTERNAL_PARAMS from a JSON Schema's properties and required list.

    Also drops per-property `x-options-*` keys: they tell the CLARIFICATION path
    which read tool lists a parameter's valid values — code-consumed, like
    `emit`. Showing the model a lookup it cannot perform only invites it to try.
    (`x-aliases` stays: vocabulary normalization is for the model.)
    """
    schema = copy.deepcopy(params or {})
    props = schema.get("properties")
    if isinstance(props, dict):
        for name in INTERNAL_PARAMS:
            props.pop(name, None)

        def _strip_options(prop: Dict[str, Any]) -> None:
            # Recurse: object params carry inner properties (create_user's
            # user_data), and a nested x-options-* must not leak to the model
            # any more than a top-level one.
            for key in [k for k in prop if k.startswith("x-options-")]:
                prop.pop(key, None)
            inner = prop.get("properties")
            if isinstance(inner, dict):
                for sub in inner.values():
                    if isinstance(sub, dict):
                        _strip_options(sub)

        for prop in props.values():
            if isinstance(prop, dict):
                _strip_options(prop)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [f for f in required if f not in INTERNAL_PARAMS]
    return schema


def planner_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    """Schema variant sent to the planning LLM: same properties, but no `required` list.

    Marking a field required pressures the model to fill it with *something* —
    placeholders ("user@example.com"), empty strings, or words lifted from the
    request ("email"). With no required pressure it naturally omits values the
    user didn't provide; server-side validation against the real schema then
    routes genuinely missing fields into the clarification loop.

    Internal params (`emit`) are stripped too: showing the model a callback slot
    it cannot fill only invites it to invent one.
    """
    schema = strip_internal_params(params)
    schema.pop("required", None)
    return schema


def _format_tool_examples(row: Dict[str, Any], n: int) -> str:
    """Render up to `n` of a tool's curated examples as a description suffix.

    Each line pairs a request with the arguments it produces, so the model learns
    the argument SHAPE and — because every curated query names the values its
    arguments use — that values come FROM the request, never invented. Returns ""
    when n <= 0 or the tool has no usable examples, leaving the description
    untouched.
    """
    if n <= 0:
        return ""
    lines = []
    for ex in (row.get("examples") or [])[:n]:
        query = (ex.get("user_query") or "").strip()
        args = ex.get("arguments")
        if not query or not isinstance(args, dict):
            continue
        # Same reason as planner_schema: never demonstrate filling in a param
        # the user cannot supply — an example outweighs a rule.
        args = {k: v for k, v in args.items() if k not in INTERNAL_PARAMS}
        lines.append(f'- "{query}" -> {json.dumps(args, ensure_ascii=False, sort_keys=True)}')
    if not lines:
        return ""
    heading = "Example call:" if len(lines) == 1 else "Example calls:"
    return "\n\n" + heading + "\n" + "\n".join(lines)


def _load_mixin_tools(package: str, mixin: str) -> List[Dict[str, Any]]:
    """
    Load tools from config/registry/{package}/{mixin}.json and convert to OpenAI format.
    Filters out tools missing from the curated allowlist (config/allowed_tools.txt)
    and mutating tools when ALLOW_MUTATING_TOOLS is False.
    Appends FES_TOOL_EXAMPLES few-shot examples per tool (default 1; 0 = none).
    """
    path = REGISTRY_DIR / package / f"{mixin}.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load mixin tools %s/%s: %s", package, mixin, exc)
        return []

    allowed = allowed_tool_ids()

    tools = []
    for row in rows:
        if allowed is not None and row.get("tool_id") not in allowed:
            continue
        if not ALLOW_MUTATING_TOOLS and row.get("mutates"):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": row["tool_id"],
                    "description": row.get("description", "") + _format_tool_examples(row, TOOL_EXAMPLES_COUNT),
                    "parameters": planner_schema(row.get("parameters") or {"type": "object", "properties": {}}),
                },
            }
        )
    return tools


def _reachable_packages_and_mixins() -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """Which packages and mixins still contain a tool the agent may actually call.

    The registry is GENERATED from the SDK and the allowlist is CURATED on top,
    so a package can be emptied entirely by delisting — encryption, mergetool
    and metadata all went to zero on 2026-09-01. Level 3 has always applied the
    allowlist, but Levels 1 and 2 read the generated index directly, so the
    router was still offered packages it could not reach: it picks one, walks
    down, finds nothing, and the route is spent. Five of fourteen L1 options
    were dead ends.

    Derived at runtime rather than filtered into the generated index on
    purpose: the allowlist is mtime-cached and re-read per turn precisely so a
    hand-edit takes effect without a rebuild. Baking it into config/registry/
    would silently reintroduce the rebuild requirement.

    Mirrors both gates Level 3 applies — the allowlist (already applied by
    _load_registry_rows) and the server-side mutation switch — so a package
    holding only writes disappears when mutations are off.
    """
    packages: Set[str] = set()
    mixins: Set[Tuple[str, str]] = set()
    for row in _load_registry_rows():
        if not ALLOW_MUTATING_TOOLS and row.get("mutates"):
            continue
        pkg = row.get("module") or ""
        if not pkg:
            continue
        packages.add(pkg)
        # Mirrors build_registry_hierarchical's stem rule: "access_management.users"
        # -> "users"; a package-level tool with no sub-module -> "_base".
        sub = row.get("sub_module") or pkg
        mixins.add((pkg, sub.split(".", 1)[1] if "." in sub and sub != pkg else "_base"))
    return packages, mixins


def _reachable_mixin_tool_names() -> Dict[Tuple[str, str], List[str]]:
    """(package, mixin) -> the short names of the tools it exposes.

    Same pass and same gates as _reachable_packages_and_mixins, so a delisted or
    mutation-blocked tool is never advertised at Level 1.

    Level 1 used to show a package blurb plus its module names and blurbs. That
    is prose, and prose competes badly: pysisense 2.1.0 added a `perspectives`
    module to `datamodel` whose description accurately says it keeps "a subset
    of its tables and columns", which made `datamodel` the best PROSE match for
    "show me the columns of a datamodel" — while the tool that answers it,
    get_datamodel_columns, lives in `access_management`. The router picked
    datamodel, found no columns tool, and settled for get_all_datamodel: 318
    data models returned for a question about columns (live 2026-09-04, three
    integration tests).

    Names end the argument. "columns" appears in one module's PROSE and in two
    modules' TOOL NAMES, and the tool names are what the caller actually gets.
    Generated from the registry, so they stay true through every SDK refresh —
    the same reason the module names are here rather than a hand-written hint,
    and unlike a doc correction they do not need to be removed when upstream
    moves the method (see pysisense_fix.md #9: these three tools are in a class
    that does not describe them, which is why the prose was ambiguous at all).
    """
    names: Dict[Tuple[str, str], List[str]] = {}
    for row in _load_registry_rows():
        if not ALLOW_MUTATING_TOOLS and row.get("mutates"):
            continue
        pkg = row.get("module") or ""
        tool_id = row.get("tool_id") or ""
        if not pkg or not tool_id:
            continue
        sub = row.get("sub_module") or pkg
        mixin = sub.split(".", 1)[1] if "." in sub and sub != pkg else "_base"
        names.setdefault((pkg, mixin), []).append(tool_id.split(".", 1)[-1])
    return {k: sorted(v) for k, v in names.items()}


def _load_all_package_tools(package: str) -> List[Dict[str, Any]]:
    """
    Load all tools for a package by combining every mixin file.
    Used for migration mode (all ~9 tools in one shot, no navigation needed).
    """
    pkg_dir = REGISTRY_DIR / package
    if not pkg_dir.is_dir():
        logger.warning("Package directory not found: %s", pkg_dir)
        return []
    tools = []
    for mixin_file in sorted(pkg_dir.glob("*.json")):
        if mixin_file.name == "index.json":
            continue
        tools.extend(_load_mixin_tools(package, mixin_file.stem))
    return tools


async def _navigate_to_tools(
    latest_user_message: Dict[str, Any],
    history: List[Dict[str, Any]],
    trace_id: Optional[str],
    mode: str = "chat",
) -> Tuple[List[Dict[str, Any]], str, str, int]:
    """
    3-level navigation: package → mixin → tools.

    Level 1: LLM picks a package from config/registry/index.json descriptions.
    Level 2: LLM picks a mixin from {package}/index.json (skipped if only 1 mixin).
    Level 3: tools loaded from {package}/{mixin}.json — returned for the planning call.

    `mode` scopes the Level 1 menu. The turn's tool universe is already filtered
    by mode once at entry (llm_agent.call_llm_with_tools), but navigation does
    not read that list — it rebuilds a menu from config/registry/ on disk, so
    the scoping never reached it. In chat mode that left `migration` as one of
    the offered packages: the router could pick it, Level 3 would load migration
    tools, and the execution choke point would then strip every one of them,
    leaving the step with an empty menu. Never unsafe — `_tool_matches_mode`
    holds — but a wasted route, and `migration` is precisely the package the
    router reaches for when someone types "move" or "copy".

    This is the mirror of the existing rule that migration mode does not walk
    the tree at all (see _navigate_for_step). That direction is the dangerous
    one, because a chat tool selected there gets no credentials; this one only
    costs a turn.

    Returns (tools, chosen_package, chosen_mixin, total_routing_ms).
    Returns ([], "", "", ms) on any failure so the caller can fall back.
    """
    total_ms = 0

    # Level 1 — pick package
    index = _load_registry_index()
    packages = index.get("packages", {})
    if not packages:
        logger.warning("Registry index empty — cannot navigate")
        return [], "", "", 0

    # Offer only packages that still hold a callable tool. Without this the
    # router spends its one choice on a package the allowlist has emptied.
    reachable_pkgs, reachable_mixins = _reachable_packages_and_mixins()
    # …and only packages this MODE can execute. One shared filter for both
    # reasons, so the router's menu always matches what the turn could run.
    reachable_pkgs = {p for p in reachable_pkgs if (p == MIGRATION_PACKAGE) == (mode == "migration")}
    dropped = [p for p in packages if p not in reachable_pkgs]
    packages = {p: info for p, info in packages.items() if p in reachable_pkgs}
    if dropped:
        logger.debug(
            "Level 1: hiding %d package(s) with no exposed tools: %s", len(dropped), ", ".join(sorted(dropped))
        )
    if not packages:
        logger.warning("Every package is empty after the allowlist — cannot navigate")
        return [], "", "", 0

    # Package blurb + the names of the modules inside it + the tools each one
    # exposes. The blurb alone is a prose summary that can omit whole
    # capabilities — routing then never even offers the right package, and the
    # tool cannot be reached at any later level. Prose also COMPETES badly: two
    # packages can both describe "columns" honestly while only one holds a
    # columns tool (see _reachable_mixin_tool_names). All of it is generated
    # data, not hand-written hints, so it stays true through every SDK refresh.
    mixin_tools = _reachable_mixin_tool_names()

    def _pkg_desc(pkg: str, info: Dict[str, Any]) -> str:
        desc = info.get("description", "")
        # Emptied mixins are hidden here too: advertising a module the router
        # cannot reach is the same dead end one level down.
        mods = {n: b for n, b in (info.get("modules") or {}).items() if (pkg, n) in reachable_mixins}
        if not mods:
            return desc
        lines = []
        for name, blurb in sorted(mods.items()):
            tools = mixin_tools.get((pkg, name)) or []
            # Names BEFORE the blurb, deliberately. Some blurbs run past 150
            # characters, and a tool list tacked on the end sits behind all of
            # that prose — the thing meant to settle the ambiguity ends up
            # weighed least. Name, then contents, then description.
            head = f"  - {name}" + (f" ({', '.join(tools)})" if tools else "")
            lines.append(f"{head}: {blurb}")
        return f"{desc}\nContains:\n" + "\n".join(lines)

    pkg_descs = {pkg: _pkg_desc(pkg, info) for pkg, info in packages.items()}
    chosen_pkg, ms1 = await _route_to_module(latest_user_message, history, pkg_descs, trace_id)
    total_ms += ms1

    if chosen_pkg == "__unclear__":
        return [], "__unclear__", "", total_ms

    if not chosen_pkg:
        logger.warning("Level 1 navigation: no package selected")
        return [], "", "", total_ms

    logger.info("Level 1: chose package=%s (%dms)", chosen_pkg, ms1)

    # Level 2 — pick mixin (skip if only 1)
    pkg_index = _load_package_index(chosen_pkg)
    # Same filter as Level 1: a mixin whose tools are all delisted must not be
    # offered. Doing it here also means the single-mixin shortcut below fires
    # when only ONE REACHABLE mixin remains, saving a routing call the router
    # would otherwise spend choosing between live and dead options.
    modules = {n: b for n, b in (pkg_index.get("modules") or {}).items() if (chosen_pkg, n) in reachable_mixins}

    if not modules:
        logger.warning("Package %s has no modules with exposed tools", chosen_pkg)
        return [], chosen_pkg, "", total_ms

    if len(modules) == 1:
        chosen_mixin = list(modules.keys())[0]
        logger.info("Level 2 skipped — single mixin in %s: %s", chosen_pkg, chosen_mixin)
    else:
        chosen_mixin, ms2 = await _route_to_module(latest_user_message, history, modules, trace_id)
        total_ms += ms2
        if chosen_mixin == "__unclear__":
            # Router saw no clear intent at mixin level — propagate the same
            # short-circuit as a Level 1 unclear instead of falling through to a
            # failed file load (0 tools → full-registry fallback → forced tool call).
            logger.info("Level 2 navigation: unclear intent in %s — short-circuiting", chosen_pkg)
            return [], "__unclear__", "", total_ms
        if not chosen_mixin:
            logger.warning("Level 2 navigation: no mixin selected in %s", chosen_pkg)
            return [], chosen_pkg, "", total_ms
        logger.info("Level 2: chose mixin=%s (%dms)", chosen_mixin, ms2)

    # Load Level 3 tools
    tools = _load_mixin_tools(chosen_pkg, chosen_mixin)
    logger.info(
        "Navigation complete: %s → %s → %d tools (total routing_ms=%d)",
        chosen_pkg,
        chosen_mixin,
        len(tools),
        total_ms,
    )
    return tools, chosen_pkg, chosen_mixin, total_ms


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------
def _pick_tool_calls_from_llm_response(
    data: Dict[str, Any],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Extract assistant content and tool_calls from an OpenAI-style response."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, []
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None, []
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    return (
        content if isinstance(content, str) else None,
        tool_calls if isinstance(tool_calls, list) else [],
    )


def _extract_latest_user_message(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get the last user message from a full UI conversation history."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m
    raise ValueError("No user message found for LLM planning call.")


# -----------------------------------------------------------------------------
# Raw LLM call
# -----------------------------------------------------------------------------
def _extract_tool_choice(resp: Dict[str, Any]) -> Tuple[str, str]:
    """The tool call(s) the model returned, for the per-call CSV row.

    Records the model's CHOICE, not what executed — a pick that later fails
    validation or is discarded on backtrack still leaves a row, which is what
    a cross-model accuracy comparison needs. Multiple tool_calls (migration's
    single-shot plan) join with ";" / a JSON list. Args cannot contain
    credentials (they are injected after the LLM call), but they pass the
    scrubber anyway.
    """
    try:
        msg = ((resp.get("choices") or [{}])[0].get("message")) or {}
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return "", ""
        names = ";".join((tc.get("function") or {}).get("name") or "" for tc in tcs)
        parsed = []
        for tc in tcs:
            raw = (tc.get("function") or {}).get("arguments") or "{}"
            try:
                parsed.append(_scrub_secrets(json.loads(raw)))
            except Exception:  # noqa: BLE001 — unparseable args are still evidence
                parsed.append(raw)
        args = json.dumps(parsed[0] if len(parsed) == 1 else parsed, ensure_ascii=False)
        return names, args
    except Exception:  # noqa: BLE001 — tracing must never break a call
        return "", ""


async def call_llm_raw(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    trace_id: Optional[str] = None,
    tool_choice: Optional[str] = None,
    label: str = "",
) -> Dict[str, Any]:
    """
    Make a single LLM call via LiteLLM and return the response as a plain dict.

    When tools are provided, tool_choice defaults to "required" so the tool-selection
    call always selects a tool rather than answering in free text. Pass tool_choice
    explicitly (e.g. "auto") to let it decline — used on the
    clarification-resume turn, where a decline signals the user changed topic.
    Providers: azure, databricks, huggingface.
    """
    kwargs: Dict[str, Any] = {
        "model": LLM_CONFIG.model,
        "messages": messages,
        "api_key": LLM_CONFIG.api_key,
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
        "timeout": LLM_CONFIG.timeout_seconds,
        "num_retries": MAX_LLM_HTTP_RETRIES,
    }

    if LLM_CONFIG.api_base:
        kwargs["api_base"] = LLM_CONFIG.api_base
    if LLM_CONFIG.api_version:
        kwargs["api_version"] = LLM_CONFIG.api_version

    if tools:
        # OpenAI rejects a tools array longer than 128 outright (HTTP 400
        # "array too long"), which turns the whole turn into a keyword-fallback
        # answer. Exposing 146 tools put chat mode at 137, so any path that
        # hands over the unrouted full list — routing returning nothing, a
        # widened backtrack — failed hard (live 2026-08-29). Routing normally
        # keeps this call at ~10 tools; the cap only ever bites on those
        # fallback paths, and a truncated menu still beats no call at all.
        # Logged at WARNING because a trimmed menu can hide the right tool.
        if len(tools) > MAX_TOOLS_PER_CALL:
            logger.warning(
                "Tool list of %d exceeds the provider cap of %d — sending the first %d. "
                "Routing failed to narrow this call; the right tool may have been trimmed.",
                len(tools),
                MAX_TOOLS_PER_CALL,
                MAX_TOOLS_PER_CALL,
            )
            tools = tools[:MAX_TOOLS_PER_CALL]
        kwargs["tools"] = tools
        # "required" forces the tool-selection call to always emit a tool_call; "auto" lets it
        # decline (return plain text). litellm.drop_params=True silently drops this
        # for providers that don't support it.
        kwargs["tool_choice"] = tool_choice or "required"

    if trace_id or label:
        # LangSmith metadata: name the run by its kind (route / plan / decide /
        # verify / ...) and tag the turn id for filtering. We deliberately do NOT
        # set the reserved `trace_id` key: LiteLLM would keep our turn id as the
        # LangSmith trace_id while generating a different run_id with no parent,
        # producing a dotted_order whose first segment != trace_id — which
        # LangSmith rejects (HTTP 400). Omitting it lets LiteLLM default
        # trace_id = run_id, so each call is a valid standalone trace. Per-turn
        # grouping lives in llm_calls.csv (grouped by trace_id). No creds/data.
        kwargs["metadata"] = {
            "run_name": label or "llm-call",
            "call_type": label or "unknown",
            "turn_id": trace_id or "",
        }

    logger.info(
        "LLM call start: kind=%s model=%s messages=%d tools=%d",
        label or "unknown",
        LLM_CONFIG.model,
        len(messages),
        len(tools or []),
    )
    _log_json("LLM request kwargs (full, scrubbed)", {k: v for k, v in kwargs.items() if k != "api_key"})

    # The sub-task THIS call saw: the last user-role message. On single-step
    # turns it equals the turn text; on multi-step turns it is the step text —
    # which is what lets a CSV row attribute a bad pick to its sub-question.
    _step_text = next(
        (str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )

    _t0 = time.perf_counter()
    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as exc:
        _ms = int((time.perf_counter() - _t0) * 1000)
        write_llm_call(
            call_type=label,
            n_messages=len(messages),
            n_tools=len(tools or []),
            latency_ms=_ms,
            ok=False,
            error=str(exc),
            step_text=_step_text,
        )
        log_llm_child(label, messages, None, _ms, n_tools=len(tools or []), error=str(exc))
        raise
    data = response.model_dump()
    _usage = data.get("usage") or {}
    _ms = int((time.perf_counter() - _t0) * 1000)
    _tool_selected, _tool_args = _extract_tool_choice(data)
    _tin = _usage.get("prompt_tokens", 0) or 0
    _tout = _usage.get("completion_tokens", 0) or 0
    try:
        # $ for this call from litellm's model pricing map. None (NOT 0.0)
        # when the model isn't in the map: zero would read as "free" or
        # "broken math" — unknown must look unknown. Tokens stay exact.
        _cost = float(litellm.completion_cost(completion_response=response))
    except Exception:  # noqa: BLE001 — pricing must never break a call
        _cost = None
    add_turn_usage(_tin, _tout, _cost)
    write_llm_call(
        call_type=label,
        n_messages=len(messages),
        n_tools=len(tools or []),
        latency_ms=_ms,
        tokens_in=_tin,
        tokens_out=_tout,
        cost=_cost,
        ok=True,
        step_text=_step_text,
        tool_selected=_tool_selected,
        tool_args=_tool_args,
    )
    log_llm_child(label, messages, data, _ms, n_tools=len(tools or []))
    _log_json("LLM raw response (full)", data)
    return data


# -----------------------------------------------------------------------------
# Fallback: direct tool (only for when planning fails hard)
# -----------------------------------------------------------------------------
async def _fallback_direct_tool(
    user_text: str, mcp_client: McpClient, mode: str = "chat"
) -> Tuple[str, Dict[str, Any]]:
    """Run a safe, simple tool based on keywords if planning fails.

    Migration mode has no safe fallback and must not guess. Every one of its
    nine tools mutates a live target environment, so there is nothing here that
    could be run unattended, and the chat tools below are unreachable in that
    mode anyway — they would go out with no credentials, since the client only
    injects source_*/target_* for the migration module. Refuse and say so.
    """
    text = (user_text or "").lower()

    if mode == "migration":
        logger.info("Planning failed in migration mode; refusing to guess a fallback tool.")
        result = {
            "ok": False,
            "error": (
                "The planning step could not work out which migration to run, and there is no safe "
                "fallback in migration mode — every migration writes to the target environment, so "
                "nothing runs without your approval. Please rephrase what you want migrated "
                "(for example, 'migrate the Sales Team group' or 'migrate all dashboards')."
            ),
            "error_type": "PlanningFailed",
        }
        return result["error"], result

    # Keep fallback tools read-only (no mutations).
    if "user" in text:
        tool_id = "access_management.get_users_all"
        args: Dict[str, Any] = {}
    elif "dashboard" in text:
        tool_id = "dashboard.get_dashboards_all"
        args = {}
    elif "data model" in text or "datamodel" in text or "data models" in text:
        tool_id = "datamodel.get_all_datamodel"
        args = {}
    else:
        result = {
            "ok": False,
            "error": (
                "The planning step could not select a tool, and no safe fallback match was possible. "
                "Please rephrase your request (for example, 'show all users' or 'list dashboards')."
            ),
            "error_type": "PlanningFailed",
        }
        return result["error"], result

    logger.info("Fallback: executing tool directly without planning: %s", tool_id)
    result = await mcp_client.invoke_tool(tool_id, args)

    data = result.get("result")
    if isinstance(data, list):
        summary = (
            "The planning step failed, so a keyword-based fallback was used.\n\n"
            f"Executed `{tool_id}` and retrieved **{len(data)}** records. The full result is available in the UI."
        )
    else:
        summary = (
            "The planning step failed, so a keyword-based fallback was used.\n\n"
            f"Executed `{tool_id}`. The result is not a simple table, so the raw payload is provided."
        )

    return summary, result
