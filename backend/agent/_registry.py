"""
backend/agent/_registry.py

Tool registry loading and result payload processing.

What lives here:
  - REGISTRY_PATH and mtime-cached _load_registry_rows()
  - _describe_tool_result() — local result description (used when summarization is off)
  - _shrink_for_llm() — generic payload shrinker for LLM summarization
  - _safe_json_loads() — best-effort JSON parse helper
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ._config import ROOT_DIR, _make_module_logger

logger = _make_module_logger("backend.agent.llm_registry", "llm_registry.log")

_registry_env = os.getenv("PYSISENSE_REGISTRY_PATH")
REGISTRY_PATH: Path = (
    Path(_registry_env) if _registry_env else ROOT_DIR / "config" / "tools.registry.with_examples.json"
)

_allowlist_env = os.getenv("FES_TOOL_ALLOWLIST")
ALLOWLIST_PATH: Path = Path(_allowlist_env) if _allowlist_env else ROOT_DIR / "config" / "allowed_tools.txt"

_allowlist_cache_mtime: Optional[float] = None
_allowlist_cache_ids: Optional[Set[str]] = None
_allowlist_missing_warned = False


def allowed_tool_ids() -> Optional[Set[str]]:
    """Tool ids the curated allowlist permits, or None when there is no allowlist.

    None means "no allowlist in force — allow everything the registry has".
    Deleting config/allowed_tools.txt therefore DISABLES the filter rather than
    hiding every tool, so a missing file can never silently empty the surface.

    mtime-cached like the registry itself, so hand-editing the file takes effect
    on the next turn without restarting the backend.
    """
    global _allowlist_cache_mtime, _allowlist_cache_ids, _allowlist_missing_warned

    if not ALLOWLIST_PATH.exists():
        if not _allowlist_missing_warned:
            logger.warning(
                "Tool allowlist not found at %s — ALL registry tools are exposed. "
                "Create it with: python scripts/04_generate_tool_allowlist.py --init",
                ALLOWLIST_PATH.resolve(),
            )
            _allowlist_missing_warned = True
        _allowlist_cache_mtime = None
        _allowlist_cache_ids = None
        return None

    try:
        mtime = ALLOWLIST_PATH.stat().st_mtime
    except Exception:
        mtime = None

    if mtime is not None and _allowlist_cache_mtime == mtime and _allowlist_cache_ids is not None:
        return _allowlist_cache_ids

    try:
        ids: Set[str] = set()
        for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
            # Strip trailing comments ("tool_id  # [write] description") and blanks.
            entry = line.split("#", 1)[0].strip()
            if entry:
                ids.add(entry)
    except Exception as exc:
        logger.exception("Failed to read tool allowlist (%s) — allowing all tools: %s", ALLOWLIST_PATH, exc)
        _allowlist_cache_mtime = mtime
        _allowlist_cache_ids = None
        return None

    _allowlist_cache_mtime = mtime
    _allowlist_cache_ids = ids
    logger.info("Loaded tool allowlist: %d tool(s) permitted (path=%s)", len(ids), ALLOWLIST_PATH.resolve())
    return ids


def _filter_by_allowlist(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop registry rows whose tool_id is not on the curated allowlist."""
    allowed = allowed_tool_ids()
    if allowed is None:
        return list(rows)
    return [r for r in rows if r.get("tool_id") in allowed]


_registry_cache_mtime: Optional[float] = None
_registry_cache_rows: List[Dict[str, Any]] = []


def _load_registry_rows() -> List[Dict[str, Any]]:
    """Load tool registry JSON from disk with a simple mtime cache.

    Rows are filtered through the curated allowlist on the way out (not on the
    way into the cache), so an allowlist edit is picked up even when the registry
    file itself has not changed.
    """
    global _registry_cache_mtime, _registry_cache_rows

    if not REGISTRY_PATH.exists():
        logger.warning("Tool registry not found at %s", REGISTRY_PATH.resolve())
        _registry_cache_mtime = None
        _registry_cache_rows = []
        return []

    try:
        mtime = REGISTRY_PATH.stat().st_mtime
    except Exception:
        mtime = None

    if mtime is not None and _registry_cache_mtime == mtime and _registry_cache_rows:
        return _filter_by_allowlist(_registry_cache_rows)

    try:
        raw = REGISTRY_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            logger.error("Registry JSON is not a list (path=%s)", REGISTRY_PATH.resolve())
            _registry_cache_rows = []
            _registry_cache_mtime = mtime
            return []
        _registry_cache_rows = data
        _registry_cache_mtime = mtime
        logger.info("Loaded registry with %d entries (path=%s)", len(data), REGISTRY_PATH.resolve())
        return _filter_by_allowlist(data)
    except Exception as exc:
        logger.exception("Failed to load registry JSON: %s", exc)
        _registry_cache_rows = []
        _registry_cache_mtime = mtime
        return []


def _describe_tool_result(tool_name: str, result: Optional[Dict[str, Any]]) -> str:
    """Generate a human-readable result description without calling the LLM."""
    if not result or not isinstance(result, dict):
        return f"`{tool_name}` ran. Result shown above."

    if not result.get("ok", True):
        # Surface the tool's OWN words. When a schema cannot express a
        # precondition, the SDK's error is the only accurate account of what
        # went wrong — swallowing it leaves the user with "failed" and nothing
        # actionable. Code path, so this never sends result data to the LLM.
        err = str(result.get("error") or "").strip()
        return f"`{tool_name}` failed — {err}" if err else f"`{tool_name}` failed. Details shown above."

    data = result.get("result")

    if isinstance(data, dict) and "error" in data:
        return f"`{tool_name}` returned an error — {data['error']}"

    if isinstance(data, list):
        n = len(data)
        if n == 0:
            return f"`{tool_name}` returned no results."
        noun = "result" if n == 1 else "results"
        return f"Found {n} {noun} from `{tool_name}`. Results shown above."

    if isinstance(data, dict):
        # "Got a response" reads as uncertainty. `ok` was true, so say so —
        # this is the line a user sees after approving a create or an update
        # with summarization off, and it should confirm the thing happened.
        return f"`{tool_name}` succeeded. Details shown above."

    return f"`{tool_name}` completed. Result shown above."


MAX_LIST_ITEMS_FOR_LLM = 20
MAX_KEYS_PER_OBJECT_FOR_LLM = 10
MAX_DEPTH_FOR_LLM = 8
MAX_STRING_LENGTH_FOR_LLM = 300
MAX_TOTAL_LENGTH_FOR_LLM = 10_000
TRUNCATION_NOTE_KEY = "_truncated"


def _shrink_for_llm(
    value: Any,
    *,
    max_list_items: int = MAX_LIST_ITEMS_FOR_LLM,
    max_keys_per_object: int = MAX_KEYS_PER_OBJECT_FOR_LLM,
    max_depth: int = MAX_DEPTH_FOR_LLM,
    max_string_length: int = MAX_STRING_LENGTH_FOR_LLM,
    max_total_length: int = MAX_TOTAL_LENGTH_FOR_LLM,
) -> Any:
    """Shrink tool results before sending them to the LLM for summarization."""
    budget = {"remaining": max_total_length}

    def take(n: int) -> None:
        budget["remaining"] = max(0, budget["remaining"] - n)

    def inner(obj: Any, depth: int) -> Any:
        if budget["remaining"] <= 0:
            return "... [truncated due to max_total_length]"

        if isinstance(obj, str):
            s = obj
            if len(s) > max_string_length:
                s = s[:max_string_length] + "... [truncated]"
            take(len(s))
            return s

        if isinstance(obj, (int, float, bool)) or obj is None:
            take(len(str(obj)))
            return obj

        if isinstance(obj, list):
            out: List[Any] = []
            total = len(obj)
            for item in obj[:max_list_items]:
                if budget["remaining"] <= 0:
                    break
                out.append(inner(item, depth + 1))
            if total > max_list_items:
                note = f"... [{total - max_list_items} more items omitted for summarization]"
                take(len(note))
                out.append(note)
            take(2 + len(out))
            return out

        if isinstance(obj, dict):
            if depth >= max_depth:
                summary_text = f"Nested content limited for summarization (object with {len(obj)} keys)"
                take(len(summary_text))
                return {"_summary": summary_text}

            out_dict: Dict[str, Any] = {}
            items = list(obj.items())
            total_keys = len(items)

            for idx, (k, v) in enumerate(items):
                if idx >= max_keys_per_object or budget["remaining"] <= 0:
                    break
                ks = str(k)
                take(len(ks))
                out_dict[ks] = inner(v, depth + 1)

            if total_keys > max_keys_per_object:
                note = f"{total_keys - max_keys_per_object} additional fields omitted for summarization"
                out_dict["_truncated_keys"] = note
                take(len(note))

            take(2 + len(out_dict))
            return out_dict

        s = repr(obj)
        if len(s) > max_string_length:
            s = s[:max_string_length] + "... [truncated]"
        take(len(s))
        return s

    shrunk = inner(value, depth=0)

    if budget["remaining"] <= 0:
        if isinstance(shrunk, dict):
            shrunk.setdefault(
                TRUNCATION_NOTE_KEY,
                "Payload limited due to summarization size constraints; only partial content shown.",
            )
        else:
            shrunk = {
                TRUNCATION_NOTE_KEY: (
                    "Payload limited due to summarization size constraints; only partial content shown."
                ),
                "partial": shrunk,
            }

    return shrunk


def _safe_json_loads(text: Any, default: Any) -> Any:
    """Best-effort JSON parse helper."""
    if not isinstance(text, str) or not text.strip():
        return default
    try:
        return json.loads(text)
    except Exception:
        return default
