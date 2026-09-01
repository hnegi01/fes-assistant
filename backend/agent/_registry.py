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


def _effective_ok(result: Any) -> bool:
    """A tool call is only as successful as the SDK says it is.

    The wrapper's `ok` means "the call executed" — transport truth. Some SDK
    methods additionally return their own verdict inside the payload
    (`ok`/`status`): a bulk migration that writes nothing raises no exception,
    it RETURNS a failure report. Found live 2026-08-14: migrate_all_users
    failed 66/66 ("username/email already exists"), the payload said
    `ok: false, status: failed`, and the run log printed "succeeded" because
    only the wrapper was consulted. When the payload explicitly carries a
    verdict, believe it; when it carries none, the wrapper's word stands.

    `success` is the same verdict under a different name. The SDK's reference
    resolvers report a miss as
    {"success": False, "status_code": 404, "datamodel_id": None, ..., "error": ...}.
    That dict carries real payload keys, so the MCP boundary's error-envelope
    matcher correctly declines it — which left it landing here as a success.
    Found 2026-08-29 while checking the 1.1.0 error contract, when both
    resolvers were exposed tools.

    They were delisted on 2026-08-31 (plumbing: nobody asks an assistant to
    resolve a reference, and neither was ever selected), so no exposed tool is
    known to return this shape today. The check stays anyway — it costs one
    comparison, `success` is a plain-English verdict any SDK method might adopt,
    and the failure mode it guards is a failed call reported as a success, which
    is the one we keep having to fix.
    """
    if not isinstance(result, dict) or not result.get("ok"):
        return False
    payload = result.get("result")
    if isinstance(payload, dict):
        if payload.get("ok") is False or payload.get("success") is False:
            return False
        status = payload.get("status")
        if isinstance(status, str) and status.strip().lower() == "failed":
            return False
        # Migration summaries come in TWO shapes and only one announces itself.
        # migrate_all_users reports `ok`/`status` + *_count and is caught above;
        # migrate_dashboards returns bare LISTS — {"succeeded": [], "skipped":
        # [], "failed": [{"title", "source_id", "reason"}]} — with no verdict
        # field at all. Live 2026-09-01: a dashboard migration parsed as
        # "succeeded=0, skipped=0, failed=1" and the user was told
        # "`migration.migrate_dashboards` succeeded."
        #
        # Nothing succeeded and something failed = a failed call. A PARTIAL run
        # stays ok=True and is named by _partial_outcome_note instead, matching
        # how partial outcomes are handled everywhere else.
        failed, succeeded = payload.get("failed"), payload.get("succeeded")
        if isinstance(failed, list) and failed and isinstance(succeeded, list) and not succeeded:
            return False
    return True


def _payload_failure_reason(payload: Any) -> str:
    """The SDK's own account of a payload-level failure, in its own words —
    never our interpretation. Empty string when the payload offers none."""
    if not isinstance(payload, dict):
        return ""
    # A payload that failed and says why, in one field — the resolver contract
    # ({"success": False, ..., "error": "..."}). Checked first: it is the SDK's
    # own sentence about this failure, which beats anything reconstructed from
    # counts below.
    own = payload.get("error")
    if isinstance(own, str) and own.strip():
        return own.strip()
    raw = payload.get("raw_error")
    if isinstance(raw, dict):
        err = raw.get("error")
        msg = (err or {}).get("message") if isinstance(err, dict) else raw.get("message")
        if msg:
            return str(msg).strip()
    # List-shaped migration summary: the reason lives on each failed item.
    failed_list = payload.get("failed")
    if isinstance(failed_list, list) and failed_list:
        reasons = []
        for item in failed_list[:3]:
            if isinstance(item, dict):
                what = item.get("title") or item.get("name") or item.get("source_id") or "one item"
                why = str(item.get("reason") or item.get("error") or "").strip()
                reasons.append(f"{what} ({why})" if why else str(what))
        more = f" and {len(failed_list) - 3} more" if len(failed_list) > 3 else ""
        if reasons:
            return f"{len(failed_list)} failed: {', '.join(reasons)}{more}"

    succeeded = payload.get("succeeded_count", payload.get("success_count"))
    failed = payload.get("failed_count")
    if failed is not None:
        total = payload.get("total_count")
        if succeeded is not None and total is not None:
            return f"{succeeded} of {total} succeeded, {failed} failed"
        return f"{failed} failed"
    status = payload.get("status")
    return str(status).strip() if status else ""


def _partial_outcome_note(payload: Any, limit: int = 3) -> str:
    """What a successful call did NOT do, in the SDK's own words. "" when all of it happened.

    pysisense 2.0 reports partial outcomes in-band rather than swallowing them:
    `errors` on get_unused_columns_bulk (references that would not resolve) and
    `skipped` on the share writers (parties Sisense refused or silently ignored,
    each with its own `reason`). Both ride on a SUCCESS return, so nothing else
    in this module would mention them.

    Named items only — the identifiers came from the user's own request, which
    is the same narrow justification the failure-reason exception rests on. No
    reason is inferred: if the SDK gave one we quote it, otherwise we name the
    item and stop.
    """
    if not isinstance(payload, dict):
        return ""

    parts: List[str] = []
    for key, verb in (
        ("errors", "could not be processed"),
        ("failed", "failed"),
        ("skipped", "was skipped"),
    ):
        items = payload.get(key)
        if not isinstance(items, list) or not items:
            continue
        labels: List[str] = []
        for item in items[:limit]:
            if isinstance(item, dict):
                name = item.get("ref") or item.get("name") or item.get("title") or item.get("id")
                why = str(item.get("error") or item.get("reason") or "").strip()
                labels.append(f"{name} ({why})" if name and why else str(name or why or "one item"))
            else:
                labels.append(str(item))
        more = f" and {len(items) - limit} more" if len(items) > limit else ""
        parts.append(f"{len(items)} {verb}: {', '.join(labels)}{more}")
    return "; ".join(parts)


def _failed_titles_sample(payload: Any, limit: int = 3) -> str:
    """Up to `limit` failed item titles from the SDK's own `failed` list, as a
    parenthetical (" (failures include: A, B, C, …)"). Empty string when the
    payload has no such list — never inferred from anything else."""
    if not isinstance(payload, dict):
        return ""
    failed = payload.get("failed")
    if not isinstance(failed, list):
        return ""
    titles = [str(f.get("title")) for f in failed if isinstance(f, dict) and f.get("title")]
    if not titles:
        return ""
    shown = ", ".join(titles[:limit])
    more = f", +{len(titles) - limit} more" if len(titles) > limit else ""
    return f" (failures include: {shown}{more})"


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

    # The call executed, but the SDK's own report says it failed (or partially
    # failed). Say so in the SDK's words — "succeeded" over a payload that
    # reads `ok: false` is the lie a user cannot detect without opening JSON.
    if not _effective_ok(result):
        reason = _payload_failure_reason(data)
        succeeded = data.get("succeeded_count", data.get("success_count")) if isinstance(data, dict) else None
        failed = data.get("failed_count") if isinstance(data, dict) else None
        # Counter-shaped summaries give a bare count, so the titles are the only
        # way to know WHICH ones failed. List-shaped summaries already name them
        # — with the SDK's own reason — inside `reason`, so appending the sample
        # there would say it twice. Decided from the payload, not by inspecting
        # the sentence we just built.
        named_already = isinstance(data, dict) and isinstance(data.get("failed"), list) and data["failed"]
        sample = "" if named_already else _failed_titles_sample(data)
        if succeeded and failed:
            return (
                f"`{tool_name}` completed with failures — {succeeded} succeeded, {failed} failed{sample}. "
                "Details shown above."
            )
        return (
            f"`{tool_name}` failed — {reason}{sample}. Details shown above."
            if reason
            else (f"`{tool_name}` failed{sample}. Details shown above.")
        )

    if isinstance(data, dict) and "error" in data:
        return f"`{tool_name}` returned an error — {data['error']}"

    if isinstance(data, list):
        n = len(data)
        if n == 0:
            return f"`{tool_name}` returned no results."
        noun = "result" if n == 1 else "results"
        return f"Found {n} {noun} from `{tool_name}`. Results shown above."

    # A pysisense 2.0 call can SUCCEED while part of what was asked for did not
    # happen: a typo'd model among valid ones (`errors`), or a share Sisense
    # silently dropped for an inactive user (`skipped`). The SDK reports both
    # in-band so they are not invisible — but this line is the whole reply with
    # summarization off, so a partial outcome that reads as plain success here
    # is exactly the wrong answer we spent three rounds getting fixed upstream.
    # The SDK's own per-item reason goes on screen; no interpretation added.
    if isinstance(data, dict):
        partial = _partial_outcome_note(data)
        if partial:
            rows = data.get("results")
            done = f"{len(rows)} returned" if isinstance(rows, list) else "the rest completed"
            return f"`{tool_name}` partly succeeded — {done}, but {partial}. Details shown above."

    if isinstance(data, dict):
        # A successful report that carries the SDK's own counters gets them in
        # the line — "295 of 295 migrated" is the deterministic summary a user
        # asked for after approving a bulk write; no LLM, no interpretation.
        succeeded = data.get("succeeded_count", data.get("success_count"))
        total = data.get("total_count")
        if succeeded is not None and total is not None:
            return f"`{tool_name}` succeeded — {succeeded} of {total} migrated. Details shown above."
        if succeeded is not None:
            return f"`{tool_name}` succeeded — {succeeded} migrated. Details shown above."
        # "Got a response" reads as uncertainty. `ok` was true, so say so —
        # this is the line a user sees after approving a create or an update
        # with summarization off, and it should confirm the thing happened.
        return f"`{tool_name}` succeeded. Details shown above."

    return f"`{tool_name}` completed. Result shown above."


# Size guards for summarization — NOT privacy guards. The privacy boundary is
# the summarization switch itself; with it off none of this data is sent at all.
#
# The TOTAL budget is the guard that should bite, because it truncates at the
# point it runs out and says so. The per-object key cap is a blunt instrument:
# it drops whichever fields fall last in dict order, per object, invisibly.
#
# Raised 2026-09-01 after a live eval caught it silently deleting data. The
# pysisense 2.0 canonical user row is 12 keys; the cap was 10, so `GROUPS` and
# `GROUP_IDS` were dropped from every summarized user record and the agent
# answered "no group information is provided in the user record" — honestly,
# because we had removed it. Dashboard records are 28 keys and had been losing
# 18 of them on every call since long before that.
#
# The two move together. Measured on real record shapes, 20 dashboard rows at
# full width need ~12.6k chars, so raising keys alone would have collapsed a
# 20-row list to a single row against the old 10k budget — trading a silent
# failure for a louder one.
MAX_LIST_ITEMS_FOR_LLM = 20
MAX_KEYS_PER_OBJECT_FOR_LLM = 30
MAX_DEPTH_FOR_LLM = 8
MAX_STRING_LENGTH_FOR_LLM = 300
MAX_TOTAL_LENGTH_FOR_LLM = 16_000
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
