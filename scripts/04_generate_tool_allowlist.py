"""
Generate / audit config/allowed_tools.txt — the curated tool surface.

The allowlist is HAND-EDITED and authoritative: only tool_ids listed there are
ever loaded into the agent registry, offered to the tool-selection LLM, or
dispatched by the MCP server. The registry itself is machine-generated from the
PySisense SDK (01_build_registry_from_sdk.py), so a refresh can introduce SDK
methods that should never reach a user. Those stay invisible until someone adds
a line here.

Default mode is a DRIFT AUDIT — it never touches the file:

    python scripts/04_generate_tool_allowlist.py

      - tools in the registry but NOT in the allowlist  (new/unreviewed → hidden)
      - tools in the allowlist but NOT in the registry  (stale lines → dead)

To create the file the first time (refuses to clobber hand edits):

    python scripts/04_generate_tool_allowlist.py --init
    python scripts/04_generate_tool_allowlist.py --init --force   # overwrite anyway
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "tools.registry.with_examples.json"
ALLOWLIST = ROOT / "config" / "allowed_tools.txt"

DESC_WIDTH = 70

HEADER = """\
# Curated tool surface for fes-assistant — hand-edit this file.
#
# One tool_id per line. To REMOVE a tool from the agent + MCP surface, delete
# its line or comment it out with '#'. Tools not listed here are never exposed,
# so new SDK methods added by a registry refresh stay hidden until you add them.
#
# Enforced in three places, all reading this same file:
#   backend/agent/_registry.py   filters TOOL_REGISTRY + the planner's catalog
#   backend/agent/_routing.py    filters the tool menu the selection LLM sees
#   mcp_server/tools_core.py     filters TOOLS_BY_ID — dispatch-level enforcement
#
# Override the path with FES_TOOL_ALLOWLIST. If this file is MISSING, every
# registry tool is allowed and a warning is logged — deleting it disables the
# allowlist rather than hiding everything.
#
# [write] tools mutate state. They stay subject to the existing gates on top of
# this list: the UI approval dialog (the backend's pending_confirmation /
# approval-consumption flow) and the MCP server's PYSISENSE_ALLOW_MUTATIONS
# kill-switch.
#
# The migration module is included: migration mode is a first-class feature of
# this app (source + target deployments). Drop those lines for a single-tenant
# deployment.
#
# Regenerate/audit with: python scripts/04_generate_tool_allowlist.py
"""


def load_registry() -> List[Dict[str, Any]]:
    if not REGISTRY.exists():
        print(f"registry not found: {REGISTRY}")
        raise SystemExit(1)
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def parse_allowlist(path: Path) -> List[str]:
    """Read tool_ids, ignoring blank lines, full-line comments and trailing comments."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def one_liner(text: Any, width: int = DESC_WIDTH) -> str:
    first = (str(text or "").strip().splitlines() or [""])[0].strip()
    return first if len(first) <= width else first[: width - 3].rstrip() + "..."


def render(rows: List[Dict[str, Any]]) -> str:
    by_module: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_module[row.get("module") or "_unknown"].append(row)

    out = [HEADER]
    for module in sorted(by_module):
        tools = by_module[module]
        reads = sorted((t for t in tools if not t.get("mutates")), key=lambda t: t["tool_id"])
        writes = sorted((t for t in tools if t.get("mutates")), key=lambda t: t["tool_id"])
        out.append(f"\n# ===== {module} ({len(reads)} read / {len(writes)} write) =====")

        pad = max((len(t["tool_id"]) for t in tools), default=0)
        for t in reads:
            out.append(f"{t['tool_id']:<{pad}}  # {one_liner(t.get('description'))}")
        for t in writes:
            out.append(f"{t['tool_id']:<{pad}}  # [write] {one_liner(t.get('description'))}")

    return "\n".join(out) + "\n"


def main() -> int:
    rows = load_registry()
    registry_ids = {r["tool_id"] for r in rows if r.get("tool_id")}

    if "--init" in sys.argv:
        if ALLOWLIST.exists() and "--force" not in sys.argv:
            print(f"{ALLOWLIST} already exists — refusing to overwrite hand edits.")
            print("Run the audit (no flags) to see drift, or pass --force to regenerate.")
            return 1
        ALLOWLIST.write_text(render(rows), encoding="utf-8")
        print(f"wrote {ALLOWLIST} with {len(registry_ids)} tools")
        return 0

    if not ALLOWLIST.exists():
        print(f"{ALLOWLIST} does not exist — the allowlist is INACTIVE (all tools allowed).")
        print("Create it with: python scripts/04_generate_tool_allowlist.py --init")
        return 0

    listed = parse_allowlist(ALLOWLIST)
    listed_set = set(listed)

    unlisted = sorted(registry_ids - listed_set)
    stale = sorted(listed_set - registry_ids)
    dupes = sorted({t for t in listed if listed.count(t) > 1})

    print(f"allowlist: {len(listed_set)} tool(s) listed   registry: {len(registry_ids)} tool(s)\n")
    if unlisted:
        print(f"IN REGISTRY, NOT LISTED — currently hidden ({len(unlisted)}):")
        for t in unlisted:
            mutates = next((r.get("mutates") for r in rows if r["tool_id"] == t), False)
            print(f"  {'[write] ' if mutates else ''}{t}")
        print()
    if stale:
        print(f"LISTED, NOT IN REGISTRY — dead lines ({len(stale)}):")
        for t in stale:
            print(f"  {t}")
        print()
    if dupes:
        print(f"DUPLICATE LINES ({len(dupes)}):")
        for t in dupes:
            print(f"  {t}")
        print()
    if not (unlisted or stale or dupes):
        print("in sync — every registry tool is listed and every line resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
