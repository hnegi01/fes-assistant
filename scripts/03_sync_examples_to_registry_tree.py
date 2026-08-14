"""
Sync the `examples` field from the flat registry into the 3-level registry tree.

    config/tools.registry.with_examples.json   →   config/registry/{package}/{mixin}.json

`build_registry_hierarchical()` (01_build_registry_from_sdk.py) already copies
examples into the tree, but it imports the pysisense SDK to introspect classes.
When only the examples change — e.g. a curation pass over `user_query` text —
this script propagates them by tool_id without needing the SDK installed.

Nothing but `examples` is touched: descriptions, parameters and mutates stay as
the generator wrote them, so a later full regeneration is still authoritative.

    python scripts/03_sync_examples_to_registry_tree.py            # dry run
    python scripts/03_sync_examples_to_registry_tree.py --write
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAT = ROOT / "config" / "tools.registry.with_examples.json"
TREE = ROOT / "config" / "registry"


def main() -> int:
    write = "--write" in sys.argv

    if not FLAT.exists():
        print(f"flat registry not found: {FLAT}")
        return 1
    if not TREE.is_dir():
        print(f"registry tree not found: {TREE}")
        return 1

    flat = {r["tool_id"]: r for r in json.loads(FLAT.read_text(encoding="utf-8")) if r.get("tool_id")}

    files_changed = 0
    tools_updated = 0
    missing: list[str] = []

    for mixin_file in sorted(TREE.glob("*/*.json")):
        if mixin_file.name == "index.json":
            continue

        rows = json.loads(mixin_file.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue

        dirty = False
        for row in rows:
            tid = row.get("tool_id")
            src = flat.get(tid)
            if src is None:
                missing.append(tid or "<no tool_id>")
                continue
            new_examples = src.get("examples")
            if new_examples and row.get("examples") != new_examples:
                row["examples"] = new_examples
                tools_updated += 1
                dirty = True

        if dirty:
            files_changed += 1
            if write:
                # Match build_registry_hierarchical()'s _write_json exactly.
                with mixin_file.open("w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2, ensure_ascii=False)

    verb = "synced" if write else "would sync"
    print(f"{verb} examples for {tools_updated} tool(s) across {files_changed} mixin file(s)")
    if missing:
        print(f"\n{len(missing)} tool(s) in the tree are absent from the flat registry:")
        for tid in sorted(set(missing)):
            print(f"  {tid}")
    if not write and files_changed:
        print("\n(dry run — re-run with --write to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
