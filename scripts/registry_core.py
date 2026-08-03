"""
scripts/registry_core.py

Shared registry utilities imported by both builder scripts and tests.

What lives here:
  - _discover_facade_classes / MODULES  — auto-discover facade classes from pysisense.__all__
  - _parse_class_docstring              — parse Modules section from a facade class docstring
  - _write_json                         — thin JSON write helper
  - build_registry_hierarchical         — write config/registry/ 3-level tree
"""

from __future__ import annotations

import inspect
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pysisense

# ---------------------------------------------------------------------------
# Facade class discovery
# ---------------------------------------------------------------------------

_SKIP_CLASSES = {"SisenseClient"}


def _discover_facade_classes() -> Dict[str, Any]:
    """
    Auto-discover facade classes from pysisense.__all__.

    Maps subpackage name → facade class, e.g.:
      "access_management" → AccessManagement
      "datamodel"         → DataModel

    The subpackage name is derived from the class's __module__:
      "pysisense.access_management" → "access_management"

    New packages added to pysisense.__all__ are picked up automatically.
    """
    modules: Dict[str, Any] = {}
    for name in pysisense.__all__:
        obj = getattr(pysisense, name, None)
        if obj is None or not inspect.isclass(obj):
            continue
        if name in _SKIP_CLASSES:
            continue
        mod_path = getattr(obj, "__module__", "") or ""
        parts = mod_path.split(".")
        subpkg = parts[1] if len(parts) >= 2 and parts[0] == "pysisense" else name.lower()
        modules[subpkg] = obj
    return modules


MODULES: Dict[str, Any] = _discover_facade_classes()

# ---------------------------------------------------------------------------
# Class docstring parser
# ---------------------------------------------------------------------------

# Module header in a NumPy-style "Modules" section: "users :" or "users:"
_MODULE_HEADER_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*$")


def _parse_class_docstring(klass: type) -> Dict[str, Any]:
    """
    Extract the overall description and Modules map from a facade class docstring.

    Returns:
        {
            "description": "First paragraphs of the class docstring.",
            "modules": {"users": "User CRUD — ...", "groups": "Group membership — ..."},
        }

    The "Modules" section must be a NumPy-style block:
        Modules
        -------
        users :
            User CRUD — ...
        groups :
            Group membership — ...
    """
    doc = inspect.getdoc(klass) or ""
    if not doc:
        return {"description": "", "modules": {}}

    lines = doc.splitlines()

    # Description: every line before the "Modules" heading
    desc_lines: List[str] = []
    modules_start: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip() == "Modules":
            modules_start = i
            break
        desc_lines.append(line)
    description = "\n".join(desc_lines).strip()

    if modules_start is None:
        return {"description": description, "modules": {}}

    # Parse the NumPy Modules section
    modules: Dict[str, str] = {}
    cur_name: Optional[str] = None
    cur_desc: List[str] = []
    seen_sep = False

    def _flush() -> None:
        nonlocal cur_name, cur_desc
        if cur_name:
            modules[cur_name] = " ".join(cur_desc).strip()
            cur_name = None
            cur_desc = []

    for line in lines[modules_start + 1 :]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if not seen_sep:
            if stripped and set(stripped) <= {"-"}:
                seen_sep = True
            continue

        if not stripped:
            continue

        if indent == 0:
            m = _MODULE_HEADER_RE.match(line)
            if m:
                _flush()
                cur_name = m.group(1)
                cur_desc = []
            else:
                _flush()
                break
        elif cur_name and stripped:
            cur_desc.append(stripped)

    _flush()
    return {"description": description, "modules": modules}


# ---------------------------------------------------------------------------
# JSON write helper
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Hierarchical registry builder
# ---------------------------------------------------------------------------


def build_registry_hierarchical(
    registry: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> None:
    """
    Write a 3-level hierarchical registry under config/registry/:

        config/registry/
            index.json                   # all packages: name → {class, description}
            {package}/
                index.json               # modules map: name → one-liner description
                {module}.json            # tool schemas for that mixin (includes examples when present)

    The top-level index.json is what the LLM sees at Level 1 (pick a package).
    Each package/index.json is what the LLM sees at Level 2 (pick a module/mixin).
    Each package/{module}.json is what the LLM sees at Level 3 (pick and call a tool).

    If tools in the registry carry an "examples" field (populated by
    02_add_llm_examples_to_registry.py), examples are included in Level 3 files.
    With 3-level navigation the LLM only sees a small mixin's tools at once,
    so the per-tool token cost of examples is affordable.

    output_dir overrides the default config/registry/ path (used in tests).
    """
    if output_dir is None:
        root_dir = Path(__file__).resolve().parents[1]
        output_dir = root_dir / "config" / "registry"
    reg_dir = output_dir
    reg_dir.mkdir(parents=True, exist_ok=True)

    sdk_version = getattr(pysisense, "__version__", "unknown")
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # Parse class docstrings and group flat registry tools by (package, mixin stem)
    pkg_info: Dict[str, Any] = {}
    tools_by_pkg_mod: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for pkg_name, klass in MODULES.items():
        cls_doc = _parse_class_docstring(klass)
        pkg_info[pkg_name] = {
            "class": klass.__name__,
            "description": cls_doc["description"],
            "modules": cls_doc["modules"],
        }
        tools_by_pkg_mod[pkg_name] = {}

    for tool in registry:
        pkg = tool.get("module", "")
        sub = tool.get("sub_module", pkg)
        # "access_management.users" → "users"; "access_management" → "_base"
        mod_stem = sub.split(".", 1)[1] if "." in sub and sub != pkg else "_base"

        entry: Dict[str, Any] = {
            "tool_id": tool["tool_id"],
            "description": tool["description"],
            "parameters": tool["parameters"],
            "mutates": tool["mutates"],
        }
        # Include examples when available — affordable at Level 3 since only
        # one mixin's tools (~5-10) are loaded per planning call.
        if tool.get("examples"):
            entry["examples"] = tool["examples"]

        tools_by_pkg_mod.setdefault(pkg, {}).setdefault(mod_stem, []).append(entry)

    # Fill in missing modules descriptions from tool IDs when class has no docstring.
    # The LLM uses these one-liners at Level 2 to pick the right mixin.
    for pkg_name, info in pkg_info.items():
        if not info["modules"]:
            pkg_tools = tools_by_pkg_mod.get(pkg_name, {})
            if len(pkg_tools) > 1:  # only worth deriving when multiple mixins exist
                derived: Dict[str, str] = {}
                for stem, stem_tools in sorted(pkg_tools.items()):
                    method_names = [t["tool_id"].split(".", 1)[1] for t in stem_tools[:3]]
                    summary = ", ".join(method_names)
                    if len(stem_tools) > 3:
                        summary += f" (+{len(stem_tools) - 3} more)"
                    derived[stem] = summary
                info["modules"] = derived

    # Write top-level index.json (Level 1)
    _write_json(
        reg_dir / "index.json",
        {
            "sdk_version": sdk_version,
            "updated_at": now_iso,
            "packages": {
                pkg: {"class": info["class"], "description": info["description"]} for pkg, info in pkg_info.items()
            },
        },
    )

    # Write per-package files (Level 2 + Level 3)
    total_mod_files = 0
    for pkg_name, info in pkg_info.items():
        pkg_dir = reg_dir / pkg_name
        pkg_dir.mkdir(exist_ok=True)

        _write_json(
            pkg_dir / "index.json",
            {
                "package": pkg_name,
                "class": info["class"],
                "modules": info["modules"],
            },
        )

        for mod_stem, tools in tools_by_pkg_mod.get(pkg_name, {}).items():
            if tools:
                _write_json(pkg_dir / f"{mod_stem}.json", tools)
                total_mod_files += 1

    print(f"Hierarchical registry → {reg_dir}/")
    print(f"  {len(pkg_info)} packages, {total_mod_files} module files")
