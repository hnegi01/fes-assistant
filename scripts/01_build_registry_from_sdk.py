import inspect
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pysisense
from pysisense import *  # noqa: F401, F403 — brings in all facade classes defined in __all__

# Maps registry module key → facade class.
# Module key is what appears in tool_id ("access.get_all_users") and in the registry
# "module" field — keep existing keys stable so tool_ids don't change.
# To add a new module: add ONE line here.
MODULES: Dict[str, Any] = {
    "access": AccessManagement,  # noqa: F405, F821
    "blox": Blox,  # noqa: F405, F821
    "custom_code": CustomCode,  # noqa: F405, F821
    "dashboard": Dashboard,  # noqa: F405, F821
    "datamodel": DataModel,  # noqa: F405, F821
    "encryption": Encryption,  # noqa: F405, F821
    "folder": Folder,  # noqa: F405, F821
    "metadata": Metadata,  # noqa: F405, F821
    "migration": Migration,  # noqa: F405, F821
    "plugins": Plugins,  # noqa: F405, F821
    "queries": Queries,  # noqa: F405, F821
    "wellcheck": WellCheck,  # noqa: F405, F821
}

# ---------------------------------------------------------------------------
# Helper: parse parameter meta from docstring (Google-style + NumPy-style)
# ---------------------------------------------------------------------------

# Google-style param lines like:
#   action (str, optional): Determines how to handle...
_GOOGLE_PARAM_LINE_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]+)\)\s*:\s*(.*)$",
    re.MULTILINE,
)

# NumPy-style param header lines like:
#   action : str, optional
_NUMPY_PARAM_LINE_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^$]+)$")


def _parse_param_doc_meta_google(doc: str) -> Dict[str, Dict[str, str]]:
    """
    Parse a Google-style Parameters section, e.g.:

        Parameters:
            action (str, optional): Determines how to handle...
                Wrapped line continues here.
            folder_name (str): The target folder whose ownership needs to be
                changed.

    Returns:
        {
          "action": {"type": "str", "description": "Determines how to handle... Wrapped line continues here."},
          "folder_name": {"type": "str", "description": "The target folder whose ownership needs to be changed."},
          ...
        }
    """
    if not doc:
        return {}

    lines = doc.splitlines()
    meta: Dict[str, Dict[str, str]] = {}

    in_params = False
    current_name = None
    current_type = None
    current_desc_parts: List[str] = []

    def flush_current() -> None:
        nonlocal current_name, current_type, current_desc_parts
        if current_name:
            desc = " ".join(current_desc_parts).strip()
            meta[current_name] = {
                "type": current_type or "",
                "description": desc,
            }
        current_name = None
        current_type = None
        current_desc_parts = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Detect start of Parameters: block
        if not in_params:
            if stripped.lower().startswith("parameters:"):
                in_params = True
            continue

        # Detect end of Parameters: section when another top-level section starts
        if stripped.endswith(":") and stripped.split(":", 1)[0] in {
            "Returns",
            "Return",
            "Raises",
            "Notes",
            "Examples",
        }:
            flush_current()
            break

        # Try to match a new parameter header line
        m = _GOOGLE_PARAM_LINE_RE.match(line)
        if m:
            # Flush previous param if any
            flush_current()

            current_name = m.group(1)
            type_part = m.group(2)
            current_type = type_part.split(",")[0].strip()
            first_desc = m.group(3).strip()
            current_desc_parts = [first_desc] if first_desc else []
            continue

        # Otherwise, if we are inside a param and this is an indented non-empty line,
        # treat it as a continuation of the description.
        if current_name and stripped:
            # Any indented non-empty line after the header is considered continuation
            if line.startswith(" ") or line.startswith("\t"):
                current_desc_parts.append(stripped)

    # Flush last param at end of doc
    flush_current()

    return meta


def _parse_param_doc_meta_numpy(doc: str) -> Dict[str, Dict[str, str]]:
    """
    Parse a NumPy-style Parameters section, e.g.:

        Parameters
        ----------
        action : str, optional
            Determines how to handle...
            Wrapped line continues here.
        folder_name : str
            The target folder whose ownership needs to be changed.

    Returns the same structure as the Google-style parser.
    """
    if not doc:
        return {}

    lines = doc.splitlines()
    meta: Dict[str, Dict[str, str]] = {}

    in_params = False
    seen_separator = False
    current_name = None
    current_type = None
    current_desc_parts: List[str] = []

    def flush_current() -> None:
        nonlocal current_name, current_type, current_desc_parts
        if current_name:
            desc = " ".join(current_desc_parts).strip()
            meta[current_name] = {
                "type": current_type or "",
                "description": desc,
            }
        current_name = None
        current_type = None
        current_desc_parts = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if not in_params:
            # Look for a NumPy-style "Parameters" heading (no colon)
            if stripped == "Parameters":
                in_params = True
                seen_separator = False
            continue

        # After "Parameters", NumPy usually has a "------" separator; skip it
        if in_params and not seen_separator:
            if set(stripped) == {"-"} and stripped:
                seen_separator = True
            # Skip until we see the separator; do not parse headers yet
            continue

        # End of Parameters section: another top-level section heading with no indent
        if indent == 0 and stripped in {
            "Returns",
            "Yields",
            "Raises",
            "Notes",
            "Examples",
        }:
            flush_current()
            break

        # Try to match a new NumPy-style parameter header line
        m = _NUMPY_PARAM_LINE_RE.match(line)
        if m and indent == 0:
            flush_current()
            current_name = m.group(1)
            type_text = m.group(2)
            current_type = type_text.split(",")[0].strip()
            current_desc_parts = []
            continue

        # If we are inside a param and see an indented non-empty line, treat as description continuation
        if current_name and stripped and indent >= 4:
            current_desc_parts.append(stripped)

    flush_current()
    return meta


def _parse_param_doc_meta(doc: str) -> Dict[str, Dict[str, str]]:
    """
    Combine Google-style and NumPy-style parameter metadata.

    If both styles define the same param, NumPy-style wins (on the assumption
    that newer methods may use NumPy style).
    """
    google_meta = _parse_param_doc_meta_google(doc)
    numpy_meta = _parse_param_doc_meta_numpy(doc)

    combined = dict(google_meta)
    for name, info in numpy_meta.items():
        combined[name] = info

    return combined


# ---------------------------------------------------------------------------
# Type inference helpers
# ---------------------------------------------------------------------------


def _schema_type_from_default(default: Any) -> Dict[str, Any]:
    """
    Infer a JSON Schema fragment from a Python default value.
    Returns dict with at least {"type": "<...>"} and possibly "items".
    """
    if default is inspect._empty or default is None:
        # Unknown from default alone
        return {"type": "string"}

    if isinstance(default, bool):
        return {"type": "boolean"}

    # bool is also int in Python, so check bool first
    if isinstance(default, int):
        return {"type": "integer"}

    if isinstance(default, float):
        return {"type": "number"}

    if isinstance(default, (list, tuple)):
        # Assume list of strings by default
        return {"type": "array", "items": {"type": "string"}}

    if isinstance(default, dict):
        return {"type": "object"}

    return {"type": "string"}


def _schema_type_from_doc_hint(doc_hint: str) -> Dict[str, Any]:
    """
    Map a docstring type token (e.g. 'list', 'dict', 'bool') to a JSON Schema fragment.
    """
    if not doc_hint:
        return {"type": "string"}

    t = doc_hint.strip().lower()

    # Collections
    if "list" in t or "tuple" in t or "sequence" in t:
        return {"type": "array", "items": {"type": "string"}}

    if "dict" in t or "mapping" in t:
        return {"type": "object"}

    # Scalars
    if t.startswith("bool") or "boolean" in t:
        return {"type": "boolean"}

    if t.startswith("int") or t.startswith("integer"):
        return {"type": "integer"}

    if t.startswith("float") or t.startswith("double"):
        return {"type": "number"}

    if t.startswith("str") or t.startswith("string"):
        return {"type": "string"}

    # Default
    return {"type": "string"}


def _apply_name_heuristics(param_name: str, schema_piece: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic, scalable heuristics based on parameter name.
    No per-tool hardcoding.

    Examples:
    - *_ids, *_names, *_list → arrays of strings
    - provider_connection_map → object, etc. (generic pattern names)
    """
    name_l = param_name.lower()
    t = schema_piece.get("type")

    # Heuristic: names that look like collections → arrays of strings
    if t == "string":
        if (
            name_l.endswith("_ids")
            or name_l.endswith("_id_list")
            or name_l.endswith("_names")
            or name_l.endswith("_name_list")
            or name_l.endswith("_list")
        ):
            schema_piece["type"] = "array"
            schema_piece["items"] = {"type": "string"}

    # Generic mapping-style names → objects (if still string)
    if t == "string" and name_l.endswith("_map"):
        schema_piece["type"] = "object"

    return schema_piece


# ---------------------------------------------------------------------------
# JSON schema builder from signature + docstring (generic only)
# ---------------------------------------------------------------------------


def json_schema_from_signature(
    sig: inspect.Signature,
    doc: str,
) -> dict:
    """
    Build a JSON schema for a method based on:
    - Python signature defaults (bool/int/list/dict → type inference)
    - Docstring param type hints + multi-line param descriptions
    - Generic name-based heuristics (e.g. *_ids → array)
    """
    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    doc_meta = _parse_param_doc_meta(doc)

    for name, p in sig.parameters.items():
        if name == "self":
            continue

        # Start with type from default
        schema_piece = _schema_type_from_default(p.default)

        # Refine from docstring if default did not give us anything useful
        meta = doc_meta.get(name)
        if meta:
            doc_type = meta.get("type")
            if (p.default is inspect._empty or p.default is None) and doc_type:
                hint_piece = _schema_type_from_doc_hint(doc_type)
                schema_piece.update(hint_piece)

        # Ensure arrays always have "items"
        if schema_piece.get("type") == "array" and "items" not in schema_piece:
            schema_piece["items"] = {"type": "string"}

        # Apply generic name-based heuristics (no per-tool logic)
        schema_piece = _apply_name_heuristics(name, schema_piece)

        # Param-level description: prefer docstring text if available
        if meta and meta.get("description"):
            schema_piece["description"] = meta["description"]
        elif "description" not in schema_piece:
            schema_piece["description"] = f"{name} parameter"

        properties[name] = schema_piece

        # Required if no default at all
        if p.default is inspect._empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# ---------------------------------------------------------------------------
# Mutate detection + tags
# ---------------------------------------------------------------------------

# Allow inflected forms and names like "delete_user", "Deletes", "Deleting"
_MUTATE_PAT = re.compile(
    r"\b(create|update|delete|remove|assign|set|share|build|schedule|migrate|post|patch|put)\w*\b",
    re.IGNORECASE,
)

_READ_PREFIXES = (
    "get_",
    "list_",
    "fetch_",
    "find_",
    "count_",
    "preview_",
    "show_",
    "check_",
    "describe_",
)

_MUTATE_PREFIXES = (
    "create_",
    "update_",
    "delete_",
    "remove_",
    "assign_",
    "set_",
    "share_",
    "build_",
    "schedule_",
    "migrate_",
    "post_",
    "patch_",
    "put_",
    "add_",
    "upload_",
)


def is_mutating(name: str, doc: str) -> bool:
    """
    Heuristic for whether a method mutates server state.

    - Methods starting with read-style prefixes (get/list/find/preview/show/check)
      are treated as non-mutating.
    - Methods starting with mutation-style prefixes (create/update/delete/...)
      are treated as mutating.
    - Otherwise, we fall back to a regex search over the name + docstring.
    """
    lowered = name.lower()

    # 1) Read-style prefixes → non-mutating
    if lowered.startswith(_READ_PREFIXES):
        return False

    # 2) Mutation-style prefixes → mutating
    if lowered.startswith(_MUTATE_PREFIXES):
        return True

    # 3) Fallback: look for mutate verbs in name + doc (including inflected forms)
    text = f"{name} {doc or ''}"
    return bool(_MUTATE_PAT.search(text))


def infer_tags(module: str, method: str, mutates: bool) -> list:
    tags = [module]
    m = method.lower()

    if "user" in m:
        tags.append("users")
    if "group" in m:
        tags.append("groups")
    if "dashboard" in m:
        tags.append("dashboards")
    if "model" in m or "datamodel" in m:
        tags.append("datamodel")

    tags.append("write" if mutates else "read")

    # de-duplicate, keep order
    out, seen = [], set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out[:6]


# ---------------------------------------------------------------------------
# SCHEMA_RULES + apply_schema_rules (shared semantics) hardcoded patches
# ---------------------------------------------------------------------------

SCHEMA_RULES: Dict[str, Dict[str, Any]] = {
    # Create DataModel → constrain datamodel_type
    "datamodel.create_datamodel": {
        "patch": {
            "parameters.properties.datamodel_type.enum": ["extract", "live"],
            "parameters.properties.datamodel_type.x-aliases": {
                "extract": ["ec", "elasticube", "elastic cube", "cube", "elastic-cube"],
                "live": ["realtime", "real-time", "live model"],
            },
            "parameters.properties.datamodel_type.description": (
                "Either 'extract' (Elasticube/EC) or 'live'. If user says 'elasticube' or 'ec', normalize to 'extract'."
            ),
        }
    },
    # Deploy/Build DataModel → constrain build_type / schema_origin / row_limit type
    "datamodel.deploy_datamodel": {
        "patch": {
            "parameters.properties.build_type.enum": ["full", "by_table"],
            "parameters.properties.build_type.x-aliases": {
                "full": ["build", "rebuild", "start", "run", "execute", "refresh"],
                "by_table": ["by-table", "table-wise", "incremental-tables"],
            },
            "parameters.properties.build_type.description": (
                "Build strategy for extract models. Omit for live/publish."
            ),
            "parameters.properties.schema_origin.enum": ["latest", "schema_changes"],
            "parameters.properties.row_limit.type": "integer",
            "parameters.properties.row_limit.minimum": 1,
        }
    },
    # Setup DataModel – enums + rich tables schema
    "datamodel.setup_datamodel": {
        "patch": {
            "parameters.properties.datamodel_type.enum": ["extract", "live"],
            "parameters.properties.datamodel_type.x-aliases": {
                "extract": ["ec", "elasticube", "elastic cube", "cube", "elastic-cube"],
                "live": ["realtime", "real-time", "live model"],
            },
            # Override the auto-generated `tables` schema with a rich object definition
            "parameters.properties.tables": {
                "type": "array",
                "description": (
                    "List of tables to add. For 'live' models, build_behavior_config is ignored. "
                    "For 'extract' models, set build_behavior_config as needed."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "database_name": {
                            "type": "string",
                            "description": (
                                "Optional override of the table's database. Defaults to top-level "
                                "database_name if omitted."
                            ),
                        },
                        "schema_name": {
                            "type": "string",
                            "description": (
                                "Optional override of the table's schema. Defaults to top-level schema_name if omitted."
                            ),
                        },
                        "table_name": {
                            "type": "string",
                            "description": ("Physical table name to add, or a logical name when using import_query."),
                        },
                        "import_query": {
                            "type": "string",
                            "description": (
                                "Optional custom SQL (executed as-is). Use fully-qualified tables: schema.table "
                                "(Databricks: `schema`.`table`)."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional table description.",
                        },
                        "tags": {
                            "type": "array",
                            "description": "Optional list of tags for the table.",
                            "items": {"type": "string"},
                        },
                        "build_behavior_config": {
                            "type": "object",
                            "description": (
                                "Extract models only; omit for 'live'. For 'increment' mode, column_name is required."
                            ),
                            "properties": {
                                "mode": {
                                    "type": "string",
                                    "enum": ["replace", "replace_changes", "append", "increment"],
                                    "description": "Table build behavior for extract models.",
                                },
                                "column_name": {
                                    "type": "string",
                                    "description": ("Required when mode='increment'; ignored otherwise."),
                                },
                            },
                        },
                    },
                    "required": ["table_name"],
                },
                "minItems": 1,
            },
        }
    },
    # Migration – all dashboards
    "migration.migrate_all_dashboards": {
        "patch": {
            "parameters.properties.action.enum": ["skip", "overwrite", "duplicate"],
        }
    },
    # Migration – all datamodels
    "migration.migrate_all_datamodels": {
        "patch": {
            "parameters.properties.dependencies.items.enum": [
                "dataSecurity",
                "formulas",
                "hierarchies",
                "perspectives",
            ],
            "parameters.properties.action.enum": ["overwrite", "duplicate"],
        }
    },
    # Migration – single dashboard
    "migration.migrate_dashboards": {
        "patch": {
            "parameters.properties.action.enum": ["skip", "overwrite", "duplicate"],
        }
    },
    # Migration – single datamodel
    "migration.migrate_datamodels": {
        "patch": {
            # dependencies: specific dependency types
            "parameters.properties.dependencies.items.enum": [
                "dataSecurity",
                "formulas",
                "hierarchies",
                "perspectives",
            ],
            # action: overwrite vs duplicate
            "parameters.properties.action.enum": ["overwrite", "duplicate"],
        }
    },
}


def _walk_and_set(d: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """
    Create/overwrite a nested key in dict given a dotted path (e.g.,
    'parameters.properties.row_limit.minimum').
    """
    parts = dotted_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_schema_rules(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutates the tool dict in-place to inject enums/aliases/type hints
    based on SCHEMA_RULES. If a path does not exist, it is created.
    """
    tool_id = tool.get("tool_id", "")
    rules = SCHEMA_RULES.get(tool_id)
    if not rules:
        return tool

    params = tool.get("parameters")
    if not isinstance(params, dict):
        # Ensure parameters object exists for patching
        params = {"type": "object", "properties": {}, "required": []}
        tool["parameters"] = params

    for dotted, val in rules.get("patch", {}).items():
        _walk_and_set(tool, dotted, val)

    return tool


# ---------------------------------------------------------------------------
# Sub-module helpers (for two-stage hierarchical routing)
# ---------------------------------------------------------------------------


def _get_defining_mixin(klass: type, method_name: str) -> type:
    """
    Walk the MRO to find the first class (other than the facade and object)
    that actually defines method_name in its own __dict__.
    Falls back to klass if nothing else owns it.
    """
    for cls in klass.__mro__:
        if cls is object or cls is klass:
            continue
        if method_name in cls.__dict__:
            return cls
    return klass


def _mixin_to_sub_module(module_key: str, mixin_class: type) -> str:
    """
    Derive a sub_module string from the mixin's source file name.
    e.g. UsersMixin defined in users.py → "access.users"
         DataModelCoreMixin defined in core.py → "datamodel.core"
    Falls back to module_key when the file cannot be determined.
    """
    try:
        filepath = inspect.getfile(mixin_class)
        stem = Path(filepath).stem  # filename without .py extension
        if stem == "__init__":
            return module_key  # method defined on the facade class itself
        return f"{module_key}.{stem}"
    except (TypeError, OSError):
        return module_key


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_registry() -> list:
    sdk_version = getattr(pysisense, "__version__", "unknown")
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    registry: List[Dict[str, Any]] = []

    for module_name, klass in MODULES.items():
        klass_name = klass.__name__

        # Introspect class methods (instance methods appear as functions here)
        for name, func in inspect.getmembers(klass, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue

            doc = (inspect.getdoc(func) or "").strip()
            one_liner = (doc.splitlines()[0] if doc else "No description.").strip()
            sig = inspect.signature(func)

            tool_id = f"{module_name}.{name}"
            schema = json_schema_from_signature(sig, doc)
            mutates = is_mutating(name, doc)
            tags = infer_tags(module_name, name, mutates)
            defining_mixin = _get_defining_mixin(klass, name)
            sub_module = _mixin_to_sub_module(module_name, defining_mixin)

            tool: Dict[str, Any] = {
                "tool_id": tool_id,
                "module": module_name,
                "sub_module": sub_module,
                "class": klass_name,
                "method": name,
                "description": one_liner,
                "full_doc": doc,  # Keep full docstring for downstream use
                "parameters": schema,
                "mutates": mutates,
                "tags": tags,
                "sdk_version": sdk_version,
                "updated_at": now_iso,
                # placeholder for later enrichment by generate_example.py
                "examples": [],
            }

            # Apply schema-level overrides (enums, aliases, extra descriptions)
            tool = apply_schema_rules(tool)

            registry.append(tool)

    return registry


def main() -> None:
    registry = build_registry()

    root_dir = Path(__file__).resolve().parents[1]
    config_dir = root_dir / "config"
    config_dir.mkdir(exist_ok=True)

    out_file = config_dir / "tools.registry.json"

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Built registry for {len(registry)} tools → {out_file}")
    print("Sample:")
    print(json.dumps(registry[:3], indent=2))


if __name__ == "__main__":
    main()
