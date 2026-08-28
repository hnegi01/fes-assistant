import inspect
import re
import typing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pysisense

from .registry_core import (
    MODULES,
    _parse_class_docstring,  # noqa: F401 — re-exported for test_registry_builder.py compat
    _write_json,
)

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
# Inline format marker → JSON Schema "format"
# ---------------------------------------------------------------------------

# A trailing "(format: <name>)" marker in a parameter description, e.g.
#   "Email address of the user to retrieve. (format: email)"
# The marker is translated into a JSON Schema `format` keyword and stripped
# from the visible description. The name is passed through verbatim (no
# allow-list) so the SDK docstring remains the single source of truth.
_FORMAT_MARKER_RE = re.compile(r"\s*\(format:\s*([a-zA-Z0-9_-]+)\s*\)\s*$")


def _split_format_marker(description: str) -> tuple:
    """Return (clean_description, format_or_None) for a parameter description."""
    if not description:
        return description, None
    m = _FORMAT_MARKER_RE.search(description)
    if not m:
        return description, None
    fmt = m.group(1).strip()
    clean = description[: m.start()].rstrip()
    return clean, fmt


# ---------------------------------------------------------------------------
# JSON schema builder from signature + docstring (generic only)
# ---------------------------------------------------------------------------


# Signature params that must never reach the registry: no caller can supply them.
# `emit` is the SDK's progress callback — the MCP server injects it for streaming
# tools and drops any client-supplied value. Leaving it in the schema only gives
# the tool-selection LLM a slot to invent a fake callback for.
_INTERNAL_PARAMS = {"self", "emit"}


def _schema_from_annotation(ann: Any) -> Optional[Dict[str, Any]]:
    """JSON Schema fragment from a resolved type annotation, or None when the
    annotation carries no usable information (Any, missing, exotic unions) —
    None sends the caller down the older default/docstring/heuristic chain.

    This is the consumer half of pysisense >= 1.1's introspection contracts
    (payload TypedDicts + Literal enums). On 1.0.x nothing here fires beyond
    plain scalars/containers, which resolve to the same types the old chain
    inferred — verified by a full registry rebuild producing a byte-identical
    surface.

    - TypedDict → object schema with per-field properties and `required` from
      __required_keys__ (typing_extensions-aware: the SDK's payloads may
      subclass typing_extensions.TypedDict, which typing.is_typeddict misses)
    - Literal[...] → enum (+ scalar type from the values)
    - Optional[X] / X | None → schema of X (a None default already marks the
      param optional; JSON Schema needs no null union for that)
    - list/tuple/set[X] → array with recursively-typed items
    - dict[...] → object; str/bool/int/float → their scalar types
    """
    import types
    import typing

    import typing_extensions

    if ann is None or ann is inspect._empty or ann is typing.Any:
        return None
    if typing_extensions.is_typeddict(ann):
        try:
            hints = typing_extensions.get_type_hints(ann)
        except Exception:  # noqa: BLE001 — unresolvable forward refs → no info
            return {"type": "object"}
        props = {k: (_schema_from_annotation(sub) or {"type": "string"}) for k, sub in hints.items()}
        req = sorted(getattr(ann, "__required_keys__", frozenset()))
        schema: Dict[str, Any] = {"type": "object", "properties": props}
        if req:
            schema["required"] = req
        return schema
    origin = typing.get_origin(ann)
    if origin is typing.Literal:
        vals = list(typing.get_args(ann))
        if all(isinstance(v, str) for v in vals):
            return {"type": "string", "enum": vals}
        if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            return {"type": "integer", "enum": vals}
        return None
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in typing.get_args(ann) if a is not type(None)]
        return _schema_from_annotation(non_none[0]) if len(non_none) == 1 else None
    if origin in (list, tuple, set) or ann in (list, tuple, set):
        args = typing.get_args(ann)
        items = _schema_from_annotation(args[0]) if args else None
        return {"type": "array", "items": items or {"type": "string"}}
    if origin is dict or ann is dict:
        return {"type": "object"}
    if ann is bool:
        return {"type": "boolean"}
    if ann is int:
        return {"type": "integer"}
    if ann is float:
        return {"type": "number"}
    if ann is str:
        return {"type": "string"}
    return None


def json_schema_from_signature(
    sig: inspect.Signature,
    doc: str,
    type_hints: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Build a JSON schema for a method based on:
    - Resolved type annotations, when the caller supplies them (TypedDict
      payloads, Literal enums — the pysisense >= 1.1 contracts); an
      annotation that yields concrete information wins outright
    - Python signature defaults (bool/int/list/dict → type inference)
    - Docstring param type hints + multi-line param descriptions
    - Generic name-based heuristics (e.g. *_ids → array)
    """
    properties: Dict[str, Dict[str, Any]] = {}
    required: List[str] = []

    doc_meta = _parse_param_doc_meta(doc)

    for name, p in sig.parameters.items():
        if name in _INTERNAL_PARAMS:
            continue

        meta = doc_meta.get(name)
        ann_piece = _schema_from_annotation((type_hints or {}).get(name))
        if ann_piece is not None:
            # The annotation is the SDK's own declaration — heuristics and
            # docstring TYPE hints never override it (descriptions still apply).
            schema_piece = dict(ann_piece)
        else:
            # Start with type from default
            schema_piece = _schema_type_from_default(p.default)

            # Refine from docstring if default did not give us anything useful
            if meta:
                doc_type = meta.get("type")
                if (p.default is inspect._empty or p.default is None) and doc_type:
                    hint_piece = _schema_type_from_doc_hint(doc_type)
                    schema_piece.update(hint_piece)

            # Apply generic name-based heuristics (no per-tool logic)
            schema_piece = _apply_name_heuristics(name, schema_piece)

        # Ensure arrays always have "items"
        if schema_piece.get("type") == "array" and "items" not in schema_piece:
            schema_piece["items"] = {"type": "string"}

        # Param-level description: prefer docstring text if available.
        # Extract any trailing "(format: <name>)" marker into a schema `format`.
        if meta and meta.get("description"):
            desc, fmt = _split_format_marker(meta["description"])
            schema_piece["description"] = desc
            if fmt:
                # For array params the format applies per element → put it on items.
                if schema_piece.get("type") == "array":
                    schema_piece.setdefault("items", {"type": "string"})["format"] = fmt
                else:
                    schema_piece["format"] = fmt
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
    # Create User → rich user_data schema. The SDK signature is `user_data: dict`,
    # so the generated schema can't see inside it: "create user himanshu negi"
    # passed validation with neither email nor role, gated, and failed in the SDK
    # ("Role 'None' not found", live 2026-08-27). Field list and requiredness
    # mirror the SDK docstring + code (pysisense/access_management/users.py::
    # create_user — role provably required, email required by the API).
    # The clarification path walks one level into object params
    # (_missing_required_fields), so missing inner fields clarify up front.
    "access_management.create_user": {
        "patch": {
            "parameters.properties.user_data": {
                "type": "object",
                "description": (
                    "The new user's details. Required: email and role. "
                    "Optional: firstName, lastName, groups, preferences."
                ),
                "properties": {
                    "email": {
                        "type": "string",
                        "format": "email",
                        "description": "The new user's email address",
                    },
                    "role": {
                        "type": "string",
                        "description": "Role name to assign (matched case-insensitively)",
                        "x-options-tool": "access_management.get_roles",
                    },
                    "firstName": {"type": "string", "description": "The user's first name"},
                    "lastName": {"type": "string", "description": "The user's last name"},
                    "groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Group names to assign the user to",
                    },
                    "preferences": {"type": "object", "description": "User preference settings"},
                },
                "required": ["email", "role"],
            },
        }
    },
    # Update User → rich user_data schema, NO inner required: the SDK docstring
    # says "Only include fields you want to change", so every inner field is
    # legitimately optional. The value of the patch is the model emitting
    # canonical field names instead of guessing.
    "access_management.update_user": {
        "patch": {
            "parameters.properties.user_data": {
                "type": "object",
                "description": "Fields to update — include only what should change.",
                "properties": {
                    "email": {
                        "type": "string",
                        "format": "email",
                        "description": "New email address for the user",
                    },
                    "userName": {"type": "string", "description": "New username/login name"},
                    "firstName": {"type": "string", "description": "New first name"},
                    "lastName": {"type": "string", "description": "New last name"},
                    "role": {
                        "type": "string",
                        "description": "New role name (matched case-insensitively)",
                        "x-options-tool": "access_management.get_roles",
                    },
                    "groups": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Group names replacing the user's group assignments",
                    },
                    "preferences": {"type": "object", "description": "User preference settings"},
                },
            },
        }
    },
    # Restore plugin snapshot → the docstring documents the shape explicitly:
    # "containing at minimum a 'plugins' key with a list of folderName values".
    "plugins.restore_snapshot": {
        "patch": {
            "parameters.properties.snapshot": {
                "type": "object",
                "description": "A snapshot as returned by save_snapshot.",
                "properties": {
                    "plugins": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "folderName values of the plugins that should be enabled; all others are disabled"
                        ),
                    },
                },
                "required": ["plugins"],
            },
        }
    },
    # Generate connection payload → datasource_type is a documented closed set;
    # connection_params fields are documented PER TYPE (different required sets
    # per provider), so the union is described without a flat inner `required` —
    # a wrong flat list would demand Athena keys from a Databricks request.
    "datamodel.generate_connections_payload": {
        "patch": {
            "parameters.properties.datasource_type.enum": ["Athena", "RedShift", "BigQuery", "DataBricks"],
            "parameters.properties.connection_params": {
                "type": "object",
                "description": (
                    "Connection details; which keys are required depends on datasource_type "
                    "(Athena: name, region, s3_output_location, aws_access_key, aws_secret_key. "
                    "DataBricks: name, connection_string, token. "
                    "BigQuery: name, service_account_key_path. "
                    "RedShift: server, username, password.)"
                ),
                "properties": {
                    "name": {"type": "string", "description": "Connection name"},
                    "description": {"type": "string", "description": "Connection description"},
                    "region": {"type": "string", "description": "AWS region (Athena)"},
                    "s3_output_location": {"type": "string", "description": "S3 output location (Athena)"},
                    "aws_access_key": {"type": "string", "description": "AWS access key (Athena)"},
                    "aws_secret_key": {"type": "string", "description": "AWS secret key (Athena)"},
                    "connection_string": {"type": "string", "description": "JDBC connection string (DataBricks)"},
                    "token": {"type": "string", "description": "Access token (DataBricks)"},
                    "service_account_key_path": {
                        "type": "string",
                        "description": "Service account key file path (BigQuery)",
                    },
                    "server": {"type": "string", "description": "Server host (RedShift)"},
                    "username": {"type": "string", "description": "Username (RedShift)"},
                    "password": {"type": "string", "description": "Password (RedShift)"},
                    "schema": {"type": "string", "description": "Schema name"},
                    "database": {"type": "string", "description": "Database name (BigQuery)"},
                },
            },
        }
    },
    # Create/update connection → canonical field names from the docstrings; no
    # inner `required` (create's docstring lists fields without marking any,
    # and update is PATCH semantics — only include what should change).
    "datamodel.create_connections": {
        "patch": {
            "parameters.properties.connection_payload": {
                "type": "object",
                "description": ("Connection configuration, typically produced by generate_connections_payload."),
                "properties": {
                    "provider": {"type": "string", "description": "Connector/provider name"},
                    "name": {"type": "string", "description": "Connection name"},
                    "description": {"type": "string", "description": "Connection description"},
                    "parameters": {"type": "object", "description": "Provider-specific connection parameters"},
                    "enabled": {"type": "boolean", "description": "Whether the connection is enabled"},
                    "supportedModelTypes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Model types the connection supports",
                    },
                },
            },
        }
    },
    "datamodel.update_connection": {
        "patch": {
            "parameters.properties.connection_data": {
                "type": "object",
                "description": "Fields to update — include only what should change.",
                "properties": {
                    "name": {"type": "string", "description": "New connection name"},
                    "provider": {"type": "string", "description": "Connector/provider name"},
                    "parameters": {"type": "object", "description": "Provider-specific connection parameters"},
                },
            },
        }
    },
    # Create notebook → documented example fields; requiredness not documented,
    # so none is invented.
    "custom_code.create_notebook": {
        "patch": {
            "parameters.properties.notebook_data": {
                "type": "object",
                "description": "Notebook creation payload.",
                "properties": {
                    "notebookType": {
                        "type": "string",
                        "description": "Notebook type (for example CustomCodeTransformation)",
                    },
                    "displayName": {"type": "string", "description": "Display name for the notebook"},
                },
            },
        }
    },
    # Saved formula measure → documented example fields, Sisense metadata format.
    "metadata.add_datasource_measure": {
        "patch": {
            "parameters.properties.measure": {
                "type": "object",
                "description": "Measure object in Sisense metadata format.",
                "properties": {
                    "datasource": {"type": "string", "description": "Datasource/datamodel the measure belongs to"},
                    "table": {"type": "string", "description": "Table the measure reads from"},
                    "column": {"type": "string", "description": "Column the measure reads from"},
                    "expression": {"type": "string", "description": "The measure's formula expression"},
                },
            },
        }
    },
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
    # Value lists mirror the SDK docstring (pysisense/datamodel/build.py::deploy_datamodel).
    "datamodel.deploy_datamodel": {
        "patch": {
            "parameters.properties.build_type.enum": ["full", "by_table", "schema_changes"],
            "parameters.properties.build_type.x-aliases": {
                "full": ["build", "rebuild", "start", "run", "execute", "refresh"],
                "by_table": ["by-table", "table-wise", "incremental-tables"],
                "schema_changes": ["schema-changes", "changes-only", "delta"],
            },
            "parameters.properties.build_type.description": (
                "Build strategy for extract models. Omit for live/publish."
            ),
            "parameters.properties.schema_origin.enum": ["latest", "running"],
            "parameters.properties.row_limit.type": "integer",
            "parameters.properties.row_limit.minimum": 1,
        }
    },
    # Create Dataset → option menu for the connection + deploy nudge
    "datamodel.create_dataset": {
        "patch": {
            "parameters.properties.connection_name.x-options-tool": "datamodel.get_connections",
            "parameters.properties.connection_name.x-options-note": (
                "Or let me know if you want to create a new connection first."
            ),
            "x-followup": {
                "note": "The change isn't queryable until the model is built or published.",
                "ask_template": "Deploy '{datamodel_name}'.",
            },
        }
    },
    # Create Table → deploy nudge (the SDK self-resolves dataset/connection from datamodel_name)
    "datamodel.create_table": {
        "patch": {
            "x-followup": {
                "note": "The change isn't queryable until the model is built or published.",
                "ask_template": "Deploy '{datamodel_name}'.",
            },
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
            # Option menu for the connection: the CLARIFICATION path (not the model)
            # reads x-options-tool, runs the listed READ tool, and renders the
            # results as choices in the code-built question. x-options-* keys are
            # stripped from every model-facing schema (strip_internal_params).
            "parameters.properties.connection_name.x-options-tool": "datamodel.get_connections",
            "parameters.properties.connection_name.x-options-note": (
                "Or let me know if you want to create a new connection first."
            ),
            # Deploy nudge: appended to the final reply IN CODE after a successful
            # run (never auto-executed — deploying is a mutation the user did not
            # ask for). ask_template is formatted from the executed call's args.
            "x-followup": {
                "note": "The model isn't queryable until it's built or published.",
                "ask_template": "Deploy '{datamodel_name}'.",
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

# Tool IDs excluded from the registry entirely.
# Add here when a method's output is incompatible with the app's rendering pipeline.
_EXCLUDED_TOOL_IDS: frozenset = frozenset(
    {
        "wellcheck.run_full_wellcheck",  # nested multi-section output; use individual checks instead
    }
)


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
            # PEP 702 deprecated aliases (pysisense keeps every renamed method
            # as a decorated wrapper for one minor version) — a rename must not
            # become two tools for one operation. The registry carries only the
            # live name; the allowlist's DEPRECATED section keeps the history.
            if getattr(func, "__deprecated__", None):
                continue

            doc = (inspect.getdoc(func) or "").strip()
            one_liner = (doc.splitlines()[0] if doc else "No description.").strip()
            sig = inspect.signature(func)

            tool_id = f"{module_name}.{name}"
            if tool_id in _EXCLUDED_TOOL_IDS:
                continue
            try:
                # Resolve string annotations (SDK modules use `from __future__
                # import annotations`) so TypedDict/Literal contracts are real
                # classes the schema builder can introspect.
                type_hints = typing.get_type_hints(func)
            except Exception:  # noqa: BLE001 — unresolvable hints → old chain
                type_hints = None
            schema = json_schema_from_signature(sig, doc, type_hints=type_hints)
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
    _write_json(out_file, registry)
    print(f"Flat registry: {len(registry)} tools → {out_file}")
    print("Run 02_add_llm_examples_to_registry to generate the hierarchical registry with examples.")


if __name__ == "__main__":
    main()
