"""
The two pieces of rename tooling added 2026-08-28, after the PySisense SDK
announced method renames (get_connections -> get_connections_all etc.):

  - scripts/04 --apply: dead allowlist lines are DELETED (they can never expose
    anything), new registry tools are STAGED as commented lines (exposing stays
    a human decision — uncommenting — but never a hand-copied id).
  - scripts/02 port_renamed_examples: curated examples follow a rename instead
    of silently regenerating as uncurated text, matched only when unambiguous.
"""

import importlib
import inspect
import typing

import pytest
import typing_extensions

s04 = importlib.import_module("scripts.04_generate_tool_allowlist")
s02 = importlib.import_module("scripts.02_add_llm_examples_to_registry")


# ---------------------------------------------------------------------------
# scripts/04 --apply
# ---------------------------------------------------------------------------

ROWS = [
    {
        "tool_id": "datamodel.get_connections_all",
        "module": "datamodel",
        "mutates": False,
        "description": "Retrieve all connections.",
    },
    {
        "tool_id": "access_management.get_users_all",
        "module": "access_management",
        "mutates": False,
        "description": "Retrieve all users.",
    },
    {
        "tool_id": "datamodel.delete_datamodel",
        "module": "datamodel",
        "mutates": True,
        "description": "Delete a data model.",
    },
]
IDS = {r["tool_id"] for r in ROWS}


@pytest.fixture()
def allowlist(tmp_path, monkeypatch):
    path = tmp_path / "allowed_tools.txt"
    monkeypatch.setattr(s04, "ALLOWLIST", path)
    return path


def _apply(path):
    return s04.apply_reconcile(ROWS, IDS)


def test_apply_moves_dead_lines_to_deprecated_with_version(allowlist):
    allowlist.write_text(
        "# ===== datamodel =====\n"
        "datamodel.get_connections      # Retrieve all connections.\n"  # renamed away — dead
        "access_management.get_users_all  # Retrieve all users.\n"  # alive
        "# datamodel.gone_tool  # staged long ago, since deleted\n"  # dead staged line
        "# prose comment that mentions nothing tool-shaped\n"
    )
    _apply(allowlist)
    text = allowlist.read_text()
    dep_at = text.index(s04.DEPRECATED_HEADER)
    # dead lines live ONLY below the deprecated header, commented, with their
    # original trailing comments preserved and a version batch marker above.
    assert "# datamodel.get_connections " in text[dep_at:]
    assert "Retrieve all connections." in text[dep_at:], "original comment kept as history"
    assert "# datamodel.gone_tool" in text[dep_at:]
    assert "removed in pysisense" in text[dep_at:]
    assert "get_connections " not in text[:dep_at], "no dead line remains in the live body"
    assert "access_management.get_users_all" in text[:dep_at]
    assert "prose comment" in text[:dep_at], "prose comments are never touched"
    # nothing in the deprecated section is ever active
    active = {ln.split("#")[0].strip() for ln in text.splitlines() if ln.split("#")[0].strip()}
    assert active == {"access_management.get_users_all"}


def test_deprecated_section_survives_reruns_without_duplication(allowlist):
    allowlist.write_text(
        "access_management.get_users_all  # Retrieve all users.\n"
        "datamodel.get_connections  # Retrieve all connections.\n"
    )
    _apply(allowlist)
    _apply(allowlist)
    text = allowlist.read_text()
    assert text.count("datamodel.get_connections ") == 1, "moved once, never re-moved or duplicated"


def test_apply_stages_new_tools_commented_with_write_tag(allowlist):
    allowlist.write_text("access_management.get_users_all  # Retrieve all users.\n")
    _apply(allowlist)
    text = allowlist.read_text()
    assert s04.STAGING_HEADER in text
    assert "new in pysisense" in text
    staged = [ln for ln in text.splitlines() if ln.startswith("# datamodel.")]
    assert any("get_connections_all" in ln for ln in staged), "new read tool staged commented"
    assert any("delete_datamodel" in ln and "[write]" in ln for ln in staged), "write tools carry the tag"
    active = {ln.split("#")[0].strip() for ln in text.splitlines() if ln.split("#")[0].strip()}
    assert active == {"access_management.get_users_all"}, "staging must not expose anything"


def test_apply_is_idempotent(allowlist):
    allowlist.write_text("access_management.get_users_all  # Retrieve all users.\n")
    _apply(allowlist)
    once = allowlist.read_text()
    _apply(allowlist)
    assert allowlist.read_text() == once, "second --apply must change nothing"


def test_apply_leaves_deliberately_hidden_commented_tools_alone(allowlist):
    # create_datamodel pattern: a commented line with a delist rationale must
    # neither be deleted (its id exists in the registry) nor re-staged.
    allowlist.write_text(
        "access_management.get_users_all  # Retrieve all users.\n"
        "# datamodel.delete_datamodel — delisted: reason recorded here\n"
        "# datamodel.get_connections_all  # staged earlier\n"
    )
    _apply(allowlist)
    text = allowlist.read_text()
    assert text.count("delete_datamodel") == 1
    assert text.count("get_connections_all") == 1


# ---------------------------------------------------------------------------
# scripts/02 port_renamed_examples
# ---------------------------------------------------------------------------


def _base(tool_id, module="datamodel", props=(), required=()):
    return {
        "tool_id": tool_id,
        "module": module,
        "parameters": {
            "type": "object",
            "properties": {p: {"type": "string"} for p in props},
            "required": list(required),
        },
    }


def test_rename_ports_curated_examples():
    base = [_base("datamodel.get_connections_all")]
    existing = {
        "datamodel.get_connections": {
            **_base("datamodel.get_connections"),
            "examples": [{"user_query": "Show all connections."}],
        }
    }
    s02.port_renamed_examples(base, existing)
    assert existing["datamodel.get_connections_all"]["examples"][0]["user_query"] == "Show all connections."


def test_dissimilar_names_never_port_even_with_identical_empty_schemas():
    # Many no-arg read tools share an empty schema — shape alone must not match.
    base = [_base("access_management.get_groups_all", module="access_management")]
    existing = {
        "access_management.get_folders_index": {
            **_base("access_management.get_folders_index", module="access_management"),
            "examples": [{"user_query": "x"}],
        }
    }
    s02.port_renamed_examples(base, existing)
    assert "access_management.get_groups_all" not in existing


def test_ambiguous_candidates_never_port():
    base = [_base("datamodel.get_connection_all")]
    existing = {
        "datamodel.get_connection": {**_base("datamodel.get_connection"), "examples": [{"user_query": "a"}]},
        "datamodel.get_connections": {**_base("datamodel.get_connections"), "examples": [{"user_query": "b"}]},
    }
    s02.port_renamed_examples(base, existing)
    assert "datamodel.get_connection_all" not in existing, "two plausible sources = no port"


def test_schema_change_blocks_the_port():
    # Same-ish name but the new tool takes different params: not a pure rename.
    base = [_base("datamodel.get_connections_all", props=("provider",), required=("provider",))]
    existing = {
        "datamodel.get_connections": {
            **_base("datamodel.get_connections"),
            "examples": [{"user_query": "x"}],
        }
    }
    s02.port_renamed_examples(base, existing)
    assert "datamodel.get_connections_all" not in existing


# ---------------------------------------------------------------------------
# scripts/01 annotation-aware schema generation (pysisense >= 1.1 contracts)
# ---------------------------------------------------------------------------

s01 = importlib.import_module("scripts.01_build_registry_from_sdk")


class _ReqHalf(typing_extensions.TypedDict):
    email: str
    role: typing_extensions.Literal["viewer", "admin"]


class _FakePayload(_ReqHalf, total=False):
    firstName: str
    groups: typing.List[str]


def _fake_method(
    self,
    user_data: _FakePayload,
    mode: typing.Literal["extract", "live"] = "extract",
    names: typing.Optional[typing.List[str]] = None,
    count: int = 0,
):
    """Do a fake thing.

    Parameters
    ----------
    user_data : dict
        The payload.
    """


def test_typeddict_annotation_becomes_nested_object_schema():
    hints = typing.get_type_hints(_fake_method)
    schema = s01.json_schema_from_signature(
        inspect.signature(_fake_method), inspect.getdoc(_fake_method) or "", type_hints=hints
    )
    ud = schema["properties"]["user_data"]
    assert ud["type"] == "object"
    assert ud["required"] == ["email", "role"]
    assert sorted(ud["properties"]) == ["email", "firstName", "groups", "role"]
    assert ud["properties"]["groups"] == {"type": "array", "items": {"type": "string"}}
    # nested Literal inside the TypedDict becomes an enum too
    assert ud["properties"]["role"]["enum"] == ["viewer", "admin"]
    # the docstring description still rides along; the doc TYPE hint ('dict')
    # does not fight the annotation
    assert ud["description"] == "The payload."


def test_literal_optional_and_scalars_resolve_from_annotations():
    hints = typing.get_type_hints(_fake_method)
    schema = s01.json_schema_from_signature(inspect.signature(_fake_method), "", type_hints=hints)
    assert schema["properties"]["mode"] == {
        "type": "string",
        "enum": ["extract", "live"],
        "description": "mode parameter",
    }
    assert schema["properties"]["names"]["type"] == "array"  # Optional[List[str]] unwraps
    assert schema["properties"]["count"]["type"] == "integer"


def test_no_hints_means_old_chain_byte_identical():
    # Without type_hints the output must be EXACTLY the pre-upgrade behavior —
    # this is the pinned-1.0.x safety property.
    schema = s01.json_schema_from_signature(inspect.signature(_fake_method), inspect.getdoc(_fake_method) or "")
    assert schema["properties"]["user_data"] == {"type": "object", "description": "The payload."}


def test_deprecated_alias_annotation_attribute_is_what_the_loop_keys_on():
    @typing_extensions.deprecated("use new_name")
    def old_name(self):
        """Old."""

    assert getattr(old_name, "__deprecated__", None) == "use new_name"
