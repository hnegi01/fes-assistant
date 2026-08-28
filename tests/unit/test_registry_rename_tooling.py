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

import pytest

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


def test_apply_deletes_dead_active_and_commented_lines(allowlist):
    allowlist.write_text(
        "# ===== datamodel =====\n"
        "datamodel.get_connections      # Retrieve all connections.\n"  # renamed away — dead
        "access_management.get_users_all  # Retrieve all users.\n"  # alive
        "# datamodel.gone_tool  # staged long ago, since deleted\n"  # dead staged line
        "# prose comment that mentions nothing tool-shaped\n"
    )
    _apply(allowlist)
    text = allowlist.read_text()
    assert "datamodel.get_connections " not in text and "gone_tool" not in text
    assert "access_management.get_users_all" in text
    assert "prose comment" in text, "prose comments are never touched"


def test_apply_stages_new_tools_commented_with_write_tag(allowlist):
    allowlist.write_text("access_management.get_users_all  # Retrieve all users.\n")
    _apply(allowlist)
    text = allowlist.read_text()
    assert s04.STAGING_HEADER in text
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
