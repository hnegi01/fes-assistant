"""
Unit tests for the registry builder script (scripts/01_build_registry_from_sdk.py).

Covers:
  - _parse_class_docstring: description extraction, Modules section parsing,
    multi-line descriptions, edge cases
  - build_registry_hierarchical: output file structure and content
  - _discover_facade_classes: auto-discovery from pysisense.__all__
"""

import json

import pytest

from scripts.registry_core import (
    _discover_facade_classes,
    _parse_class_docstring,
    build_registry_hierarchical,
)

# ---------------------------------------------------------------------------
# _parse_class_docstring
# ---------------------------------------------------------------------------


class TestParseClassDocstring:
    def test_parses_first_paragraph_as_description(self):
        class FakeClass:
            """Manage widgets on the platform.

            Covers creation, deletion, and configuration.
            """

        result = _parse_class_docstring(FakeClass)

        assert "Manage widgets on the platform." in result["description"]
        assert "Covers creation" in result["description"]

    def test_description_stops_before_modules_section(self):
        class FakeClass:
            """Top level description.

            Modules
            -------
            users :
                User management.
            """

        result = _parse_class_docstring(FakeClass)

        assert "Modules" not in result["description"]
        assert "users" not in result["description"]

    def test_parses_single_module(self):
        class FakeClass:
            """Description.

            Modules
            -------
            users :
                User CRUD operations.
            """

        result = _parse_class_docstring(FakeClass)

        assert result["modules"] == {"users": "User CRUD operations."}

    def test_parses_multiple_modules(self):
        class FakeClass:
            """Description.

            Modules
            -------
            users :
                User management.
            groups :
                Group operations.
            admin :
                Admin tasks.
            """

        result = _parse_class_docstring(FakeClass)

        assert set(result["modules"].keys()) == {"users", "groups", "admin"}
        assert result["modules"]["groups"] == "Group operations."

    def test_multi_line_module_description_is_joined(self):
        class FakeClass:
            """Description.

            Modules
            -------
            users :
                User CRUD — get, create, update users;
                resolve by email or ID.
            """

        result = _parse_class_docstring(FakeClass)

        desc = result["modules"]["users"]
        assert "get, create, update users;" in desc
        assert "resolve by email or ID." in desc
        assert "\n" not in desc

    def test_no_modules_section_returns_empty_dict(self):
        class FakeClass:
            """Just a plain description."""

        result = _parse_class_docstring(FakeClass)

        assert result["description"] == "Just a plain description."
        assert result["modules"] == {}

    def test_empty_docstring(self):
        class FakeClass:
            pass

        result = _parse_class_docstring(FakeClass)

        assert result["description"] == ""
        assert result["modules"] == {}

    def test_multi_line_module_description_is_joined_with_space(self):
        class FakeClass:
            """Description.

            Modules
            -------
            core :
                First sentence.
                Second sentence.
            """

        result = _parse_class_docstring(FakeClass)

        assert result["modules"]["core"] == "First sentence. Second sentence."

    @pytest.mark.skipif(
        __import__("pysisense").__version__ < "1.0.0",
        reason="requires pysisense >= 1.0.0 with class-level docstrings",
    )
    def test_real_access_management_class(self):
        from pysisense import AccessManagement

        result = _parse_class_docstring(AccessManagement)

        assert result["description"]
        assert len(result["modules"]) >= 3
        assert "users" in result["modules"]
        assert "groups" in result["modules"]


# ---------------------------------------------------------------------------
# build_registry_hierarchical
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_registry():
    return [
        {
            "tool_id": "access_management.get_users_all",
            "module": "access_management",
            "sub_module": "access_management.users",
            "description": "Get all users.",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "mutates": False,
        },
        {
            "tool_id": "access_management.create_user",
            "module": "access_management",
            "sub_module": "access_management.users",
            "description": "Create a user.",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "mutates": True,
        },
        {
            "tool_id": "access_management.get_groups_all",
            "module": "access_management",
            "sub_module": "access_management.groups",
            "description": "Get all groups.",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "mutates": False,
        },
        {
            "tool_id": "dashboard.get_dashboards_all",
            "module": "dashboard",
            "sub_module": "dashboard.dashboards",
            "description": "Get all dashboards.",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "mutates": False,
        },
    ]


@pytest.fixture()
def fake_modules(monkeypatch):
    class FakeAccess:
        """Manage access.

        Modules
        -------
        users :
            User operations.
        groups :
            Group operations.
        """

    class FakeDashboard:
        """Manage dashboards.

        Modules
        -------
        dashboards :
            Dashboard listing and sharing.
        """

    import scripts.registry_core as _core

    fake = {"access_management": FakeAccess, "dashboard": FakeDashboard}
    monkeypatch.setattr(_core, "MODULES", fake)
    return fake


class TestBuildRegistryHierarchical:
    def test_writes_top_level_index(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        index = json.loads((tmp_path / "index.json").read_text())
        assert "packages" in index
        assert "access_management" in index["packages"]
        assert "dashboard" in index["packages"]

    def test_top_level_index_has_description(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        index = json.loads((tmp_path / "index.json").read_text())
        assert index["packages"]["access_management"]["description"]

    def test_top_level_index_has_sdk_metadata(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        index = json.loads((tmp_path / "index.json").read_text())
        assert "sdk_version" in index
        assert "updated_at" in index

    def test_writes_package_index_with_modules(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        pkg_index = json.loads((tmp_path / "access_management" / "index.json").read_text())
        assert pkg_index["package"] == "access_management"
        assert "users" in pkg_index["modules"]
        assert "groups" in pkg_index["modules"]

    def test_writes_module_tool_files(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        users_file = tmp_path / "access_management" / "users.json"
        assert users_file.exists()
        tool_ids = [t["tool_id"] for t in json.loads(users_file.read_text())]
        assert "access_management.get_users_all" in tool_ids
        assert "access_management.create_user" in tool_ids

    def test_module_tool_files_have_minimal_fields_only(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        tools = json.loads((tmp_path / "access_management" / "users.json").read_text())
        for tool in tools:
            assert set(tool.keys()) == {"tool_id", "description", "parameters", "mutates"}

    def test_tools_split_into_separate_mixin_files(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        am_dir = tmp_path / "access_management"
        stems = {p.stem for p in am_dir.glob("*.json") if p.name != "index.json"}
        assert "users" in stems
        assert "groups" in stems

    def test_packages_in_separate_subdirectories(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        assert (tmp_path / "access_management").is_dir()
        assert (tmp_path / "dashboard").is_dir()

    def test_groups_tools_under_correct_package(self, tmp_path, fake_registry, fake_modules):
        build_registry_hierarchical(fake_registry, output_dir=tmp_path)

        dashboard_tools = json.loads((tmp_path / "dashboard" / "dashboards.json").read_text())
        assert all(t["tool_id"].startswith("dashboard.") for t in dashboard_tools)


# ---------------------------------------------------------------------------
# _discover_facade_classes
# ---------------------------------------------------------------------------


class TestDiscoverFacadeClasses:
    def test_discovers_all_expected_packages(self):
        import inspect as _inspect

        import pysisense as _pysisense

        modules = _discover_facade_classes()

        # Compute the expected set from the SDK itself, never a hardcoded
        # list. pysisense >= 1.1 exports FACADES — the explicit tuple of
        # tool-bearing classes — and __all__ additionally carries the TypedDict
        # payload classes (module "pysisense.payloads"), which are data
        # contracts, not facades: deriving from __all__ here would demand a
        # bogus "payloads" package. Older SDKs have no FACADES and fall back.
        facades = getattr(_pysisense, "FACADES", None)
        if facades:
            expected = {c.__module__.split(".")[1] for c in facades if c.__name__ != "SisenseClient"}
        else:
            expected = {
                getattr(_pysisense, name).__module__.split(".")[1]
                for name in _pysisense.__all__
                if _inspect.isclass(getattr(_pysisense, name, None)) and name != "SisenseClient"
            }
        assert set(modules.keys()) == expected

    def test_skips_sisense_client(self):
        modules = _discover_facade_classes()

        assert "SisenseClient" not in modules
        # also not under a subpackage key
        for obj in modules.values():
            assert obj.__name__ != "SisenseClient"

    def test_all_values_are_classes(self):
        import inspect

        modules = _discover_facade_classes()

        for obj in modules.values():
            assert inspect.isclass(obj)

    def test_keys_match_subpackage_names(self):
        modules = _discover_facade_classes()

        for key, klass in modules.items():
            mod_path = getattr(klass, "__module__", "")
            subpkg = mod_path.split(".")[1] if "." in mod_path else ""
            assert key == subpkg, f"{klass.__name__}: expected key '{subpkg}', got '{key}'"


# ---------------------------------------------------------------------------
# SCHEMA_RULES drift guard
# ---------------------------------------------------------------------------
#
# SCHEMA_RULES hand-patches enums/aliases/rich schemas onto specific tools, and
# apply_schema_rules() creates missing paths blindly (_walk_and_set). If the
# SDK renames or drops a patched method or parameter, a rebuild would silently
# write the patch onto a ghost property the SDK no longer has. These tests make
# that drift fail here instead.


def _builder():
    import importlib

    return importlib.import_module("scripts.01_build_registry_from_sdk")


def _patched_param_names(patch: dict) -> set:
    """The parameter name each dotted patch path targets.

    Tool-level keys (x-followup and friends) don't address a parameter and are
    skipped — anything else must sit under parameters.properties.<param>.
    """
    names = set()
    for dotted in patch:
        parts = dotted.split(".")
        if len(parts) == 1:
            assert parts[0].startswith("x-"), f"unexpected tool-level patch key: {dotted}"
            continue
        assert parts[:2] == ["parameters", "properties"], f"unexpected patch path shape: {dotted}"
        names.add(parts[2])
    return names


class TestSchemaRulesDrift:
    def test_every_patched_tool_exists_in_sdk(self):
        import inspect

        mod = _builder()
        from scripts.registry_core import MODULES

        for tool_id in mod.SCHEMA_RULES:
            module_name, method_name = tool_id.split(".", 1)
            assert module_name in MODULES, f"{tool_id}: module '{module_name}' not discovered from pysisense"
            klass = MODULES[module_name]
            # Same lens build_registry() uses to enumerate tools (line ~755).
            public_funcs = {
                name for name, _ in inspect.getmembers(klass, predicate=inspect.isfunction) if not name.startswith("_")
            }
            assert method_name in public_funcs, f"{tool_id}: method '{method_name}' not found on {klass.__name__}"

    def test_every_patched_parameter_exists_in_signature(self):
        import inspect

        mod = _builder()
        from scripts.registry_core import MODULES

        for tool_id, rules in mod.SCHEMA_RULES.items():
            module_name, method_name = tool_id.split(".", 1)
            func = getattr(MODULES[module_name], method_name)
            sig_params = {p for p in inspect.signature(func).parameters if p != "self"}
            for param in _patched_param_names(rules.get("patch", {})):
                assert param in sig_params, (
                    f"{tool_id}: SCHEMA_RULES patches '{param}' but the SDK signature has "
                    f"{sorted(sig_params)} — stale patch would create a ghost property"
                )

    def test_shipped_registry_carries_the_patches(self):
        import backend.agent._registry as registry_m

        mod = _builder()
        rows = json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))
        by_id = {row["tool_id"]: row for row in rows}

        for tool_id, rules in mod.SCHEMA_RULES.items():
            assert tool_id in by_id, f"{tool_id}: patched in SCHEMA_RULES but absent from the shipped registry"
            tool = by_id[tool_id]
            for dotted, expected in rules.get("patch", {}).items():
                cur = tool
                for part in dotted.split("."):
                    assert isinstance(cur, dict) and part in cur, (
                        f"{tool_id}: shipped registry is missing '{dotted}' — "
                        "SCHEMA_RULES changed without regenerating the registry?"
                    )
                    cur = cur[part]
                assert cur == expected, (
                    f"{tool_id}: shipped registry value at '{dotted}' differs from SCHEMA_RULES — "
                    "regenerate the registry (scripts/01) to sync them"
                )


class TestPackageDocCorrections:
    """Corrections strip capability claims that are factually wrong about a
    package — the Level 1 router picks a package from its description alone, so
    a false claim silently misroutes every such request (wellcheck advertised
    "unused columns" it does not have; requests landed on island tables).

    These are stand-ins for upstream docstring fixes, so each one must expire:
    when the SDK stops making the claim, the correction is dead weight that
    hides what the docstring now says."""

    def test_every_correction_still_matches_the_sdk_docstring(self):

        from scripts.registry_core import _PACKAGE_DOC_CORRECTIONS, MODULES, _parse_class_docstring

        stale = {}
        for pkg, phrases in _PACKAGE_DOC_CORRECTIONS.items():
            klass = MODULES.get(pkg)
            assert klass is not None, f"correction targets unknown package {pkg!r}"
            doc = _parse_class_docstring(klass)["description"]
            missing = [p for p in phrases if p not in doc]
            if missing:
                stale[pkg] = missing
        assert stale == {}, f"the SDK no longer makes these claims — delete the corrections: {stale}"

    def test_corrected_claim_is_absent_from_the_shipped_index(self):
        import json
        from pathlib import Path

        from scripts.registry_core import _PACKAGE_DOC_CORRECTIONS

        idx = json.loads(
            (Path(__file__).resolve().parents[2] / "config" / "registry" / "index.json").read_text(encoding="utf-8")
        )["packages"]
        for pkg, phrases in _PACKAGE_DOC_CORRECTIONS.items():
            desc = idx.get(pkg, {}).get("description", "")
            for phrase in phrases:
                assert phrase not in desc, f"{pkg} index still carries the corrected claim {phrase!r}"
