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

        # Compute expected set directly from __all__ so this test stays
        # correct across SDK versions without a hardcoded package list.
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
