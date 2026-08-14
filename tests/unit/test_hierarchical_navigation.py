"""
Unit tests for 3-level hierarchical navigation.

Covers:
  - _load_mixin_tools: OpenAI format, mutating filter, missing file
  - _load_all_package_tools: combines mixin files, skips index.json
  - _load_registry_index: real file + missing file fallback
  - _load_package_index: real file + missing file fallback
  - _navigate_to_tools: full success, single-mixin skip, Level 1/2 failure,
    LLM error, empty registry, latency field
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

import backend.agent._routing as routing_m
from backend.agent._routing import (
    _load_all_package_tools,
    _load_mixin_tools,
    _load_package_index,
    _load_registry_index,
    _navigate_to_tools,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _llm_response(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content, "tool_calls": None}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


# ---------------------------------------------------------------------------
# Shared fixture: minimal fake registry tree
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_registry_dir(tmp_path, monkeypatch):
    """
    Minimal config/registry/ tree in tmp_path.

        index.json                         (2 packages)
        access_management/
            index.json                     (2 modules: users, groups)
            users.json                     (get_users_all[read], create_user[mutates])
            groups.json                    (get_groups_all[read])
        encryption/
            index.json                     (1 module: core — tests Level 2 skip)
            core.json                      (get_encryption_status[read])
    """
    reg_dir = tmp_path / "registry"
    reg_dir.mkdir()

    (reg_dir / "index.json").write_text(
        json.dumps(
            {
                "sdk_version": "1.0.0",
                "updated_at": "2025-01-01T00:00:00Z",
                "packages": {
                    "access_management": {
                        "class": "AccessManagement",
                        "description": "Manage users, groups, and access control.",
                    },
                    "encryption": {
                        "class": "Encryption",
                        "description": "Manage encryption keys and settings.",
                    },
                },
            }
        )
    )

    am_dir = reg_dir / "access_management"
    am_dir.mkdir()
    (am_dir / "index.json").write_text(
        json.dumps(
            {
                "package": "access_management",
                "class": "AccessManagement",
                "modules": {
                    "users": "User CRUD — get, create, update, deactivate users.",
                    "groups": "Group membership — list groups, add or remove users.",
                },
            }
        )
    )
    (am_dir / "users.json").write_text(
        json.dumps(
            [
                {
                    "tool_id": "access_management.get_users_all",
                    "description": "Get all users.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "mutates": False,
                },
                {
                    "tool_id": "access_management.create_user",
                    "description": "Create a user.",
                    "parameters": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                        "required": ["email"],
                    },
                    "mutates": True,
                },
            ]
        )
    )
    (am_dir / "groups.json").write_text(
        json.dumps(
            [
                {
                    "tool_id": "access_management.get_groups_all",
                    "description": "Get all groups.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "mutates": False,
                },
            ]
        )
    )

    enc_dir = reg_dir / "encryption"
    enc_dir.mkdir()
    (enc_dir / "index.json").write_text(
        json.dumps(
            {
                "package": "encryption",
                "class": "Encryption",
                "modules": {"core": "Encryption key management and status checks."},
            }
        )
    )
    (enc_dir / "core.json").write_text(
        json.dumps(
            [
                {
                    "tool_id": "encryption.get_encryption_status",
                    "description": "Get encryption status.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "mutates": False,
                },
            ]
        )
    )

    monkeypatch.setattr(routing_m, "REGISTRY_DIR", reg_dir)
    # These tool_ids are fabricated, so the real config/allowed_tools.txt would
    # filter every one of them out. None = "no allowlist in force" — the
    # allowlist itself is covered by test_tool_allowlist.py.
    monkeypatch.setattr(routing_m, "allowed_tool_ids", lambda: None)
    return reg_dir


# ---------------------------------------------------------------------------
# _load_mixin_tools
# ---------------------------------------------------------------------------


class TestLoadMixinTools:
    def test_returns_openai_format(self, fake_registry_dir):
        tools = _load_mixin_tools("access_management", "users")
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]
            assert "description" in t["function"]
            assert "parameters" in t["function"]

    def test_tool_names_match_tool_ids(self, fake_registry_dir):
        tools = _load_mixin_tools("access_management", "users")
        names = {t["function"]["name"] for t in tools}
        assert "access_management.get_users_all" in names
        assert "access_management.create_user" in names

    def test_filters_mutating_when_not_allowed(self, fake_registry_dir, monkeypatch):
        monkeypatch.setattr(routing_m, "ALLOW_MUTATING_TOOLS", False)
        tools = _load_mixin_tools("access_management", "users")
        names = [t["function"]["name"] for t in tools]
        assert "access_management.get_users_all" in names
        assert "access_management.create_user" not in names

    def test_includes_mutating_when_allowed(self, fake_registry_dir, monkeypatch):
        monkeypatch.setattr(routing_m, "ALLOW_MUTATING_TOOLS", True)
        tools = _load_mixin_tools("access_management", "users")
        names = [t["function"]["name"] for t in tools]
        assert "access_management.create_user" in names

    def test_returns_empty_on_missing_mixin_file(self, fake_registry_dir):
        tools = _load_mixin_tools("access_management", "nonexistent")
        assert tools == []

    def test_returns_empty_for_missing_package(self, fake_registry_dir):
        tools = _load_mixin_tools("nonexistent_pkg", "core")
        assert tools == []

    def test_single_mixin_file_loads_correctly(self, fake_registry_dir):
        tools = _load_mixin_tools("access_management", "groups")
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "access_management.get_groups_all"


# ---------------------------------------------------------------------------
# _load_all_package_tools
# ---------------------------------------------------------------------------


class TestLoadAllPackageTools:
    def test_combines_all_mixin_files(self, fake_registry_dir):
        tools = _load_all_package_tools("access_management")
        ids = {t["function"]["name"] for t in tools}
        assert "access_management.get_users_all" in ids
        assert "access_management.get_groups_all" in ids

    def test_skips_index_json(self, fake_registry_dir):
        tools = _load_all_package_tools("access_management")
        for t in tools:
            assert t["function"]["name"] != "index"

    def test_returns_empty_for_missing_directory(self, fake_registry_dir):
        tools = _load_all_package_tools("nonexistent_pkg")
        assert tools == []

    def test_single_mixin_package(self, fake_registry_dir):
        tools = _load_all_package_tools("encryption")
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "encryption.get_encryption_status"

    def test_total_count_matches_sum_of_mixins(self, fake_registry_dir):
        users = _load_mixin_tools("access_management", "users")
        groups = _load_mixin_tools("access_management", "groups")
        all_tools = _load_all_package_tools("access_management")
        assert len(all_tools) == len(users) + len(groups)


# ---------------------------------------------------------------------------
# _load_registry_index
# ---------------------------------------------------------------------------


class TestLoadRegistryIndex:
    def test_real_index_has_packages(self):
        result = _load_registry_index()
        assert "packages" in result
        assert len(result["packages"]) > 0

    def test_real_index_has_sdk_version(self):
        result = _load_registry_index()
        assert "sdk_version" in result

    def test_real_index_has_access_management(self):
        result = _load_registry_index()
        assert "access_management" in result["packages"]

    def test_returns_empty_on_missing_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(routing_m, "REGISTRY_DIR", tmp_path / "nonexistent")
        assert _load_registry_index() == {}

    def test_fake_index_structure(self, fake_registry_dir):
        result = _load_registry_index()
        assert "access_management" in result["packages"]
        assert "encryption" in result["packages"]
        assert result["packages"]["access_management"]["description"]


# ---------------------------------------------------------------------------
# _load_package_index
# ---------------------------------------------------------------------------


class TestLoadPackageIndex:
    def test_real_access_management_has_modules(self):
        result = _load_package_index("access_management")
        assert "modules" in result
        assert len(result["modules"]) > 0

    def test_returns_empty_on_missing_package(self, monkeypatch, tmp_path):
        monkeypatch.setattr(routing_m, "REGISTRY_DIR", tmp_path / "nonexistent")
        assert _load_package_index("access_management") == {}

    def test_fake_package_index_modules(self, fake_registry_dir):
        result = _load_package_index("access_management")
        assert "users" in result["modules"]
        assert "groups" in result["modules"]

    def test_single_mixin_package_index(self, fake_registry_dir):
        result = _load_package_index("encryption")
        assert list(result["modules"].keys()) == ["core"]


# ---------------------------------------------------------------------------
# _navigate_to_tools
# ---------------------------------------------------------------------------


class TestNavigateToTools:
    def test_full_navigation_success(self, fake_registry_dir):
        with patch.object(
            routing_m,
            "call_llm_raw",
            new=AsyncMock(
                side_effect=[
                    _llm_response("access_management"),
                    _llm_response("users"),
                ]
            ),
        ):
            tools, pkg, mixin, ms = run(
                _navigate_to_tools(
                    {"role": "user", "content": "show all users"},
                    [],
                    None,
                )
            )

        assert pkg == "access_management"
        assert mixin == "users"
        assert len(tools) > 0

    def test_single_mixin_skips_level2(self, fake_registry_dir):
        with patch.object(
            routing_m,
            "call_llm_raw",
            new=AsyncMock(return_value=_llm_response("encryption")),
        ) as mock_llm:
            tools, pkg, mixin, ms = run(
                _navigate_to_tools(
                    {"role": "user", "content": "check encryption"},
                    [],
                    None,
                )
            )

        assert pkg == "encryption"
        assert mixin == "core"
        assert mock_llm.call_count == 1  # Level 2 was skipped

    def test_level1_unrecognised_response_returns_empty(self, fake_registry_dir):
        with patch.object(routing_m, "call_llm_raw", new=AsyncMock(return_value=_llm_response("unknown_garbage"))):
            tools, pkg, mixin, ms = run(
                _navigate_to_tools(
                    {"role": "user", "content": "do something"},
                    [],
                    None,
                )
            )

        assert tools == []
        assert pkg == ""

    def test_level1_llm_error_returns_empty(self, fake_registry_dir):
        with patch.object(routing_m, "call_llm_raw", new=AsyncMock(side_effect=RuntimeError("timeout"))):
            tools, pkg, mixin, ms = run(
                _navigate_to_tools(
                    {"role": "user", "content": "show users"},
                    [],
                    None,
                )
            )

        assert tools == []

    def test_level2_unrecognised_response_returns_empty_tools(self, fake_registry_dir):
        with patch.object(
            routing_m,
            "call_llm_raw",
            new=AsyncMock(
                side_effect=[
                    _llm_response("access_management"),
                    _llm_response("unknown_mixin"),
                ]
            ),
        ):
            tools, pkg, mixin, ms = run(
                _navigate_to_tools(
                    {"role": "user", "content": "show users"},
                    [],
                    None,
                )
            )

        assert tools == []
        assert pkg == "access_management"
        assert mixin == ""

    def test_empty_registry_dir_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(routing_m, "REGISTRY_DIR", tmp_path / "nonexistent")
        tools, pkg, mixin, ms = run(
            _navigate_to_tools(
                {"role": "user", "content": "show users"},
                [],
                None,
            )
        )
        assert tools == []
        assert pkg == ""

    def test_returns_latency_ms_as_int(self, fake_registry_dir):
        with patch.object(
            routing_m,
            "call_llm_raw",
            new=AsyncMock(
                side_effect=[
                    _llm_response("access_management"),
                    _llm_response("users"),
                ]
            ),
        ):
            _, _, _, ms = run(
                _navigate_to_tools(
                    {"role": "user", "content": "list users"},
                    [],
                    None,
                )
            )

        assert isinstance(ms, int)
        assert ms >= 0

    def test_navigation_tools_are_openai_format(self, fake_registry_dir):
        with patch.object(
            routing_m,
            "call_llm_raw",
            new=AsyncMock(
                side_effect=[
                    _llm_response("access_management"),
                    _llm_response("groups"),
                ]
            ),
        ):
            tools, _, _, _ = run(
                _navigate_to_tools(
                    {"role": "user", "content": "list all groups"},
                    [],
                    None,
                )
            )

        assert len(tools) > 0
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]

    def test_history_passed_through_to_llm(self, fake_registry_dir):
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        captured_calls = []

        async def mock_llm(messages, **kwargs):
            captured_calls.append(messages)
            if len(captured_calls) == 1:
                return _llm_response("access_management")
            return _llm_response("users")

        with patch.object(routing_m, "call_llm_raw", new=mock_llm):
            run(
                _navigate_to_tools(
                    {"role": "user", "content": "show users"},
                    history,
                    None,
                )
            )

        # Both Level 1 and Level 2 calls should include history
        for call_messages in captured_calls:
            contents = [m["content"] for m in call_messages]
            assert any("previous question" in c for c in contents)
