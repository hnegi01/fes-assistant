"""
Unit tests for allowlist enforcement inside the MCP server.

The MCP server is the dispatch boundary: a delisted tool must be unreachable
even from a client that is not our backend, so tools_core enforces the same
config/allowed_tools.txt independently of backend/agent/_registry.py.

Covers:
  - _load_allowlist: parsing, relative-path resolution, missing → None
  - TOOLS_BY_ID is filtered at import, so list_tools() and _resolve_sdk_callable
    both refuse a delisted tool
"""

import importlib
import json

import pytest

import mcp_server.tools_core as tools_core


# ---------------------------------------------------------------------------
# _load_allowlist
# ---------------------------------------------------------------------------
class TestLoadAllowlist:
    def test_parses_ids_and_strips_comments(self, tmp_path):
        path = tmp_path / "allowed.txt"
        path.write_text(
            "# header\n\na.one  # [write] mutating\nb.two\n# c.three disabled\n",
            encoding="utf-8",
        )
        assert tools_core._load_allowlist(str(path)) == {"a.one", "b.two"}

    def test_missing_file_returns_none(self, tmp_path):
        assert tools_core._load_allowlist(str(tmp_path / "nope.txt")) is None

    def test_relative_path_resolves_against_repo_root(self):
        """The default value is relative ("config/allowed_tools.txt"), and the
        MCP server's cwd is not guaranteed to be the repo root."""
        result = tools_core._load_allowlist("config/allowed_tools.txt")
        assert result is None or isinstance(result, set)
        if (tools_core.ROOT_DIR / "config" / "allowed_tools.txt").exists():
            assert result, "shipped allowlist resolved to an empty set"

    def test_empty_file_is_an_empty_surface(self, tmp_path):
        path = tmp_path / "allowed.txt"
        path.write_text("# nothing enabled\n", encoding="utf-8")
        assert tools_core._load_allowlist(str(path)) == set()

    def test_unreadable_path_returns_none(self, tmp_path):
        bad = tmp_path / "allowed.txt"
        bad.mkdir()
        assert tools_core._load_allowlist(str(bad)) is None


# ---------------------------------------------------------------------------
# Import-time filtering of the dispatch surface
# ---------------------------------------------------------------------------
class TestDispatchSurface:
    @pytest.fixture()
    def reloaded(self, tmp_path, monkeypatch):
        """Reimport tools_core with a two-tool allowlist and a matching registry."""
        rows = json.loads((tools_core.ROOT_DIR / "config" / "tools.registry.with_examples.json").read_text())
        keep = ["dashboard.get_all_dashboards", "dashboard.rename_dashboard"]
        subset = [r for r in rows if r["tool_id"] in keep]
        assert len(subset) == len(keep), "fixture assumes these tools exist in the registry"

        registry = tmp_path / "registry.json"
        registry.write_text(json.dumps(subset + [r for r in rows if r["tool_id"] == "dashboard.export_dashboard"]))

        allowlist = tmp_path / "allowed.txt"
        allowlist.write_text("\n".join(keep) + "\n# dashboard.export_dashboard is disabled\n")

        monkeypatch.setenv("PYSISENSE_REGISTRY_PATH", str(registry))
        monkeypatch.setenv("FES_TOOL_ALLOWLIST", str(allowlist))
        module = importlib.reload(tools_core)
        yield module
        # Restore the real module state for any later test in the session.
        monkeypatch.undo()
        importlib.reload(tools_core)

    def test_tools_by_id_excludes_delisted(self, reloaded):
        assert set(reloaded.TOOLS_BY_ID) == {
            "dashboard.get_all_dashboards",
            "dashboard.rename_dashboard",
        }

    def test_list_tools_excludes_delisted(self, reloaded):
        assert "dashboard.export_dashboard" not in {t["tool_id"] for t in reloaded.list_tools()}

    def test_delisted_tool_cannot_be_dispatched(self, reloaded):
        with pytest.raises(ValueError, match="Unknown tool_id"):
            reloaded._resolve_sdk_callable("dashboard.export_dashboard", {})

    def test_listed_tool_is_still_reachable(self, reloaded):
        assert "dashboard.get_all_dashboards" in reloaded.TOOLS_BY_ID
