"""
Unit tests for the curated tool allowlist (config/allowed_tools.txt).

Covers:
  - allowed_tool_ids: parsing (comments, blanks, trailing comments), mtime
    reload, missing file → None (allow all), unreadable file → None
  - _filter_by_allowlist / _load_registry_rows: the agent registry is filtered
  - _load_mixin_tools: the tool menu shown to the selection LLM is filtered
  - the allowlist composes with (does not replace) the mutating-tool filter

The MCP server enforces the same file independently at import time
(mcp_server/tools_core.py::_load_allowlist), which is covered by
test_mcp_tool_allowlist.py.
"""

import json
import os

import pytest

import backend.agent._registry as registry_m
import backend.agent._routing as routing_m


@pytest.fixture()
def allowlist_file(tmp_path, monkeypatch):
    """Point the loader at a temp allowlist and clear its mtime cache."""

    def _write(text: str):
        path = tmp_path / "allowed_tools.txt"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(registry_m, "ALLOWLIST_PATH", path)
        monkeypatch.setattr(registry_m, "_allowlist_cache_mtime", None)
        monkeypatch.setattr(registry_m, "_allowlist_cache_ids", None)
        return path

    return _write


@pytest.fixture()
def no_allowlist(tmp_path, monkeypatch):
    """Point the loader at a path that does not exist."""
    monkeypatch.setattr(registry_m, "ALLOWLIST_PATH", tmp_path / "nope.txt")
    monkeypatch.setattr(registry_m, "_allowlist_cache_mtime", None)
    monkeypatch.setattr(registry_m, "_allowlist_cache_ids", None)
    monkeypatch.setattr(registry_m, "_allowlist_missing_warned", False)


# ---------------------------------------------------------------------------
# allowed_tool_ids — parsing
# ---------------------------------------------------------------------------
class TestAllowedToolIdsParsing:
    def test_plain_lines(self, allowlist_file):
        allowlist_file("a.one\nb.two\n")
        assert registry_m.allowed_tool_ids() == {"a.one", "b.two"}

    def test_ignores_blank_lines_and_full_line_comments(self, allowlist_file):
        allowlist_file("# header\n\na.one\n\n#  b.two is disabled\nc.three\n")
        assert registry_m.allowed_tool_ids() == {"a.one", "c.three"}

    def test_strips_trailing_comments(self, allowlist_file):
        allowlist_file("a.one   # [write] does a thing\nb.two  # read\n")
        assert registry_m.allowed_tool_ids() == {"a.one", "b.two"}

    def test_commented_out_tool_is_not_allowed(self, allowlist_file):
        allowlist_file("a.one\n# b.two\n")
        assert registry_m.allowed_tool_ids() == {"a.one"}

    def test_empty_file_allows_nothing(self, allowlist_file):
        """An empty file is an explicit empty surface — distinct from no file."""
        allowlist_file("# everything disabled\n")
        assert registry_m.allowed_tool_ids() == set()

    def test_duplicate_lines_collapse(self, allowlist_file):
        allowlist_file("a.one\na.one\nb.two\n")
        assert registry_m.allowed_tool_ids() == {"a.one", "b.two"}


# ---------------------------------------------------------------------------
# allowed_tool_ids — missing file and caching
# ---------------------------------------------------------------------------
class TestAllowedToolIdsLifecycle:
    def test_missing_file_returns_none_meaning_allow_all(self, no_allowlist):
        assert registry_m.allowed_tool_ids() is None

    def test_missing_file_warns_once(self, no_allowlist, caplog):
        with caplog.at_level("WARNING"):
            registry_m.allowed_tool_ids()
            registry_m.allowed_tool_ids()
            registry_m.allowed_tool_ids()
        assert sum("allowlist not found" in r.message for r in caplog.records) == 1

    def test_edit_is_picked_up_without_restart(self, allowlist_file):
        path = allowlist_file("a.one\n")
        assert registry_m.allowed_tool_ids() == {"a.one"}
        # Force a distinct mtime so the cache is invalidated deterministically.
        st = path.stat()
        path.write_text("a.one\nb.two\n", encoding="utf-8")
        os.utime(path, (st.st_atime + 10, st.st_mtime + 10))
        assert registry_m.allowed_tool_ids() == {"a.one", "b.two"}

    def test_unreadable_file_falls_back_to_allow_all(self, tmp_path, monkeypatch):
        # A directory where a file is expected: exists() is True, read_text raises.
        bad = tmp_path / "allowed_tools.txt"
        bad.mkdir()
        monkeypatch.setattr(registry_m, "ALLOWLIST_PATH", bad)
        monkeypatch.setattr(registry_m, "_allowlist_cache_mtime", None)
        monkeypatch.setattr(registry_m, "_allowlist_cache_ids", None)
        assert registry_m.allowed_tool_ids() is None


# ---------------------------------------------------------------------------
# Enforcement: agent registry
# ---------------------------------------------------------------------------
class TestRegistryFiltering:
    ROWS = [
        {"tool_id": "a.read", "module": "a", "mutates": False},
        {"tool_id": "a.write", "module": "a", "mutates": True},
        {"tool_id": "b.read", "module": "b", "mutates": False},
    ]

    def test_filter_keeps_only_listed(self, allowlist_file):
        allowlist_file("a.read\nb.read\n")
        kept = [r["tool_id"] for r in registry_m._filter_by_allowlist(self.ROWS)]
        assert kept == ["a.read", "b.read"]

    def test_filter_is_passthrough_when_no_allowlist(self, no_allowlist):
        kept = [r["tool_id"] for r in registry_m._filter_by_allowlist(self.ROWS)]
        assert kept == ["a.read", "a.write", "b.read"]

    def test_load_registry_rows_applies_allowlist(self, tmp_path, monkeypatch, allowlist_file):
        reg = tmp_path / "registry.json"
        reg.write_text(json.dumps(self.ROWS), encoding="utf-8")
        monkeypatch.setattr(registry_m, "REGISTRY_PATH", reg)
        monkeypatch.setattr(registry_m, "_registry_cache_mtime", None)
        monkeypatch.setattr(registry_m, "_registry_cache_rows", [])
        allowlist_file("b.read\n")
        assert [r["tool_id"] for r in registry_m._load_registry_rows()] == ["b.read"]

    def test_allowlist_edit_beats_registry_mtime_cache(self, tmp_path, monkeypatch, allowlist_file):
        """The filter runs on the way OUT of the cache, so an allowlist-only
        change takes effect even though the registry file never changed."""
        reg = tmp_path / "registry.json"
        reg.write_text(json.dumps(self.ROWS), encoding="utf-8")
        monkeypatch.setattr(registry_m, "REGISTRY_PATH", reg)
        monkeypatch.setattr(registry_m, "_registry_cache_mtime", None)
        monkeypatch.setattr(registry_m, "_registry_cache_rows", [])

        allowlist_file("a.read\n")
        assert [r["tool_id"] for r in registry_m._load_registry_rows()] == ["a.read"]

        # Registry is now warm in the cache; only the allowlist changes.
        allowlist_file("a.read\nb.read\n")
        assert [r["tool_id"] for r in registry_m._load_registry_rows()] == ["a.read", "b.read"]


# ---------------------------------------------------------------------------
# Enforcement: the tool menu the selection LLM sees
# ---------------------------------------------------------------------------
class TestMixinToolFiltering:
    @pytest.fixture()
    def mixin_dir(self, tmp_path, monkeypatch):
        pkg = tmp_path / "registry" / "demo"
        pkg.mkdir(parents=True)
        (pkg / "core.json").write_text(
            json.dumps(
                [
                    {"tool_id": "demo.read", "description": "Read.", "parameters": {}, "mutates": False},
                    {"tool_id": "demo.write", "description": "Write.", "parameters": {}, "mutates": True},
                    {"tool_id": "demo.hidden", "description": "Hidden.", "parameters": {}, "mutates": False},
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(routing_m, "REGISTRY_DIR", tmp_path / "registry")
        return pkg

    def _names(self):
        return [t["function"]["name"] for t in routing_m._load_mixin_tools("demo", "core")]

    def test_unlisted_tool_never_reaches_the_llm(self, mixin_dir, allowlist_file):
        allowlist_file("demo.read\ndemo.write\n")
        assert self._names() == ["demo.read", "demo.write"]

    def test_no_allowlist_exposes_all(self, mixin_dir, no_allowlist):
        assert self._names() == ["demo.read", "demo.write", "demo.hidden"]

    def test_composes_with_mutating_filter(self, mixin_dir, allowlist_file, monkeypatch):
        """The allowlist does not replace ALLOW_MUTATING_TOOLS — both apply."""
        allowlist_file("demo.read\ndemo.write\n")
        monkeypatch.setattr(routing_m, "ALLOW_MUTATING_TOOLS", False)
        assert self._names() == ["demo.read"]

    def test_allowlist_cannot_re_enable_a_mutating_tool(self, mixin_dir, allowlist_file, monkeypatch):
        allowlist_file("demo.write\n")
        monkeypatch.setattr(routing_m, "ALLOW_MUTATING_TOOLS", False)
        assert self._names() == []


# ---------------------------------------------------------------------------
# The shipped file itself
# ---------------------------------------------------------------------------
class TestShippedAllowlist:
    def test_every_line_resolves_to_a_real_tool(self):
        """A typo'd or stale line silently hides nothing but wastes a review —
        catch drift between the allowlist and the registry in CI."""
        if not registry_m.ALLOWLIST_PATH.exists():
            pytest.skip("no allowlist shipped")
        listed = set()
        for line in registry_m.ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                listed.add(entry)
        rows = json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))
        known = {r["tool_id"] for r in rows}
        assert listed - known == set(), "allowlist lines with no matching registry tool"
