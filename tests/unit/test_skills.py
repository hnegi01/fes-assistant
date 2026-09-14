"""
Skills — the loader, the planner hook, and the drift guards.

Three layers, matching the three things that can be wrong (docs/design/skills.md §10):

  - THE SHIPPED FILES: every skills/*/SKILL.md parses, its frontmatter obeys the
    contract, every tool it names exists in the registry AND is allowlisted, and
    every tool the body mentions is declared. A rename or delisting upstream
    fails HERE, in CI, not in a customer's workflow months later.
  - THE LOADER: malformed files are excluded and logged, never shipped
    half-broken; a missing directory means no skills; the mtime cache
    invalidates when a file changes.
  - THE PLANNER HOOK: with skills loaded the catalog message carries the index;
    a `SKILL: <name>` reply triggers exactly one second pass with the body;
    an unknown name is dropped and the rest planned as usual; no skills means
    a prompt byte-identical to the pre-skills one.

Also here because the skill work exposed it: the `# [write]` comments in
allowed_tools.txt must agree with the registry's `mutates` flags. Script 04
writes them when it STAGES a tool and never refreshes them, so a generator fix
leaves a stale marker behind (two read-only perspective tools read `[write]` on
2026-09-13). Nothing enforces from the comment — but it is what a human reads
before uncommenting a line.
"""

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import backend.agent._registry as registry_m
import backend.agent._skills as skills_m
import backend.agent.llm_agent as A

ROOT = Path(__file__).resolve().parents[2]


def _run(coro):
    """Run a coroutine on the shared loop. NOT _run(): that closes the
    loop when it returns, and every later test in the session that calls
    asyncio.get_event_loop() then fails with 'no current event loop' — nine
    unrelated tests went red the first time this file ran ahead of them."""
    return asyncio.get_event_loop().run_until_complete(coro)


SHIPPED = sorted((ROOT / "skills").glob("*/SKILL.md"))

# Sisense's role vocabulary, both forms (internal / display), as get_roles
# returns them on 2.x. A skill's requires_role must be one of these — a typo
# here would make the pre-plan role check (design §6) reject every user.
KNOWN_SISENSE_ROLES = {
    "admin",
    "sysAdmin",
    "super",
    "tenantAdmin",
    "dataAdmin",
    "dataDesigner",
    "dashboardDesigner",
    "contributor",
    "designer",
    "viewer",
    "consumer",
    "viewerPlus",
}


def _registry_rows():
    return json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))


def _allowlist_lines():
    return registry_m.ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# Drift guards over the shipped skills
# ---------------------------------------------------------------------------
class TestShippedSkills:
    def test_at_least_one_skill_ships(self):
        assert SHIPPED, "skills/ is empty — the first skill should be here"

    @pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.parent.name)
    def test_parses_and_matches_directory(self, path):
        skill = skills_m.parse_skill_file(path)
        assert skill.name == path.parent.name
        assert skill.version >= 1
        assert skill.description and len(skill.description) < 400, "description is the one line the planner sees"

    @pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.parent.name)
    def test_role_is_a_real_sisense_role(self, path):
        skill = skills_m.parse_skill_file(path)
        if skill.requires_role is not None:
            assert skill.requires_role in KNOWN_SISENSE_ROLES, skill.requires_role

    @pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.parent.name)
    def test_every_tool_exists_and_is_allowlisted(self, path):
        skill = skills_m.parse_skill_file(path)
        registry_ids = {r["tool_id"] for r in _registry_rows()}
        allowed = registry_m.allowed_tool_ids()
        problem = skills_m.validate_against_surface(skill, registry_ids, allowed)
        assert problem is None, f"{skill.name}: {problem}"

    @pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.parent.name)
    def test_body_has_the_five_sections(self, path):
        body = skills_m.parse_skill_file(path).body
        for heading in ("## When this applies", "## Procedure", "## Never", "## On failure", "## Report"):
            assert heading in body, f"{path.parent.name} is missing '{heading}'"

    @pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.parent.name)
    def test_never_names_ownership_tools(self, path):
        """Design §2 rule 4: no skill may plan an ownership change. Enforced by
        the tools list — this pins that no shipped skill has quietly added one."""
        skill = skills_m.parse_skill_file(path)
        forbidden = [t for t in skill.tools if "owner" in t or "ownership" in t]
        assert forbidden == [], forbidden

    def test_loader_loads_every_shipped_skill(self, monkeypatch):
        monkeypatch.setattr(skills_m, "SKILLS_ENABLED", True)
        monkeypatch.setattr(skills_m, "_cache_key", None)
        loaded = skills_m.load_skills(
            registry_ids={r["tool_id"] for r in _registry_rows()},
            allowed=registry_m.allowed_tool_ids(),
        )
        assert set(loaded) == {p.parent.name for p in SHIPPED}


class TestAllowlistCommentsAgreeWithRegistry:
    def test_write_markers_match_mutates(self):
        mutates = {r["tool_id"]: bool(r.get("mutates")) for r in _registry_rows()}
        disagreements = []
        for line in _allowlist_lines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tool_id = stripped.split("#", 1)[0].strip()
            marked = "[write]" in stripped
            if tool_id in mutates and marked != mutates[tool_id]:
                disagreements.append(
                    (tool_id, "comment says write" if marked else "comment says read", mutates[tool_id])
                )
        assert disagreements == [], (
            "allowed_tools.txt comments disagree with the registry's mutates flag "
            f"(script 04 writes them at staging time and never refreshes): {disagreements}"
        )


# ---------------------------------------------------------------------------
# Loader behaviour on a fixture directory
# ---------------------------------------------------------------------------
GOOD = """---
name: {name}
description: A fixture skill.
version: 1
tools:
  - datamodel.get_all_datamodel
---

## When this applies
Always, in tests.

## Procedure
1. List the models (`datamodel.get_all_datamodel`).

## Never
- Nothing.

## On failure
- Report.

## Report
- The list.
"""


def _write(tmp: Path, name: str, text: str) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "SKILL.md"
    f.write_text(text, encoding="utf-8")
    return f


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_m, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(skills_m, "SKILLS_ENABLED", True)
    monkeypatch.setattr(skills_m, "_cache_key", None)
    monkeypatch.setattr(skills_m, "_cache", {})
    return tmp_path


class TestLoader:
    def test_missing_directory_means_no_skills(self, tmp_path, monkeypatch):
        monkeypatch.setattr(skills_m, "SKILLS_DIR", tmp_path / "nope")
        monkeypatch.setattr(skills_m, "SKILLS_ENABLED", True)
        monkeypatch.setattr(skills_m, "_cache_key", None)
        assert skills_m.load_skills() == {}

    def test_disabled_means_no_skills_even_when_present(self, skills_dir, monkeypatch):
        _write(skills_dir, "one", GOOD.format(name="one"))
        monkeypatch.setattr(skills_m, "SKILLS_ENABLED", False)
        assert skills_m.load_skills() == {}

    def test_good_skill_loads(self, skills_dir):
        _write(skills_dir, "one", GOOD.format(name="one"))
        loaded = skills_m.load_skills()
        assert list(loaded) == ["one"]
        assert loaded["one"].tools == ("datamodel.get_all_datamodel",)

    @pytest.mark.parametrize(
        "mutate, reason",
        [
            (lambda t: t.replace("---\nname", "name", 1), "missing YAML frontmatter"),
            (lambda t: t.replace("name: one", "name: other"), "does not match its directory"),
            (lambda t: t.replace("version: 1", "version: one"), "positive integer"),
            (lambda t: t.replace("tools:\n  - datamodel.get_all_datamodel\n", "tools: []\n"), "non-empty list"),
            (lambda t: t.replace("version: 1", "version: 1\nbogus: yes"), "unknown frontmatter key"),
            (lambda t: t.replace("version: 1", "version: 1\nguardrails:\n  - id: not-a-thing"), "unknown guardrail"),
            (lambda t: t.split("---\n\n")[0] + "---\n\n", "body is empty"),
        ],
    )
    def test_defects_are_excluded_with_a_reason(self, skills_dir, mutate, reason, caplog):
        _write(skills_dir, "one", mutate(GOOD.format(name="one")))
        with caplog.at_level("ERROR", logger=skills_m.logger.name):
            loaded = skills_m.load_skills()
        assert loaded == {}
        assert any(reason in rec.getMessage() for rec in caplog.records), [r.getMessage() for r in caplog.records]

    def test_a_bad_skill_does_not_take_a_good_one_down(self, skills_dir):
        _write(skills_dir, "good", GOOD.format(name="good"))
        _write(skills_dir, "bad", "not even frontmatter")
        assert list(skills_m.load_skills()) == ["good"]

    def test_tool_not_in_registry_is_excluded(self, skills_dir, caplog):
        _write(skills_dir, "one", GOOD.format(name="one"))
        with caplog.at_level("ERROR", logger=skills_m.logger.name):
            loaded = skills_m.load_skills(registry_ids={"something.else"}, allowed=None)
        assert loaded == {}
        assert any("not in the registry" in r.getMessage() for r in caplog.records)

    def test_tool_not_allowlisted_is_excluded_with_that_reason(self, skills_dir, caplog):
        _write(skills_dir, "one", GOOD.format(name="one"))
        with caplog.at_level("ERROR", logger=skills_m.logger.name):
            loaded = skills_m.load_skills(registry_ids={"datamodel.get_all_datamodel"}, allowed={"other.tool"})
        assert loaded == {}
        assert any("not allowlisted" in r.getMessage() for r in caplog.records)

    def test_body_tool_not_declared_is_excluded(self, skills_dir, caplog):
        text = GOOD.format(name="one").replace(
            "1. List the models (`datamodel.get_all_datamodel`).",
            "1. List the models (`datamodel.get_all_datamodel`).\n2. Then delete one (`datamodel.delete_datamodel`).",
        )
        _write(skills_dir, "one", text)
        with caplog.at_level("ERROR", logger=skills_m.logger.name):
            loaded = skills_m.load_skills(
                registry_ids={"datamodel.get_all_datamodel", "datamodel.delete_datamodel"}, allowed=None
            )
        assert loaded == {}
        assert any("does not declare" in r.getMessage() for r in caplog.records)

    def test_cache_invalidates_on_file_change(self, skills_dir):
        f = _write(skills_dir, "one", GOOD.format(name="one"))
        assert skills_m.load_skills()["one"].version == 1
        import os
        import time

        f.write_text(GOOD.format(name="one").replace("version: 1", "version: 2"), encoding="utf-8")
        os.utime(f, (time.time() + 5, time.time() + 5))  # guarantee a distinct mtime
        assert skills_m.load_skills()["one"].version == 2

    def test_cache_hit_returns_same_object(self, skills_dir):
        _write(skills_dir, "one", GOOD.format(name="one"))
        first = skills_m.load_skills()
        assert skills_m.load_skills() is first


# ---------------------------------------------------------------------------
# The directive parser
# ---------------------------------------------------------------------------
class TestDirective:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("SKILL: optimize-datamodel-for-ai-assistant", "optimize-datamodel-for-ai-assistant"),
            ("\n\n  SKILL:  one  \n", "one"),
            ("SKILL: one\n1. then a plan", "one"),
            ("1. do the thing\nSKILL: one", None),  # not first → not a directive
            ("skill: one", None),  # case matters; the planner is told the exact form
            ("SKILL: has space", None),
            ("", None),
            (None, None),
        ],
    )
    def test_parse(self, text, expected):
        assert skills_m.parse_skill_directive(text) == expected

    def test_strip_removes_only_a_leading_directive(self):
        assert skills_m.strip_skill_directive("SKILL: x\n1. a\n2. b") == "1. a\n2. b"
        assert skills_m.strip_skill_directive("1. a\nSKILL: x") == "1. a\nSKILL: x"


# ---------------------------------------------------------------------------
# The planner hook
# ---------------------------------------------------------------------------
def _llm_reply(text: str):
    return {"choices": [{"message": {"content": text}}]}


class TestPlannerHook:
    @pytest.fixture
    def one_skill(self, monkeypatch):
        skill = skills_m.Skill(
            name="one",
            description="Do the one thing.",
            version=3,
            tools=("datamodel.get_all_datamodel",),
            body="## Procedure\n1. List models (`datamodel.get_all_datamodel`).",
            path=Path("/fixture/one/SKILL.md"),
        )
        monkeypatch.setattr(A, "_load_skills_for_planner", lambda mode: {"one": skill} if mode == "chat" else {})
        monkeypatch.setattr(A, "_capability_catalog", lambda mode: "- datamodel.get_all_datamodel: List models")
        return skill

    def test_no_skills_means_prompt_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(A, "_load_skills_for_planner", lambda mode: {})
        monkeypatch.setattr(A, "_capability_catalog", lambda mode: "- x.y: z")
        llm = AsyncMock(return_value=_llm_reply("1. do x"))
        monkeypatch.setattr(A, "call_llm_raw", llm)
        steps = _run(A._make_plan("do x", "chat", [], "t"))
        assert steps == ["do x"]
        assert llm.await_count == 1
        catalog_msg = llm.await_args.args[0][1]["content"]
        assert catalog_msg == "Operation catalog:\n- x.y: z", "no skills → byte-identical catalog message"

    def test_index_is_offered_when_skills_exist(self, one_skill, monkeypatch):
        llm = AsyncMock(return_value=_llm_reply("1. do x"))
        monkeypatch.setattr(A, "call_llm_raw", llm)
        _run(A._make_plan("do x", "chat", [], "t"))
        catalog_msg = llm.await_args.args[0][1]["content"]
        assert "Procedures available" in catalog_msg
        assert "- one: Do the one thing." in catalog_msg
        assert "SKILL: <name>" in catalog_msg

    def test_directive_triggers_exactly_one_second_pass_with_the_body(self, one_skill, monkeypatch):
        plan_json = '{"steps": [{"id": "1", "tool": "datamodel.get_all_datamodel", "args": {}}]}'
        llm = AsyncMock(side_effect=[_llm_reply("SKILL: one"), _llm_reply(plan_json)])
        monkeypatch.setattr(A, "call_llm_raw", llm)
        monkeypatch.setattr(A, "TOOL_REGISTRY", {"datamodel.get_all_datamodel": {"parameters": {"type": "object"}}})

        detailed = _run(A._make_plan_detailed("do the one thing", "chat", [], "t"))
        assert detailed["skill"] is one_skill
        assert detailed["skill_plan"] == {"steps": [{"id": "1", "tool": "datamodel.get_all_datamodel", "args": {}}]}
        assert detailed["steps"], "prose rendering of the typed plan, for the UI"
        assert llm.await_count == 2
        second = llm.await_args_list[1]
        assert second.kwargs["label"] == "planner_skill"
        system_texts = [m["content"] for m in second.args[0] if m["role"] == "system"]
        assert any(one_skill.body in t for t in system_texts), "second pass must carry the skill body"
        assert any('procedure "one"' in t for t in system_texts)
        assert any("Schemas of the procedure" in t for t in system_texts), "and its tools' schemas"
        assert not any("Procedures available" in t for t in system_texts), "second pass must NOT re-offer the index"

    def test_non_json_second_pass_degrades_to_the_loop(self, one_skill, monkeypatch):
        llm = AsyncMock(side_effect=[_llm_reply("SKILL: one"), _llm_reply("1. list models\n2. done")])
        monkeypatch.setattr(A, "call_llm_raw", llm)
        detailed = _run(A._make_plan_detailed("do the one thing", "chat", [], "t"))
        assert detailed["skill"] is None and detailed["skill_plan"] is None
        assert detailed["steps"] == ["list models", "done"], "the prose still becomes an ordinary plan"

    def test_make_plan_wrapper_returns_steps_only(self, one_skill, monkeypatch):
        llm = AsyncMock(return_value=_llm_reply("1. do x"))
        monkeypatch.setattr(A, "call_llm_raw", llm)
        assert _run(A._make_plan("do x", "chat", [], "t")) == ["do x"]

    def test_unknown_skill_name_is_dropped_and_rest_planned(self, one_skill, monkeypatch):
        llm = AsyncMock(return_value=_llm_reply("SKILL: nope\n1. do x\n2. do y"))
        monkeypatch.setattr(A, "call_llm_raw", llm)
        steps = _run(A._make_plan("do x and y", "chat", [], "t"))
        assert steps == ["do x", "do y"]
        assert llm.await_count == 1, "an unknown name must not trigger a second pass"

    def test_migration_mode_never_sees_skills(self, monkeypatch):
        called = {}

        def fake_load(mode):
            called["mode"] = mode
            return {}

        monkeypatch.setattr(A, "_load_skills_for_planner", fake_load)
        monkeypatch.setattr(A, "_capability_catalog", lambda mode: "- m.t: migrate")
        llm = AsyncMock(return_value=_llm_reply("1. migrate"))
        monkeypatch.setattr(A, "call_llm_raw", llm)
        _run(A._make_plan("migrate", "migration", [], "t"))
        assert "Procedures available" not in llm.await_args.args[0][1]["content"]

    def test_real_loader_hides_skills_from_migration(self):
        assert A._load_skills_for_planner("migration") == {}


# ---------------------------------------------------------------------------
# The prompt constant the second pass uses
# ---------------------------------------------------------------------------
def test_skill_plan_prompt_states_the_contract():
    from backend.agent._prompts import SKILL_PLAN_SYSTEM_PROMPT as P

    text = P.format(name="x")
    assert '"x"' in text, "the procedure name is interpolated"
    assert "stop early" in text.lower()
    assert "args_from" in text and "for_each" in text and "when" in text
    assert re.search(r"never a guess", text, re.I)
    assert '{"steps": []}' in text, "the no-match escape hatch"
