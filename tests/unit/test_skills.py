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
        assert "Skills available" in catalog_msg
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
        assert any('Skill "one"' in t for t in system_texts)
        assert any("Schemas of the skill" in t for t in system_texts), "and its tools' schemas"
        assert not any("Skills available" in t for t in system_texts), "second pass must NOT re-offer the index"

    def test_non_json_second_pass_is_reported_not_routed(self, one_skill, monkeypatch):
        """The request matched a skill; prose from the second pass must NOT be
        handed to the ordinary loop (it once became an unrelated mutation gate).
        skill_flow gets an `error` hand-off and answers honestly."""
        llm = AsyncMock(side_effect=[_llm_reply("SKILL: one"), _llm_reply("1. list models\n2. done")])
        monkeypatch.setattr(A, "call_llm_raw", llm)
        detailed = _run(A._make_plan_detailed("do the one thing", "chat", [], "t"))
        assert detailed["skill"] is one_skill
        assert detailed["skill_plan"] == {"error": "1. list models\n2. done"}

    def test_parser_tolerates_a_raw_newline_inside_a_string(self):
        plan = A._parse_skill_plan_json(
            '{"steps": [{"id": "1", "tool": "a.b", "args_ask": {"name": "line one\nline two"}}]}'
        )
        assert plan["steps"][0]["args_ask"]["name"] == "line one\nline two"

    def test_planner_body_hides_the_code_facing_sections(self):
        sk = skills_m.Skill(
            name="x",
            description="d",
            version=1,
            tools=("a.b",),
            path=Path("/x/x/SKILL.md"),
            body="## Procedure\n1. a\n\n## Ask\nlong {a.count}\n\n## Approval\nlong {a.count}\n\n## Report\n- r",
        )
        assert sk.planner_body == "## Procedure\n1. a\n\n## Report\n- r"
        assert "## Ask" in sk.body

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
        assert "Skills available" not in llm.await_args.args[0][1]["content"]

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


class TestSkillAsksForUserInput:
    """A procedure can refuse to plan until the user supplies a value only they
    can give (a name that cannot be changed later). The planner's `ask` becomes
    a hand-off the runtime turns into a clarifying question."""

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
        monkeypatch.setattr(A, "TOOL_REGISTRY", {"datamodel.get_all_datamodel": {"parameters": {"type": "object"}}})
        return skill

    def _turn(self, tid):
        from backend.agent._config import begin_turn_output, set_current_turn

        set_current_turn(tid, "q")
        begin_turn_output(tid)

    def test_ask_becomes_a_handoff_with_attempt_one(self, one_skill, monkeypatch):
        from backend.agent._config import pop_turn_output

        self._turn("t-ask-1")
        llm = AsyncMock(
            side_effect=[_llm_reply("SKILL: one"), _llm_reply('{"ask": "What should the perspective be called?"}')]
        )
        monkeypatch.setattr(A, "call_llm_raw", llm)
        detailed = _run(A._make_plan_detailed("optimize M", "chat", [], "t-ask-1"))
        assert detailed["skill"] is one_skill
        assert detailed["skill_plan"] == {"ask": "What should the perspective be called?", "attempts": 1}
        out = pop_turn_output("t-ask-1")
        assert out["skill_handoff"] == {"skill_name": "one", "skill_plan": detailed["skill_plan"]}

    def test_attempts_carry_over_from_the_resume_path(self, one_skill, monkeypatch):
        from backend.agent._config import pop_turn_output, turn_output

        self._turn("t-ask-2")
        turn_output()["skill_clarify_attempts"] = 1  # what call_llm_with_tools sets on a skill-clarification resume
        llm = AsyncMock(side_effect=[_llm_reply("SKILL: one"), _llm_reply('{"ask": "Still: the name?"}')])
        monkeypatch.setattr(A, "call_llm_raw", llm)
        detailed = _run(A._make_plan_detailed("dunno", "chat", [], "t-ask-2"))
        assert detailed["skill_plan"]["attempts"] == 2
        assert "skill_clarify_attempts" not in pop_turn_output("t-ask-2"), "consumed, never leaks to the response"

    def test_a_real_plan_after_a_question_drops_the_counter(self, one_skill, monkeypatch):
        from backend.agent._config import pop_turn_output, turn_output

        self._turn("t-ask-3")
        turn_output()["skill_clarify_attempts"] = 1
        plan_json = '{"steps": [{"id": "1", "tool": "datamodel.get_all_datamodel", "args": {}}]}'
        llm = AsyncMock(side_effect=[_llm_reply("SKILL: one"), _llm_reply(plan_json)])
        monkeypatch.setattr(A, "call_llm_raw", llm)
        detailed = _run(A._make_plan_detailed("call it X", "chat", [], "t-ask-3"))
        assert "steps" in detailed["skill_plan"]
        assert "skill_clarify_attempts" not in pop_turn_output("t-ask-3")

    def test_parser_accepts_ask_and_steps_only(self):
        assert A._parse_skill_plan_json('{"ask": "name?"}') == {"ask": "name?"}
        assert A._parse_skill_plan_json('```json\n{"steps": []}\n```') == {"steps": []}
        assert A._parse_skill_plan_json('{"plan": 1}') is None


def test_approval_section_is_parsed_from_the_body(tmp_path):
    d = tmp_path / "with-approval"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: with-approval\ndescription: x\nversion: 1\ntools: [datamodel.get_all_datamodel]\n---\n"
        "## When this applies\nAlways.\n\n## Procedure\n1. List (`datamodel.get_all_datamodel`).\n\n"
        "## Approval\nI will list {get_all_datamodel.count} models. Approve?\n\n## Report\n- done\n",
        encoding="utf-8",
    )
    sk = skills_m.parse_skill_file(d / "SKILL.md")
    assert sk.approval == "I will list {get_all_datamodel.count} models. Approve?"
    assert "## Approval" in sk.body, "the body is left whole"


def test_ask_section_is_parsed(tmp_path):
    d = tmp_path / "asker"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: asker\ndescription: x\nversion: 1\ntools: [datamodel.get_all_datamodel]\n---\n"
        "## Procedure\n1. List (`datamodel.get_all_datamodel`).\n\n## Ask\nFound {get_all_datamodel.count}. Name?\n",
        encoding="utf-8",
    )
    assert skills_m.parse_skill_file(d / "SKILL.md").ask == "Found {get_all_datamodel.count}. Name?"


def test_parser_tolerates_python_literals():
    plan = A._parse_skill_plan_json('{"steps": [{"id": "1", "tool": "a.b", "args": {"detailed": True, "x": None}}]}')
    assert plan["steps"][0]["args"] == {"detailed": True, "x": None}


def test_step_labels_are_parsed_and_must_name_declared_tools(tmp_path):
    d = tmp_path / "labelled"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: labelled\ndescription: x\nversion: 1\ntools: [datamodel.get_all_datamodel]\n"
        "step_labels:\n  datamodel.get_all_datamodel: Listing your   data models\n---\n## Procedure\n1. List.\n",
        encoding="utf-8",
    )
    sk = skills_m.parse_skill_file(d / "SKILL.md")
    assert sk.step_labels == {"datamodel.get_all_datamodel": "Listing your data models"}
    (d / "SKILL.md").write_text(
        "---\nname: labelled\ndescription: x\nversion: 1\ntools: [datamodel.get_all_datamodel]\n"
        "step_labels:\n  dashboard.get_dashboards: nope\n---\n## Procedure\n1. List.\n",
        encoding="utf-8",
    )
    with pytest.raises(skills_m.SkillError, match="step_labels names tools not declared"):
        skills_m.parse_skill_file(d / "SKILL.md")


class TestEmptySkillsIsNeverSilent:
    """Zero skills must always say why.

    2.7.0 shipped `skills/` into the backend image with every `SKILL.md`
    filtered out by `.dockerignore`'s `**/*.md`. The loader returned {} through
    a silent early exit, so the deployed feature was indistinguishable from a
    planner that simply chose not to use it: no error, no warning, nothing in
    llm_skills.log but the logger's own init line. Every empty outcome is now
    announced once.
    """

    @staticmethod
    def _captured(monkeypatch, tmp_path, **env):
        import logging

        from backend.agent import _skills

        records: list[str] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler = _Collect(level=logging.WARNING)
        monkeypatch.setattr(_skills, "_last_empty_reason", None, raising=False)
        for key, value in env.items():
            monkeypatch.setattr(_skills, key, value, raising=False)
        _skills.logger.addHandler(handler)
        try:
            result = _skills.load_skills()
            second = _skills.load_skills()
        finally:
            _skills.logger.removeHandler(handler)
        return result, second, records

    def test_directory_present_but_no_skill_files_warns(self, monkeypatch, tmp_path):
        (tmp_path / "some-skill").mkdir()
        result, _, records = self._captured(monkeypatch, tmp_path, SKILLS_DIR=tmp_path, SKILLS_ENABLED=True)
        assert result == {}
        assert any("contains no SKILL.md" in r for r in records), records

    def test_missing_directory_warns(self, monkeypatch, tmp_path):
        gone = tmp_path / "not-here"
        result, _, records = self._captured(monkeypatch, tmp_path, SKILLS_DIR=gone, SKILLS_ENABLED=True)
        assert result == {}
        assert any("does not exist" in r for r in records), records

    def test_disabled_warns(self, monkeypatch, tmp_path):
        result, _, records = self._captured(monkeypatch, tmp_path, SKILLS_DIR=tmp_path, SKILLS_ENABLED=False)
        assert result == {}
        assert any("disabled" in r for r in records), records

    def test_the_reason_is_logged_once_not_every_turn(self, monkeypatch, tmp_path):
        (tmp_path / "some-skill").mkdir()
        _, _, records = self._captured(monkeypatch, tmp_path, SKILLS_DIR=tmp_path, SKILLS_ENABLED=True)
        assert len(records) == 1, f"a per-turn warning would flood the log: {records}"
