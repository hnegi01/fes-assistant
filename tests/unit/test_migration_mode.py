"""
Unit tests for migration mode through the V2 agentic loop.

Migration mode predates the V2 loop and had no coverage of how the two meet.
Everything here is about that seam (LLM + router mocked; no creds, no network):

  - mode filtering: the planner's capability catalog is migration-only in
    migration mode, and migration-free in chat mode
  - routing bypass: migration mode skips two-stage navigation and loads all
    migration tools in one shot
  - fan-out: independent steps never run concurrently in migration mode
  - the gate: every migration tool mutates, so nothing runs unapproved (the
    plan-level approval itself is covered in test_migration_flow.py)
  - dependency ordering: the mode context prompt states groups → users →
    datamodels → dashboards, and the planner receives it
  - approval disclosure: the unset optional settings the schema declares are
    surfaced in code on every path, and nothing about scope is claimed
  - engine parity: the same behaviour under FES_AGENT_ENGINE=langgraph
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

import backend.agent.llm_agent as m
from backend.agent._prompts import MIGRATION_PLAN_SYSTEM_PROMPT, MIGRATION_PLANNING_CONTEXT_PROMPT

MIGRATE_GROUPS_SCHEMA = {
    "type": "object",
    "properties": {"group_name_list": {"type": "array", "items": {"type": "string"}}},
    "required": ["group_name_list"],
}
MIGRATE_USERS_SCHEMA = {
    "type": "object",
    "properties": {"user_name_list": {"type": "array", "items": {"type": "string"}}},
    "required": ["user_name_list"],
}
MIGRATE_DASHBOARDS_SCHEMA = {
    "type": "object",
    "properties": {
        "dashboard_ids": {"type": "array", "items": {"type": "string"}},
        "dashboard_names": {"type": "array", "items": {"type": "string"}},
        "action": {"type": "string", "enum": ["skip", "overwrite", "duplicate"]},
        "republish": {"type": "boolean"},
    },
    "required": [],
}
GET_USER_SCHEMA = {
    "type": "object",
    "properties": {"user_email": {"type": "string"}},
    "required": ["user_email"],
}


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    monkeypatch.setattr(
        m,
        "TOOL_REGISTRY",
        {
            "migration.migrate_groups": {
                "module": "migration",
                "method": "migrate_groups",
                "mutates": True,
                "description": "Migrate specific groups from source to target.",
                "parameters": MIGRATE_GROUPS_SCHEMA,
            },
            "migration.migrate_users": {
                "module": "migration",
                "method": "migrate_users",
                "mutates": True,
                "description": "Migrate specific users from source to target.",
                "parameters": MIGRATE_USERS_SCHEMA,
            },
            "migration.migrate_dashboards": {
                "module": "migration",
                "method": "migrate_dashboards",
                "mutates": True,
                "description": "Migrate dashboards from source to target.",
                "parameters": MIGRATE_DASHBOARDS_SCHEMA,
            },
            "migration.migrate_all_users": {
                "module": "migration",
                "method": "migrate_all_users",
                "mutates": True,
                "description": "Migrate all users from source to target.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            "access_management.get_user": {
                "module": "access_management",
                "method": "get_user",
                "mutates": False,
                "description": "Retrieve a user by email.",
                "parameters": GET_USER_SCHEMA,
            },
        },
    )
    monkeypatch.setattr(m, "ALLOW_SUMMARIZATION", True)
    monkeypatch.setattr(m, "REQUIRE_MUTATION_CONFIRM", True)
    monkeypatch.setattr(m, "VERIFY_GOAL", False)
    # Off by default here: these tests pin exact LLM response lists, and the
    # completeness check consumes one. Its own behaviour is covered in
    # TestCompletenessCheck (test_migration_flow.py).
    monkeypatch.setattr(m, "MIGRATION_COMPLETENESS_CHECK", False)


@pytest.fixture(autouse=True)
def reset_pending():
    m.LAST_PENDING_LOOP = None
    m.LAST_PENDING_CLARIFICATION = None
    m.LAST_TOOL_RESULT = None
    m.LAST_STEP_RESULTS = []
    yield
    m.LAST_PENDING_LOOP = None
    m.LAST_PENDING_CLARIFICATION = None
    m.LAST_TOOL_RESULT = None
    m.LAST_STEP_RESULTS = []


def _tool_def(name, schema):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": schema}}


def _plan_resp(tool_id, arguments_json, call_id="c1"):
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": call_id, "function": {"name": tool_id, "arguments": arguments_json}}],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _text_resp(text):
    return {
        "choices": [{"message": {"content": text, "tool_calls": []}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _plan_call(tool_id, args, cid="c1"):
    return {"id": cid, "type": "function", "function": {"name": tool_id, "arguments": json.dumps(args)}}


def mf_plan_key(calls):
    """Approval key for a whole migration plan (see migration_flow)."""
    import backend.agent.migration_flow as mf

    return m._approval_key(mf.PLAN_TOOL_ID, mf.plan_arguments(calls))


MIGRATION_TOOLS = [
    _tool_def("migration.migrate_groups", MIGRATE_GROUPS_SCHEMA),
    _tool_def("migration.migrate_users", MIGRATE_USERS_SCHEMA),
    _tool_def("migration.migrate_dashboards", MIGRATE_DASHBOARDS_SCHEMA),
]


@pytest.fixture(params=["custom", "langgraph"])
def engine(request, monkeypatch):
    monkeypatch.setenv("FES_AGENT_ENGINE", request.param)
    return request.param


# ---------------------------------------------------------------------------
# 1) Mode filtering — the planner's catalog
# ---------------------------------------------------------------------------
class TestCapabilityCatalogIsModeFiltered:
    def test_migration_mode_lists_only_migration_tools(self):
        catalog = m._capability_catalog("migration")
        assert "migration.migrate_groups" in catalog
        assert "access_management.get_user" not in catalog

    def test_chat_mode_excludes_migration_tools(self):
        catalog = m._capability_catalog("chat")
        assert "access_management.get_user" in catalog
        assert "migration.migrate_groups" not in catalog

    def test_every_migration_tool_appears(self):
        catalog = m._capability_catalog("migration")
        expected = [t for t, meta in m.TOOL_REGISTRY.items() if meta["module"] == "migration"]
        for tool_id in expected:
            assert tool_id in catalog


# ---------------------------------------------------------------------------
# 2) The dependency-ordering rule lives in the mode context prompt
# ---------------------------------------------------------------------------
class TestDependencyOrdering:
    """The rule is stated as a PRINCIPLE — "migrate what is referenced before
    what references it" — not a ranked list of tool names, and it lives ONCE, as
    a primary instruction in MIGRATION_PLAN_SYSTEM_PROMPT. History: while it sat
    only in the secondary context prompt, it lost to the borrowed single-tool
    selection prompt 12 runs out of 12."""

    @staticmethod
    def _flat(text):
        return " ".join(text.lower().split())

    def test_the_plan_prompt_states_the_principle(self):
        text = self._flat(MIGRATION_PLAN_SYSTEM_PROMPT)
        assert "migrate what is referenced before whatever references it" in text

    def test_the_plan_prompt_judges_by_meaning_not_name(self):
        """Without this a new tool falls outside the rule entirely."""
        text = self._flat(MIGRATION_PLAN_SYSTEM_PROMPT)
        assert "by what it actually moves, not by its name" in text
        assert "a tool you have not seen before" in text

    def test_the_plan_prompt_gives_the_reference_directions(self):
        text = self._flat(MIGRATION_PLAN_SYSTEM_PROMPT)
        assert "a user is assigned to groups" in text
        assert "a dashboard queries a datamodel" in text
        assert "granted to users and groups" in text

    def test_the_plan_prompt_forbids_inventing_asset_kinds(self):
        assert "do not emit a call for anything they did not ask for" in self._flat(MIGRATION_PLAN_SYSTEM_PROMPT)

    def test_the_rule_is_stated_exactly_once(self):
        """One prompt, everything once. The earlier duplicated arrangement was
        defended on 6-12-run samples that were not statistically significant."""
        text = self._flat(MIGRATION_PLAN_SYSTEM_PROMPT)
        assert text.count("migrate what is referenced before whatever references it") == 1
        assert "referenced" not in self._flat(MIGRATION_PLANNING_CONTEXT_PROMPT)


# ---------------------------------------------------------------------------
# 3) Routing bypass — migration mode never calls the two-stage navigator
# ---------------------------------------------------------------------------
def test_migration_mode_skips_two_stage_routing(engine):
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {"migrated": 1}})
    args = {"group_name_list": ["Sales Team"]}
    key = m._approval_key("migration.migrate_groups", args)

    nav = AsyncMock()
    raw = AsyncMock(
        side_effect=[
            _plan_resp("migration.migrate_groups", json.dumps(args)),
            _text_resp("Migrated the Sales Team group."),
        ]
    )
    # Migration approves the whole PLAN, not a single tool (migration_flow).
    key = mf_plan_key([_plan_call("migration.migrate_groups", args)])

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    with (
        patch.object(m, "_navigate_to_tools", new=nav),
        patch.object(m, "call_llm_raw", new=raw),
        patch.object(m, "_make_plan", new=_one_step_plan),
        patch.object(m, "_load_all_package_tools", return_value=MIGRATION_TOOLS) as load_all,
    ):
        reply = run(
            m.call_llm_with_tools(
                [{"role": "user", "content": "migrate the Sales Team group"}],
                MIGRATION_TOOLS,
                client,
                approved_mutations={key},
                allow_summarization=True,
            )
        )

    nav.assert_not_called(), "migration mode must not run L1/L2 navigation"
    load_all.assert_not_called(), "the 9 tools are already in hand — no need to re-read the registry tree"
    # The tool-selection call got exactly the turn's scoped tool list.
    assert raw.await_args_list[0].kwargs["tools"] == MIGRATION_TOOLS
    assert reply == "Migrated the Sales Team group."


def test_turn_tool_list_is_rescoped_to_the_mode(engine):
    """_select_tools_for_mode falls back to ALL tools when its filter comes up
    empty (a broken-registry safety valve). Migration mode must not inherit chat
    tools from that fallback, so the loop re-filters what it was handed."""
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {"migrated": 1}})
    contaminated = MIGRATION_TOOLS + [_tool_def("access_management.get_user", GET_USER_SCHEMA)]
    args = {"group_name_list": ["Sales Team"]}

    raw = AsyncMock(
        side_effect=[
            _plan_resp("migration.migrate_groups", json.dumps(args)),
            _text_resp("Migrated the Sales Team group."),
        ]
    )

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    with (
        patch.object(m, "_navigate_to_tools", new=AsyncMock()),
        patch.object(m, "call_llm_raw", new=raw),
        patch.object(m, "_make_plan", new=_one_step_plan),
    ):
        run(
            m.call_llm_with_tools(
                [{"role": "user", "content": "migrate the Sales Team group"}],
                contaminated,
                client,
                approved_mutations={m._approval_key("migration.migrate_groups", args)},
                allow_summarization=True,
            )
        )

    offered = [t["function"]["name"] for t in raw.await_args_list[0].kwargs["tools"]]
    assert "access_management.get_user" not in offered
    assert offered == [t["function"]["name"] for t in MIGRATION_TOOLS]


# ---------------------------------------------------------------------------
# 4) Fan-out is disabled in migration mode
# ---------------------------------------------------------------------------
_FANOUT_LOG = "Fan-out: running"


def test_migration_mode_never_fans_out(engine, monkeypatch, caplog):
    """Two independent plan steps fan out in chat mode. In migration mode they
    must not: every migration tool mutates, and the gate is one-at-a-time."""
    monkeypatch.setattr(m, "MAX_PARALLEL_STEPS", 3)
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {"migrated": 1}})

    async def _two_step_plan(user_text, mode, history, trace_id):
        return ["migrate the Sales Team group", "migrate the dashboards"]

    raw = AsyncMock(
        side_effect=[
            _plan_resp("migration.migrate_groups", '{"group_name_list":["Sales Team"]}'),
            _text_resp("This will migrate the Sales Team group."),  # gate explanation
        ]
    )

    with caplog.at_level("INFO"):
        with (
            patch.object(m, "call_llm_raw", new=raw),
            patch.object(m, "_make_plan", new=_two_step_plan),
            patch.object(m, "_load_all_package_tools", return_value=MIGRATION_TOOLS),
        ):
            run(
                m.call_llm_with_tools(
                    [{"role": "user", "content": "migrate the Sales Team group and the dashboards"}],
                    MIGRATION_TOOLS,
                    client,
                    allow_summarization=True,
                )
            )

    assert _FANOUT_LOG not in caplog.text, "migration mode never runs steps concurrently"
    # Gated as a whole plan before anything runs.
    client.invoke_tool.assert_not_awaited()
    assert m.LAST_PENDING_LOOP["plan"], "the plan awaits approval"


def test_chat_mode_does_fan_out(engine, monkeypatch, caplog):
    """Control for the test above: the same two-step shape in chat mode DOES
    fan out, so the assertion there is capable of failing."""
    monkeypatch.setattr(m, "MAX_PARALLEL_STEPS", 3)
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {"email": "a@b.com"}})
    read_tools = [_tool_def("access_management.get_user", GET_USER_SCHEMA)]

    async def _two_step_plan(user_text, mode, history, trace_id):
        return ["look up user a@b.com", "look up user c@d.com"]

    with caplog.at_level("INFO"):
        with (
            patch.object(
                m, "_navigate_to_tools", new=AsyncMock(return_value=(read_tools, "access_management", "users", 0))
            ),
            patch.object(
                m,
                "call_llm_raw",
                new=AsyncMock(return_value=_plan_resp("access_management.get_user", '{"user_email":"a@b.com"}')),
            ),
            patch.object(m, "_make_plan", new=_two_step_plan),
        ):
            run(
                m.call_llm_with_tools(
                    [{"role": "user", "content": "look up two users"}],
                    read_tools,
                    client,
                    allow_summarization=True,
                )
            )

    assert _FANOUT_LOG in caplog.text


# ---------------------------------------------------------------------------
# 5) The gate: two mutations in one plan → two separate approvals
# ---------------------------------------------------------------------------
def test_first_migration_step_pauses_for_approval(engine):
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {}})

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    raw = AsyncMock(
        side_effect=[
            _plan_resp("migration.migrate_groups", '{"group_name_list":["Sales Team"]}'),
            _text_resp("This will copy the Sales Team group to the target environment."),
        ]
    )
    with (
        patch.object(m, "call_llm_raw", new=raw),
        patch.object(m, "_make_plan", new=_one_step_plan),
        patch.object(m, "_load_all_package_tools", return_value=MIGRATION_TOOLS),
    ):
        reply = run(
            m.call_llm_with_tools(
                [{"role": "user", "content": "migrate the Sales Team group"}],
                MIGRATION_TOOLS,
                client,
                allow_summarization=True,
            )
        )

    client.invoke_tool.assert_not_awaited(), "nothing may run before approval"
    # One approval covers the plan; migration_flow owns the details.
    assert m.LAST_TOOL_RESULT["pending_confirmation"]["tool_id"] == "migration.plan"
    steps = m.LAST_TOOL_RESULT["pending_confirmation"]["arguments"]["steps"]
    assert [s["tool"] for s in steps] == ["migration.migrate_groups"]
    assert "Sales Team" in reply


# ---------------------------------------------------------------------------
# 5b) The mode boundary is structural, not conventional
# ---------------------------------------------------------------------------
class TestModeBoundaryIsEnforcedAtExecution:
    """Filtering at each call site is a convention any new path can forget —
    and one already did. The execution choke point refuses off-mode tools
    outright, so migration mode can only ever reach the 9 migration tools."""

    @pytest.mark.parametrize(
        "tool_id,mode,expected",
        [
            ("migration.migrate_groups", "migration", True),
            ("migration.migrate_groups", "chat", False),
            ("access_management.get_user", "chat", True),
            ("access_management.get_user", "migration", False),
        ],
    )
    def test_reachability(self, tool_id, mode, expected):
        assert m._tool_matches_mode(tool_id, mode) is expected

    def test_chat_tool_is_refused_in_migration_mode(self):
        client = AsyncMock()
        client.invoke_tool = AsyncMock(return_value={"ok": True, "result": []})
        result = run(m._invoke_tool_traced(client, "access_management.get_user", {"user_email": "a@b"}, "migration"))

        client.invoke_tool.assert_not_awaited(), "the call must never reach the MCP client"
        assert result["ok"] is False
        assert "not available in migration mode" in result["error"]

    def test_migration_tool_is_refused_in_chat_mode(self):
        client = AsyncMock()
        client.invoke_tool = AsyncMock(return_value={"ok": True, "result": []})
        result = run(m._invoke_tool_traced(client, "migration.migrate_groups", {}, "chat"))

        client.invoke_tool.assert_not_awaited()
        assert result["ok"] is False

    def test_matching_tool_executes_normally(self):
        client = AsyncMock()
        client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {"migrated": 1}})
        result = run(m._invoke_tool_traced(client, "migration.migrate_groups", {}, "migration"))

        client.invoke_tool.assert_awaited_once()
        assert result["ok"] is True

    def test_keyword_fallback_never_runs_a_chat_tool_in_migration_mode(self):
        """The planning-failure fallback calls the MCP client directly, so it
        bypasses the execution guard entirely. In migration mode 'migrate all
        users' matches its "user" keyword and would have run a chat tool with
        no credentials attached."""
        client = AsyncMock()
        client.invoke_tool = AsyncMock(return_value={"ok": True, "result": []})
        summary, result = run(m._fallback_direct_tool("migrate all users", client, "migration"))

        client.invoke_tool.assert_not_awaited()
        assert result["ok"] is False
        assert result["error_type"] == "PlanningFailed"
        assert "no safe fallback in migration mode" in summary

    def test_keyword_fallback_still_works_in_chat_mode(self):
        client = AsyncMock()
        client.invoke_tool = AsyncMock(return_value={"ok": True, "result": [{"id": 1}]})
        _summary, result = run(m._fallback_direct_tool("show me all users", client, "chat"))

        client.invoke_tool.assert_awaited_once()
        assert client.invoke_tool.await_args.args[0] == "access_management.get_users_all"
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# 6) Approval disclosure — optional selectors still get surfaced
# ---------------------------------------------------------------------------
class TestApprovalDisclosure:
    DASH = "migration.migrate_dashboards"

    def _meta(self, tool_id):
        return m.TOOL_REGISTRY[tool_id]

    def test_unset_options_are_listed_with_their_allowed_values(self):
        note = m._approval_disclosure(self.DASH, self._meta(self.DASH), {"dashboard_names": ["Sales"]})
        assert "`action`" in note
        assert "skip / overwrite / duplicate" in note

    def test_unset_target_params_are_listed_as_the_facts_they_are(self):
        """The schema marks them optional and this call left them unset. That is
        stated; what the operation does about it is not, because no tool
        definition says."""
        note = m._options_note(self.DASH, self._meta(self.DASH), {})
        assert "`dashboard_ids`" in note and "`dashboard_names`" in note

    def test_no_claim_is_made_about_scope(self):
        for args in ({}, {"dashboard_names": ["Sales"]}):
            note = m._approval_disclosure(self.DASH, self._meta(self.DASH), args).lower()
            for banned in ("every", "all of them", "target list", "cannot be narrowed", "⚠️"):
                assert banned not in note, f"disclosure must not predict scope: {banned!r}"

    def test_both_surfaces_offer_the_same_params(self):
        """The clarification question renders inline and the approval dialog
        renders a block — different layouts, but they must never disagree about
        WHICH params they name, or the feature has drifted into two features."""
        schema = self._meta(self.DASH)["parameters"]
        specs = m._optional_specs(schema, {}, m._OPTIONALS_IN_APPROVAL)
        inline = m._optionals_inline(specs)
        block = m._options_note(self.DASH, self._meta(self.DASH), {})
        for name, _enum, _summary in specs:
            assert f"`{name}`" in inline
            assert f"`{name}`" in block

    def test_enum_values_appear_in_both_renderings(self):
        specs = m._optional_specs(self._meta(self.DASH)["parameters"], {}, m._OPTIONALS_IN_APPROVAL)
        assert "skip / overwrite / duplicate" in m._optionals_inline(specs)
        assert "skip / overwrite / duplicate" in m._optionals_block(specs)

    def test_rst_markup_is_stripped_from_descriptions(self):
        """SDK docstrings carry ``literal`` markup that leaked verbatim into
        clarification questions."""
        assert m._clean_desc("When ``True``, request with ``force=true``.") == (
            "When 'True', request with 'force=true'."
        )

    def test_bulk_tool_with_no_params_gets_no_disclosure(self):
        """migrate_all_users takes no arguments at all. Nothing in its
        definition supports a scope claim, so nothing is said."""
        tid = "migration.migrate_all_users"
        assert m._approval_disclosure(tid, self._meta(tid), {}) == ""

    def test_fully_specified_call_gets_no_disclosure(self):
        tid = "migration.migrate_groups"
        assert m._approval_disclosure(tid, self._meta(tid), {"group_name_list": ["Sales Team"]}) == ""

    def test_disclosure_survives_an_llm_failure(self):
        """The fallback template must carry the disclosure too — that is the
        path taken exactly when something is already going wrong."""
        with patch.object(m, "call_llm_raw", new=AsyncMock(side_effect=RuntimeError("boom"))):
            text = run(
                m._generate_mutation_explanation(self.DASH, self._meta(self.DASH), {"dashboard_names": ["S"]}, "t")
            )
        assert "Optional settings, not set" in text
        assert "`action`" in text

    def test_disclosure_is_appended_to_the_llm_explanation(self):
        with patch.object(m, "call_llm_raw", new=AsyncMock(return_value=_text_resp("This migrates dashboards."))):
            text = run(
                m._generate_mutation_explanation(self.DASH, self._meta(self.DASH), {"dashboard_names": ["S"]}, "t")
            )
        assert text.startswith("This migrates dashboards.")
        assert "Optional settings, not set" in text

    def test_absent_target_list_is_not_warned_about_here(self):
        """It is an SDK precondition, not a scope choice: the turn asks WHICH
        dashboards long before an approval dialog exists. Warning 'this will run
        without a target list' would describe something that never happens."""
        note = m._approval_disclosure(self.DASH, self._meta(self.DASH), {})
        assert "target list" not in note.lower()
        assert "⚠️" not in note

    def test_block_lines_carry_the_schema_description_and_default(self):
        """The block states what a setting does and what the SDK does without
        it — both read straight from the generated schema, never invented."""
        meta = {
            "parameters": {
                "type": "object",
                "properties": {
                    "republish": {
                        "type": "boolean",
                        "description": "Whether to republish dashboards after migration. Default: False.",
                    }
                },
                "required": [],
            }
        }
        note = m._options_note("migration.migrate_all_dashboards", meta, {})
        assert "Whether to republish dashboards after migration" in note
        assert "Default: False" in note

    def test_default_sentence_is_pulled_forward(self):
        assert m._param_summary(
            {"description": "Whether to republish dashboards after migration. Default: False."}
        ) == ("Whether to republish dashboards after migration. Default: False")
        assert m._param_summary({"description": "Only the first sentence. The rest is manual prose."}) == (
            "Only the first sentence"
        )
        assert m._param_summary({}) == ""

    def test_example_query_is_offered_as_a_phrasing_hint(self):
        """examples[0].user_query is curated natural language — shown as 'how
        you could re-ask', never as a claim about what this call will do."""
        query = "Migrate dashboards d1 and d2 but skip any that already exist?"
        meta = {
            **self._meta(self.DASH),
            "examples": [{"user_query": query, "arguments": {"action": "skip"}}],
        }
        note = m._options_note(self.DASH, meta, {})
        assert "For example, you could ask:" in note
        assert query in note

    def test_no_example_means_no_hint(self):
        meta = {"parameters": self._meta(self.DASH)["parameters"]}  # same schema, no examples key
        note = m._options_note(self.DASH, meta, {})
        assert note  # optionals still listed
        assert "For example" not in note

    def test_heading_override_attributes_the_block(self):
        heading = "**Optional settings for step 3 — Migrates all dashboards (not set)**"
        note = m._options_note(self.DASH, self._meta(self.DASH), {}, heading=heading)
        assert note.startswith(heading)
        assert "Optional settings, not set" not in note

    def test_inline_rendering_stays_compact(self):
        """Clarification questions keep names + enums only — descriptions and
        example hints belong to the dialog, where the user is deciding."""
        schema = {
            "type": "object",
            "properties": {"republish": {"type": "boolean", "description": "Whether to republish. Default: False."}},
            "required": [],
        }
        specs = m._optional_specs(schema, {}, m._OPTIONALS_IN_APPROVAL)
        inline = m._optionals_inline(specs)
        assert "`republish`" in inline
        assert "Default:" not in inline
        assert "For example" not in inline


# ---------------------------------------------------------------------------
# 7) Filled-value semantics
# ---------------------------------------------------------------------------
class TestIsFilled:
    """Drives which optional params get reported as unset."""

    @pytest.mark.parametrize("value", [None, "", "   ", [], {}])
    def test_unfilled_values(self, value):
        assert m._is_filled(value) is False

    @pytest.mark.parametrize("value", ["Sales", ["a"], {"k": "v"}, 0, False])
    def test_filled_values(self, value):
        assert m._is_filled(value) is True

    def test_credentials_are_never_offered_as_options(self):
        schema = {"properties": {"source_domain": {}, "action": {}}, "required": []}
        assert [n for n, *_ in m._optional_specs(schema, {}, 6)] == ["action"]


# ---------------------------------------------------------------------------
# 8) Invalid argument values are still a hard block in migration mode
# ---------------------------------------------------------------------------
def test_invalid_enum_value_blocks_instead_of_running(engine):
    client = AsyncMock()
    client.invoke_tool = AsyncMock(return_value={"ok": True, "result": {}})

    async def _one_step_plan(user_text, mode, history, trace_id):
        return [user_text]

    raw = AsyncMock(
        side_effect=[_plan_resp("migration.migrate_dashboards", '{"dashboard_ids":["d1"],"action":"replace"}')]
    )
    with (
        patch.object(m, "call_llm_raw", new=raw),
        patch.object(m, "_make_plan", new=_one_step_plan),
        patch.object(m, "_load_all_package_tools", return_value=MIGRATION_TOOLS),
    ):
        reply = run(
            m.call_llm_with_tools(
                [{"role": "user", "content": "migrate dashboard d1 and replace existing ones"}],
                MIGRATION_TOOLS,
                client,
                allow_summarization=True,
            )
        )

    client.invoke_tool.assert_not_awaited(), "an invalid enum must never reach the SDK"
    # Wording-agnostic: the reply must name the bad value and the allowed ones.
    assert "replace" in reply and "skip" in reply and "overwrite" in reply
