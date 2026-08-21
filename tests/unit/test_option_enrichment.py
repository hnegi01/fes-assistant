"""
Clarification option menus (x-options-tool) and follow-up nudges (x-followup).

Both are SCHEMA_RULES-curated, code-consumed per-tool facts:
  - x-options-tool on a param: when that param is missing, the CLARIFICATION
    path runs the named READ tool. The question TEXT carries only the count
    and a list-on-request offer — in EVERY summarization mode, because the
    question is an assistant message and message content re-enters LLM prompts
    via history. The example NAMES go to `display_hints`: a screen-only
    channel the UI renders under the reply and never stores in content.
  - x-followup on a tool: after a successful run, a deploy-style nudge is
    appended to the final reply in code — suggested, never executed.

Guard tests pin the curated data against the shipped registry (a renamed
lookup tool fails here, not in a user's clarification); behavior tests pin
the mechanism against a fixture registry.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

import backend.agent._registry as registry_m
import backend.agent.llm_agent as m
from backend.agent._config import begin_turn_output, pop_turn_output, set_current_turn
from backend.agent._routing import strip_internal_params

# ---------------------------------------------------------------------------
# fixture registry (behavior tests)
# ---------------------------------------------------------------------------

SETUP_SCHEMA = {
    "type": "object",
    "properties": {
        "datamodel_name": {"type": "string", "description": "Name of the data model."},
        "connection_name": {
            "type": "string",
            "description": "Name of the connection to use.",
            "x-options-tool": "datamodel.get_connections",
            "x-options-note": "Or let me know if you want to create a new connection first.",
        },
    },
    "required": ["datamodel_name", "connection_name"],
}

FIXTURE_REGISTRY = {
    "datamodel.setup_datamodel": {
        "tool_id": "datamodel.setup_datamodel",
        "module": "datamodel",
        "mutates": True,
        "description": "Set up a data model end to end using an existing connection.",
        "parameters": SETUP_SCHEMA,
        "x-followup": {
            "note": "The model isn't queryable until it's built or published.",
            "ask_template": "Deploy '{datamodel_name}'.",
        },
    },
    "datamodel.get_connections": {
        "tool_id": "datamodel.get_connections",
        "module": "datamodel",
        "mutates": False,
        "description": "Retrieve all connections.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "datamodel.create_dataset": {
        "tool_id": "datamodel.create_dataset",
        "module": "datamodel",
        "mutates": True,
        "description": "Create a new dataset.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch):
    monkeypatch.setattr(m, "TOOL_REGISTRY", dict(FIXTURE_REGISTRY))


class _FakeClient:
    """Just enough of McpClient for _invoke_tool_traced: an async invoke_tool."""

    def __init__(self, result):
        self.invoke_tool = AsyncMock(return_value=result)


CONNECTIONS = {"ok": True, "result": [{"name": "PROD_SNOWFLAKE"}, {"name": "STAGING_PG"}]}
SETUP_META = FIXTURE_REGISTRY["datamodel.setup_datamodel"]


# ---------------------------------------------------------------------------
# guards over the shipped curated data
# ---------------------------------------------------------------------------


def _shipped_registry():
    return json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))


def _allowed_ids():
    text = (registry_m.REGISTRY_PATH.parent / "allowed_tools.txt").read_text(encoding="utf-8")
    ids = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


class TestCuratedDataGuards:
    def test_every_options_tool_is_a_real_allowlisted_read_tool(self):
        rows = _shipped_registry()
        by_id = {r["tool_id"]: r for r in rows}
        allowed = _allowed_ids()
        found = 0
        for row in rows:
            props = (row.get("parameters") or {}).get("properties") or {}
            for pname, prop in props.items():
                opt = (prop or {}).get("x-options-tool")
                if not opt:
                    continue
                found += 1
                where = f"{row['tool_id']}.{pname}"
                assert opt in by_id, f"{where}: x-options-tool {opt!r} is not in the registry"
                assert not by_id[opt].get("mutates"), f"{where}: x-options-tool {opt!r} mutates — lookups must be reads"
                assert opt in allowed, f"{where}: x-options-tool {opt!r} is not allowlisted"
                assert by_id[opt].get("module") != "migration", (
                    f"{where}: x-options-tool {opt!r} is a migration tool — unreachable from the chat clarify path"
                )
        assert found >= 2, "expected setup_datamodel and create_dataset to carry x-options-tool"

    def test_every_followup_template_uses_only_required_params(self):
        import string

        found = 0
        for row in _shipped_registry():
            followup = row.get("x-followup")
            if not followup:
                continue
            found += 1
            assert isinstance(followup, dict), f"{row['tool_id']}: x-followup must be a dict"
            template = followup.get("ask_template") or ""
            fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
            required = set((row.get("parameters") or {}).get("required") or [])
            assert fields <= required, (
                f"{row['tool_id']}: ask_template references {sorted(fields - required)} — not required params of "
                "the tool, so a successful call may lack them and the hint silently drops"
            )
        assert found >= 3, "expected setup_datamodel, create_dataset and create_table to carry x-followup"

    def test_model_boundary_strips_x_options_keys(self):
        stripped = strip_internal_params(SETUP_SCHEMA)
        prop = stripped["properties"]["connection_name"]
        assert "x-options-tool" not in prop and "x-options-note" not in prop
        assert "x-options-tool" in SETUP_SCHEMA["properties"]["connection_name"], "must copy, not mutate"

    def test_x_aliases_survive_the_strip(self):
        params = {"type": "object", "properties": {"t": {"type": "string", "x-aliases": {"a": ["b"]}}}}
        assert strip_internal_params(params)["properties"]["t"]["x-aliases"] == {"a": ["b"]}


# ---------------------------------------------------------------------------
# clarification option menus
# ---------------------------------------------------------------------------


class TestClarificationOptionMenu:
    def _with_turn(self, trace_id):
        set_current_turn(trace_id, "q")
        begin_turn_output(trace_id)

    def test_names_go_to_display_hints_never_the_question_text(self):
        """The question text carries only the count (metadata) in EVERY mode;
        the names ride the screen-only display_hints channel."""
        self._with_turn("t-opts-1")
        client = _FakeClient(CONNECTIONS)
        question = run(
            m._generate_clarification_question(
                "datamodel.setup_datamodel", SETUP_META, ["connection_name"], {}, "t", mcp_client=client, mode="chat"
            )
        )
        assert "PROD_SNOWFLAKE" not in question and "STAGING_PG" not in question
        assert "I found 2 existing options for the connection name" in question
        assert "full list first" in question
        assert "create a new connection" in question  # the x-options-note
        # Bullets stay clean; the count paragraph comes after them (UI feedback
        # 2026-08-20), and the stiff tool-description header is gone.
        assert question.index("- Name of the connection to use") < question.index("I found 2 existing")
        assert "end to end" not in question
        client.invoke_tool.assert_awaited_once()
        assert client.invoke_tool.await_args.args[0] == "datamodel.get_connections"
        hints = pop_turn_output("t-opts-1")["display_hints"]
        assert hints == ["A few existing options for the connection name: `PROD_SNOWFLAKE`, `STAGING_PG` (of 2 total)."]

    def test_display_hint_caps_at_three_names(self):
        self._with_turn("t-opts-2")
        rows = [{"name": f"CONN_{i}"} for i in range(12)]
        client = _FakeClient({"ok": True, "result": rows})
        question = run(
            m._generate_clarification_question(
                "datamodel.setup_datamodel", SETUP_META, ["connection_name"], {}, "t", mcp_client=client, mode="chat"
            )
        )
        assert "I found 12 existing options" in question and "CONN_0" not in question
        (hint,) = pop_turn_output("t-opts-2")["display_hints"]
        assert "(of 12 total)" in hint
        assert "`CONN_2`" in hint and "CONN_3" not in hint

    def test_no_turn_context_still_renders_the_count(self):
        # Bare call with no turn slot (e.g. from a non-request context): the
        # hint recording no-ops, the question is unaffected.
        client = _FakeClient(CONNECTIONS)
        question = run(
            m._generate_clarification_question(
                "datamodel.setup_datamodel", SETUP_META, ["connection_name"], {}, "t", mcp_client=client, mode="chat"
            )
        )
        assert "I found 2 existing options" in question
        assert "PROD_SNOWFLAKE" not in question

    def test_no_client_means_plain_question_and_no_lookup(self):
        question = run(
            m._generate_clarification_question("datamodel.setup_datamodel", SETUP_META, ["connection_name"], {}, "t")
        )
        assert "I found" not in question
        assert "I need a bit more information" in question

    def test_empty_lookup_result_adds_no_line(self):
        client = _FakeClient({"ok": True, "result": []})
        question = run(
            m._generate_clarification_question(
                "datamodel.setup_datamodel", SETUP_META, ["connection_name"], {}, "t", mcp_client=client, mode="chat"
            )
        )
        assert "I found" not in question

    def test_lookup_failure_degrades_to_plain_question(self):
        client = _FakeClient(None)
        client.invoke_tool = AsyncMock(side_effect=RuntimeError("boom"))
        question = run(
            m._generate_clarification_question(
                "datamodel.setup_datamodel", SETUP_META, ["connection_name"], {}, "t", mcp_client=client, mode="chat"
            )
        )
        assert "I need a bit more information" in question
        assert "I found" not in question

    def test_mutating_lookup_tool_is_refused(self):
        meta = {
            "tool_id": "fake.tool",
            "description": "Fake.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "x-options-tool": "datamodel.create_dataset"}},
                "required": ["target"],
            },
        }
        client = _FakeClient(CONNECTIONS)
        question = run(
            m._generate_clarification_question("fake.tool", meta, ["target"], {}, "t", mcp_client=client, mode="chat")
        )
        client.invoke_tool.assert_not_awaited()
        assert "PROD_SNOWFLAKE" not in question

    def test_unknown_lookup_tool_is_skipped(self):
        meta = {
            "tool_id": "fake.tool",
            "description": "Fake.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "x-options-tool": "datamodel.gone_after_rename"}},
                "required": ["target"],
            },
        }
        client = _FakeClient(CONNECTIONS)
        question = run(
            m._generate_clarification_question("fake.tool", meta, ["target"], {}, "t", mcp_client=client, mode="chat")
        )
        client.invoke_tool.assert_not_awaited()
        assert "PROD_SNOWFLAKE" not in question


# ---------------------------------------------------------------------------
# follow-up nudges
# ---------------------------------------------------------------------------


class TestFollowupHint:
    def _turn(self, trace_id):
        set_current_turn(trace_id, "q")
        begin_turn_output(trace_id)

    def test_success_queues_hint_formatted_from_args(self):
        self._turn("t-followup-1")
        m._record_followup_hint(
            "datamodel.setup_datamodel", SETUP_META, {"datamodel_name": "Sales"}, {"ok": True, "result": {}}
        )
        tail = m._followup_tail()
        assert "Deploy 'Sales'" in tail
        assert tail.startswith("\n\n")
        pop_turn_output("t-followup-1")

    def test_failed_result_queues_nothing(self):
        self._turn("t-followup-2")
        m._record_followup_hint("datamodel.setup_datamodel", SETUP_META, {"datamodel_name": "Sales"}, {"ok": False})
        assert m._followup_tail() == ""
        pop_turn_output("t-followup-2")

    def test_tool_without_followup_queues_nothing(self):
        self._turn("t-followup-3")
        meta = FIXTURE_REGISTRY["datamodel.get_connections"]
        m._record_followup_hint("datamodel.get_connections", meta, {}, {"ok": True, "result": []})
        assert m._followup_tail() == ""
        pop_turn_output("t-followup-3")

    def test_duplicate_hints_are_deduped(self):
        self._turn("t-followup-4")
        for _ in range(2):
            m._record_followup_hint("datamodel.setup_datamodel", SETUP_META, {"datamodel_name": "Sales"}, {"ok": True})
        tail = m._followup_tail()
        assert tail.count("Deploy 'Sales'") == 1
        pop_turn_output("t-followup-4")

    def test_template_missing_arg_is_skipped_not_raised(self):
        self._turn("t-followup-5")
        m._record_followup_hint("datamodel.setup_datamodel", SETUP_META, {"other": 1}, {"ok": True})
        assert m._followup_tail() == ""
        pop_turn_output("t-followup-5")

    def test_no_turn_context_is_a_noop(self):
        m._record_followup_hint("datamodel.setup_datamodel", SETUP_META, {"datamodel_name": "X"}, {"ok": True})
        # nothing to assert beyond "did not raise" — turn_output() is None here
