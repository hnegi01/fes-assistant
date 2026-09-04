"""
Unit tests for FES_TOOL_EXAMPLES — few-shot examples on the tool-selection call.

Covers:
  - _format_tool_examples: count honoured, ordering, malformed examples skipped,
    0 → empty string
  - _load_mixin_tools: at 0 the description is byte-identical to the registry's
    (the whole point of the default — no prompt change until opted in)
  - the clamp on FES_TOOL_EXAMPLES (out of range / unparseable)
  - the shipped registry's example[0] never introduces a value its query omits,
    which is the property that makes these examples safe to show the model
"""

import json
import re

import pytest

import backend.agent._config as config_m
import backend.agent._registry as registry_m
import backend.agent._routing as routing_m

EXAMPLES = [
    {"user_query": "first query", "arguments": {"b": 2, "a": 1}},
    {"user_query": "second query", "arguments": {"x": "y"}},
    {"user_query": "third query", "arguments": {"z": True}},
]


# ---------------------------------------------------------------------------
# _format_tool_examples
# ---------------------------------------------------------------------------
class TestFormatToolExamples:
    def test_zero_returns_empty_string(self):
        assert routing_m._format_tool_examples({"examples": EXAMPLES}, 0) == ""

    def test_negative_returns_empty_string(self):
        assert routing_m._format_tool_examples({"examples": EXAMPLES}, -1) == ""

    def test_one_example_uses_singular_heading(self):
        out = routing_m._format_tool_examples({"examples": EXAMPLES}, 1)
        assert "Example call:" in out and "Example calls:" not in out
        assert "first query" in out
        assert "second query" not in out

    def test_multiple_examples_use_plural_heading(self):
        out = routing_m._format_tool_examples({"examples": EXAMPLES}, 3)
        assert "Example calls:" in out
        assert out.count("\n- ") == 3

    def test_takes_examples_in_order(self):
        out = routing_m._format_tool_examples({"examples": EXAMPLES}, 2)
        assert out.index("first query") < out.index("second query")
        assert "third query" not in out

    def test_arguments_render_as_sorted_json(self):
        out = routing_m._format_tool_examples({"examples": EXAMPLES}, 1)
        assert '{"a": 1, "b": 2}' in out

    def test_no_examples_field_returns_empty(self):
        assert routing_m._format_tool_examples({}, 3) == ""
        assert routing_m._format_tool_examples({"examples": []}, 3) == ""

    def test_skips_examples_missing_query_or_arguments(self):
        rows = {
            "examples": [
                {"user_query": "", "arguments": {"a": 1}},
                {"user_query": "no args"},
                {"user_query": "args not a dict", "arguments": ["nope"]},
                {"user_query": "good one", "arguments": {"a": 1}},
            ]
        }
        out = routing_m._format_tool_examples(rows, 4)
        assert "good one" in out
        assert out.count("\n- ") == 1

    def test_all_examples_unusable_returns_empty(self):
        assert routing_m._format_tool_examples({"examples": [{"user_query": "x"}]}, 2) == ""


# ---------------------------------------------------------------------------
# _load_mixin_tools integration
# ---------------------------------------------------------------------------
class TestMixinToolsWithExamples:
    @pytest.fixture()
    def mixin_dir(self, tmp_path, monkeypatch):
        pkg = tmp_path / "registry" / "demo"
        pkg.mkdir(parents=True)
        (pkg / "core.json").write_text(
            json.dumps(
                [
                    {
                        "tool_id": "demo.read",
                        "description": "Read a thing.",
                        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
                        "mutates": False,
                        "examples": EXAMPLES,
                    }
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(routing_m, "REGISTRY_DIR", tmp_path / "registry")
        monkeypatch.setattr(routing_m, "allowed_tool_ids", lambda: None)
        return pkg

    def _desc(self, n, monkeypatch):
        monkeypatch.setattr(routing_m, "TOOL_EXAMPLES_COUNT", n)
        return routing_m._load_mixin_tools("demo", "core")[0]["function"]["description"]

    def test_zero_leaves_description_untouched(self, mixin_dir, monkeypatch):
        assert self._desc(0, monkeypatch) == "Read a thing."

    def test_one_appends_a_single_example(self, mixin_dir, monkeypatch):
        desc = self._desc(1, monkeypatch)
        assert desc.startswith("Read a thing.")
        assert "Example call:" in desc
        assert desc.count("\n- ") == 1

    def test_three_appends_three_examples(self, mixin_dir, monkeypatch):
        assert self._desc(3, monkeypatch).count("\n- ") == 3

    def test_examples_do_not_alter_the_schema(self, mixin_dir, monkeypatch):
        monkeypatch.setattr(routing_m, "TOOL_EXAMPLES_COUNT", 3)
        with_ex = routing_m._load_mixin_tools("demo", "core")[0]["function"]["parameters"]
        monkeypatch.setattr(routing_m, "TOOL_EXAMPLES_COUNT", 0)
        without = routing_m._load_mixin_tools("demo", "core")[0]["function"]["parameters"]
        assert with_ex == without

    def test_tool_with_no_examples_is_unaffected(self, tmp_path, monkeypatch):
        pkg = tmp_path / "registry" / "demo"
        pkg.mkdir(parents=True)
        (pkg / "core.json").write_text(
            json.dumps([{"tool_id": "demo.bare", "description": "Bare.", "parameters": {}, "mutates": False}]),
            encoding="utf-8",
        )
        monkeypatch.setattr(routing_m, "REGISTRY_DIR", tmp_path / "registry")
        monkeypatch.setattr(routing_m, "allowed_tool_ids", lambda: None)
        monkeypatch.setattr(routing_m, "TOOL_EXAMPLES_COUNT", 3)
        assert routing_m._load_mixin_tools("demo", "core")[0]["function"]["description"] == "Bare."


# ---------------------------------------------------------------------------
# Env parsing / clamp
# ---------------------------------------------------------------------------
class TestExampleCountClamp:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, 0),
            ("", 0),
            ("0", 0),
            ("1", 1),
            ("3", 3),
            ("4", 3),
            ("99", 3),
            ("-5", 0),
            ("abc", 0),
            (" 2 ", 2),
        ],
    )
    def test_clamped(self, raw, expected, monkeypatch):
        if raw is None:
            monkeypatch.delenv("FES_TOOL_EXAMPLES", raising=False)
        else:
            monkeypatch.setenv("FES_TOOL_EXAMPLES", raw)
        assert config_m._env_int_clamped("FES_TOOL_EXAMPLES", 0, 0, 3) == expected


# ---------------------------------------------------------------------------
# The shipped registry's curated examples
# ---------------------------------------------------------------------------
class TestShippedExampleQuality:
    """example[0] is the one FES_TOOL_EXAMPLES=1 shows the model, so its
    arguments must never contain an identity value the query does not mention.
    An example that invents an email or an ObjectId would demonstrate exactly
    what PLANNING_SYSTEM_PROMPT forbids, and a demonstration beats a rule."""

    IDENT = re.compile(r"(name|user|owner|email|title|id)s?$", re.I)

    def _invented(self, example):
        query = (example.get("user_query") or "").lower()
        found = []

        def walk(key, value):
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(k, v)
            elif isinstance(value, list):
                for v in value:
                    walk(key, v)
            elif isinstance(value, str):
                looks_identity = "@" in value or self.IDENT.search(str(key))
                if looks_identity and len(value) > 2 and value.lower() not in query:
                    found.append(f"{key}={value!r}")

        for k, v in (example.get("arguments") or {}).items():
            walk(k, v)
        return found

    def test_first_example_never_invents_an_identity_value(self):
        rows = json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))
        allowed = registry_m.allowed_tool_ids()
        offenders = {}
        for row in rows:
            # Exposed tools only — see the note in test_internal_params.py:
            # an unexposed tool's example is never shown to anyone, and the
            # bar applies the moment it is allowlisted.
            if allowed is not None and row["tool_id"] not in allowed:
                continue
            examples = row.get("examples") or []
            if not examples:
                continue
            invented = self._invented(examples[0])
            if invented:
                offenders[row["tool_id"]] = invented
        assert offenders == {}, f"example[0] invents values absent from its query: {offenders}"

    # Values the query never states are FINE when they are plainly a flag or
    # enum the tool's own schema offers — "user", "text", "oid", "sum". A
    # reader derives those from the request. They are NOT fine when they name
    # part of a customer's schema: "[Region]", "sales.date", "2023-Q3". Those
    # cannot be derived from anything, so an example carrying one demonstrates
    # inventing a field name — the exact move the planner prompt forbids.
    # A declared enum is the precise version of that test — the schema itself
    # says the value is one the tool offers, so no derivation is needed. Falls
    # back to the shape heuristic for the nested payloads whose inner fields
    # carry no enum (a share's `type`, a datasecurity rule's `datatype`).
    ENUMISH = re.compile(r"[a-z][a-z_]{0,11}\Z")

    @staticmethod
    def _declared_enum_values(schema):
        found = set()

        def walk(node):
            if isinstance(node, dict):
                for value in node.get("enum") or []:
                    if isinstance(value, str):
                        found.add(value.lower())
                for child in node.values():
                    walk(child)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(schema)
        return found

    def _invented_references(self, example, schema=None):
        query = (example.get("user_query") or "").lower()
        enums = self._declared_enum_values(schema or {})
        found = []

        def walk(path, value):
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(f"{path}.{k}" if path else str(k), v)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    walk(f"{path}[{i}]", v)
            elif isinstance(value, str) and value.strip():
                if value.lower() in enums or self.ENUMISH.match(value):
                    return
                if value.lower() not in query:
                    found.append(f"{path}={value!r}")

        for k, v in (example.get("arguments") or {}).items():
            walk(str(k), v)
        return found

    def test_first_example_never_invents_a_schema_reference(self):
        """The identity guard above is keyed on the PARAMETER NAME, so it only
        sees values under name/user/owner/email/title/id. That let a whole
        class through: queries.elasticube_run_jaql_query shipped a
        `jaql_payload` naming [Region], [Sales] and [Year] for the query
        "Fetch total sales by region for 2023", under keys called `field` and
        `dimension` (found 2026-09-03; both queries tools are now withheld in
        allowed_tools.txt). Nested values need checking on their own merits,
        whatever the key is called."""
        rows = json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))
        allowed = registry_m.allowed_tool_ids()
        offenders = {}
        for row in rows:
            if allowed is not None and row["tool_id"] not in allowed:
                continue
            examples = row.get("examples") or []
            if not examples:
                continue
            invented = self._invented_references(examples[0], row.get("parameters"))
            if invented:
                offenders[row["tool_id"]] = invented
        assert offenders == {}, f"example[0] references schema its query never names: {offenders}"

    def test_every_tool_has_at_least_one_usable_example(self):
        rows = json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))
        missing = [r["tool_id"] for r in rows if not routing_m._format_tool_examples(r, 1)]
        assert missing == [], f"tools with no renderable example: {missing}"

    QUESTION_OPENERS = ("how ", "what ", "who ", "which ", "can you", "could you")

    def test_first_example_is_an_imperative_command(self):
        """example[0] is shown to users in dialogs and clarifications as 'how
        you could phrase this' — it must model what the user would TYPE next
        (a command), not a question about it. Curated 2026-08-14; this pins
        the style against future registry regenerations."""
        rows = json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))
        allowed = registry_m.allowed_tool_ids()
        offenders = {}
        for row in rows:
            # Exposed tools only — see the note in test_internal_params.py:
            # an unexposed tool's example is never shown to anyone, and the
            # bar applies the moment it is allowlisted.
            if allowed is not None and row["tool_id"] not in allowed:
                continue
            examples = row.get("examples") or []
            if not examples:
                continue
            q = (examples[0].get("user_query") or "").strip()
            if q.endswith("?") or q.lower().startswith(self.QUESTION_OPENERS):
                offenders[row["tool_id"]] = q
        assert offenders == {}, f"example[0] is question-phrased: {offenders}"

    def _unverbalized_numbers(self, example):
        query = (example.get("user_query") or "").lower()
        found = []

        def walk(key, value):
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(k, v)
            elif isinstance(value, list):
                for v in value:
                    walk(key, v)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if str(value) not in query:
                    found.append(f"{key}={value}")

        for k, v in (example.get("arguments") or {}).items():
            walk(k, v)
        return found

    def test_first_example_never_sets_a_number_its_query_omits(self):
        """The identity check catches invented names/ids; this catches invented
        NUMBERS (batch_size: 10 in an example whose sentence never mentions
        batches) — the other way a few-shot demonstrates filling in values the
        user did not give."""
        rows = json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))
        allowed = registry_m.allowed_tool_ids()
        offenders = {}
        for row in rows:
            # Exposed tools only — see the note in test_internal_params.py:
            # an unexposed tool's example is never shown to anyone, and the
            # bar applies the moment it is allowlisted.
            if allowed is not None and row["tool_id"] not in allowed:
                continue
            examples = row.get("examples") or []
            if not examples:
                continue
            unverbalized = self._unverbalized_numbers(examples[0])
            if unverbalized:
                offenders[row["tool_id"]] = unverbalized
        assert offenders == {}, f"example[0] sets numbers absent from its query: {offenders}"


# ---------------------------------------------------------------------------
# Examples must be VALID, not just well-phrased
# ---------------------------------------------------------------------------
class TestShippedExamplesValidateAgainstTheirSchema:
    """An example's arguments must satisfy its own tool's JSON Schema.

    The quality checks above police what an example SAYS; this polices whether
    it would actually run. The gap was real: the two datasecurity tools carried
    examples whose payload was an array of JSON STRINGS, correct against the
    old (wrong) items:string schema and invalid once the generator learned to
    read the SDK's list[dict] annotation — a curated demonstration of a call
    the validator would reject. Two others passed explicit nulls for optional
    params, teaching the model to send null instead of omitting the key.

    Examples are preserved by tool_id across rebuilds (script 02), so a schema
    correction never refreshes them — nothing but this test closes that loop.
    """

    def test_every_example_argument_set_matches_its_tool_schema(self):
        import jsonschema

        rows = json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))
        allowed = registry_m.allowed_tool_ids()
        offenders = {}
        for row in rows:
            tool_id = row.get("tool_id")
            if allowed is not None and tool_id not in allowed:
                continue
            schema = row.get("parameters") or {}
            for i, example in enumerate(row.get("examples") or []):
                args = example.get("arguments")
                if not isinstance(args, dict):
                    continue
                try:
                    jsonschema.validate(instance=args, schema=schema)
                except jsonschema.ValidationError as exc:
                    offenders[f"{tool_id} example[{i}]"] = exc.message[:120]
        assert offenders == {}, f"curated examples that fail their own schema: {offenders}"
