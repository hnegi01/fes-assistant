"""
Unit tests for INTERNAL_PARAMS — signature params no caller can ever supply.

`emit` is the SDK's progress callback. The MCP server injects it itself for
streaming tools and drops whatever a client sends (tools_core), and it strips
it from its advertised schema list (server.py). The backend never learned the
same trick, so the tool-selection LLM was shown an `emit` slot on 6 tools — and
three shipped example[0] entries actually demonstrated filling it in, teaching
the model to invent a callback object.

Nothing broke at execution time; this is about not putting an unfillable slot in
front of the model. Covered:
  - strip_internal_params / planner_schema drop it from properties and required
  - _format_tool_examples never renders it, whatever the data says
  - the shipped registry (flat + tree) carries no trace of it
  - the registry builder skips it, so a rebuild cannot reintroduce it
"""

import importlib
import inspect
import json

import pytest

import backend.agent._registry as registry_m
import backend.agent._routing as routing_m

SCHEMA_WITH_EMIT = {
    "type": "object",
    "properties": {
        "group_name_list": {"type": "array", "items": {"type": "string"}},
        "emit": {"type": "object", "description": "Optional callback invoked with progress events."},
    },
    "required": ["group_name_list", "emit"],
}


# ---------------------------------------------------------------------------
# strip_internal_params / planner_schema
# ---------------------------------------------------------------------------
class TestStripInternalParams:
    def test_removes_emit_from_properties(self):
        out = routing_m.strip_internal_params(SCHEMA_WITH_EMIT)
        assert "emit" not in out["properties"]

    def test_keeps_real_properties(self):
        out = routing_m.strip_internal_params(SCHEMA_WITH_EMIT)
        assert "group_name_list" in out["properties"]

    def test_removes_emit_from_required(self):
        out = routing_m.strip_internal_params(SCHEMA_WITH_EMIT)
        assert out["required"] == ["group_name_list"]

    def test_does_not_mutate_the_input(self):
        original = json.dumps(SCHEMA_WITH_EMIT, sort_keys=True)
        routing_m.strip_internal_params(SCHEMA_WITH_EMIT)
        assert json.dumps(SCHEMA_WITH_EMIT, sort_keys=True) == original

    @pytest.mark.parametrize("params", [None, {}, {"type": "object"}, {"properties": {}}])
    def test_tolerates_degenerate_schemas(self, params):
        routing_m.strip_internal_params(params)  # must not raise

    def test_planner_schema_drops_emit_and_required(self):
        out = routing_m.planner_schema(SCHEMA_WITH_EMIT)
        assert "emit" not in out["properties"]
        assert "required" not in out


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
class TestExamplesNeverShowInternalParams:
    def test_emit_is_stripped_from_rendered_arguments(self):
        row = {
            "examples": [
                {
                    "user_query": "migrate all groups and track progress",
                    "arguments": {"emit": {"type": "function"}, "batch_size": 10},
                }
            ]
        }
        out = routing_m._format_tool_examples(row, 1)
        assert "emit" not in out
        assert "batch_size" in out

    def test_example_reduced_to_no_arguments_still_renders(self):
        row = {"examples": [{"user_query": "migrate all groups", "arguments": {"emit": {}}}]}
        out = routing_m._format_tool_examples(row, 1)
        assert "-> {}" in out


# ---------------------------------------------------------------------------
# The shipped registry data
# ---------------------------------------------------------------------------
def _flat_rows():
    return json.loads(registry_m.REGISTRY_PATH.read_text(encoding="utf-8"))


def _tree_rows():
    for path in sorted(routing_m.REGISTRY_DIR.rglob("*.json")):
        if path.name == "index.json":
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            for row in rows:
                yield path.name, row


class TestShippedRegistryIsClean:
    def test_no_internal_param_in_flat_schemas(self):
        offenders = [
            r["tool_id"]
            for r in _flat_rows()
            if set((r.get("parameters") or {}).get("properties") or {}) & routing_m.INTERNAL_PARAMS
        ]
        assert offenders == [], f"internal params in registry schemas: {offenders}"

    def test_no_internal_param_in_flat_examples(self):
        # Allowlisted tools only: examples reach a consumer (a user in a dialog,
        # the model at FES_TOOL_EXAMPLES>=1) exclusively for EXPOSED tools, and
        # an SDK refresh lands dozens of unexposed tools with auto-generated
        # examples (49 in the pysisense 1.1.0 bump). Gating on the allowlist
        # makes this "curate the example before you expose the tool" — the
        # check fires the moment a line is uncommented, which is the moment it
        # starts mattering.
        import backend.agent._registry as registry_m

        allowed = registry_m.allowed_tool_ids()
        offenders = [
            r["tool_id"]
            for r in _flat_rows()
            if allowed is None or r["tool_id"] in allowed
            for ex in (r.get("examples") or [])
            if set(ex.get("arguments") or {}) & routing_m.INTERNAL_PARAMS
        ]
        assert offenders == [], f"internal params demonstrated in examples: {offenders}"

    def test_no_internal_param_in_the_registry_tree(self):
        offenders = [
            f"{name}:{row.get('tool_id')}"
            for name, row in _tree_rows()
            if set((row.get("parameters") or {}).get("properties") or {}) & routing_m.INTERNAL_PARAMS
        ]
        assert offenders == [], f"internal params in the routing tree: {offenders}"


class TestBuilderSkipsInternalParams:
    """A registry rebuild must not put `emit` back — the data fix above is only
    durable if the generator agrees."""

    def test_builder_skip_set_covers_internal_params(self):
        mod = importlib.import_module("scripts.01_build_registry_from_sdk")
        assert routing_m.INTERNAL_PARAMS <= mod._INTERNAL_PARAMS
        assert "self" in mod._INTERNAL_PARAMS

    def test_builder_omits_internal_params_from_a_built_schema(self):
        mod = importlib.import_module("scripts.01_build_registry_from_sdk")

        def migrate_all_groups(self, emit=None, batch_size=10):
            """Migrate groups.

            :param emit: Optional progress callback.
            :param batch_size: How many per batch.
            """

        schema = mod.json_schema_from_signature(inspect.signature(migrate_all_groups), migrate_all_groups.__doc__)
        assert "emit" not in schema["properties"]
        assert "self" not in schema["properties"]
        assert "batch_size" in schema["properties"]
