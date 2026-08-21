"""Prompts carry principles, never scenario patches (CLAUDE.md rule).

Pins the 2026-08-17 audit fix so the forced logic cannot quietly return:
PLANNING_SYSTEM_PROMPT (the CHAT tool-selection prompt) once hardcoded seven
migration-only parameter names and an "all dependencies" -> 4-value-list
mapping that duplicated a schema enum and contradicted the SDK's own default
(omitting the parameter already means "all"). Failures become eval cases,
not prompt patches.
"""

from backend.agent._prompts import PLANNING_SYSTEM_PROMPT


def test_no_hardcoded_parameter_names():
    """Concrete parameter names belong in schemas the model is shown, not in
    prompt prose that drifts when the SDK renames one."""
    for name in [
        "group_name_list",
        "user_name_list",
        "dashboard_names",
        "dashboard_ids",
        "datamodel_names",
        "datamodel_ids",
    ]:
        assert name not in PLANNING_SYSTEM_PROMPT, f"hardcoded param name {name!r} is back in the chat prompt"


def test_no_dependencies_scenario_patch():
    """The enum lives in the attached schema; a prompt copy goes stale the day
    the SDK adds a fifth dependency type."""
    for token in ["dataSecurity", "hierarchies", "perspectives", "all dependencies"]:
        assert token not in PLANNING_SYSTEM_PROMPT, f"dependencies scenario patch {token!r} is back"


def test_array_rule_is_schema_derived():
    """The replacement principle: keyed to schema type, covers tools that do
    not exist yet."""
    assert 'schema type is "array"' in PLANNING_SYSTEM_PROMPT
