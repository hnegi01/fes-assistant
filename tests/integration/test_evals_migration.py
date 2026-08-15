"""
Migration-mode EVALS — a regression battery of real prompts that once failed.

The migration counterpart to test_evals_planner.py. Separate file, separate
harness: migration mode is its own thing end to end (its own credentials, its
own 9-tool surface, its own mutation gate), and the two batteries assert on
different evidence. Sharing one harness meant bending the chat one out of shape.

  chat evals      assert on which tools EXECUTED
  migration evals assert on the PLAN the agent proposed

Every migration tool mutates, and migration takes ONE approval for the whole
plan (migration_flow), so a migration turn stops at the dialog having executed
nothing, with the proposed steps in `pending_confirmation.arguments.steps`.
These cases send no `approved_keys`, deliberately and always: the agent's CHOICE
is what is under test, and nothing is ever written to a real target environment.

Because every case stops at the gate (or a clarification), credentials are
never used: this battery needs a running stack and LLM credentials, but the
migration_config in integration_config.yaml can be PLACEHOLDER values. The
execution mechanics behind the gate (approval consume, sequential run,
single-use re-gate, cancel-drop) are unit-tested with everything mocked —
tests/unit/test_migration_flow.py and test_approval_single_use.py.
Do not add an approving case here — mutation lifecycle belongs in
test_mutation_lifecycle.py, which creates the asset it later destroys.

Adding a case = appending one dict to EVAL_CASES (no new code):
  prompt            - the user message, verbatim
  expect_steps_any  - some planned step's tool_id must contain one of these
  forbid_steps      - no planned step may have exactly this tool_id
  expect_step_order - these fragments must appear in this relative order
  expect_step_args  - [tool_fragment, arg_name, value] a planned step must carry
  expect_reply_any  - reply must contain at least one of these (case-insensitive)
  forbid_reply      - reply must contain none of these (case-insensitive)
  origin            - date + one line on the failure this guards against

Run (cred-gated; skipped unless integration_config.yaml has migration_config):
    pytest tests/integration/test_evals_migration.py -m eval -v

Like all LLM-judgment tests these are non-deterministic: re-run a single failure
once before treating it as real (see tests/integration/README.md).
"""

import uuid

import httpx
import pytest

EVAL_CASES = [
    {
        "id": "vague-dashboards-must-not-blind-call",
        "prompt": "migrate dashboards to the target environment",
        # No dashboards named. migrate_dashboards RAISES when given neither
        # dashboard_ids nor dashboard_names — it does NOT migrate everything —
        # and both param descriptions carry the SDK's own wording, "Provide
        # either `dashboard_ids` or `dashboard_names`", so the model has been
        # told. Acceptable: ask which dashboards, or gate the bulk tool (a
        # defensible reading of a bare "dashboards"). Unacceptable: gate
        # migrate_dashboards with empty arguments, spending the user's approval
        # on a call that cannot run.
        "expect_steps_any": [],
        "forbid_steps": ["migration.migrate_dashboards"],
        "expect_reply_any": [],
        # The SDK's precondition error must never be what the user ends up with.
        "forbid_reply": ["provide either", "'dashboard_ids'"],
        "origin": "2026-08-07: model emitted migrate_dashboards({}) — passes schema validation "
        "(every selector is optional) and then raises inside the SDK. Guarded here rather than "
        "by hand-written rules in the generated registry.",
    },
    {
        "id": "all-dashboards-picks-the-bulk-tool",
        "prompt": "migrate all dashboards to the target environment",
        # "all" has a dedicated tool that needs no target list. Reaching for the
        # per-dashboard tool is the same confusion as the case above.
        "expect_steps_any": ["migration.migrate_all_dashboards"],
        "forbid_steps": [],
        "expect_reply_any": [],
        "forbid_reply": ["provide either"],
        "origin": "2026-08-07: migrate_all_dashboards vs migrate_dashboards — the bulk tool is "
        "the only one that can honour a bare 'all'.",
    },
    {
        # 2026-08-08: this case used to read "the users and also the groups" —
        # plural and generic, which steers the model onto the BULK tools, where it
        # happened to order correctly. It was green while the targeted tools got
        # it backwards 12 runs out of 12. Naming a specific group is what exposes
        # that path, so the wording here is deliberate: do not generalise it.
        "id": "ordering-groups-before-users",
        "prompt": "migrate the users and also the Sales Team group to the target environment",
        # Asked users-first on purpose. A user cannot be assigned to a group that
        # does not exist yet. Nothing in code enforces this any more — the rule
        # is a principle in MIGRATION_PLANNING_CONTEXT_PROMPT and the planner
        # applies it — so this case is the only automated check on an ordering
        # mistake that otherwise fails SILENTLY (users arrive without their
        # group assignments, no error).
        "expect_steps_any": [],
        "forbid_steps": [],
        "expect_step_order": ["group", "user"],
        "expect_reply_any": [],
        "forbid_reply": [],
        "origin": "2026-08-07: dependency ordering must survive the user naming the assets in "
        "the wrong order; enforced by prompt only, so this eval is the safety net.",
    },
    {
        "id": "dashboard-shares-are-a-flag-not-a-second-step",
        "prompt": "migrate the 'Sales Overview' dashboard to the target environment, including its shares",
        # migrate_dashboard_shares needs SOURCE and TARGET dashboard ids, and the
        # target ids do not exist until the dashboard has been migrated — so it
        # cannot be planned up front as a second step. migrate_dashboards has a
        # migrate_share flag and the SDK does the source→target id mapping
        # internally. Planning two steps here produces a second call whose
        # required arguments nobody can supply.
        "expect_steps_any": ["migration.migrate_dashboards"],
        "forbid_steps": ["migration.migrate_dashboard_shares"],
        "expect_step_args": [["migrate_dashboards", "migrate_share", True]],
        "expect_reply_any": [],
        "forbid_reply": [],
        "origin": "2026-08-07: single-shot plans every call up front, so a step needing a value "
        "an earlier step produces cannot work — shares must be a flag on the dashboard migration.",
    },
    {
        "id": "bare-dashboard-shares-must-clarify-not-improvise",
        "prompt": "migrate the dashboard shares to the target environment",
        # migrate_dashboard_shares REQUIRES source and target dashboard IDs and
        # migration mode has no read tools to resolve them, so the only correct
        # outcome for this prompt is a clarifying question. Any gate is wrong:
        # empty ID lists now count as missing (clarify), and non-empty ones
        # could only be invented. The seen failure was worse — the dependency
        # guidance ("users must exist before shares") pulled the planner into
        # migrating an asset kind the user never mentioned.
        "expect_steps_any": [],
        "forbid_steps": [
            "migration.migrate_all_users",
            "migration.migrate_users",
            "migration.migrate_all_groups",
            "migration.migrate_groups",
            "migration.migrate_all_dashboards",
            "migration.migrate_dashboards",
            "migration.migrate_all_datamodels",
            "migration.migrate_datamodels",
            "migration.migrate_dashboard_shares",
        ],
        "expect_reply_any": ["dashboard id"],
        "forbid_reply": [],
        "origin": "2026-08-14: planner emitted migrate_all_users({}) for a bare shares request "
        "(~1 in 6 runs); two other runs gated migrate_dashboard_shares with EMPTY id lists, "
        "which passed the missing-required check until _is_missing learned that [] means absent.",
    },
    {
        "id": "three-kinds-all-planned-bulk-and-ordered",
        "prompt": "migrate the dashboards, the users and the groups to the target environment",
        # The completeness probe (M2 in the live battery): three asset kinds
        # named means three steps — a dropped kind fails SILENTLY (the user
        # believes it migrated). Nothing specific named, so all three must be
        # the bulk tools, ordered groups → users → dashboards regardless of the
        # order the user said them in.
        "expect_steps_any": [],
        "forbid_steps": [
            "migration.migrate_groups",
            "migration.migrate_users",
            "migration.migrate_dashboards",
        ],
        "expect_step_order": ["group", "user", "dashboard"],
        "expect_reply_any": [],
        "forbid_reply": [],
        "origin": "2026-08-14: live M2 run — verified three bulk steps in dependency order; "
        "kept as the automated completeness check (an omitted kind is the quiet failure mode "
        "FES_MIGRATION_COMPLETENESS_CHECK exists for, and this eval is the always-on guard).",
    },
    {
        "id": "named-group-picks-targeted-tool-with-exact-name",
        "prompt": "migrate the QA Team group to the target environment",
        # A named asset must use the targeted tool and pass EXACTLY the name
        # given — not the bulk tool (blast radius: everything), not a variant
        # of the name (extraction, never invention).
        "expect_steps_any": ["migration.migrate_groups"],
        "forbid_steps": ["migration.migrate_all_groups"],
        "expect_step_args": [["migrate_groups", "group_name_list", ["QA Team"]]],
        "expect_reply_any": [],
        "forbid_reply": [],
        "origin": "2026-08-14: live M5 run (throwaway group) — the single-step write path; "
        "the gate showed exactly the named group and executed only it on approval.",
    },
    {
        "id": "two-named-assets-one-plan-group-before-user",
        "prompt": "migrate the QA Team group and the user qa.user@example.com to the target environment",
        # Two named assets = ONE plan with both steps (a second dialog after the
        # first step means batch approval broke), targeted tools with the exact
        # names, group ordered before user.
        "expect_steps_any": [],
        "forbid_steps": ["migration.migrate_all_groups", "migration.migrate_all_users"],
        "expect_step_order": ["group", "user"],
        "expect_step_args": [
            ["migrate_groups", "group_name_list", ["QA Team"]],
            ["migrate_users", "user_name_list", ["qa.user@example.com"]],
        ],
        "expect_reply_any": [],
        "forbid_reply": [],
        "origin": "2026-08-14: live M6 run — both steps executed sequentially off a single "
        "approval; this eval pins the plan shape that makes that possible.",
    },
    {
        "id": "unasked-asset-kinds-are-not-added-to-the-plan",
        "prompt": "migrate all users and all datamodels to the target environment",
        # Two kinds asked = two steps. The planner added migrate_all_groups
        # unasked, pulled by the dependency guidance ("groups before users") —
        # the same improvisation family as the bare-shares case. The contract
        # (MIGRATION_PLANNING_CONTEXT_PROMPT) is explicit: never emit a call
        # for something the user did not ask for; the dialog exists so a HUMAN
        # can widen scope, the planner must not do it silently.
        "expect_steps_any": [],
        "forbid_steps": [
            "migration.migrate_all_groups",
            "migration.migrate_groups",
            "migration.migrate_all_dashboards",
            "migration.migrate_dashboards",
        ],
        "expect_step_order": ["user", "datamodel"],
        "expect_reply_any": [],
        "forbid_reply": [],
        "origin": "2026-08-14: live progress-test run — 'migrate all users and datamodels' "
        "planned THREE steps (groups added unasked). Caught by the user in the dialog.",
    },
]


def _migration_turn(backend_url, migration_config, prompt, allow_summarization=True):
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": f"eval-mig-{uuid.uuid4()}",
            "messages": [{"role": "user", "content": prompt}],
            "user_input": prompt,
            "mode": "migration",
            "migration_config": migration_config,
            # Never approve. The gate is where these cases stop, by design.
            "approved_keys": [],
            "allow_summarization": allow_summarization,
        },
        timeout=180,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.integration
@pytest.mark.eval
@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_migration_eval(backend_url, migration_config, case):
    body = _migration_turn(backend_url, migration_config, case["prompt"], case.get("allow_summarization", True))
    reply = (body.get("reply") or "").lower()
    pending = (body.get("tool_result") or {}).get("pending_confirmation") or {}
    steps = (pending.get("arguments") or {}).get("steps") or []
    planned = [str(s.get("tool") or "") for s in steps]
    executed = [str(s.get("tool_id") or "") for s in (body.get("step_results") or [])]

    ctx = f"\n  origin: {case['origin']}\n  planned: {planned}\n  executed: {executed}\n  reply: {reply[:300]!r}"

    # Safety net for the whole battery, not a per-case option: no migration
    # eval may ever execute a mutation against a real target.
    assert not executed, f"a migration eval executed a tool — approvals must never be sent{ctx}"

    if case["expect_steps_any"]:
        assert any(frag in t for t in planned for frag in case["expect_steps_any"]), (
            f"no planned step matches one of {case['expect_steps_any']}{ctx}"
        )
    for frag in case["forbid_steps"]:
        # Exact ids: "migration.migrate_dashboards" must not ban
        # "migration.migrate_dashboard_shares" by accident.
        assert frag not in planned, f"{frag!r} must not be planned{ctx}"
    if case.get("expect_step_order"):
        positions = []
        for frag in case["expect_step_order"]:
            hit = next((i for i, t in enumerate(planned) if frag in t), None)
            assert hit is not None, f"no planned step matches {frag!r}{ctx}"
            positions.append(hit)
        assert positions == sorted(positions), f"steps must run in the order {case['expect_step_order']}{ctx}"
    for frag, arg_name, expected in case.get("expect_step_args", []):
        match = next((s for s in steps if frag in str(s.get("tool") or "")), None)
        assert match is not None, f"no planned step matches {frag!r}{ctx}"
        assert (match.get("arguments") or {}).get(arg_name) == expected, (
            f"{frag} should have been planned with {arg_name}={expected!r}{ctx}"
        )
    if case["expect_reply_any"]:
        assert any(frag in reply for frag in case["expect_reply_any"]), (
            f"reply lacks all of {case['expect_reply_any']}{ctx}"
        )
    for frag in case["forbid_reply"]:
        assert frag not in reply, f"reply contains forbidden {frag!r}{ctx}"
