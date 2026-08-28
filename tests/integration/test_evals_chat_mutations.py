"""
Chat mutation-planning EVALS — do gated calls carry the right arguments?

The third eval battery, and the third distinct kind of evidence:

  test_evals_planner.py    chat, read-only — which tools EXECUTED
  test_evals_migration.py  migration       — the PLAN that was proposed
  this file                chat, mutating  — the ARGUMENTS of the gated call

A mutating chat turn stops at the approval dialog having executed nothing, so
`step_results` is empty and the planner battery's assertions have nothing to look
at. What matters here is the payload the agent was *about* to send: a create that
silently loses a field is a successful-looking call that produces the wrong
object, or an SDK error the user has to decode.

These cases send no `approved_keys`, deliberately and always — the agent's
proposed arguments are under test and nothing is ever written to the tenant. Do
not add an approving case here; mutation lifecycle belongs in
test_mutation_lifecycle.py, which creates the asset it later destroys.

Adding a case = appending one dict to EVAL_CASES (no new code):
  prompt           - the user message, verbatim
  expect_gated     - the paused mutation's tool_id must contain this fragment
  expect_args      - [[dotted.path, value], ...] the gated arguments must carry
  forbid_arg_paths - [dotted.path, ...] that must NOT be set (no invented values)
  forbid_reply     - reply must contain none of these (case-insensitive)
  origin           - date + one line on the failure this guards against

Run (cred-gated like all integration tests; skipped without config):
    pytest tests/integration/test_evals_chat_mutations.py -m eval -v

Like all LLM-judgment tests these are non-deterministic: re-run a single failure
once before treating it as real (see tests/integration/README.md).
"""

import json
import uuid
from pathlib import Path

import httpx
import pytest

# Addresses that exist nowhere. Nothing here is approved, so nothing is created,
# but the names make it obvious in a log that no real user was targeted.
PROBE = "probe.eval.donotcreate@sisense-test.com"

EVAL_CASES = [
    {
        "id": "create-user-with-role-keeps-the-role",
        "prompt": f"create a user {PROBE} with the viewer role",
        # 2026-08-07, live: the planner split this into "1. Create a user with
        # the email ..." + "2. Assign the viewer role to ...". create_user takes
        # the role in the SAME call (it resolves the name to a roleId itself), so
        # step 1 went out with only an email and the SDK failed with
        # "Role 'None' not found in roles_mapping". `user_data` is a free-form
        # object in the schema, so validation and the clarification loop are both
        # blind to a missing field inside it — this eval is the only check.
        "expect_gated": "access_management.create_user",
        "expect_args": [["user_data.email", PROBE], ["user_data.role", "viewer"]],
        "forbid_arg_paths": [],
        "forbid_reply": ["role 'none'", "not found in roles_mapping"],
        "origin": "2026-08-07: planner split create-with-role into two steps; step 1 lost the "
        "role and the SDK rejected it. One ask, one operation.",
    },
    {
        "id": "create-user-does-not-invent-fields",
        "prompt": f"create a user {PROBE} with the viewer role",
        # The user gave an email and a role and nothing else. A first name, a
        # group, or a password conjured to fill the payload would create a user
        # that does not match what was asked for — and the approver would have to
        # notice a wrong value rather than a missing one.
        "expect_gated": "access_management.create_user",
        "expect_args": [],
        "forbid_arg_paths": ["user_data.password", "user_data.groups"],
        "forbid_reply": [],
        "origin": "2026-08-07: PLANNING_SYSTEM_PROMPT forbids invented values; a mutation payload "
        "is where an invented one does real damage.",
    },
    {
        "id": "delete-user-targets-exactly-the-named-user",
        "prompt": f"delete the user {PROBE}",
        # The single most destructive shape in chat mode. The gated arguments must
        # name that user and no other — a normalised, truncated or substituted
        # address means the approval dialog describes one thing and the call does
        # another.
        "expect_gated": "access_management.delete_user",
        "expect_args": [["user_name", PROBE]],
        "forbid_arg_paths": [],
        "forbid_reply": [],
        "origin": "2026-08-07: added alongside the create-user case — a delete must never widen "
        "or alter its target between the request and the payload.",
    },
    {
        "id": "full-setup-request-lands-on-setup-datamodel",
        "prompt": (
            "Set up an extract data model called 'EVAL_PROBE_MODEL_DONOTCREATE' using the "
            "'EVAL_PROBE_CONN' connection against the 'EVAL_PROBE_DB' database and 'public' "
            "schema, with the customers table."
        ),
        # A from-scratch request naming tables and a connection must select the
        # SDK's composite (create model + dataset + tables in one call), not a
        # building block. create_datamodel is delisted (an unrescuable skeleton),
        # but create_dataset/create_table remain choosable and would leave the
        # request half-done.
        "expect_gated": "datamodel.setup_datamodel",
        "expect_args": [["datamodel_name", "EVAL_PROBE_MODEL_DONOTCREATE"], ["datamodel_type", "extract"]],
        "forbid_arg_paths": [],
        "forbid_reply": [],
        "origin": "2026-08-20: create-vs-setup tool-picking review — the composite covers "
        "from-scratch creation; building blocks mis-picked here leave a dead-end model.",
    },
    {
        "id": "add-table-request-lands-on-create-table",
        "prompt": "Add a new table named 'eval_probe_table' to the 'EVAL_PROBE_MODEL_DONOTCREATE' data model.",
        # The standalone add-a-table case create_table exists for: the SDK
        # self-resolves dataset/connection/schema from datamodel_name, so this
        # must NOT route to setup_datamodel (which would try to create the model
        # and fail on the existing name).
        "expect_gated": "datamodel.create_table",
        "expect_args": [["datamodel_name", "EVAL_PROBE_MODEL_DONOTCREATE"], ["table_name", "eval_probe_table"]],
        "forbid_arg_paths": [],
        "forbid_reply": [],
        "origin": "2026-08-20: create-vs-setup review — add-a-table is create_table's one "
        "legitimate standalone job; the composite would error on the existing model.",
    },
    {
        "id": "dashboard-sharing-is-a-share-not-a-role-change",
        "prompt": "share dashboard 6a90b533a689f5da2f617354 with the user "
        "probe.eval.donotcreate@sisense-test.com as viewer",
        # Granting access to ONE dashboard must gate add_dashboard_shares.
        # access_management.update_user(role=viewer) looks superficially similar
        # in a dialog and silently widens the user's access across the WHOLE
        # deployment instead — a far broader mutation than the one asked for.
        #
        # KNOWN GAP (live 2026-08-28, deterministic): the phrasing "give X view
        # access to dashboard <id>" — nearly verbatim add_dashboard_shares' own
        # curated example — routes to access_management/users, so the selection
        # step never sees the sharing tool and picks update_user. Root cause is
        # the SDK class docstrings the L1 index is generated from:
        # access_management advertises "dashboard ownership transfer ...
        # reporting on dashboard shares" while dashboard advertises "share
        # management for users and groups". Proven by experiment — sharpening
        # those two descriptions routes it correctly 4/4 — so the fix belongs in
        # the SDK docstrings (index.json is generated; a hand-edit is dropped on
        # the next rebuild). This case pins the phrasings that DO work; the
        # "give ... view access" wording joins the prompt once the SDK lands.
        "expect_gated": "dashboard.add_dashboard_shares",
        "expect_args": [["dashboard_id", "6a90b533a689f5da2f617354"]],
        "forbid_arg_paths": [],
        "forbid_reply": [],
        "origin": "2026-08-28: live write test — 'give X view access to dashboard Y' gated "
        "update_user(role=viewer), a deployment-wide role change instead of one dashboard share.",
    },
]


def _dig(obj, dotted):
    """Fetch a nested value by dotted path; a sentinel when any hop is absent."""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return ...
        cur = cur[part]
    return cur


def _chat_turn(backend_url, tenant_config, prompt):
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": f"eval-mut-{uuid.uuid4()}",
            "messages": [{"role": "user", "content": prompt}],
            "user_input": prompt,
            "mode": "chat",
            "tenant_config": tenant_config,
            # Never approve. The gate is where these cases stop, by design.
            "approved_keys": [],
            "allow_summarization": True,
        },
        timeout=180,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.integration
@pytest.mark.eval
@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_chat_mutation_eval(backend_url, tenant_config, case):
    body = _chat_turn(backend_url, tenant_config, case["prompt"])
    reply = (body.get("reply") or "").lower()
    pending = (body.get("tool_result") or {}).get("pending_confirmation") or {}
    gated = str(pending.get("tool_id") or "")
    args = pending.get("arguments") or {}
    executed = [str(s.get("tool_id") or "") for s in (body.get("step_results") or [])]

    ctx = f"\n  origin: {case['origin']}\n  gated: {gated!r}\n  args: {args}\n  reply: {reply[:200]!r}"

    # Safety net for the whole battery, not a per-case option: no chat mutation
    # eval may ever execute a MUTATION against a real tenant. Reads are allowed
    # — the planner may legitimately add a lookup step before the gated write
    # (seen live 2026-08-20: a verify-the-connection read ran ahead of the
    # gated setup_datamodel), and no approvals are ever sent, so the backend's
    # gate blocks every write regardless. Mutability comes from the registry,
    # never from the tool name.
    _registry_by_id = {
        r["tool_id"]: r
        for r in json.loads((Path(__file__).parents[2] / "config" / "tools.registry.with_examples.json").read_text())
    }
    executed_mutations = [t for t in executed if _registry_by_id.get(t, {}).get("mutates")]
    assert not executed_mutations, f"a mutation eval EXECUTED a mutating tool — approvals must never be sent{ctx}"

    assert case["expect_gated"] in gated, f"expected {case['expect_gated']!r} to be gated{ctx}"

    for path, expected in case["expect_args"]:
        got = _dig(args, path)
        assert got is not ..., f"{path} was not set at all{ctx}"
        # String values compare case-insensitively: the SDK's own contract is
        # case-insensitive where it matters (create_user docstring: "the role
        # field is matched case-insensitively"), and the model flips casing
        # ('viewer'/'VIEWER') run to run. These cases guard that a VALUE
        # survives into the gated call — not its capitalisation.
        if isinstance(got, str) and isinstance(expected, str):
            assert got.lower() == expected.lower(), f"{path} should be {expected!r}, got {got!r}{ctx}"
        else:
            assert got == expected, f"{path} should be {expected!r}, got {got!r}{ctx}"

    for path in case["forbid_arg_paths"]:
        assert _dig(args, path) is ..., f"{path} was invented — the user never gave it{ctx}"

    for frag in case["forbid_reply"]:
        assert frag not in reply, f"reply contains forbidden {frag!r}{ctx}"
