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

import uuid

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
    # eval may ever execute a mutation against a real tenant.
    assert not executed, f"a mutation eval executed a tool — approvals must never be sent{ctx}"

    assert case["expect_gated"] in gated, f"expected {case['expect_gated']!r} to be gated{ctx}"

    for path, expected in case["expect_args"]:
        got = _dig(args, path)
        assert got is not ..., f"{path} was not set at all{ctx}"
        assert got == expected, f"{path} should be {expected!r}, got {got!r}{ctx}"

    for path in case["forbid_arg_paths"]:
        assert _dig(args, path) is ..., f"{path} was invented — the user never gave it{ctx}"

    for frag in case["forbid_reply"]:
        assert frag not in reply, f"reply contains forbidden {frag!r}{ctx}"
