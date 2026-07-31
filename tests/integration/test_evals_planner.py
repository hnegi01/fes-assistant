"""
Planner-behaviour EVALS — a regression battery of real prompts that once failed.

This is not a wiring test (that's the other integration files): each case here
records a live failure we diagnosed, and asserts the *strategy* the agent should
use — which tools it picks / avoids, and what its answer must (not) claim. The
point is to end prompt whack-a-mole: any prompt change is validated against ALL
recorded scenarios at once, instead of fixing one and silently breaking another.

Adding a case = appending one dict to EVAL_CASES (no new code):
  prompt               - the user message, verbatim
  expect_tools_any     - at least one executed tool id must contain one of these
  expect_tools_all     - EVERY fragment here must match some executed tool id
  expect_min_steps     - at least this many tool executions (step_results)
  forbid_tools         - no executed tool id may contain any of these
  expect_reply_any     - reply must contain at least one of these (case-insensitive)
  forbid_reply         - reply must contain none of these (case-insensitive)
  origin               - date + one line on the failure this guards against

Run (cred-gated like all integration tests; skipped without config):
    pytest tests/integration/test_evals_planner.py -m eval -v

Like all LLM-judgment tests these are non-deterministic: re-run a single failure
once before treating it as real (see tests/integration/README.md).
"""

import uuid

import httpx
import pytest

EVAL_CASES = [
    {
        "id": "role-lookup-then-users-with-role",
        "prompt": "what role does himanshu.negi@sisense.com has? " "Also find all the users belong to that role",
        # Must resolve the user's record; a role is answered from user data.
        "expect_tools_any": ["get_user"],
        # 2026-07-30 failure: checker pushed users_per_group("admin") — treated
        # the admin ROLE as a GROUP; "group not found" polluted the answer.
        "forbid_tools": ["users_per_group"],
        "expect_reply_any": ["admin"],
        "forbid_reply": ["not found", "does not exist"],
        "origin": "2026-07-30: role treated as group; wrong extra step users_per_group('admin')",
    },
    {
        "id": "group-membership-via-user-record",
        "prompt": "can you tell which group himanshu.negi@sisense.com belongs to "
        "and then show all user belonging to that group",
        # Must fetch the user's record first (it lists GROUP_NAMES), then the group.
        "expect_tools_any": ["get_user"],
        # 2026-07-31 failure: enumerated ALL groups (users_per_group_all) and
        # scanned for the user instead of reading their record; dead-ended.
        "forbid_tools": ["users_per_group_all"],
        "expect_reply_any": ["everyone"],
        "forbid_reply": ["does not appear in any group", "cannot determine"],
        "origin": "2026-07-31: enumerated all groups to find one user's membership",
    },
    {
        "id": "group-membership-summ-off-blocks-honestly",
        "prompt": "can you tell which group himanshu.negi@sisense.com belongs to "
        "and then show all user belonging to that group",
        "allow_summarization": False,
        # Summ OFF: the group name lives in step 1's result, which the LLM must
        # not see. Correct outcome = honest BLOCKED ("turn summarization on"),
        # raw results shown. The reply must NEVER name the group — if "everyone"
        # appears, result data leaked to the LLM or was fabricated.
        "expect_tools_any": ["get_user"],
        "forbid_tools": [],
        "expect_reply_any": ["summarization", "can't see", "cannot see"],
        "forbid_reply": ["everyone"],
        "origin": "2026-07-31: nodata decide said CONTINUE instead of BLOCKED; planner "
        "filled group_name with the user's email (doomed call) before blocking honestly",
    },
    {
        "id": "group-membership-typo-email-honest-failure",
        "prompt": "can you tell which group himanshu.negi@sisense belongs to "
        "and then show all user belonging to that group",
        # Typo'd email (missing .com): the honest outcome is "user not found" —
        # NOT a scan-based claim that the user belongs to no group.
        "expect_tools_any": ["get_user"],
        "forbid_tools": [],
        "expect_reply_any": ["not found", "couldn't find", "could not find", "no user"],
        "forbid_reply": ["does not appear in any group"],
        "origin": "2026-07-31: typo'd email produced a misleading 'in no group' from a bulk scan",
    },
    {
        "id": "adaptive-role-chain-two-steps",
        "prompt": "get details of user gowtham.senthilkumar@sisense.com, "
        "then list all other users who have that same role",
        # Dependent chain: step 2 needs step 1's returned role. Must resolve the
        # user first, then pull users-with-roles; never treat the role as a group.
        "expect_tools_any": ["get_user"],
        "expect_min_steps": 2,
        "forbid_tools": ["users_per_group"],
        "expect_reply_any": ["sysadmin", "admin"],
        "forbid_reply": ["not found"],
        "origin": "2026-07-30/31: adaptive value-passing chain, verified live during Step 8",
    },
    {
        "id": "fanout-three-independent-steps",
        "prompt": "list all datamodels, all user groups, and also all folders",
        # Three independent parts -> the orchestrator plans 3 untagged steps and
        # they fan out concurrently; all three domains must actually execute.
        "expect_tools_any": ["folder"],
        "expect_tools_all": ["group", "folder"],
        "expect_min_steps": 3,
        "forbid_tools": [],
        "expect_reply_any": ["folder"],
        "forbid_reply": [],
        "origin": "2026-07-31: parallel fan-out (level 1+2) — 3-part request, verified live ~10s",
    },
]


def _turn(backend_url, tenant_config, prompt, allow_summarization=True):
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": f"eval-{uuid.uuid4()}",
            "messages": [{"role": "user", "content": prompt}],
            "user_input": prompt,
            "mode": "chat",
            "tenant_config": tenant_config,
            "allow_summarization": allow_summarization,
        },
        timeout=180,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.integration
@pytest.mark.eval
@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_planner_eval(backend_url, tenant_config, case):
    body = _turn(backend_url, tenant_config, case["prompt"], case.get("allow_summarization", True))
    reply = (body.get("reply") or "").lower()
    tools = [str(s.get("tool_id") or "") for s in (body.get("step_results") or [])]

    ctx = f"\n  origin: {case['origin']}\n  tools: {tools}\n  reply: {reply[:300]!r}"

    if case["expect_tools_any"]:
        assert any(
            frag in t for t in tools for frag in case["expect_tools_any"]
        ), f"expected a tool matching one of {case['expect_tools_any']}{ctx}"
    for frag in case.get("expect_tools_all", []):
        assert any(frag in t for t in tools), f"no executed tool matches {frag!r}{ctx}"
    if case.get("expect_min_steps"):
        assert (
            len(tools) >= case["expect_min_steps"]
        ), f"expected >= {case['expect_min_steps']} executed steps, got {len(tools)}{ctx}"
    for frag in case["forbid_tools"]:
        # Exact-fragment check, but don't let e.g. "users_per_group" ban
        # "users_per_group_all" unless explicitly listed — match whole ids.
        assert not any(t.endswith(frag) or t == frag for t in tools), f"forbidden tool {frag!r} was executed{ctx}"
    if case["expect_reply_any"]:
        assert any(
            frag in reply for frag in case["expect_reply_any"]
        ), f"reply lacks all of {case['expect_reply_any']}{ctx}"
    for frag in case["forbid_reply"]:
        assert frag not in reply, f"reply contains forbidden {frag!r}{ctx}"
