"""
Planner-behaviour EVALS — a regression battery of real prompts that once failed.

This is not a wiring test (that's the other integration files): each case here
records a live failure we diagnosed, and asserts the *strategy* the agent should
use — which tools it picks / avoids, and what its answer must (not) claim. The
point is to end prompt whack-a-mole: any prompt change is validated against ALL
recorded scenarios at once, instead of fixing one and silently breaking another.

Adding a case = appending one dict to EVAL_CASES (no new code):
  prompt               - the user message, verbatim
  history              - optional prior messages ([{role, content}, ...]) sent
                         before the prompt, for cases about conversation state
  allow_summarization  - True (default), False, or "both". Production mostly
                         runs summarization OFF, so every guard must hold there
                         too: use "both" when ONE set of checks is valid under
                         both settings (clarifications; honest failures — the
                         error reason reaches the LLM even when off); write a
                         summ-off SIBLING case when the correct behavior
                         differs (dependent chains complete on / block
                         honestly off). Discover off-checks by driving the
                         prompt live, never by assuming.
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

import re
import uuid

import httpx
import pytest

EVAL_CASES = [
    {
        "id": "role-lookup-then-users-with-role",
        "prompt": "what role does {user_a_email} has? Also find all the users belong to that role",
        # Must resolve the user's record; a role is answered from user data.
        "expect_tools_any": ["get_user"],
        # 2026-07-30 failure: checker pushed users_per_group("admin") — treated
        # the admin ROLE as a GROUP; "group not found" polluted the answer.
        "forbid_tools": ["users_per_group"],
        "expect_reply_any": ["{user_a_role}", "{user_a_role_alt}"],
        "forbid_reply": ["not found", "does not exist"],
        "origin": "2026-07-30: role treated as group; wrong extra step users_per_group('admin')",
    },
    {
        "id": "role-lookup-summ-off-blocks-honestly",
        "prompt": "what role does {user_a_email} has? Also find all the users belong to that role",
        "allow_summarization": False,
        # Summ-off sibling of the case above: the role lives in step 1's result,
        # which the LLM must not see. Correct = fetch the user, then block
        # honestly. Naming the role data-blind means leak or invention.
        "expect_tools_any": ["get_user"],
        "forbid_tools": ["users_per_group"],
        "expect_reply_any": ["summarization"],
        "forbid_reply": ["{user_a_role}", "{user_a_role_alt}"],
        "origin": "2026-08-21: summ-off coverage sweep — every guard must hold data-blind",
    },
    {
        "id": "group-membership-via-user-record",
        "prompt": "can you tell which group {user_a_email} belongs to and then show all user belonging to that group",
        # Must fetch the user's record first (it lists GROUP_NAMES), then the group.
        "expect_tools_any": ["get_user"],
        # 2026-07-31 failure: enumerated ALL groups (users_per_group_all) and
        # scanned for the user instead of reading their record; dead-ended.
        "forbid_tools": ["users_per_group_all"],
        "expect_reply_any": ["{user_a_group}"],
        "forbid_reply": ["does not appear in any group", "cannot determine"],
        "origin": "2026-07-31: enumerated all groups to find one user's membership",
    },
    {
        "id": "group-membership-summ-off-blocks-honestly",
        "prompt": "can you tell which group {user_a_email} belongs to and then show all user belonging to that group",
        "allow_summarization": False,
        # Summ OFF: the group name lives in step 1's result, which the LLM must
        # not see. Correct outcome = honest BLOCKED ("turn summarization on"),
        # raw results shown. The reply must NEVER name the group — if "everyone"
        # appears, result data leaked to the LLM or was fabricated.
        "expect_tools_any": ["get_user"],
        "forbid_tools": [],
        "expect_reply_any": ["summarization", "can't see", "cannot see"],
        "forbid_reply": ["{user_a_group}"],
        "origin": "2026-07-31: nodata decide said CONTINUE instead of BLOCKED; planner "
        "filled group_name with the user's email (doomed call) before blocking honestly",
    },
    {
        "id": "group-membership-typo-email-honest-failure",
        "prompt": "can you tell which group {user_a_email_typo} belongs to "
        "and then show all user belonging to that group",
        # Runs under BOTH settings with one set of checks: failure reasons are
        # the one summ-off exception (the error string reaches the LLM), so
        # "not found" must appear either way — and summ-off is exactly when a
        # data-blind model is most tempted to invent (verified live 2026-08-21).
        "allow_summarization": "both",
        # Typo'd email (missing .com): the honest outcome is "user not found" —
        # NOT a scan-based claim that the user belongs to no group.
        "expect_tools_any": ["get_user"],
        "forbid_tools": [],
        # "does not exist" added 2026-08-15: equally honest phrasing, seen live
        # in an otherwise-correct reply — the case is about honesty, not wording.
        "expect_reply_any": ["not found", "couldn't find", "could not find", "no user", "does not exist"],
        "forbid_reply": ["does not appear in any group"],
        "origin": "2026-07-31: typo'd email produced a misleading 'in no group' from a bulk scan",
    },
    {
        "id": "adaptive-role-chain-two-steps",
        "prompt": "get details of user {user_b_email}, then list all other users who have that same role",
        # Dependent chain: step 2 needs step 1's returned role. Must resolve the
        # user first, then pull users-with-roles; never treat the role as a group.
        "expect_tools_any": ["get_user"],
        "expect_min_steps": 2,
        "forbid_tools": ["users_per_group"],
        "expect_reply_any": ["{user_b_role}", "{user_b_role_alt}"],
        "forbid_reply": ["not found"],
        "origin": "2026-07-30/31: adaptive value-passing chain, verified live during Step 8",
    },
    {
        "id": "adaptive-role-chain-summ-off-blocks-honestly",
        "prompt": "get details of user {user_b_email}, then list all other users who have that same role",
        "allow_summarization": False,
        # Summ-off sibling: the role filter needs step 1's data. Correct =
        # fetch the user, do any data-free parts, block honestly on the filter
        # — never name the role (that would be leak or invention).
        "expect_tools_any": ["get_user"],
        "forbid_tools": ["users_per_group"],
        "expect_reply_any": ["summarization"],
        "forbid_reply": ["{user_b_role}", "{user_b_role_alt}"],
        "origin": "2026-08-21: summ-off coverage sweep — every guard must hold data-blind",
    },
    {
        "id": "fanout-three-independent-steps",
        "prompt": "list all datamodels, all user groups, and also all folders",
        # Independent steps need no result data, so this must work under BOTH
        # settings with one set of checks: summ-off's code-built reply names
        # the executed tools, so "folder" appears either way (verified live
        # 2026-08-21).
        "allow_summarization": "both",
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
    {
        "id": "bare-create-after-stale-clarification-still-clarifies",
        "prompt": "create datamodel abc",
        # Clarifying needs no result data — one set of checks, BOTH settings.
        "allow_summarization": "both",
        # A prior clarify exchange sits in history but is no longer pending
        # server-side (backend restart wiped the session; the user changed topic
        # instead of answering). Seen live 2026-08-20: the planner role-played
        # the assistant — its "plan" was a clarifying QUESTION, routing said
        # none, and the useful question was discarded for the generic fallback.
        "history": [
            {"role": "user", "content": "set up a new extract datamodel called UI_TEST_MODEL"},
            {
                "role": "assistant",
                "content": (
                    "I need a bit more information to run this.\n\n"
                    "**Set up a data model end to end using an existing connection** needs:\n"
                    "- Name of the connection to use\n- Name of the data source database\n"
                    "- Name of the data source schema\n- List of tables to add"
                ),
            },
        ],
        "expect_tools_any": [],
        "forbid_tools": [],
        "expect_reply_any": ["i need", "more information", "more details", "needs:", "could you provide"],
        "forbid_reply": ["didn't quite understand"],
        "origin": "2026-08-20: stale clarify exchange in history derailed the planner into "
        "answering as the assistant; the fallback then discarded its question.",
    },
    {
        "id": "create-request-over-pending-clarification-stays-on-create",
        "prompt": "create datamodel abc with xyz connection",
        # Live failure was with summarization off; clarifying needs no result
        # data, so the same checks must hold under BOTH settings.
        "allow_summarization": "both",
        # The user abandoned a pending setup_datamodel clarification by asking
        # for a DIFFERENT datamodel. Correct: treat it as a fresh create — pin
        # setup_datamodel, keep name=abc + connection=xyz, clarify for the rest.
        # Seen live 2026-08-21 (UI, resume declined the pinned tool): the fresh
        # planner went exploring instead — get_datasources ×2 + get_elasticubes
        # fanned out, then the turn blocked on the summ-off dependency gate.
        # Non-deterministic (2 API repros behaved); this pins the invariant.
        # Fresh session per case = no server-side pin; guards the history shape.
        "history": [
            {"role": "user", "content": "set up a datamodel called FES_V2_Manual"},
            {
                "role": "assistant",
                "content": (
                    "I need a bit more information to run this:\n\n"
                    "- Type of the data model\n- Name of the connection to use\n"
                    "- Name of the data source database\n- Name of the data source schema\n"
                    "- List of tables to add\n\nOptionally, you can also include: `dataset_name`."
                ),
            },
        ],
        "expect_tools_any": [],
        "forbid_tools": ["get_datasources", "get_elasticubes"],
        "expect_reply_any": ["i need", "more information", "more details", "needs:", "could you provide"],
        "forbid_reply": ["didn't quite understand"],
        "origin": "2026-08-21: topic-change to a second create request mid-clarification sent "
        "the planner into discovery reads instead of clarifying the new create.",
    },
    {
        "id": "create-user-without-email-or-role-clarifies-not-gates",
        "prompt": "create user himanshu negi",
        # user_data is an SDK `dict` param whose requirements live one level
        # down (email + role). Before the rich SCHEMA_RULES schema + the nested
        # walk in _missing_required_fields, this passed validation, GATED, the
        # user approved, and the SDK failed ("Role 'None' not found", live
        # 2026-08-27) — a wasted approval on a doomed call. Correct: clarify up
        # front, no gate, nothing executed. The anchor is ROLE: it carries
        # x-options-tool, so a value the user never said always counts as
        # missing. Email is NOT asserted — the model sometimes invents a
        # plausible address (himanshu.negi@sisense.com, derived from the name),
        # which no deterministic rule can brand; the approval dialog's arg
        # disclosure is that residual's net. Clarifying needs no result data →
        # both settings.
        "allow_summarization": "both",
        "expect_tools_any": [],
        "forbid_tools": ["access_management.create_user"],
        "expect_reply_any": ["role"],
        "forbid_reply": ["didn't quite understand", "not found"],
        "origin": "2026-08-27: create_user gated with neither email nor role inside user_data; "
        "approval was spent on a call the SDK was guaranteed to reject.",
    },
    {
        "id": "off-topic-request-is-denied-not-force-fit",
        "prompt": "write me a poem about databases",
        # Not a Sisense task: deny, execute nothing. Before the
        # refuse-what-the-catalog-cannot-do planner rule, this force-fit the
        # nearest tool — a notebook-create — and asked for its payload. Denying
        # needs no result data, so one set of checks holds under BOTH settings.
        "allow_summarization": "both",
        "expect_tools_any": [],
        "forbid_tools": ["custom_code.create_notebook", "custom_code.update_notebook"],
        "expect_reply_any": ["sisense", "didn't quite understand"],
        "forbid_reply": ["notebook", "i need a bit more information"],
        "origin": "2026-08-21: off-topic request routed to a notebook-creation tool and "
        "asked for its creation payload.",
    },
    {
        "id": "named-object-property-is-one-step-not-resolve-then-fetch",
        "prompt": "get columns from dashboard called test",
        # The columns tool takes the dashboard NAME — the user's own words —
        # yet with listing turns in history the planner reliably (4/4 live,
        # 0/4 fresh) planned "get the dashboard by name, THEN its columns
        # [needs-prior-result]", inventing an id-dependency; under summ-off
        # the dependency gate then skipped the tagged step and the turn died
        # with "turn summarization on" for a one-step request. Root cause was
        # our own resolve-the-named-entity-first rule read as mandating a
        # two-step resolve; the resolving-is-ONE-step rule pins the intent.
        # History is the trigger, so the case carries the reproducing fixture.
        "allow_summarization": False,
        "history": [
            {"role": "user", "content": "get all dashboard"},
            {
                "role": "assistant",
                "content": "Found 4 results from `dashboard.get_all_dashboards`. Results shown above.",
            },
            {"role": "user", "content": "get users"},
            {
                "role": "assistant",
                "content": "Found 1 result from `access_management.get_users_all`. Results shown above.",
            },
        ],
        "expect_tools_any": ["dashboard.get_dashboard_columns"],
        "forbid_tools": [],
        "expect_reply_any": [],
        "forbid_reply": ["turn summarization on"],
        "origin": "2026-08-27: history-triggered resolve-then-fetch plan blocked a one-step "
        "summ-off request; live EC2, reproduced 4/4 with this fixture, 5/5 fixed.",
    },
]


def _turn(backend_url, tenant_config, prompt, allow_summarization=True, history=None):
    resp = httpx.post(
        f"{backend_url}/agent/turn",
        json={
            "session_id": f"eval-{uuid.uuid4()}",
            "messages": [*(history or []), {"role": "user", "content": prompt}],
            "user_input": prompt,
            "mode": "chat",
            "tenant_config": tenant_config,
            "allow_summarization": allow_summarization,
        },
        timeout=180,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _expand_summarization_runs(cases):
    """allow_summarization: "both" runs the case twice — once per setting —
    with the SAME checks (use sibling cases when the correct behavior differs)."""
    runs = []
    for case in cases:
        summ = case.get("allow_summarization", True)
        if summ == "both":
            for flag in (True, False):
                suffix = "summ-on" if flag else "summ-off"
                runs.append(({**case, "allow_summarization": flag}, f"{case['id']}-{suffix}"))
        else:
            runs.append((case, case["id"]))
    return runs


_EVAL_RUNS = _expand_summarization_runs(EVAL_CASES)


class _SafeIds(dict):
    """format_map helper: unknown placeholders pass through untouched, so case
    text may contain literal braces without breaking resolution."""

    def __missing__(self, key):
        return "{" + key + "}"


def _resolve_identities(case, identities):
    """Fill {user_a_email}-style placeholders from the gitignored config.

    The repo is public: committed cases carry placeholders, never real users.
    Reply-check fragments are lowercased (the harness compares against the
    lowercased reply)."""
    ids = _SafeIds(identities)

    def _fmt(v):
        return v.format_map(ids)

    out = dict(case)
    out["prompt"] = _fmt(case["prompt"])
    if case.get("history"):
        out["history"] = [{**m, "content": _fmt(m["content"])} for m in case["history"]]
    for key in ("expect_reply_any", "forbid_reply"):
        out[key] = [_fmt(f).lower() for f in case.get(key, [])]
    return out


@pytest.mark.integration
@pytest.mark.eval
@pytest.mark.parametrize("case", [r[0] for r in _EVAL_RUNS], ids=[r[1] for r in _EVAL_RUNS])
def test_planner_eval(backend_url, tenant_config, eval_identities, case):
    case = _resolve_identities(case, eval_identities)
    body = _turn(
        backend_url,
        tenant_config,
        case["prompt"],
        case.get("allow_summarization", True),
        history=case.get("history"),
    )
    reply = (body.get("reply") or "").lower()
    tools = [str(s.get("tool_id") or "") for s in (body.get("step_results") or [])]

    ctx = f"\n  origin: {case['origin']}\n  tools: {tools}\n  reply: {reply[:300]!r}"

    if case["expect_tools_any"]:
        assert any(frag in t for t in tools for frag in case["expect_tools_any"]), (
            f"expected a tool matching one of {case['expect_tools_any']}{ctx}"
        )
    for frag in case.get("expect_tools_all", []):
        assert any(frag in t for t in tools), f"no executed tool matches {frag!r}{ctx}"
    if case.get("expect_min_steps"):
        assert len(tools) >= case["expect_min_steps"], (
            f"expected >= {case['expect_min_steps']} executed steps, got {len(tools)}{ctx}"
        )
    for frag in case["forbid_tools"]:
        # Exact-fragment check, but don't let e.g. "users_per_group" ban
        # "users_per_group_all" unless explicitly listed — match whole ids.
        assert not any(t.endswith(frag) or t == frag for t in tools), f"forbidden tool {frag!r} was executed{ctx}"
    # Reply fragments match ignoring punctuation and spacing: Sisense renders
    # one role three ways — "super" (internal), "sysAdmin" (ROLE_NAME in user
    # listings) and "Sys. Admin" (displayName from get_roles) — and which one
    # a reply quotes depends on the tool that fetched it. The identity pair
    # ({user_a_role}/{user_a_role_alt}) covers internal-vs-display; squashing
    # covers the punctuation variants, so "sys. admin" satisfies "sysadmin"
    # without the case enumerating spellings forever.
    squashed = re.sub(r"[^a-z0-9]", "", reply)

    def _in_reply(frag: str) -> bool:
        return frag in reply or re.sub(r"[^a-z0-9]", "", frag) in squashed

    if case["expect_reply_any"]:
        assert any(_in_reply(frag) for frag in case["expect_reply_any"]), (
            f"reply lacks all of {case['expect_reply_any']}{ctx}"
        )
    for frag in case["forbid_reply"]:
        # Squashed here too: a leak spelled "Sys. Admin" is still a leak.
        assert not _in_reply(frag), f"reply contains forbidden {frag!r}{ctx}"
