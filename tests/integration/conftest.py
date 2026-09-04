"""
Shared fixtures for integration tests.

Credentials live in ONE place — tests/integration/integration_config.yaml
(gitignored). Copy integration_config.example.yaml to that name and fill it in.

If the config file is missing or still holds the example placeholder values,
every integration test is SKIPPED rather than failed. That keeps CI and any
unconfigured machine green while still letting these tests run end-to-end
locally before a release.

Usage in a test:

    @pytest.mark.integration
    def test_something(backend_url, tenant_config):
        ...
"""

import re
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "integration_config.yaml"

# Markers of an unfilled template — if any appears, treat config as not set up.
_PLACEHOLDER_MARKERS = ("your-real-token", "your-sisense-instance.example.com")


def _load_config() -> Dict[str, Any]:
    """Load integration_config.yaml, or skip the test if it's unusable."""
    if not _CONFIG_PATH.exists():
        pytest.skip(
            f"integration_config.yaml not found at {_CONFIG_PATH}. "
            f"Copy integration_config.example.yaml and fill in credentials."
        )

    with _CONFIG_PATH.open() as fh:
        data = yaml.safe_load(fh) or {}

    tenant = data.get("tenant_config") or {}
    domain = str(tenant.get("domain", ""))
    token = str(tenant.get("token", ""))
    if not domain or not token or any(mark in (domain + token) for mark in _PLACEHOLDER_MARKERS):
        pytest.skip(
            "integration_config.yaml is not configured (still has placeholder "
            "values). Fill in a real Sisense domain and token to run these tests."
        )

    return data


@pytest.fixture(scope="session")
def integration_config() -> Dict[str, Any]:
    """Full parsed integration config (skips if missing/unconfigured)."""
    return _load_config()


@pytest.fixture(scope="session")
def backend_url(integration_config) -> str:
    return integration_config.get("backend_url", "http://localhost:8001")


@pytest.fixture(scope="session")
def tenant_config(integration_config) -> Dict[str, Any]:
    """Normalized to the shape /agent/turn expects. The yaml documents the
    human-facing key `verify_ssl`, but the backend/SDK read `ssl` — passing the
    section verbatim silently dropped it, and the SDK defaulted to https
    (is_ssl=True). Invisible while every test tenant used verify_ssl: true;
    surfaced on an http-only sandbox (connection resets, 2026-08-27)."""
    raw = integration_config["tenant_config"]
    domain = str(raw["domain"]).strip().rstrip("/")
    if domain and "://" not in domain:
        domain = f"https://{domain}"  # bare domains default to https, like the UI
    # Port handling lives in the SDK: with ssl=False it calls
    # http://<domain>:30845 (Linux default; a `port` config key overrides).
    return {
        "domain": domain,
        "token": raw["token"],
        "ssl": raw.get("verify_ssl", True),
    }


def _side(raw: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """One side of a migration pair, in the shape the backend actually reads."""
    domain = str(raw.get(f"{prefix}_url", "")).strip().rstrip("/")
    if domain and "://" not in domain:
        domain = f"https://{domain}"  # bare domains default to https, like the UI
    return {"domain": domain, "token": raw.get(f"{prefix}_token"), "ssl": raw.get(f"{prefix}_verify_ssl", True)}


@pytest.fixture(scope="session")
def migration_config(integration_config) -> Dict[str, Any]:
    """Normalized to the shape /agent/turn expects: {"source": {...}, "target": {...}}.

    The yaml documents flat, human-facing keys (source_url, source_verify_ssl),
    but McpClient._with_migration reads a NESTED pair and maps domain/token/ssl
    onto source_*/target_* arguments — the same shape frontend/app.py builds.
    Passing the yaml section verbatim meant the SDK received no credentials at
    all: "Migration tools require source_domain and source_token to be
    provided."

    This is the migration twin of the tenant_config bug fixed 2026-08-27, and it
    hid for the same reason — the migration battery deliberately sends no
    approved_keys, so every case stops at the approval gate and nothing ever
    executes. Planning never touches credentials, so 11 cases passed against a
    config that could not have run. Found 2026-09-01 on the first real
    source→target migration.
    """
    cfg = integration_config.get("migration_config")
    if not cfg:
        pytest.skip("migration_config not set in integration_config.yaml")
    normalized = {"source": _side(cfg, "source"), "target": _side(cfg, "target")}
    for side in ("source", "target"):
        if not normalized[side]["domain"] or not normalized[side]["token"]:
            pytest.skip(f"migration_config {side} is missing a url or token")
    return normalized


# What a discovered identity MUST have: other members sharing its role and its
# group, so "list everyone with that role" and "show all users in that group"
# have real answers rather than being trivially satisfied.
_ROLE_MEMBERS_MIN = 2
_GROUP_MEMBERS_MIN = 2
# What it IDEALLY has — a preference the sort expresses, never a filter. Small
# is better: a 59-member group makes the reply enormous and the turn slow for an
# assertion that only checks the group's name appears. But rejecting large ones
# outright left this tenant with no subject at all, so they stay eligible and
# merely sort last.
_ROLE_MEMBERS_IDEAL = 6
_GROUP_MEMBERS_IDEAL = 4
# Each candidate costs one users_per_group call to confirm membership, so stop
# after this many rather than sweeping every pairing on an 80-user tenant.
_DISCOVERY_MAX_VERIFY = 12


def _squash(value: str) -> str:
    """Lowercase, strip everything but letters and digits — the same
    normalization the eval battery's reply matcher uses, so discovery can tell
    when two identifiers would be indistinguishable to an assertion."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _member_emails(rows: Any) -> set:
    """Lowercased emails from a users_per_group result."""
    if not isinstance(rows, list):
        return set()
    out = set()
    for row in rows:
        if isinstance(row, dict):
            email = row.get("EMAIL") or row.get("email") or row.get("USER_NAME")
            if email:
                out.add(str(email).lower())
    return out


def _discover_eval_identities(
    tenant: Dict[str, Any],
    prefer_a: str | None = None,
    prefer_b: str | None = None,
    prefer_datamodel: str | None = None,
) -> Dict[str, str]:
    """Find identities matching what the chat eval battery needs, on THIS tenant.

    Read-only — get_users_all, users_per_group, get_all_datamodel. Skips (never
    fails) when the tenant cannot supply a usable subject, matching how the
    rest of this file treats an environment it cannot work with.

    The group is VERIFIED from the group side before being accepted: the user
    record and the group listing are known to disagree about membership
    (`Everyone` is filtered out of some user records upstream), so trusting
    GROUPS alone can hand the battery a group the agent will not report.
    """
    from collections import Counter

    from pysisense import AccessManagement, DataModel, SisenseClient

    client = SisenseClient(
        config_file=None,
        domain=tenant["domain"],
        token=tenant["token"],
        is_ssl=tenant.get("ssl", True),
    )
    access = AccessManagement(client)

    users = access.get_users_all()
    if not isinstance(users, list) or not users:
        pytest.skip(f"cannot discover eval identities — get_users_all returned {users!r}")

    # Sorted so a given tenant always yields the same subject: an eval battery
    # that silently changes who it asks about is not a regression detector.
    active = sorted(
        (u for u in users if u.get("IS_ACTIVE") and u.get("EMAIL") and u.get("ROLE_NAME")),
        key=lambda u: str(u["EMAIL"]).lower(),
    )
    if not active:
        pytest.skip("cannot discover eval identities — tenant has no active users with a role")

    role_members = Counter(str(u["ROLE_NAME"]) for u in active)
    group_members = Counter(g for u in active for g in (u.get("GROUPS") or []))

    # Groups nearly everyone is in. Detected by share of the tenant, not by
    # name, so this holds on a deployment that renames or adds one.
    tenant_wide = {g for g, n in group_members.items() if n >= 0.9 * len(active)}

    def _assertable_group(user: Dict[str, Any]) -> tuple:
        """The ONE group a reply about this user must name: (tier, group|None).

        Two properties make a subject sound, and both were learned the hard way
        live on 2026-09-03:

        UNAMBIGUOUS. The group-membership case asserts the reply names
        {user_a_group}, so a user in several groups turns it into a coin flip —
        the agent answered with "admins" for a user pinned to "Solution
        Consultants" and failed a case it had got right. The subject must have
        exactly one group of its own. (A purpose-made user in `Everyone` only
        was the original fixture precisely because it has this property.)

        DISTINGUISHABLE FROM THE EMAIL. The summ-off case forbids the group name
        from appearing, and the battery matches with punctuation stripped, so
        `assaftest2@sisense.com` in group `assaf_test_2` makes it unfalsifiable:
        both squash to `assaftest2`, and the honest reply names the user, so
        "the group leaked" fires on a reply that leaked nothing.
        """
        groups = [g for g in (user.get("GROUPS") or []) if group_members[g] >= _GROUP_MEMBERS_MIN]
        own = [g for g in groups if g not in tenant_wide]
        if len(own) == 1:
            chosen, tier = own[0], 0
        elif not own and groups:
            # Only in the tenant-wide group(s) — still unambiguous, but the
            # agent may name either of them, so rank it below a real group.
            chosen, tier = sorted(groups, key=lambda g: (group_members[g], g))[0], 1
        else:
            return (2, None)
        email_squashed = _squash(str(user.get("EMAIL") or ""))
        chosen_squashed = _squash(chosen)
        if not chosen_squashed or chosen_squashed in email_squashed or email_squashed in chosen_squashed:
            return (2, None)
        return (tier, chosen)

    # Only two things are REQUIRED — a role with other members, and one
    # unambiguous, non-colliding group. Size is a PREFERENCE expressed by the
    # sort, never a filter: requiring a small role rejected `admin` (29 of the
    # 74 active users here) and discovery found nobody at all. A tenant whose
    # only groups are large should still get a subject, just not a pretty one.
    candidates = []
    for user in active:
        if role_members[str(user["ROLE_NAME"])] < _ROLE_MEMBERS_MIN:
            continue
        tier, group = _assertable_group(user)
        if not group:
            continue
        candidates.append(
            (
                tier,
                abs(group_members[group] - _GROUP_MEMBERS_IDEAL),
                abs(role_members[str(user["ROLE_NAME"])] - _ROLE_MEMBERS_IDEAL),
                str(user["EMAIL"]).lower(),
                user,
                group,
            )
        )
    # Deterministic: same tenant, same subject. An eval battery that silently
    # changes who it asks about is not a regression detector.
    candidates.sort(key=lambda c: c[:4])
    if prefer_a:
        wanted = prefer_a.strip().lower()
        candidates.sort(key=lambda c: c[3] != wanted)

    user_a: Dict[str, Any] | None = None
    group_a: str | None = None
    for *_, user, group in candidates[:_DISCOVERY_MAX_VERIFY]:
        if str(user["EMAIL"]).lower() in _member_emails(access.users_per_group(group)):
            user_a, group_a = user, group
            break

    if not user_a:
        pytest.skip(
            "cannot discover eval identities — no active user on this tenant has a role with "
            f"{_ROLE_MEMBERS_MIN}+ members and a group of {_GROUP_MEMBERS_MIN}+ whose name is "
            f"distinguishable from their email (checked {len(candidates[:_DISCOVERY_MAX_VERIFY])} candidates)"
        )

    # user_b drives "list all OTHER users with this role", so it only needs a
    # shared role. A different role from user_a broadens what the battery
    # covers; the same user is an acceptable fallback.
    wanted_b = (prefer_b or "").strip().lower()
    user_b = next(
        (
            u
            for u in sorted(active, key=lambda u: str(u["EMAIL"]).lower() != wanted_b)
            if role_members[str(u["ROLE_NAME"])] >= _ROLE_MEMBERS_MIN
            and (str(u["EMAIL"]).lower() == wanted_b or str(u["ROLE_NAME"]) != str(user_a["ROLE_NAME"]))
        ),
        user_a,
    )

    def _role_pair(user: Dict[str, Any]) -> tuple:
        """The two vocabularies for one role. Sisense names the same role twice
        ('super' internally, 'sysAdmin' for display) and which one a reply
        quotes depends on which tool fetched it, so cases accept either."""
        primary = str(user["ROLE_NAME"])
        alt = str(user.get("ROLE_RAW_NAME") or user.get("ROLE_DISPLAY_NAME") or primary)
        return primary, (alt if alt.lower() != primary.lower() else primary)

    role_a, role_a_alt = _role_pair(user_a)
    role_b, role_b_alt = _role_pair(user_b)

    resolved = {
        "user_a_email": str(user_a["EMAIL"]),
        "user_a_group": str(group_a),
        "user_a_role": role_a,
        "user_a_role_alt": role_a_alt,
        "user_b_email": str(user_b["EMAIL"]),
        "user_b_role": role_b,
        "user_b_role_alt": role_b_alt,
    }

    # datamodel_name is per-case (absent = those cases skip themselves), so a
    # tenant with no models must not take the whole battery down.
    models = DataModel(client).get_all_datamodel()
    titles = (
        [str(m.get("title") or m.get("name")) for m in models if isinstance(m, dict)]
        if isinstance(models, list)
        else []
    )
    titles = [t for t in titles if t and t != "None"]
    if prefer_datamodel and prefer_datamodel in titles:
        resolved["datamodel_name"] = prefer_datamodel
    elif titles:
        resolved["datamodel_name"] = sorted(titles)[0]

    return resolved


@pytest.fixture(scope="session")
def eval_identities(integration_config, tenant_config) -> Dict[str, str]:
    """Real identities on the tenant that the chat eval battery asserts on.

    The repo is public, so no real user appears in committed test data —
    eval cases carry {user_a_email}-style placeholders, and these are
    DISCOVERED FROM THE LIVE TENANT rather than written down.

    They used to be hardcoded in the gitignored config, and that broke the
    moment the sandbox was swapped (2026-09-03): the file still named a user
    who had been deleted, so four cases asserted on a role nobody held and
    failed while the agent was behaving perfectly — it said the user did not
    exist, which was true. The identities were never the point. What the cases
    need is a SHAPE: a user whose role has other members (so "list everyone
    with that role" has an answer) and a group that user verifiably belongs to.
    Every tenant has those, so find them per-environment instead of pinning
    them to one.

    `eval_identities` in the config is now an optional PREFERENCE, not the
    source of truth: it may name which user to prefer, and that is honoured
    while the user still exists. Their role and group always come off the live
    record, because those attributes are exactly what went stale.

    user_a_email_typo stays derived (the email minus its final dot-segment) to
    keep the two forms in sync."""
    prefs = integration_config.get("eval_identities") or {}
    resolved = _discover_eval_identities(
        tenant_config,
        prefer_a=str(prefs.get("user_a_email") or "") or None,
        prefer_b=str(prefs.get("user_b_email") or "") or None,
        prefer_datamodel=str(prefs.get("datamodel_name") or "") or None,
    )
    resolved["user_a_email_typo"] = resolved["user_a_email"].rsplit(".", 1)[0]
    return resolved
