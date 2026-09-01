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


@pytest.fixture(scope="session")
def eval_identities(integration_config) -> Dict[str, str]:
    """Real identities on the tenant that the chat eval battery asserts on.

    The repo is public, so no real user appears in committed test data —
    eval cases carry {user_a_email}-style placeholders resolved from this
    gitignored config section. user_a_email_typo is derived (the email with
    its final dot-segment dropped) to keep the two forms in sync."""
    ids = integration_config.get("eval_identities")
    if not ids:
        pytest.skip("eval_identities not set in integration_config.yaml (see the example file)")
    required = ("user_a_email", "user_a_group", "user_a_role", "user_b_email", "user_b_role")
    # A Sisense role has an internal name and a display name for the SAME role
    # ("super" is shown as "sysAdmin"); which one a reply quotes depends on the
    # tool that fetched it, so cases assert on either. Optional: falls back to
    # the primary when a deployment has no distinct internal name.
    optional = ("user_a_role_alt", "user_b_role_alt")
    # Keys some cases need but most don't. Absent = those cases skip themselves
    # (see needs_identities in test_evals_planner.py); adding one here never
    # skips the whole battery on a config that predates it.
    per_case = ("datamodel_name",)
    missing = [k for k in required if not ids.get(k)]
    if missing:
        pytest.skip(f"eval_identities missing keys: {missing}")
    resolved = {k: str(ids[k]) for k in required}
    for k in optional:
        resolved[k] = str(ids.get(k) or resolved[k.replace("_alt", "")])
    for k in per_case:
        if ids.get(k):
            resolved[k] = str(ids[k])
    resolved["user_a_email_typo"] = resolved["user_a_email"].rsplit(".", 1)[0]
    return resolved
