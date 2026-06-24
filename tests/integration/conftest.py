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
    return integration_config["tenant_config"]


@pytest.fixture(scope="session")
def migration_config(integration_config) -> Dict[str, Any]:
    cfg = integration_config.get("migration_config")
    if not cfg:
        pytest.skip("migration_config not set in integration_config.yaml")
    return cfg
