"""The two secret scrubbers must stay identical, and must catch real field names.

They drifted: `backend/agent/_config.py` substring-matched its markers while
`mcp_server/tools_core.py` exact-matched most of them. So `aws_secret_key`,
`aws_access_key`, `db_password`, `client_secret`, `private_key` and
`connection_string` were redacted on one side and written in CLEARTEXT on the
other — into `logs/server_mutations.log`, the always-on mutation audit, every
time an allowlisted connection tool ran. Those are exactly the fields
`datamodel.create_connections` and `datamodel.update_connection` take, and the
registry's own curated examples for them carry AWS keys and a BigQuery service
account private key.

Two properties are pinned here: the two implementations agree on every input,
and both catch the credential names that actually reach them.
"""

from __future__ import annotations

import pytest

from backend.agent._config import _scrub_secrets as backend_scrub
from mcp_server.tools_core import _scrub_secrets as mcp_scrub

REDACTED = "***REDACTED***"

# Field names that MUST be redacted. Every one of these is either taken by an
# allowlisted tool or is a common variant that a future SDK release could add.
MUST_REDACT = [
    "token",
    "Token",
    "source_token",
    "target_token",
    "access_token",
    "refresh_token",
    "password",
    "Password",
    "db_password",
    "user_password",
    "passwd",
    "pwd",
    "secret",
    "client_secret",
    "aws_secret_key",
    "secret_key",
    "SECRET",
    "api_key",
    "API_KEY",
    "api-key",
    "apikey",
    "x_api_key",
    "authorization",
    "Authorization",
    "auth",
    "credential",
    "credentials",
    "aws_credentials",
    "private_key",
    "privateKey",
    "access_key",
    "aws_access_key",
    "accessKey",
    "connection_string",
    "connectionString",
    "passphrase",
    "key",
    "pass",
]

# Field names that must NOT be redacted — over-redaction destroys the
# debuggability the logs exist for.
MUST_KEEP = [
    "dashboard",
    "title",
    "name",
    "author",
    "keys",
    "monkey",
    "keyword",
    "tool_id",
    "datamodel",
    "oid",
    "email",
    "module",
    "passthrough_count",
]


class TestBothImplementationsAgree:
    @pytest.mark.parametrize("field", MUST_REDACT + MUST_KEEP)
    def test_same_verdict_on_both_sides(self, field: str) -> None:
        payload = {field: "sensitive"}
        assert backend_scrub(payload) == mcp_scrub(payload), (
            f"scrubbers disagree on {field!r} — they must stay byte-identical; "
            f"backend={backend_scrub(payload)} mcp={mcp_scrub(payload)}"
        )

    def test_agree_on_a_realistic_nested_connection_payload(self) -> None:
        payload = {
            "connection": {
                "provider": "S3",
                "aws_access_key": "AKIA…",
                "aws_secret_key": "…",
                "bucket": "analytics",
            },
            "datasets": [{"name": "orders", "db_password": "…"}],
            "datamodel": "Sample Retail",
        }
        assert backend_scrub(payload) == mcp_scrub(payload)


class TestCredentialsAreRedacted:
    @pytest.mark.parametrize("scrub", [backend_scrub, mcp_scrub], ids=["backend", "mcp"])
    @pytest.mark.parametrize("field", MUST_REDACT)
    def test_redacted(self, scrub, field: str) -> None:
        assert scrub({field: "sensitive"})[field] == REDACTED

    @pytest.mark.parametrize("scrub", [backend_scrub, mcp_scrub], ids=["backend", "mcp"])
    @pytest.mark.parametrize("field", MUST_KEEP)
    def test_kept(self, scrub, field: str) -> None:
        assert scrub({field: "visible"})[field] == "visible"

    @pytest.mark.parametrize("scrub", [backend_scrub, mcp_scrub], ids=["backend", "mcp"])
    def test_redaction_reaches_nested_dicts_and_lists(self, scrub) -> None:
        out = scrub({"a": {"b": {"aws_secret_key": "s"}}, "c": [{"password": "s"}, {"ok": "v"}]})
        assert out["a"]["b"]["aws_secret_key"] == REDACTED
        assert out["c"][0]["password"] == REDACTED
        assert out["c"][1]["ok"] == "v"

    @pytest.mark.parametrize("scrub", [backend_scrub, mcp_scrub], ids=["backend", "mcp"])
    def test_non_dict_values_pass_through(self, scrub) -> None:
        assert scrub("plain") == "plain"
        assert scrub([1, 2]) == [1, 2]
        assert scrub(None) is None
