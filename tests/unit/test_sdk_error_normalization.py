"""
SDK failure reports → ok=False, at both layers that classify a result.

PySisense signals failure in three shapes, and each is caught in a different
place — so this file covers both classifiers together, because a shape that
falls between them is reported to the user as a success:

  {"error": ...}                     MCP boundary  (_sdk_error_payload)
  {"error": ..., "status_code": ...} MCP boundary  — the 1.1.0 HTTP contract
                                     from utils._extract_error_message
  {"success": False, ..., "error":}  backend       (_effective_ok) — the
                                     reference resolvers, which carry real
                                     payload keys and so are correctly NOT
                                     error envelopes

Some PySisense methods report failure by RETURNING a dict instead of raising.
`_sdk_error_payload` is the single place that turns those into `ok: False`, and
everything downstream trusts the resulting flag: the agent's decide call, the
recovery ladder, the critic, and — critically — the summarization-OFF path,
where the payload never reaches the LLM and `ok` is nearly all it has.

So the failure mode this file guards is asymmetric and both directions are real:

  too strict  — a failure envelope the matcher does not recognise is reported as
                a SUCCESS with no rows. Summ-off, the model is told `ok: true`
                and nothing else, and answers as if the call worked.
  too loose   — a data row that happens to carry an "error" field is read as a
                failed call, turning a good result into a false failure.

The matcher was once `list(keys) == ["error"]`, which is safe against the second
and wide open to the first: the SDK enriched the envelope with
`failed_references` (2026-08-29) and every such failure silently became a
success. These tests pin both edges so the next enrichment fails loudly here
rather than quietly in a reply.
"""

from backend.agent._registry import _effective_ok, _payload_failure_reason
from mcp_server.tools_core import _sdk_error_payload

TOOL = "access_management.get_unused_columns_bulk"


class TestFailureEnvelopesAreRecognised:
    def test_bare_error_dict(self):
        """The original shape — a lone error key."""
        out = _sdk_error_payload(TOOL, {"error": "boom"})
        assert out and out["ok"] is False and out["error"] == "boom"
        assert out["error_type"] == "SDKError"

    def test_error_dict_wrapped_in_a_single_item_list(self):
        """List-returning methods wrap their error report in a 1-item list."""
        out = _sdk_error_payload(TOOL, [{"error": "boom"}])
        assert out and out["ok"] is False and out["error"] == "boom"

    def test_enriched_envelope_with_failed_references(self):
        """The shape that exposed the sole-key bug.

        pysisense get_unused_columns_bulk, asked about a data model that does
        not exist: Sisense answers 404 ElasticubeNotFound, the SDK resolver
        packages it, and bulk returns the message PLUS the per-reference detail.
        Two keys — which the old `list(keys) == ["error"]` test rejected, so the
        call came back ok=true with zero rows and the agent reported the missing
        data model as clean.
        """
        result = {
            "error": "None of the given data model references could be processed — 'Slaes_Analytics': 404 not found",
            "failed_references": [{"ref": "Slaes_Analytics", "error": "404 not found"}],
        }
        out = _sdk_error_payload(TOOL, result)
        assert out is not None, "enriched failure envelope must normalise to ok=False"
        assert out["ok"] is False
        assert "Slaes_Analytics" in out["error"]

    def test_enriched_envelope_wrapped_in_a_list(self):
        result = [{"error": "nope", "failed_references": [{"ref": "X", "error": "404"}]}]
        out = _sdk_error_payload(TOOL, result)
        assert out and out["ok"] is False

    def test_http_failure_contract_with_status_code(self):
        """The shape pysisense 1.1.0 returns for any HTTP error.

        utils._extract_error_message documents it as: {"error": str} always,
        plus {"status_code": int} when an HTTP status is available. It is built
        at runtime rather than written as a literal, which is why a source scan
        for error-dicts mostly turns up the one-key form and this one is easy
        to miss. Two keys — so the sole-key matcher rejected it, and every 401 /
        403 / 500 on a path using the helper was reported as a successful call.

        This is the shape the role-token permission matrix will generate on
        purpose, so it has to classify correctly before that work means anything.
        """
        out = _sdk_error_payload(
            TOOL, {"error": "Failed to retrieve dashboards: Access denied (HTTP 403)", "status_code": 403}
        )
        assert out is not None, "the 1.1.0 HTTP failure contract must normalise to ok=False"
        assert out["ok"] is False and "403" in out["error"]


class TestSuccessesAreNotMisreadAsFailures:
    def test_ordinary_row_list_passes_through(self):
        rows = [{"table": "Commerce", "column": "Revenue", "used": True}]
        assert _sdk_error_payload(TOOL, rows) is None

    def test_single_row_list_is_not_an_envelope(self):
        """A 1-item list is unwrapped before matching, so a lone DATA row must
        still survive — the unwrap exists for error reports, not for results."""
        assert _sdk_error_payload(TOOL, [{"table": "T", "column": "C", "used": False}]) is None

    def test_data_row_carrying_an_incidental_error_field_is_not_a_failure(self):
        """Why the key set is closed rather than `"error" in candidate`.

        A per-item status row can legitimately carry an error field; reading it
        as a failed CALL would discard a result that succeeded.
        """
        row = {"build_id": "abc", "status": "done", "error": "row 12 skipped"}
        assert _sdk_error_payload(TOOL, [row]) is None

    def test_empty_error_value_is_not_a_failure(self):
        """`error: None` is the SDK saying "no error", not a failure report."""
        assert _sdk_error_payload(TOOL, {"error": None, "status_code": 200}) is None
        assert _sdk_error_payload(TOOL, {"error": ""}) is None

    def test_non_dict_results_pass_through(self):
        assert _sdk_error_payload(TOOL, None) is None
        assert _sdk_error_payload(TOOL, []) is None
        assert _sdk_error_payload(TOOL, "some string") is None


class TestResolverVerdictsAreHonoured:
    """The reference resolvers report a miss with a `success` flag, not an
    `error` envelope — and they are exposed tools (allowed_tools.txt lines 77
    and 99), so a user can reach them directly.

    Their failure dict carries real payload keys (datamodel_id, datamodel_title),
    so the MCP boundary correctly declines to read it as an error envelope —
    widening that matcher to catch it would make it read data rows as failures.
    The verdict belongs where the other payload verdicts live: _effective_ok,
    which already believes `ok: False` and `status: "failed"`.
    """

    RESOLVER_MISS = {
        "success": False,
        "status_code": 404,
        "datamodel_id": None,
        "datamodel_title": None,
        "error": "Failed to resolve data model by title. Status: 404, Error: ElasticubeNotFound",
    }
    RESOLVER_HIT = {
        "success": True,
        "status_code": 200,
        "datamodel_id": "abc123",
        "datamodel_title": "Sample ECommerce",
        "error": None,
    }

    def test_mcp_boundary_correctly_declines_the_resolver_shape(self):
        """Not a bug — payload keys mean it is not an error envelope."""
        assert _sdk_error_payload("datamodel.resolve_datamodel_reference", self.RESOLVER_MISS) is None

    def test_success_false_is_a_failure(self):
        assert _effective_ok({"tool_id": "t", "ok": True, "result": self.RESOLVER_MISS}) is False

    def test_success_true_is_a_success(self):
        assert _effective_ok({"tool_id": "t", "ok": True, "result": self.RESOLVER_HIT}) is True

    def test_the_reason_is_the_sdks_own_sentence(self):
        """Summarization OFF sends only {tool, ok, error} to the model, so this
        string is the entire account of what went wrong. It must be the SDK's,
        never reconstructed by us."""
        assert _payload_failure_reason(self.RESOLVER_MISS).startswith("Failed to resolve data model by title")

    def test_a_successful_payload_contributes_no_reason(self):
        assert _payload_failure_reason(self.RESOLVER_HIT) == ""

    def test_absent_verdict_leaves_the_wrappers_word_standing(self):
        """Most payloads carry no verdict at all; those must stay successful."""
        assert _effective_ok({"tool_id": "t", "ok": True, "result": [{"table": "T", "column": "C"}]}) is True
        assert _effective_ok({"tool_id": "t", "ok": True, "result": {"title": "Sample ECommerce"}}) is True
