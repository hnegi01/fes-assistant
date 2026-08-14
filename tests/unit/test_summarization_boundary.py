"""
The summarization kill switch — what may reach the MODEL when it is off.

`allow_summarization=false` is a privacy switch over result DATA, enforced in
code and never by asking the LLM nicely. With it off, a step contributes
`{tool, ok, count}` to the LLM's history — plus `error` when it FAILED.

That failure exception is deliberate (2026-08-08) and it is the interesting line
to guard. Without it the loop is blind exactly when it needs to think: a failed
create left the decide call with `ok: false` alone and it invented a cause. An
error normally restates what the user already typed, so it rarely adds anything
the model has not seen — but not never, and that residual exposure is documented
in README.md rather than hidden.

What must NEVER cross on a summ-off turn is result data: rows, field values,
payloads from a SUCCESSFUL call. That is the boundary these tests defend.

The person always sees everything either way: the reply names the failure
(`_describe_tool_result`, a code path) and the UI shows the raw payload.
"""

import json

import pytest

import backend.agent._registry as registry_m
import backend.agent.llm_agent as m

SECRET = "jane.doe@acme.com"
FAILED = {
    "tool_id": "access_management.get_user",
    "ok": False,
    "error": f"user {SECRET} not found in tenant acme-prod",
    "error_type": "ValueError",
}
SUCCEEDED = {
    "tool_id": "access_management.get_users_all",
    "ok": True,
    "result": [{"email": SECRET, "role": "admin"}, {"email": "bob@acme.com", "role": "viewer"}],
}


def _call(tool_id):
    return {"id": "c1", "type": "function", "function": {"name": tool_id, "arguments": "{}"}}


# ---------------------------------------------------------------------------
# _metadata_record — the only shape allowed through
# ---------------------------------------------------------------------------
class TestMetadataRecord:
    def test_successful_step_exposes_only_tool_ok_count(self):
        rec = m._metadata_record("access_management.get_users_all", SUCCEEDED)
        assert set(rec) == {"tool", "ok", "count"}
        assert rec == {"tool": "access_management.get_users_all", "ok": True, "count": 2}

    def test_failed_step_exposes_the_reason_and_nothing_more(self):
        """The narrow exception: a failure reason crosses so recovery can reason
        from the truth instead of guessing. Still no payload, no rows."""
        rec = m._metadata_record("access_management.get_user", FAILED)
        assert set(rec) == {"tool", "ok", "error"}
        assert rec["ok"] is False
        assert rec["error"] == FAILED["error"]

    def test_a_successful_step_never_carries_an_error_key(self):
        rec = m._metadata_record("access_management.get_users_all", SUCCEEDED)
        assert "error" not in rec

    def test_row_data_never_appears(self):
        rec = m._metadata_record("access_management.get_users_all", SUCCEEDED)
        assert SECRET not in json.dumps(rec)

    def test_count_only_for_list_payloads(self):
        rec = m._metadata_record("x.y", {"ok": True, "result": {"email": SECRET}})
        assert set(rec) == {"tool", "ok"}

    @pytest.mark.parametrize("result", [None, "a string", 42, [], {"ok": True}])
    def test_tolerates_odd_payloads(self, result):
        rec = m._metadata_record("x.y", result)
        assert set(rec) <= {"tool", "ok", "count"}


# ---------------------------------------------------------------------------
# _transcript_step — the single point the boundary is enforced at
# ---------------------------------------------------------------------------
class TestTranscriptBoundary:
    def _text(self, result, summ_on):
        return json.dumps(
            m._transcript_step(_call("access_management.get_user"), "access_management.get_user", result, summ_on)
        )

    def test_summ_off_passes_the_failure_reason(self):
        """Intentional: the decide call needs a real reason to replan on."""
        assert "not found in tenant acme-prod" in self._text(FAILED, summ_on=False)

    def test_summ_off_leaks_nothing_from_a_success(self):
        assert SECRET not in self._text(SUCCEEDED, summ_on=False)

    def test_summ_on_does_pass_the_data(self):
        """The control: with the switch ON the same call carries the payload, so
        the assertion above is testing the switch and not a broken fixture."""
        assert SECRET in self._text(SUCCEEDED, summ_on=True)

    def test_the_only_thing_summ_off_adds_beyond_metadata_is_the_reason(self):
        """Belt and braces on the exception's SCOPE: for a failure, the reason is
        allowed; for a success, nothing beyond {tool, ok, count} ever is."""
        import backend.agent.llm_agent as mod

        for tool, result in (("t.fail", FAILED), ("t.ok", SUCCEEDED)):
            rec = mod._metadata_record(tool, result)
            assert set(rec) <= {"tool", "ok", "count", "error"}
            if result["ok"]:
                assert "error" not in rec


# ---------------------------------------------------------------------------
# The user still gets the whole story — these are code paths, not LLM calls
# ---------------------------------------------------------------------------
class TestUserStillSeesTheFailure:
    def test_the_reply_names_the_error_verbatim(self):
        line = registry_m._describe_tool_result("access_management.get_user", FAILED)
        assert "not found in tenant acme-prod" in line

    def test_local_rendering_covers_every_result(self):
        out = m._describe_results_local(
            [("access_management.get_user", FAILED), ("access_management.get_users_all", SUCCEEDED)]
        )
        assert "not found in tenant acme-prod" in out
        assert "Found 2 results" in out
