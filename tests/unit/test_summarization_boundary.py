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

A second, narrower widening arrived with pysisense 2.0 (2026-08-31): a call can
now SUCCEED while part of what was asked for did not happen — a typo'd model
among valid ones (`errors`), a share Sisense silently dropped for an inactive
user (`skipped`). The SDK reports those in-band specifically so they are not
invisible, so a step may also contribute `errors_count` / `skipped_count`.

COUNTS ONLY. The names and reasons stay out, exactly as row data does; a count
is the same metadata class as `count` itself, which has always crossed. The
person still gets the full account, because the locally-rendered reply names
each item — a code path, not an LLM call.

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
# Partial outcomes (pysisense 2.0) — a SUCCESS that did not do all of it
# ---------------------------------------------------------------------------
# get_unused_columns_bulk -> {"results": [...], "errors": [{"ref", "error"}]}
# add_datamodel_shares    -> {..., "skipped": [{"name", "type", "reason"}]}
# Both ride on ok:true, so without special handling the model is told the call
# worked and nothing else — the invisible-skip bug, reproduced on our side of
# the boundary after we had it fixed upstream.
PARTIAL_BULK = {
    "tool_id": "access_management.get_unused_columns_bulk",
    "ok": True,
    "result": {
        "results": [{"table": "Commerce", "column": "Revenue", "used": True}],
        "errors": [{"ref": SECRET, "error": f"model {SECRET} not found (HTTP 404)"}],
    },
}
PARTIAL_SHARES = {
    "tool_id": "datamodel.add_datamodel_shares",
    "ok": True,
    "result": {"success": True, "new_shares": 2, "skipped": [{"name": SECRET, "type": "user", "reason": "inactive"}]},
}


class TestPartialOutcomesCrossAsCountsOnly:
    def test_rows_under_a_results_key_still_get_counted(self):
        """2.0 moved bulk rows from a bare list into {"results": [...]}. Miss it
        and `count` silently vanishes — summ-off would see only {tool, ok}."""
        rec = m._metadata_record("bulk", PARTIAL_BULK)
        assert rec["count"] == 1

    @pytest.mark.parametrize("result,key", [(PARTIAL_BULK, "errors_count"), (PARTIAL_SHARES, "skipped_count")])
    def test_the_model_learns_that_something_was_missed(self, result, key):
        rec = m._metadata_record("t", result)
        assert rec[key] == 1
        assert rec["ok"] is True, "the call did succeed; only part of it was skipped"

    @pytest.mark.parametrize("result", [PARTIAL_BULK, PARTIAL_SHARES])
    def test_but_never_which_item_or_why(self, result):
        """The whole point of the widening is a COUNT. The identifier and the
        SDK's reason are result data and stay on the user's screen."""
        rec = m._metadata_record("t", result)
        assert SECRET not in json.dumps(rec)
        assert "not found" not in json.dumps(rec) and "inactive" not in json.dumps(rec)

    @pytest.mark.parametrize("result", [PARTIAL_BULK, PARTIAL_SHARES])
    def test_nothing_beyond_the_agreed_keys_crosses(self, result):
        rec = m._metadata_record("t", result)
        assert set(rec) <= {"tool", "ok", "count", "errors_count", "skipped_count"}

    def test_a_clean_run_carries_no_partial_keys(self):
        """No crying wolf: empty errors/skipped lists must not add keys, or every
        successful bulk call would look partially failed."""
        clean = {"ok": True, "result": {"results": [{"a": 1}], "errors": []}}
        rec = m._metadata_record("bulk", clean)
        assert set(rec) == {"tool", "ok", "count"}

    @pytest.mark.parametrize(
        "result,fragment",
        [(PARTIAL_BULK, "not found"), (PARTIAL_SHARES, "inactive")],
    )
    def test_the_person_is_told_which_item_and_why(self, result, fragment):
        """The other half of the contract: what the model may not see, the user
        must. This is the locally-rendered reply — a code path, no LLM."""
        line = registry_m._describe_tool_result(result["tool_id"], result)
        assert "partly succeeded" in line
        assert SECRET in line and fragment in line


# ---------------------------------------------------------------------------
# The payload's own verdict — the SDK can fail by RETURNING a failure report
# (found live 2026-08-14: migrate_all_users 66/66 failed, wrapper ok:true,
# run log said "succeeded"). The wrapper proves the call executed; the payload
# says what actually happened, and what-actually-happened is what users get.
# ---------------------------------------------------------------------------
PAYLOAD_FAILED = {
    "ok": True,  # wrapper: the call executed
    "result": {
        "ok": False,  # the SDK's verdict
        "status": "failed",
        "success_count": 0,
        "failed_count": 66,
        "results": [{"name": SECRET, "status": "Failed"}],
        "raw_error": {"error": {"message": "username/email already exists", "status": 400}},
    },
}
PAYLOAD_PARTIAL = {
    "ok": True,
    "result": {"ok": False, "status": "failed", "succeeded_count": 228, "failed_count": 67, "total_count": 295},
}


class TestPayloadVerdict:
    def test_effective_ok_believes_the_payload(self):
        assert m._effective_ok(PAYLOAD_FAILED) is False
        assert m._effective_ok({"ok": True, "result": {"ok": True}}) is True
        assert m._effective_ok({"ok": True, "result": [1, 2]}) is True, "no verdict → wrapper stands"
        assert m._effective_ok({"ok": True, "result": {"rows": 3}}) is True, "no verdict → wrapper stands"
        assert m._effective_ok({"ok": False, "result": {"ok": True}}) is False, "wrapper failure is final"

    def test_metadata_carries_the_sdk_reason_and_nothing_more(self):
        rec = m._metadata_record("migration.migrate_all_users", PAYLOAD_FAILED)
        assert rec["ok"] is False
        assert rec["error"] == "username/email already exists"
        assert SECRET not in json.dumps(rec), "the reason crosses; the rows never do"

    def test_describe_says_failed_in_the_sdks_words(self):
        text = m._describe_tool_result("migration.migrate_all_users", PAYLOAD_FAILED)
        assert "failed" in text
        assert "username/email already exists" in text
        assert "succeeded" not in text.split("—")[0], "never label a failed payload 'succeeded'"

    def test_describe_reports_partial_failure_with_the_sdks_counts(self):
        text = m._describe_tool_result("migration.migrate_all_datamodels", PAYLOAD_PARTIAL)
        assert "completed with failures" in text
        assert "228 succeeded" in text and "67 failed" in text

    def test_describe_uses_the_sdks_counters_on_success_too(self):
        """The deterministic 'what was migrated' summary — counters from the
        payload, no LLM, no interpretation."""
        ok_result = {"ok": True, "result": {"ok": True, "succeeded_count": 295, "total_count": 295}}
        text = m._describe_tool_result("migration.migrate_all_datamodels", ok_result)
        assert "295 of 295 migrated" in text

    def test_failed_titles_come_from_the_sdks_own_list_capped_at_three(self):
        """WHICH ones failed must reach the user, capped so a 500-item failure
        does not become the whole reply.

        The wording moved on 2026-09-01: a `failed` LIST now supplies the reason
        itself (naming each item and quoting the SDK's per-item `reason`), so the
        older "failures include: …" parenthetical would repeat the same titles.
        The contract being guarded is the titles and the cap, not the phrasing —
        asserting the exact sentence would have blocked a strictly better one.
        """
        payload = {
            "ok": False,
            "failed_count": 5,
            "failed": [{"title": f"Model {i}"} for i in range(5)],
        }
        text = m._describe_tool_result("migration.migrate_all_datamodels", {"ok": True, "result": payload})
        assert "Model 0" in text and "Model 1" in text and "Model 2" in text
        assert "2 more" in text, "the cap must say how many it did not name"
        assert "Model 3" not in text and "Model 4" not in text

    def test_a_list_shaped_total_failure_is_not_called_succeeded(self):
        """migrate_dashboards returns bare lists with no ok/status/count field.
        Live 2026-09-01: succeeded=0, failed=1, and the user was told it
        succeeded — the exact failure mode this whole class exists to catch."""
        payload = {
            "succeeded": [],
            "skipped": [],
            "failed": [{"title": "Blox Multi Select function", "reason": "no source->target mapping"}],
        }
        wrapped = {"ok": True, "result": payload}
        assert m._effective_ok(wrapped) is False
        text = m._describe_tool_result("migration.migrate_dashboards", wrapped)
        assert "failed" in text
        assert "Blox Multi Select function" in text and "no source->target mapping" in text
        assert "succeeded" not in text.split("—")[0]

    def test_a_list_shaped_partial_still_counts_as_ok_but_says_what_failed(self):
        """Some landed, some did not. Calling the whole call failed would be as
        wrong as calling it clean — name the casualties and move on."""
        payload = {
            "succeeded": [{"title": "A"}],
            "skipped": [],
            "failed": [{"title": "B", "reason": "datasource missing"}],
        }
        wrapped = {"ok": True, "result": payload}
        assert m._effective_ok(wrapped) is True
        text = m._describe_tool_result("migration.migrate_dashboards", wrapped)
        assert "partly succeeded" in text and "B" in text and "datasource missing" in text


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


# ---------------------------------------------------------------------------
# Shrinker sizing — a SIZE guard that must not become a silent data filter
# ---------------------------------------------------------------------------
# Found live 2026-09-01: the per-object key cap was 10 while the pysisense 2.0
# canonical user row is 12, so GROUPS and GROUP_IDS were dropped from every
# summarized user record. The agent then told a user "no group information is
# provided in the user record" — which was honest; we had removed it. An eval
# asserting on group membership is what surfaced it.
_CANONICAL_USER_ROW = {
    "USER_ID": "u1",
    "USER_NAME": "a@b.com",
    "EMAIL": "a@b.com",
    "FIRST_NAME": "A",
    "LAST_NAME": "B",
    "IS_ACTIVE": True,
    "ROLE_ID": "r1",
    "ROLE_NAME": "sysAdmin",
    "ROLE_DISPLAY_NAME": "sysAdmin",
    "ROLE_RAW_NAME": "super",
    "GROUPS": ["Everyone"],
    "GROUP_IDS": ["g1"],
}


class TestShrinkerKeepsWholeRecords:
    def test_the_canonical_user_row_survives_intact(self):
        """Every field, not most of them. This row is what answers "which groups
        is X in", and the fields at risk are the ones dict order puts last."""
        out = registry_m._shrink_for_llm(_CANONICAL_USER_ROW)
        assert set(out) == set(_CANONICAL_USER_ROW), "the shrinker dropped fields from a single record"
        assert out["GROUPS"] == ["Everyone"]

    def test_the_key_cap_clears_the_widest_record_we_ship(self):
        """Dashboard records are the widest at 28 keys. The cap must sit above
        them, or every summarized dashboard silently loses most of itself."""
        assert registry_m.MAX_KEYS_PER_OBJECT_FOR_LLM >= 28

    def test_a_full_page_of_wide_records_still_fits_the_budget(self):
        """The two limits move together: raising the key cap without the total
        budget collapses a 20-row list to one row. Guards that pairing."""
        wide = [{f"f{i}": f"v{i}{'x' * 12}" for i in range(28)}] * registry_m.MAX_LIST_ITEMS_FOR_LLM
        out = registry_m._shrink_for_llm(wide)
        assert len(out) == registry_m.MAX_LIST_ITEMS_FOR_LLM, "budget truncated a full page of wide rows"
        assert all(len(r) == 28 for r in out if isinstance(r, dict))

    def test_oversized_payloads_are_still_bounded(self):
        """The guard must still guard — this is a size limit, not a free pass."""
        huge = [{"k": "x" * 400} for _ in range(500)]
        out = registry_m._shrink_for_llm(huge)
        assert len(json.dumps(out)) <= registry_m.MAX_TOTAL_LENGTH_FOR_LLM * 1.2
