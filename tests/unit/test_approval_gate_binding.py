"""An approval must correspond to a dialog the SERVER actually issued.

The approval key is a pure function of `(tool_id, canonical args)`
(`llm_agent._approval_key`), and the API accepted whatever `approved_keys` the
client sent. So any caller could COMPUTE a key, skip the dialog entirely, and
the mutation would execute — and `logs/mutations.log` would record it as
approved. `docs/security.md` said "nothing that writes runs without a dialog
naming the operation and its arguments", which held only for the shipped UI.

`ApprovalSet` binds the client's keys to the server's own record of what it
proposed, per session, with a TTL.

What this does NOT claim: a scripted client can still drive the dialog and then
answer it. Nothing server-side can prevent that — it is what the API is for.
What is restored is that every executed mutation matches a dialog the server
really rendered, with exactly these arguments, in this session, recently.
"""

from __future__ import annotations

import time

from backend.agent import llm_agent as A

TOOL = "dashboard.delete_dashboard"
ARGS = {"dashboard": "Q4 Revenue", "oid": "abc123"}


def _server_offered(issued: dict, tool: str = TOOL, args: dict | None = None) -> A.ApprovalSet:
    """Simulate the server emitting a dialog, then the client approving it."""
    approved = A.ApprovalSet(issued=issued)
    A.record_issued(approved, tool, args if args is not None else ARGS)
    approved.add(A._approval_key(tool, args if args is not None else ARGS))
    return approved


class TestForgedApprovals:
    def test_precomputed_key_is_rejected_when_no_dialog_was_issued(self) -> None:
        """The attack: compute the key, never see a dialog, execute anyway."""
        forged = A.ApprovalSet([A._approval_key(TOOL, ARGS)], issued={})
        assert A._consume_approval(forged, TOOL, ARGS) is False

    def test_a_rejected_forgery_is_also_discarded_so_it_cannot_be_retried(self) -> None:
        forged = A.ApprovalSet([A._approval_key(TOOL, ARGS)], issued={})
        A._consume_approval(forged, TOOL, ARGS)
        assert A._approval_key(TOOL, ARGS) not in forged

    def test_approval_for_different_arguments_than_were_offered_is_rejected(self) -> None:
        """Offered a delete of one dashboard, approve a delete of another."""
        issued: dict = {}
        approved = A.ApprovalSet(issued=issued)
        A.record_issued(approved, TOOL, ARGS)
        other = {"dashboard": "Payroll", "oid": "zzz999"}
        approved.add(A._approval_key(TOOL, other))
        assert A._consume_approval(approved, TOOL, other) is False

    def test_key_from_another_session_is_rejected(self) -> None:
        """Registries are per-session; a key issued in one is inert in another."""
        session_a: dict = {}
        offered = _server_offered(session_a)
        session_b: dict = {}
        replayed = A.ApprovalSet(set(offered), issued=session_b)
        assert A._consume_approval(replayed, TOOL, ARGS) is False


class TestLegitimateApprovals:
    def test_a_dialog_the_server_issued_is_honoured(self) -> None:
        issued: dict = {}
        approved = _server_offered(issued)
        assert A._consume_approval(approved, TOOL, ARGS) is True

    def test_still_single_use(self) -> None:
        issued: dict = {}
        approved = _server_offered(issued)
        assert A._consume_approval(approved, TOOL, ARGS) is True
        approved.add(A._approval_key(TOOL, ARGS))  # client replays it
        assert A._consume_approval(approved, TOOL, ARGS) is False

    def test_issuing_twice_then_approving_once_consumes_one(self) -> None:
        issued: dict = {}
        approved = _server_offered(issued)
        A.record_issued(approved, TOOL, ARGS)
        assert A._consume_approval(approved, TOOL, ARGS) is True
        assert issued == {}


class TestExpiry:
    def test_an_old_dialog_is_rejected(self, monkeypatch) -> None:
        issued: dict = {}
        approved = _server_offered(issued)
        issued[A._approval_key(TOOL, ARGS)] = time.time() - (A.APPROVAL_TTL_SECONDS + 60)
        assert A._consume_approval(approved, TOOL, ARGS) is False

    def test_a_fresh_dialog_inside_the_window_is_accepted(self) -> None:
        issued: dict = {}
        approved = _server_offered(issued)
        issued[A._approval_key(TOOL, ARGS)] = time.time() - 5
        assert A._consume_approval(approved, TOOL, ARGS) is True

    def test_recording_expires_stale_entries_so_the_registry_cannot_grow(self) -> None:
        issued = {("old.tool", "{}"): time.time() - (A.APPROVAL_TTL_SECONDS + 1)}
        approved = A.ApprovalSet(issued=issued)
        A.record_issued(approved, TOOL, ARGS)
        assert ("old.tool", "{}") not in issued
        assert A._approval_key(TOOL, ARGS) in issued


class TestNoRegistryIsTheLibraryPath:
    def test_a_plain_set_keeps_working(self) -> None:
        """Direct library/test use has no session, so there is nothing to bind to."""
        approved = {A._approval_key(TOOL, ARGS)}
        assert A._consume_approval(approved, TOOL, ARGS) is True

    def test_record_issued_on_a_plain_set_is_a_no_op(self) -> None:
        approved = {A._approval_key(TOOL, ARGS)}
        A.record_issued(approved, TOOL, ARGS)  # must not raise
        assert approved == {A._approval_key(TOOL, ARGS)}


class TestTheApiPathIsGuarded:
    """The whole fix rests on runtime binding a registry. Pin it."""

    def test_runtime_wraps_client_keys_in_an_approval_set(self) -> None:
        import inspect

        from backend import runtime

        src = inspect.getsource(runtime._run_turn_once)
        assert "ApprovalSet(" in src, (
            "runtime._run_turn_once must bind client-supplied approved_keys to the "
            "session's issued registry — without this the gate is advisory again"
        )

    def test_session_entry_carries_the_registry(self) -> None:
        from backend.runtime import SessionEntry

        assert "issued_approvals" in SessionEntry.__dataclass_fields__

    def test_every_dialog_site_records_what_it_issued(self) -> None:
        """A missed record_issued fails CLOSED, but it would break the product."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        for rel in (
            "backend/agent/llm_agent.py",
            "backend/agent/graph_engine.py",
            "backend/agent/migration_flow.py",
            "backend/agent/skill_flow.py",
        ):
            src = (root / rel).read_text()
            n_dialogs = src.count('"pending_confirmation": {')
            n_records = src.count("record_issued(")
            assert n_records >= n_dialogs, (
                f"{rel} constructs {n_dialogs} pending_confirmation(s) but calls "
                f"record_issued {n_records} time(s) — an unrecorded dialog can never be approved"
            )
