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


class TestTheBindingSurvivesTheWholeTurn:
    """The registry has to survive every hop, including the empty first turn.

    It did not. `call_llm_with_tools` normalised its argument with
    `approved_mutations = approved_mutations or set()`, and an empty
    ApprovalSet is FALSY — so on turn one, the turn that ISSUES the dialog and
    has zero approvals by definition, the session-bound set was replaced by a
    plain one. `record_issued` then no-oped, and turn two's legitimate approval
    was refused as forged. Shipped in 2.7.3; it blocked every write in the
    product — chat mutations, migrations and skill runs.

    The unit suite passed throughout, because the tests above call
    `_consume_approval` directly with plain sets, which take the unbound legacy
    path. Only a live two-turn run reaches the seam. These pin it here so the
    next person does not need a production outage to find it.
    """

    def test_an_empty_approval_set_is_falsy_which_is_the_whole_trap(self) -> None:
        empty = A.ApprovalSet(issued={})
        assert not empty, "if this ever becomes truthy the regression below is moot"
        assert empty.issued == {}

    def test_normalising_an_empty_set_must_not_drop_the_registry(self) -> None:
        issued: dict = {}
        approved = A.ApprovalSet(issued=issued)

        # The bug: `approved or set()` discards the binding on an empty set.
        assert getattr(approved or set(), "issued", None) is None, "documents the old behaviour"

        # The fix: an explicit None check preserves it.
        normalised = set() if approved is None else approved
        assert getattr(normalised, "issued", None) is issued

    def test_the_source_uses_an_is_none_check_not_truthiness(self) -> None:
        import inspect

        src = inspect.getsource(A.call_llm_with_tools)
        assert "approved_mutations = approved_mutations or set()" not in src, (
            "an empty ApprovalSet is falsy; `or set()` drops the session binding "
            "on the very turn that issues the dialog"
        )
        assert "if approved_mutations is None:" in src

    def test_issue_then_approve_across_two_turns_with_a_shared_registry(self) -> None:
        """End to end over the registry, the way runtime binds it per session."""
        session_registry: dict = {}

        # Turn 1: no approvals. Must still record what was offered.
        turn1 = A.ApprovalSet(issued=session_registry)
        turn1 = set() if turn1 is None else turn1  # the normalisation, fixed
        assert A._consume_approval(turn1, TOOL, ARGS) is False
        A.record_issued(turn1, TOOL, ARGS)
        assert session_registry, "turn 1 must leave a record for turn 2 to match"

        # Turn 2: client returns the key; same session registry.
        turn2 = A.ApprovalSet([A._approval_key(TOOL, ARGS)], issued=session_registry)
        assert A._consume_approval(turn2, TOOL, ARGS) is True
