"""The `|join_paths` filter: pairs of tables joinable more than one way.

This line was `**none found**` on a model where all 15 dimension pairs are
joinable five ways each — a confident, specific, FALSE statement about exactly
the thing a user relies on it for. The SDK was computing the paths and only
surfacing them when they changed the kept table set; it now reports them all
(pysisense 2.3.0), and this renders them.

Written in code rather than the template because the template language has no
min/max and no predicates, and this needs both.

Fixtures are verbatim from live runs against Governance_Optimized.
"""

from __future__ import annotations

from backend.agent.skill_flow import _render_join_paths

FACTS = [
    "Fact_admin_shared_dashoards",
    "Fact_users_groups",
    "Fact_shared_dashboards",
    "Fact_datamodels_datasets",
    "Fact_usage",
]


def _pair(frm, to, *, changes, in_use, n=5):
    return {
        "from": frm,
        "to": to,
        "changes_tables": changes,
        "resolved": True,
        "needed_by": [f"{frm}/{to}: a widget"],
        "paths": [{"via": [f], "in_use": f in in_use} for f in FACTS[:n]],
    }


class TestNothingToReport:
    def test_empty_list(self) -> None:
        assert _render_join_paths([]) == "**none found**"

    def test_empty_result_dict(self) -> None:
        assert _render_join_paths({"join_path_choices": [], "summary": {}}) == "**none found**"

    def test_missing_key_entirely(self) -> None:
        """A 2.2.0 response has no such key — must not raise."""
        assert _render_join_paths({"summary": {}}) == "**none found**"


class TestAmbiguityThatCostsNothing:
    """Case A: the ambiguity is real, but it changes no kept table."""

    ROWS = [
        _pair(
            "Dim_dashboards",
            "Dim_users",
            changes=False,
            in_use={"Fact_usage", "Fact_users_groups", "Fact_shared_dashboards", "Fact_admin_shared_dashoards"},
        )
    ] + [_pair(f"Dim_{i}", "Dim_tenants", changes=False, in_use={"Fact_usage"}) for i in range(10)]

    def test_counts_the_pairs(self) -> None:
        assert "11 pairs" in _render_join_paths(self.ROWS)

    def test_explains_why_no_route_was_chosen(self) -> None:
        """ "The trim is safe" explains nothing. Say why: every route's tables
        are required anyway, so narrowing would remove no table and would cost
        the discarded routes' join columns."""
        out = _render_join_paths(self.ROWS)
        assert "Every route's tables are required anyway" in out
        assert "no route was chosen and all are kept" in out
        assert "the trim is safe" not in out

    def test_names_the_busiest_pair_and_its_route_count(self) -> None:
        out = _render_join_paths(self.ROWS)
        assert "Dim_dashboards and Dim_users" in out
        assert "4 different fact tables" in out

    def test_never_claims_a_winner_when_several_routes_are_in_use(self) -> None:
        """Case A has four of five routes in use — 'the widgets use X' would be false."""
        assert "so that is what the perspective keeps" not in _render_join_paths(self.ROWS)

    def test_no_cost_sentence_when_nothing_was_traded_away(self) -> None:
        payload = {
            "join_path_choices": self.ROWS,
            "summary": {
                "tables_required_in_perspective": 11,
                "columns_required_in_perspective": 70,
                "tables_required_all_paths": 11,
                "columns_required_all_paths": 70,
            },
        }
        assert "Keeping every route instead" not in _render_join_paths(payload)


class TestAmbiguityThatDecidesTheTrim:
    """Case B: the choice is 4 tables vs 8."""

    PAYLOAD = {
        "join_path_choices": [
            _pair("Dim_groups", "Dim_users", changes=True, in_use={"Fact_users_groups"}),
            _pair("Dim_dashboards", "Dim_users", changes=True, in_use={"Fact_users_groups"}),
        ],
        "summary": {
            "tables_required_in_perspective": 4,
            "columns_required_in_perspective": 9,
            "tables_required_all_paths": 8,
            "columns_required_all_paths": 21,
        },
    }

    def test_flags_that_the_choice_matters(self) -> None:
        assert "here the choice matters" in _render_join_paths(self.PAYLOAD)

    def test_names_both_pairs_with_and_not_a_comma(self) -> None:
        assert "Dim_groups to Dim_users and Dim_dashboards to Dim_users" in _render_join_paths(self.PAYLOAD)

    def test_names_the_route_the_widgets_use(self) -> None:
        assert "The widgets use Fact_users_groups" in _render_join_paths(self.PAYLOAD)

    def test_reports_what_the_alternative_would_have_cost(self) -> None:
        assert "would need **8 tables and 21 columns**" in _render_join_paths(self.PAYLOAD)


class TestShapesThatMustNotBreakIt:
    def test_multi_hop_route_renders_as_a_chain(self) -> None:
        rows = [
            {
                "from": "A",
                "to": "B",
                "changes_tables": True,
                "resolved": True,
                "paths": [{"via": ["F1", "F2"], "in_use": True}, {"via": ["F3"], "in_use": False}],
            }
        ]
        assert "F1 → F2" in _render_join_paths(rows)

    def test_several_routes_in_use_are_listed_as_a_set(self) -> None:
        rows = [
            {
                "from": "A",
                "to": "B",
                "changes_tables": True,
                "resolved": True,
                "paths": [{"via": ["F1"], "in_use": True}, {"via": ["F2"], "in_use": True}],
            }
        ]
        out = _render_join_paths(rows)
        assert "F1 and F2" in out and "are what the perspective keeps" in out

    def test_route_count_range_when_pairs_differ(self) -> None:
        rows = [
            _pair("A", "B", changes=False, in_use={"Fact_usage"}, n=5),
            _pair("C", "D", changes=False, in_use={"Fact_usage"}, n=4),
        ]
        assert "4 or 5 routes each" in _render_join_paths(rows)

    def test_singular_pair_reads_correctly(self) -> None:
        rows = [_pair("A", "B", changes=False, in_use={"Fact_usage"})]
        assert "1 pair," in _render_join_paths(rows)

    def test_more_than_three_deciding_pairs_are_summarised(self) -> None:
        rows = [_pair(f"D{i}", "Dim_users", changes=True, in_use={"Fact_usage"}) for i in range(5)]
        assert "and 2 more" in _render_join_paths(rows)

    def test_non_dict_entries_are_ignored(self) -> None:
        assert _render_join_paths(["junk", None, 7]) == "**none found**"
