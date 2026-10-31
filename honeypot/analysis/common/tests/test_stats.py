# honeypot/analysis/common/tests/test_stats.py
import polars as pl
import pytest

from honeypot.analysis.common.stats import (
    OTHER_LEVEL,
    adjust_pvalues,
    category_breakdown,
    compare_groups,
)


def _results(p_values: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"property": [f"p{i}" for i in range(len(p_values))], "p_value": p_values},
        schema={"property": pl.Utf8, "p_value": pl.Float64},
    )


class TestAdjustPvalues:
    def test_matches_hand_computed_corrections(self) -> None:
        # m = 4. Bonferroni: p * 4. Holm on sorted [.01, .02, .03, .04]:
        # [.04, .06, .06, .06] after the monotone max. BH: p * m / rank,
        # then a monotone min from the top: [.04, .04, .04, .04].
        result = adjust_pvalues(_results([0.04, 0.01, 0.03, 0.02]))

        assert result["p_bonferroni"].to_list() == pytest.approx(
            [0.16, 0.04, 0.12, 0.08]
        )
        assert result["p_holm"].to_list() == pytest.approx([0.06, 0.04, 0.06, 0.06])
        assert result["q_bh"].to_list() == pytest.approx([0.04, 0.04, 0.04, 0.04])
        assert result["significant_bh"].to_list() == [True, True, True, True]

    def test_bh_is_less_strict_than_holm(self) -> None:
        # BH: every q becomes 0.035 * 5 / 4 = 0.04375 -> 4 hits. Holm: the
        # smallest p alone is already scaled to 0.01 * 5 = 0.05 -> 0 hits.
        result = adjust_pvalues(_results([0.01, 0.02, 0.03, 0.035, 0.5]))

        holm_hits = (result["p_holm"] < 0.05).sum()
        bh_hits = result["significant_bh"].sum()
        assert bh_hits > holm_hits

    def test_caps_at_one(self) -> None:
        result = adjust_pvalues(_results([0.6, 0.9]))

        assert result["p_bonferroni"].max() == 1.0
        assert result["p_holm"].max() == 1.0

    def test_untestable_rows_stay_null_and_do_not_count(self) -> None:
        result = adjust_pvalues(_results([0.01, None]))

        # Only one real test, so no inflation at all.
        assert result["p_bonferroni"].to_list() == pytest.approx([0.01, None])
        assert result["q_bh"][1] is None
        assert result["significant_bh"][1] is None


def _units(a: list[dict], b: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        [{"in_a": True, **row} for row in a] + [{"in_a": False, **row} for row in b]
    )


class TestCompareGroups:
    def test_detects_numeric_shift_with_positive_rank_biserial(self) -> None:
        df = _units(
            [{"x": v} for v in range(100, 130)], [{"x": v} for v in range(0, 30)]
        )

        row = compare_groups(df, "in_a", categorical=[], numeric=["x"]).row(
            0, named=True
        )

        assert row["test"] == "mann_whitney_u"
        assert row["effect_size"] == pytest.approx(1.0)
        assert row["p_value"] < 1e-6
        assert row["significant_bh"] is True

    def test_two_level_categorical_uses_fisher(self) -> None:
        df = _units(
            [{"flag": True}] * 20 + [{"flag": False}] * 5,
            [{"flag": True}] * 5 + [{"flag": False}] * 20,
        )

        row = compare_groups(df, "in_a", categorical=["flag"], numeric=[]).row(
            0, named=True
        )

        assert row["test"] == "fisher_exact"
        assert row["detail"].startswith("true: 80% vs 20%")
        assert 0 < row["effect_size"] < 1
        assert row["significant_bh"] is True

    def test_multi_level_categorical_uses_chi_square_and_pools_rare(self) -> None:
        a = [{"c": "NL"}] * 30 + [{"c": "US"}] * 10 + [{"c": "rare1"}]
        b = [{"c": "NL"}] * 10 + [{"c": "US"}] * 30 + [{"c": "CN"}] * 20
        b += [{"c": "rare2"}]
        df = _units(a, b)

        row = compare_groups(df, "in_a", categorical=["c"], numeric=[]).row(
            0, named=True
        )
        breakdown = category_breakdown(df, "in_a", "c")

        assert row["test"] == "chi_square"
        assert set(breakdown["level"]) == {"NL", "US", "CN", OTHER_LEVEL}
        assert breakdown.row(0, named=True)["level"] == "NL"
        assert row["detail"].startswith("NL:")

    def test_nulls_are_excluded_per_property(self) -> None:
        df = _units(
            [{"c": "x", "n": 1.0}, {"c": None, "n": 2.0}],
            [{"c": "y", "n": None}, {"c": "x", "n": 3.0}],
        )

        results = compare_groups(df, "in_a", categorical=["c"], numeric=["n"])
        counts = {r["property"]: (r["n_a"], r["n_b"]) for r in results.to_dicts()}

        assert counts == {"c": (1, 2), "n": (2, 1)}

    def test_group_without_values_is_untestable(self) -> None:
        df = _units([{"c": None}], [{"c": "x"}])

        row = compare_groups(df, "in_a", categorical=["c"], numeric=[]).row(
            0, named=True
        )

        assert row["test"] == "untestable"
        assert row["p_value"] is None

    def test_constant_numeric_is_untestable(self) -> None:
        df = _units([{"n": 1}] * 5, [{"n": 1}] * 5)

        row = compare_groups(df, "in_a", categorical=[], numeric=["n"]).row(
            0, named=True
        )

        assert row["test"] == "untestable"
        assert row["detail"] == "no variation"

    def test_null_effect_is_not_significant(self) -> None:
        same = [{"c": "x"}] * 20 + [{"c": "y"}] * 20
        df = _units(same, same)

        row = compare_groups(df, "in_a", categorical=["c"], numeric=[]).row(
            0, named=True
        )

        assert row["effect_size"] == pytest.approx(0.0)
        assert row["significant_bh"] is False
