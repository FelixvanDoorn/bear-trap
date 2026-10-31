# honeypot/analysis/common/stats.py
#
# Two-group property screening with multiple-testing correction (Data-
# Driven Security ch. 4): given one row per unit (e.g. per src_ip) and a
# boolean group column, test every listed property for a difference
# between the groups, then adjust the whole family of p-values together.
#
# - Categorical properties: chi-square test of independence on the
#   group x level contingency table, Fisher's exact test when that table
#   is 2x2 (booleans, or a categorical collapsed down to two levels).
#   Chi-square's approximation needs expected counts of roughly >= 5 per
#   cell, so rare levels are pooled into "other" first (_collapse_rare).
#   Effect size: Cramér's V (0 = no association, 1 = level fully
#   determined by group), computed from the uncorrected chi-square.
# - Numeric properties: Mann-Whitney U (no normality assumption -- these
#   are heavy-tailed counts and durations). Effect size: rank-biserial r =
#   P(a > b) - P(a < b), from -1 to 1, positive when group a tends larger.
#
# Why effect sizes matter here: p-values shrink with sample size, so with
# a few hundred units a negligible difference can still come out
# "significant". Read effect_size alongside q_bh, not p alone.
#
# Multiple testing (adjust_pvalues): Benjamini-Hochberg controls the false
# discovery rate -- the expected share of flagged properties that are
# false positives -- which suits an exploratory screen of many correlated
# properties better than controlling the chance of even one false positive
# (family-wise error rate). BH stays valid under the positive dependence
# expected between related properties (country / ASN, client version /
# hassh). Holm (FWER, uniformly more powerful than Bonferroni) and plain
# Bonferroni are reported alongside for a stricter reading.
import numpy as np
import polars as pl
from scipy import stats

OTHER_LEVEL = "other"

DEFAULT_MIN_EXPECTED = 5.0
DEFAULT_ALPHA = 0.05

RESULT_SCHEMA = {
    "property": pl.Utf8,
    "kind": pl.Utf8,
    "test": pl.Utf8,
    "n_a": pl.Int64,
    "n_b": pl.Int64,
    "statistic": pl.Float64,
    "p_value": pl.Float64,
    "effect_size": pl.Float64,
    "effect_measure": pl.Utf8,
    "detail": pl.Utf8,
}


def _level_counts(df: pl.DataFrame, group_col: str, column: str) -> pl.DataFrame:
    """One row per non-null level of `column`: count_a (group True) and
    count_b (group False)."""
    return (
        df.filter(pl.col(column).is_not_null())
        .group_by(pl.col(column).cast(pl.Utf8).alias("level"))
        .agg(
            pl.col(group_col).sum().cast(pl.Int64).alias("count_a"),
            (~pl.col(group_col)).sum().cast(pl.Int64).alias("count_b"),
        )
        .sort("level")
    )


def _collapse_rare(counts: pl.DataFrame, min_expected: float) -> pl.DataFrame:
    """Pool every level whose smallest expected cell count (under
    independence) is below `min_expected` into a single OTHER_LEVEL row.
    The pooled row is kept even if it's still below the threshold itself:
    dropping it would silently discard units, and a single small cell is
    within Cochran's rule (<= 20% of cells under 5)."""
    n_a, n_b = counts["count_a"].sum(), counts["count_b"].sum()
    total = n_a + n_b
    min_expected_cell = (pl.col("count_a") + pl.col("count_b")) * min(n_a, n_b) / total
    kept = counts.filter(min_expected_cell >= min_expected)
    rare = counts.filter(min_expected_cell < min_expected)
    if rare.height == 0:
        return kept
    pooled = pl.DataFrame(
        {
            "level": [OTHER_LEVEL],
            "count_a": [rare["count_a"].sum()],
            "count_b": [rare["count_b"].sum()],
        },
        schema=counts.schema,
    )
    return pl.concat([kept, pooled])


def _shares(counts: pl.DataFrame) -> pl.DataFrame:
    return counts.with_columns(
        (pl.col("count_a") / pl.col("count_a").sum()).alias("share_a"),
        (pl.col("count_b") / pl.col("count_b").sum()).alias("share_b"),
    )


def _untestable(column: str, kind: str, n_a: int, n_b: int, why: str) -> dict:
    return {
        "property": column,
        "kind": kind,
        "test": "untestable",
        "n_a": n_a,
        "n_b": n_b,
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "effect_measure": None,
        "detail": why,
    }


def _categorical_test(
    df: pl.DataFrame, group_col: str, column: str, min_expected: float
) -> dict:
    counts = _level_counts(df, group_col, column)
    n_a, n_b = int(counts["count_a"].sum()), int(counts["count_b"].sum())
    if n_a == 0 or n_b == 0:
        return _untestable(column, "categorical", n_a, n_b, "a group has no values")
    counts = _collapse_rare(counts, min_expected)
    if counts.height < 2:
        return _untestable(column, "categorical", n_a, n_b, "fewer than 2 levels")

    table = counts.select("count_a", "count_b").to_numpy().T
    chi2, chi2_p, _, _ = stats.chi2_contingency(table, correction=False)
    cramers_v = float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))
    if counts.height == 2:
        test = "fisher_exact"
        statistic, p_value = stats.fisher_exact(table)
    else:
        test = "chi_square"
        statistic, p_value = chi2, chi2_p

    # The level most over-represented in group a, for a readable summary.
    top = (
        _shares(counts)
        .with_columns((pl.col("share_a") - pl.col("share_b")).alias("diff"))
        .sort("diff", descending=True)
        .row(0, named=True)
    )
    detail = (
        f"{top['level']}: {top['share_a']:.0%} vs {top['share_b']:.0%} "
        f"({counts.height} levels)"
    )
    return {
        "property": column,
        "kind": "categorical",
        "test": test,
        "n_a": n_a,
        "n_b": n_b,
        "statistic": float(statistic),
        # Exact tests can land a hair above 1 from floating-point summation.
        "p_value": min(float(p_value), 1.0),
        "effect_size": cramers_v,
        "effect_measure": "cramers_v",
        "detail": detail,
    }


def _numeric_test(df: pl.DataFrame, group_col: str, column: str) -> dict:
    values = df.filter(pl.col(column).is_not_null())
    a = values.filter(pl.col(group_col))[column].cast(pl.Float64).to_numpy()
    b = values.filter(~pl.col(group_col))[column].cast(pl.Float64).to_numpy()
    if len(a) == 0 or len(b) == 0:
        return _untestable(column, "numeric", len(a), len(b), "a group has no values")
    # Every value tied -> zero rank variance and a NaN p from scipy. Nothing
    # to compare, so untestable rather than p = 1, which would also count
    # toward (and needlessly tighten) the multiple-testing correction.
    if np.unique(np.concatenate([a, b])).size < 2:
        return _untestable(column, "numeric", len(a), len(b), "no variation")

    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    rank_biserial = 2 * result.statistic / (len(a) * len(b)) - 1
    return {
        "property": column,
        "kind": "numeric",
        "test": "mann_whitney_u",
        "n_a": len(a),
        "n_b": len(b),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size": float(rank_biserial),
        "effect_measure": "rank_biserial",
        "detail": f"median {np.median(a):.4g} vs {np.median(b):.4g}",
    }


def _holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values: sort ascending, scale the i-th
    smallest (0-based) by (m - i), then enforce monotonicity."""
    m = len(p_values)
    order = np.argsort(p_values)
    scaled = np.minimum(1.0, (m - np.arange(m)) * p_values[order])
    adjusted = np.empty(m)
    adjusted[order] = np.maximum.accumulate(scaled)
    return adjusted


def adjust_pvalues(results: pl.DataFrame, alpha: float = DEFAULT_ALPHA) -> pl.DataFrame:
    """Add p_bonferroni, p_holm and q_bh (Benjamini-Hochberg) columns
    adjusting `p_value` across every row, plus significant_bh = q_bh <
    `alpha`. Rows with a null p_value (untestable) stay null and don't
    count toward the number of tests."""
    tested = results["p_value"].is_not_null().to_numpy()
    p_values = results["p_value"].to_numpy()[tested].astype(float)

    columns: dict[str, np.ndarray] = {
        "p_bonferroni": np.minimum(1.0, p_values * len(p_values)),
        "p_holm": _holm(p_values),
        "q_bh": stats.false_discovery_control(p_values, method="bh")
        if len(p_values)
        else p_values,
    }
    adjusted = results
    for name, values in columns.items():
        full = np.full(results.height, np.nan)
        full[tested] = values
        adjusted = adjusted.with_columns(
            pl.Series(name, full, dtype=pl.Float64).fill_nan(None)
        )
    return adjusted.with_columns((pl.col("q_bh") < alpha).alias("significant_bh"))


def compare_groups(
    df: pl.DataFrame,
    group_col: str,
    categorical: list[str],
    numeric: list[str],
    min_expected: float = DEFAULT_MIN_EXPECTED,
    alpha: float = DEFAULT_ALPHA,
) -> pl.DataFrame:
    """One row per property: the test run, per-group non-null counts (n_a
    = `group_col` True), statistic, raw p_value, effect size and a
    readable `detail`, plus the adjust_pvalues columns -- corrected across
    all properties at once. Boolean properties belong in `categorical`.
    Sorted by q_bh ascending (untestable rows last), then property."""
    rows = [
        _categorical_test(df, group_col, column, min_expected) for column in categorical
    ] + [_numeric_test(df, group_col, column) for column in numeric]
    results = pl.DataFrame(rows, schema=RESULT_SCHEMA)
    return adjust_pvalues(results, alpha=alpha).sort(
        ["q_bh", "property"], nulls_last=True
    )


def category_breakdown(
    df: pl.DataFrame,
    group_col: str,
    column: str,
    min_expected: float = DEFAULT_MIN_EXPECTED,
) -> pl.DataFrame:
    """Drill-down for one categorical property: the exact (collapsed)
    levels compare_groups tested, with counts and within-group shares.
    Sorted by share_a - share_b descending (most over-represented in group
    a first)."""
    counts = _collapse_rare(_level_counts(df, group_col, column), min_expected)
    return (
        _shares(counts)
        .with_columns((pl.col("share_a") - pl.col("share_b")).alias("share_diff"))
        .sort("share_diff", descending=True)
    )
