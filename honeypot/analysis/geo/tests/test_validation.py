# honeypot/analysis/geo/tests/test_validation.py
import datetime as dt

import polars as pl
import pytest

from honeypot.analysis.geo.exposure import country_counts
from honeypot.analysis.geo.regression import fit_exposure_model
from honeypot.analysis.geo.tests.fixtures import simulated_counts
from honeypot.analysis.geo.validation import (
    SPLIT_METHODS,
    bootstrap_slope,
    bootstrap_summary,
    compare_country_residuals,
    compare_slopes,
    compare_split_methods,
    cross_validate,
    drop_influential,
    fit_parts,
    influence_table,
    part_counts,
    response_robustness,
    split_ips,
    split_log_ratios,
)

_START = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def _attackers(n_countries: int = 30, seed: int = 0) -> pl.DataFrame:
    """attacker_ips-shaped rows from simulated_counts: each country's IPs
    spread over three ASNs, both node roles and protocols, and a month of
    first sightings."""
    counts = simulated_counts(n=n_countries, seed=seed)
    rows = []
    for country, ips in counts.select("country", "ips").rows():
        for i in range(ips):
            rows.append(
                {
                    "src_ip": f"{country}-{i}",
                    "country": country,
                    "asn": hash((country, i % 3)) % 10**6,
                    "as_label": f"AS{i % 3} {country}",
                    "slash24": i // 4,
                    "protocols": ["ssh", "telnet", "ssh+telnet"][i % 3],
                    "node_roles": ["agent_detector", "control_proxy"][i % 2],
                    "first_seen": _START + dt.timedelta(hours=i),
                }
            )
    return pl.DataFrame(rows)


def _space(n_countries: int = 30, seed: int = 0) -> pl.DataFrame:
    return simulated_counts(n=n_countries, seed=seed).select("country", "addresses")


@pytest.fixture(scope="module")
def attackers() -> pl.DataFrame:
    return _attackers()


class TestSplitIps:
    def test_random_ip_is_a_disjoint_cover(self, attackers) -> None:
        parts = split_ips(attackers, "random_ip")

        assert parts.height == attackers.height
        assert set(parts["part"]) == {"A", "B"}
        assert parts["src_ip"].n_unique() == attackers.height

    def test_random_asn_keeps_each_asn_on_one_side(self, attackers) -> None:
        parts = split_ips(attackers, "random_asn").join(
            attackers.select("src_ip", "asn"), on="src_ip"
        )

        assert parts.height == attackers.height
        assert parts.group_by("asn").agg(pl.col("part").n_unique())["part"].max() == 1

    def test_time_blocks_are_consecutive_and_equal(self, attackers) -> None:
        parts = split_ips(attackers, "time_blocks", n_blocks=3).join(
            attackers.select("src_ip", "first_seen"), on="src_ip"
        )
        blocks = (
            parts.group_by("part")
            .agg(
                pl.len(),
                pl.col("first_seen").min().alias("lo"),
                pl.col("first_seen").max().alias("hi"),
            )
            .sort("lo")
        )

        assert blocks.height == 3
        assert blocks["len"].max() - blocks["len"].min() <= 1
        assert (blocks["hi"].head(2) <= blocks["lo"].tail(2)).all()

    def test_node_role_and_protocol(self) -> None:
        attackers = pl.DataFrame(
            {
                "src_ip": ["a", "b", "c"],
                "node_roles": ["agent_detector+control_proxy", "control_proxy", "x"],
                "protocols": ["ssh", "ssh+telnet", "telnet"],
            }
        )

        roles = split_ips(attackers, "node_role")
        protocols = split_ips(attackers, "protocol")

        assert roles.filter(pl.col("src_ip") == "a")["part"].sort().to_list() == [
            "agent_detector",
            "control_proxy",
        ]
        # Mixed-protocol IPs belong to neither population.
        assert protocols.rows() == [("a", "ssh"), ("c", "telnet")]

    def test_unknown_method_raises(self, attackers) -> None:
        with pytest.raises(ValueError, match="Unknown split method"):
            split_ips(attackers, "by_vibes")


class TestSplitStability:
    def test_part_counts_has_every_country_per_part(self, attackers) -> None:
        counts = part_counts(attackers, split_ips(attackers, "random_ip"), _space())

        assert list(counts) == ["A", "B"]
        assert all(c.height == 30 for c in counts.values())
        assert sum(c["ips"].sum() for c in counts.values()) == attackers.height

    def test_compare_methods_one_summary_row_each(self, attackers) -> None:
        summary, pairs = compare_split_methods(
            attackers, _space(), repeats=2, min_ips=1
        )

        assert summary["method"].to_list() == SPLIT_METHODS
        pair_counts = dict(summary.select("method", "pairs").rows())
        assert pair_counts["random_ip"] == 2  # 2 seeds x 1 pair
        assert pair_counts["time_blocks"] == 1
        assert pairs["spearman"].is_between(-1, 1).all()

    def test_random_ip_halves_agree_on_real_structure(self, attackers) -> None:
        summary, _ = compare_split_methods(
            attackers, _space(), methods=["random_ip"], repeats=1, min_ips=5
        )

        # Countries' deviations come from the data, not the coin flip.
        assert summary["median_spearman"][0] > 0.5

    def test_log_ratios_long_per_part(self, attackers) -> None:
        ratios = split_log_ratios(attackers, _space(), "time_blocks")

        assert ratios.height == 60
        assert set(ratios.columns) == {
            "country",
            "observed",
            "log_ratio",
            "significant",
            "direction",
            "part",
        }


class TestResponseRobustness:
    def test_one_row_per_response(self) -> None:
        result = response_robustness(simulated_counts(), min_ips=1)

        assert result["response"].to_list() == ["ips", "slash24s", "asns"]
        assert result["spearman_vs_ips"][0] == pytest.approx(1.0)


class TestInfluence:
    def test_injected_outlier_is_most_influential(self) -> None:
        counts = simulated_counts().with_columns(
            pl.when(pl.col("country") == "C000")
            .then(pl.col("ips") * 50 + 500)
            .otherwise(pl.col("ips"))
            .alias("ips")
        )
        fit = fit_exposure_model(counts)

        table = influence_table(fit)
        dropped = drop_influential(counts, fit, k_values=(1, 2))

        assert table["country"][0] == "C000"
        assert table["cooks_d"].is_sorted(descending=True)
        assert dropped["dropped"].to_list() == [0, 1, 2]
        assert dropped["countries_dropped"][1] == "C000"


class TestBootstrap:
    def test_draws_and_percentile_summary(self) -> None:
        draws = bootstrap_slope(simulated_counts(), n_boot=20)
        summary = bootstrap_summary(draws)

        assert 15 <= draws.height <= 20
        assert summary["parameter"].to_list() == ["beta", "alpha"]
        assert (summary["ci_low"] <= summary["median"]).all()
        assert (summary["median"] <= summary["ci_high"]).all()


class TestCrossValidate:
    def test_poisson_scores_worst_on_overdispersed_data(self) -> None:
        cv = cross_validate(simulated_counts(), k=3, repeats=2)

        assert cv["folds"].to_list() == [6, 6, 6]
        assert cv["model"][-1] == "poisson"
        scores = dict(cv.select("model", "diff_vs_negbin").rows())
        assert scores["negbin"] == 0.0
        assert scores["poisson"] < 0


def test_country_counts_roundtrip_matches_attackers(attackers) -> None:
    counts = country_counts(attackers, _space())

    assert counts["ips"].sum() == attackers.height


class TestPartComparisons:
    def test_fit_parts_one_fit_per_protocol(self, attackers) -> None:
        fits = fit_parts(attackers, _space(), "protocol")

        assert list(fits) == ["ssh", "telnet"]

    def test_compare_slopes_rows_and_difference(self) -> None:
        fits = {
            "a": fit_exposure_model(simulated_counts(seed=1)),
            "b": fit_exposure_model(simulated_counts(seed=2)),
        }

        result = compare_slopes(fits)

        assert result["comparison"].to_list() == ["a", "b", "a - b"]
        assert result["test"].to_list() == ["beta = 1", "beta = 1", "equal slopes"]
        betas = result["beta"].to_list()
        assert betas[2] == pytest.approx(betas[0] - betas[1])
        # Same true slope in both: no significant difference.
        assert result["p_value"][2] > 0.01

    def test_compare_country_residuals_side_by_side(self) -> None:
        counts = simulated_counts()
        fits = {
            "ips": fit_exposure_model(counts, "ips"),
            "slash24s": fit_exposure_model(counts, "slash24s"),
        }

        result = compare_country_residuals(fits, min_observed=5)

        assert result.columns == [
            "country",
            "observed_ips",
            "ratio_ips",
            "q_ips",
            "observed_slash24s",
            "ratio_slash24s",
            "q_slash24s",
        ]
        assert (
            result.select(pl.max_horizontal("observed_ips", "observed_slash24s"))
            .to_series()
            .min()
            >= 5
        )
        assert result["ratio_ips"].is_sorted(descending=True)
