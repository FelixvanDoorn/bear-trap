# honeypot/analysis/geo/tests/test_covariates.py
import numpy as np
import polars as pl
import pytest

from honeypot.analysis.geo.covariates import (
    COVARIATE_SPECS,
    compare_specifications,
    covered,
    full_model_coefficients,
    log_correlations,
    with_indicators,
)
from honeypot.analysis.geo.tests.fixtures import nb_draw
from honeypot.analysis.geo.validation import cross_validate


def _counts(n: int = 300, seed: int = 0) -> pl.DataFrame:
    """Counts driven by internet users (elasticity 0.9), not by address
    space beyond its correlation with users; GDP is pure noise."""
    rng = np.random.default_rng(seed)
    log_addresses = rng.uniform(np.log(1e3), np.log(1e9), n)
    log_users = log_addresses + rng.normal(0, 1.0, n)
    mu = np.exp(-6 + 0.9 * log_users)
    return pl.DataFrame(
        {
            "country": [f"C{i:03d}" for i in range(n)],
            "addresses": np.round(np.exp(log_addresses)).astype(np.int64),
            "ips": nb_draw(rng, mu, 0.5).astype(np.int64),
            "internet_users": np.exp(log_users),
            "gdp_per_capita": np.exp(rng.normal(9, 1, n)),
        }
    )


class TestWithIndicators:
    def test_left_join_keeps_every_country(self) -> None:
        counts = pl.DataFrame({"country": ["AA", "BB"], "ips": [1, 2]})
        indicators = pl.DataFrame(
            {"country": ["AA"], "internet_users": [10.0], "gdp_per_capita": [5.0]}
        )

        result = with_indicators(counts, indicators)

        assert result.height == 2
        assert covered(result)["country"].to_list() == ["AA"]


class TestLogCorrelations:
    def test_matrix_and_vif(self) -> None:
        table = log_correlations(_counts())

        assert table["variable"].to_list() == [
            "addresses",
            "internet_users",
            "gdp_per_capita",
        ]
        assert table["addresses"][0] == pytest.approx(1.0)
        assert table["internet_users"][0] > 0.9
        assert (table["vif"] >= 1).all()


class TestSpecifications:
    def test_true_covariate_wins_in_and_out_of_sample(self) -> None:
        counts = _counts()

        specs = compare_specifications(counts)
        cv = cross_validate(
            counts, specs=COVARIATE_SPECS, baseline="address space", k=3, repeats=2
        )

        aic = dict(specs.select("spec", "aic").rows())
        assert specs["countries"].n_unique() == 1
        assert aic["+ internet users"] < aic["address space"]
        scores = dict(cv.select("model", "diff_vs_address space").rows())
        assert scores["+ internet users"] > 0

    def test_full_model_recovers_the_users_elasticity(self) -> None:
        coefs = dict(full_model_coefficients(_counts()).select("term", "coef").rows())

        assert coefs["log_internet_users"] == pytest.approx(0.9, abs=0.2)
        assert abs(coefs["log_gdp_per_capita"]) < 0.2
