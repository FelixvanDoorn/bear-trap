# honeypot/analysis/geo/tests/test_regression.py
import numpy as np
import polars as pl
import pytest

from honeypot.analysis.geo.regression import (
    assumption_scorecard,
    country_residuals,
    design,
    fit_exposure_model,
    frequency_table,
    heldout_loglik,
    mean_variance_table,
    model_checks,
    moment_alpha,
    overdispersion_tests,
    pearson_residuals,
    prediction_band,
    quantile_residuals,
    slope,
    variance_by_fitted,
)
from honeypot.analysis.geo.tests.fixtures import (
    WORLD_ALPHA,
    WORLD_BETA,
    simulated_counts,
)


@pytest.fixture(scope="module")
def nb_fit():
    return fit_exposure_model(simulated_counts())


@pytest.fixture(scope="module")
def poisson_fit():
    return fit_exposure_model(simulated_counts(poisson=True))


class TestDesign:
    def test_centered_log_with_optional_square(self) -> None:
        addresses = np.array([np.e, np.e**3])

        assert design(addresses, center=2.0).tolist() == [[1, -1], [1, 1]]
        assert design(addresses, center=2.0, quadratic=True)[:, 2].tolist() == [1, 1]


class TestFit:
    def test_recovers_known_slope_and_dispersion(self, nb_fit) -> None:
        s = slope(nb_fit)

        assert nb_fit.converged
        assert s["ci_low"] < WORLD_BETA < s["ci_high"]
        assert s["alpha"] == pytest.approx(WORLD_ALPHA, abs=0.2)
        assert s["p_beta_eq_1"] < 0.001

    def test_mu_at_new_addresses_matches_fitted_rows(self, nb_fit) -> None:
        assert nb_fit.mu("negbin", nb_fit.addresses) == pytest.approx(
            nb_fit.mu("negbin")
        )


class TestOverdispersionChecks:
    def test_nb_data_rejects_poisson(self, nb_fit) -> None:
        tests = overdispersion_tests(nb_fit)
        checks = dict(zip(*model_checks(nb_fit).select("check", "poisson")))

        assert tests["test"].to_list() == [
            "likelihood ratio (alpha = 0)",
            "Cameron-Trivedi",
        ]
        assert tests["p_value"][0] < 1e-6
        assert checks["Pearson dispersion"] > 5

    def test_poisson_data_passes(self, poisson_fit) -> None:
        checks = model_checks(poisson_fit)
        dispersion = dict(zip(*checks.select("check", "poisson")))

        assert overdispersion_tests(poisson_fit)["p_value"][0] > 0.01
        assert dispersion["Pearson dispersion"] == pytest.approx(1, abs=0.3)

    def test_nb_pearson_dispersion_near_one_on_nb_data(self, nb_fit) -> None:
        checks = dict(zip(*model_checks(nb_fit).select("check", "negbin")))

        assert checks["Pearson dispersion"] == pytest.approx(1, abs=0.3)


class TestMeanVariance:
    def test_equal_size_bins(self) -> None:
        table = mean_variance_table(simulated_counts(), bins=6)

        assert table["bin"].to_list() == list(range(1, 7))
        assert table["countries"].to_list() == [50] * 6
        assert table.columns == [
            "bin",
            "countries",
            "median_addresses",
            "mean",
            "variance",
            "var_to_mean",
        ]

    def test_moment_alpha_exact_on_nb_shaped_table(self) -> None:
        mean = np.array([1.0, 10.0, 100.0])
        table = pl.DataFrame({"mean": mean, "variance": mean + 0.7 * mean**2})

        assert moment_alpha(table) == pytest.approx(0.7)


class TestFrequencyTable:
    def test_bands_cover_every_country(self, nb_fit) -> None:
        freq = frequency_table(nb_fit)
        n = nb_fit.data.height

        assert freq["band"].to_list() == ["0", "1-4", "5-19", "20-99", "100+"]
        assert freq["observed"].sum() == n
        assert freq["poisson_expected"].sum() == pytest.approx(n)
        assert freq["negbin_expected"].sum() == pytest.approx(n)


class TestResiduals:
    def test_injected_outlier_is_flagged_over(self) -> None:
        counts = simulated_counts().with_columns(
            pl.when(pl.col("country") == "C000")
            .then(pl.col("ips") * 50 + 500)
            .otherwise(pl.col("ips"))
            .alias("ips")
        )

        residuals = country_residuals(fit_exposure_model(counts))

        top = residuals.row(0, named=True)
        assert top["country"] == "C000"
        assert top["direction"] == "over"
        assert top["significant"]
        assert residuals["log_ratio"].is_sorted(descending=True)

    def test_pearson_residuals_per_country(self, nb_fit) -> None:
        pearson = pearson_residuals(nb_fit)

        assert pearson.height == nb_fit.data.height
        assert abs(pearson["resid_negbin"].mean()) < 0.2


class TestPredictionBand:
    def test_band_brackets_expected_and_rises(self, nb_fit) -> None:
        band = prediction_band(nb_fit, n_points=20)

        assert band.height == 20
        assert (band["low"] <= band["expected"]).all()
        assert (band["expected"] <= band["high"]).all()
        assert band["expected"].is_sorted()


class TestHeldoutLoglik:
    def test_nb_beats_poisson_on_new_nb_data(self, nb_fit) -> None:
        new = simulated_counts(n=200, seed=1)

        nb = heldout_loglik(nb_fit, new, "negbin")
        poisson = heldout_loglik(nb_fit, new, "poisson")

        assert len(nb) == 200
        assert nb.mean() > poisson.mean()


class TestAssumptionScorecard:
    def test_nb_data_favours_negbin_on_decisive_checks(self, nb_fit) -> None:
        card = assumption_scorecard(nb_fit, mean_variance_table(nb_fit.data))
        favours = dict(card.select("check", "favours").rows())

        assert card.height == 6
        assert favours["likelihood ratio, alpha = 0"] == "negbin"
        assert favours["AIC (lower is better)"] == "negbin"
        assert favours["Pearson dispersion (~1 if variance model fits)"] == "negbin"

    def test_poisson_data_does_not_favour_negbin_by_lr(self, poisson_fit) -> None:
        card = assumption_scorecard(poisson_fit, mean_variance_table(poisson_fit.data))
        favours = dict(card.select("check", "favours").rows())

        assert favours["likelihood ratio, alpha = 0"] == "poisson"


class TestQuantileResiduals:
    def test_standard_normal_under_the_right_model_only(self, nb_fit) -> None:
        residuals = quantile_residuals(nb_fit)
        nb = residuals["qresid_negbin"].to_numpy()
        poisson = residuals["qresid_poisson"].to_numpy()

        assert abs(nb.mean()) < 0.15
        assert nb.std() == pytest.approx(1, abs=0.15)
        assert np.mean(np.abs(nb) > 2) < 0.1
        # Poisson understates the variance of NB data: far too spread.
        assert poisson.std() > 2

    def test_zero_counts_scatter_instead_of_lying_on_a_curve(self, nb_fit) -> None:
        zeros = nb_fit.data[nb_fit.response].to_numpy() == 0
        a = quantile_residuals(nb_fit, seed=0)["qresid_negbin"].to_numpy()[zeros]
        b = quantile_residuals(nb_fit, seed=1)["qresid_negbin"].to_numpy()[zeros]

        assert zeros.sum() > 5
        assert not np.allclose(a, b)


class TestVarianceByFitted:
    def test_nb_near_one_per_bin_while_poisson_climbs(self, nb_fit) -> None:
        table = variance_by_fitted(nb_fit, bins=4)

        assert table["countries"].to_list() == [75] * 4
        assert table["mean_sq_negbin"].is_between(0.4, 2.0).all()
        assert table["mean_sq_poisson"][-1] > 10 * table["mean_sq_poisson"][0]
