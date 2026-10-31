# honeypot/analysis/geo/regression.py
#
# Count regression of attacking IPs per country on IPv4 address space
# (Data-Driven Security ch. 5), plus the checks that pick the model:
#
#   log E[count] = b0 + beta * log(addresses)
#
# beta is estimated, not fixed at 1 (an offset): beta = 1 means attackers
# scale in proportion to address space, beta < 1 that big address holders
# send fewer per address. Fixing it at 1 when it isn't would make every
# small country look over-represented.
#
# Poisson assumes variance = mean. Attack counts are bursty (one campaign
# or one infected ISP adds dozens of IPs at once), so the negative binomial
# (NB2: variance = mu + alpha * mu^2) is fitted alongside, and these decide
# between them:
# - mean_variance_table: before any model, does variance grow like the
#   mean (Poisson) or like mean^2 (NB)?
# - overdispersion_tests: likelihood-ratio test of alpha = 0, and the
#   Cameron-Trivedi auxiliary regression.
# - model_checks: log-likelihood, AIC, Pearson dispersion (~1 when the
#   variance model is right), observed vs expected zero countries.
# - frequency_table: observed vs expected number of countries per count
#   band (the data behind a rootogram).
#
# Residuals (country_residuals) are tested against the NB predictive
# distribution, one two-sided test per country, BH-adjusted across all of
# them -- the same FDR reasoning as common/stats.py.
from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomialResults
from statsmodels.genmod.generalized_linear_model import GLMResults

from honeypot.analysis.common.stats import DEFAULT_ALPHA

DEFAULT_RESPONSE = "ips"

# Count bands for frequency_table: (low, high), high None = no upper bound.
FREQUENCY_BANDS: list[tuple[int, int | None]] = [
    (0, 0),
    (1, 4),
    (5, 19),
    (20, 99),
    (100, None),
]

# Added to observed and expected counts before taking a log ratio, so
# zero-count countries get a finite value.
_LOG_RATIO_PSEUDOCOUNT = 0.5


def design(addresses: np.ndarray, center: float, quadratic: bool = False) -> np.ndarray:
    """Model matrix: intercept, centered log(addresses) and optionally its
    square (a curvature check on the log-log line). Centering changes only
    the intercept's meaning (log count at the mean log size), not beta."""
    x = np.log(addresses.astype(float)) - center
    columns = [np.ones_like(x), x] + ([x**2] if quadratic else [])
    return np.column_stack(columns)


@dataclass(frozen=True)
class ExposureFit:
    """Poisson and NB2 fits of one count column on log(addresses).
    `data` keeps the rows fitted (country, addresses, response) so every
    check below works from the fit alone."""

    data: pl.DataFrame
    response: str
    center: float
    quadratic: bool
    poisson: GLMResults
    negbin: NegativeBinomialResults

    @property
    def y(self) -> np.ndarray:
        return self.data[self.response].to_numpy().astype(float)

    @property
    def addresses(self) -> np.ndarray:
        return self.data["addresses"].to_numpy()

    @property
    def converged(self) -> bool:
        return bool(self.negbin.mle_retvals.get("converged", False))

    @property
    def alpha(self) -> float:
        """NB2 dispersion: variance = mu + alpha * mu^2."""
        return float(self.negbin.params[-1])

    def mu(self, model: str, addresses: np.ndarray | None = None) -> np.ndarray:
        """Expected counts under "poisson" or "negbin", at the fitted rows
        or at new `addresses` (centered with this fit's center)."""
        exog = design(
            self.addresses if addresses is None else addresses,
            self.center,
            self.quadratic,
        )
        params = self.poisson.params if model == "poisson" else self.negbin.params
        return np.exp(exog @ params[: exog.shape[1]])


def fit_exposure_model(
    counts: pl.DataFrame,
    response: str = DEFAULT_RESPONSE,
    quadratic: bool = False,
) -> ExposureFit:
    """Poisson GLM and NB2 (maximum likelihood, alpha estimated) of
    `response` on log(addresses), over geo.exposure.country_counts output.
    The NB fit starts from the Poisson coefficients, which keeps it
    converging on sparse subsets (one split part, a bootstrap draw)."""
    data = counts.select("country", "addresses", response)
    addresses = data["addresses"].to_numpy()
    y = data[response].to_numpy().astype(float)
    center = float(np.log(addresses.astype(float)).mean())
    exog = design(addresses, center, quadratic)

    poisson = sm.GLM(y, exog, family=sm.families.Poisson()).fit()
    # The optimizer's line search can probe a negative alpha on the way to
    # the optimum (log of a negative number); those steps are rejected,
    # so the numpy warnings they raise are noise. `converged` says whether
    # the fit actually got there.
    with np.errstate(divide="ignore", invalid="ignore"):
        negbin = sm.NegativeBinomial(y, exog, loglike_method="nb2").fit(
            start_params=np.append(poisson.params, 0.5), disp=0, maxiter=500
        )
    return ExposureFit(data, response, center, quadratic, poisson, negbin)


def _nb_shape(mu: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """scipy's nbinom(n, p) parametrisation of NB2 with mean mu."""
    n = 1.0 / alpha
    return n, n / (n + mu)


def slope(fit: ExposureFit, alpha: float = DEFAULT_ALPHA) -> dict[str, float]:
    """The NB estimate of beta with its Wald CI and a z-test of beta = 1
    (proportional to address space), plus the dispersion alpha."""
    beta = float(fit.negbin.params[1])
    se = float(fit.negbin.bse[1])
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "beta": beta,
        "se": se,
        "ci_low": beta - z_crit * se,
        "ci_high": beta + z_crit * se,
        "p_beta_eq_1": float(2 * stats.norm.sf(abs(beta - 1) / se)),
        "alpha": fit.alpha,
    }


def mean_variance_table(
    counts: pl.DataFrame, response: str = DEFAULT_RESPONSE, bins: int = 8
) -> pl.DataFrame:
    """Model-free overdispersion check: countries split into `bins` equal-
    size groups by address space, with each group's mean, variance and
    variance / mean of `response`. Poisson needs variance / mean ~ 1 in
    every bin; NB2 predicts it grows as 1 + alpha * mean."""
    return (
        counts.with_columns(
            pl.col("addresses")
            .rank("ordinal")
            .qcut(bins, labels=[str(i) for i in range(1, bins + 1)])
            .cast(pl.Utf8)
            .cast(pl.Int64)
            .alias("bin")
        )
        .group_by("bin")
        .agg(
            pl.len().alias("countries"),
            pl.col("addresses").median().alias("median_addresses"),
            pl.col(response).mean().alias("mean"),
            pl.col(response).var().alias("variance"),
        )
        .with_columns((pl.col("variance") / pl.col("mean")).alias("var_to_mean"))
        .sort("bin")
    )


def moment_alpha(table: pl.DataFrame) -> float:
    """Least-squares alpha in variance = mean + alpha * mean^2 across the
    bins of mean_variance_table: a model-free estimate to compare with the
    fitted NB alpha."""
    rows = table.filter(pl.col("mean") > 0)
    mean = rows["mean"].to_numpy()
    variance = rows["variance"].to_numpy()
    return float(np.sum((variance - mean) * mean**2) / np.sum(mean**4))


def overdispersion_tests(fit: ExposureFit) -> pl.DataFrame:
    """Two tests of Poisson (alpha = 0) against NB2 (alpha > 0):
    - likelihood ratio, NB vs Poisson log-likelihood. alpha = 0 sits on
      the boundary of its range, so the chi2(1) p-value is halved.
    - Cameron-Trivedi: OLS of ((y - mu)^2 - y) / mu on mu (no intercept)
      at the Poisson fit; the slope estimates alpha. One-sided t-test.
      Least squares on squared residuals lets the few largest-mu
      countries dominate, so it has little power with ~250 countries."""
    lr = 2 * (fit.negbin.llf - fit.poisson.llf)
    mu = fit.poisson.mu
    y = fit.y
    auxiliary = sm.OLS(((y - mu) ** 2 - y) / mu, mu).fit()
    ct_alpha = float(auxiliary.params[0])
    ct_t = float(auxiliary.tvalues[0])
    return pl.DataFrame(
        {
            "test": ["likelihood ratio (alpha = 0)", "Cameron-Trivedi"],
            "statistic": [float(lr), ct_t],
            "alpha_estimate": [fit.alpha, ct_alpha],
            "p_value": [
                float(0.5 * stats.chi2.sf(lr, 1)),
                float(stats.t.sf(ct_t, auxiliary.df_resid)),
            ],
        }
    )


def _pearson_dispersion(
    y: np.ndarray, mu: np.ndarray, variance: np.ndarray, n_params: int
) -> float:
    return float(np.sum((y - mu) ** 2 / variance) / (len(y) - n_params))


def model_checks(fit: ExposureFit) -> pl.DataFrame:
    """Side-by-side fit statistics for the Poisson and NB models. Pearson
    dispersion should be ~1 when the variance model is right; the zero
    rows compare the share of countries with a count of 0 to what each
    model expects."""
    y = fit.y
    mu_p = fit.mu("poisson")
    mu_nb = fit.mu("negbin")
    n_mean = len(fit.poisson.params)
    n, p = _nb_shape(mu_nb, fit.alpha)
    return pl.DataFrame(
        {
            "check": [
                "log-likelihood",
                "AIC",
                "Pearson dispersion",
                "beta (log addresses)",
                "share of zero countries, observed",
                "share of zero countries, expected",
            ],
            "poisson": [
                float(fit.poisson.llf),
                float(fit.poisson.aic),
                _pearson_dispersion(y, mu_p, mu_p, n_mean),
                float(fit.poisson.params[1]),
                float(np.mean(y == 0)),
                float(np.mean(np.exp(-mu_p))),
            ],
            "negbin": [
                float(fit.negbin.llf),
                float(fit.negbin.aic),
                _pearson_dispersion(y, mu_nb, mu_nb + fit.alpha * mu_nb**2, n_mean + 1),
                float(fit.negbin.params[1]),
                float(np.mean(y == 0)),
                float(np.mean(stats.nbinom.pmf(0, n, p))),
            ],
        }
    )


def _band_label(low: int, high: int | None) -> str:
    if high is None:
        return f"{low}+"
    return str(low) if low == high else f"{low}-{high}"


def frequency_table(
    fit: ExposureFit,
    bands: list[tuple[int, int | None]] = FREQUENCY_BANDS,
) -> pl.DataFrame:
    """Per count band: how many countries were observed in it, and how
    many each model expects (the sum over countries of the probability of
    landing in the band)."""
    y = fit.y
    mu_p = fit.mu("poisson")
    n, p = _nb_shape(fit.mu("negbin"), fit.alpha)
    rows = []
    for low, high in bands:
        in_band = (y >= low) & (y <= (np.inf if high is None else high))
        if high is None:
            expected_p = stats.poisson.sf(low - 1, mu_p)
            expected_nb = stats.nbinom.sf(low - 1, n, p)
        else:
            expected_p = stats.poisson.cdf(high, mu_p) - stats.poisson.cdf(
                low - 1, mu_p
            )
            expected_nb = stats.nbinom.cdf(high, n, p) - stats.nbinom.cdf(low - 1, n, p)
        rows.append(
            {
                "band": _band_label(low, high),
                "observed": int(in_band.sum()),
                "poisson_expected": float(expected_p.sum()),
                "negbin_expected": float(expected_nb.sum()),
            }
        )
    return pl.DataFrame(rows)


def pearson_residuals(fit: ExposureFit) -> pl.DataFrame:
    """Per country, fitted value and Pearson residual (y - mu) / sd under
    each model. Their sum of squares is the Pearson dispersion, and the
    largest ones name the outliers. Not the best residual to *plot* for
    small counts: at y = 0 the residual is fixed by mu (a curve, not a
    scatter) and the distribution is strongly skewed, so +-2 means little
    -- use quantile_residuals for the picture and variance_by_fitted for
    the numbers."""
    y = fit.y
    mu_p = fit.mu("poisson")
    mu_nb = fit.mu("negbin")
    return fit.data.select("country").with_columns(
        pl.Series("fitted_poisson", mu_p),
        pl.Series("resid_poisson", (y - mu_p) / np.sqrt(mu_p)),
        pl.Series("fitted_negbin", mu_nb),
        pl.Series("resid_negbin", (y - mu_nb) / np.sqrt(mu_nb + fit.alpha * mu_nb**2)),
    )


def quantile_residuals(fit: ExposureFit, seed: int = 0) -> pl.DataFrame:
    """Randomized quantile residuals (Dunn & Smyth 1996) under each model:
    for a count y, draw u uniformly between F(y - 1) and F(y) of the
    model's predictive distribution and return the standard-normal
    quantile of u. If the model is right they are exactly N(0, 1) at every
    fitted value -- zeros included, which Pearson residuals can't manage --
    so a residuals-vs-fitted plot and a normal QQ plot can be read at face
    value. The randomization only spreads ties within a count; a fixed
    `seed` keeps the picture reproducible."""
    y = fit.y
    rng = np.random.default_rng(seed)
    mu_p = fit.mu("poisson")
    mu_nb = fit.mu("negbin")
    n, p = _nb_shape(mu_nb, fit.alpha)
    columns = {
        "poisson": (stats.poisson.cdf(y - 1, mu_p), stats.poisson.cdf(y, mu_p)),
        "negbin": (stats.nbinom.cdf(y - 1, n, p), stats.nbinom.cdf(y, n, p)),
    }
    # Clipped so a count far in a tail (u ~ 1 under Poisson) stays finite.
    residuals = {
        model: stats.norm.ppf(np.clip(rng.uniform(low, high), 1e-12, 1 - 1e-12))
        for model, (low, high) in columns.items()
    }
    return fit.data.select("country").with_columns(
        pl.Series("fitted_poisson", mu_p),
        pl.Series("qresid_poisson", residuals["poisson"]),
        pl.Series("fitted_negbin", mu_nb),
        pl.Series("qresid_negbin", residuals["negbin"]),
    )


def variance_by_fitted(fit: ExposureFit, bins: int = 5) -> pl.DataFrame:
    """The variance check in numbers: countries split into `bins` equal-
    size groups by NB fitted value, with the mean squared Pearson residual
    per model in each. If a model's variance function is right that mean
    is ~1 in every bin; a value climbing with the fitted count means the
    variance is understated more and more for bigger countries. The median
    is shown too: one outlier can lift a bin's mean on its own."""
    pearson = pearson_residuals(fit)
    return (
        pearson.with_columns(
            pl.col("fitted_negbin")
            .rank("ordinal")
            .qcut(bins, labels=[str(i) for i in range(1, bins + 1)])
            .cast(pl.Utf8)
            .cast(pl.Int64)
            .alias("bin")
        )
        .group_by("bin")
        .agg(
            pl.len().alias("countries"),
            pl.col("fitted_negbin").median().alias("median_fitted"),
            (pl.col("resid_poisson") ** 2).mean().alias("mean_sq_poisson"),
            (pl.col("resid_negbin") ** 2).mean().alias("mean_sq_negbin"),
            (pl.col("resid_negbin") ** 2).median().alias("median_sq_negbin"),
        )
        .sort("bin")
    )


def log_ratio(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """log((observed + 0.5) / (expected + 0.5)): over/under-representation
    on a symmetric scale that stays finite at zero counts."""
    return np.log(
        (observed + _LOG_RATIO_PSEUDOCOUNT) / (expected + _LOG_RATIO_PSEUDOCOUNT)
    )


def country_residuals(fit: ExposureFit, alpha: float = DEFAULT_ALPHA) -> pl.DataFrame:
    """Per country, the NB expected count, the ratio observed / expected,
    log_ratio, and a two-sided test of the observed count against the NB
    predictive distribution (2 * the smaller tail, capped at 1),
    BH-adjusted across all countries. `direction` is "over" or "under";
    `significant` means q_bh < alpha. Sorted by log_ratio, descending."""
    y = fit.y
    mu = fit.mu("negbin")
    n, p = _nb_shape(mu, fit.alpha)
    lower = stats.nbinom.cdf(y, n, p)
    upper = stats.nbinom.sf(y - 1, n, p)
    p_value = np.minimum(1.0, 2 * np.minimum(lower, upper))
    q = stats.false_discovery_control(p_value, method="bh")
    return (
        fit.data.rename({fit.response: "observed"})
        .with_columns(
            pl.Series("expected", mu),
            pl.Series("ratio", y / mu),
            pl.Series("log_ratio", log_ratio(y, mu)),
            pl.Series("p_value", p_value),
            pl.Series("q_bh", q),
        )
        .with_columns(
            pl.when(pl.col("observed") >= pl.col("expected"))
            .then(pl.lit("over"))
            .otherwise(pl.lit("under"))
            .alias("direction"),
            (pl.col("q_bh") < alpha).alias("significant"),
        )
        .sort("log_ratio", descending=True)
    )


def prediction_band(
    fit: ExposureFit, n_points: int = 100, level: float = 0.95
) -> pl.DataFrame:
    """NB expected count and central `level` predictive interval over a
    log-spaced grid of address-space sizes spanning the data: the band a
    country's count should fall in if it only reflected its size."""
    addresses = fit.addresses
    grid = np.geomspace(addresses.min(), addresses.max(), n_points)
    mu = fit.mu("negbin", grid)
    n, p = _nb_shape(mu, fit.alpha)
    tail = (1 - level) / 2
    return pl.DataFrame(
        {
            "addresses": grid,
            "expected": mu,
            "low": stats.nbinom.ppf(tail, n, p),
            "high": stats.nbinom.ppf(1 - tail, n, p),
        }
    )


def heldout_loglik(fit: ExposureFit, counts: pl.DataFrame, model: str) -> np.ndarray:
    """Log-likelihood of each row of `counts` (not used to fit) under
    `fit`'s "poisson" or "negbin" model -- the out-of-sample score that
    cross-validation averages."""
    y = counts[fit.response].to_numpy().astype(float)
    mu = fit.mu(model, counts["addresses"].to_numpy())
    if model == "poisson":
        return stats.poisson.logpmf(y, mu)
    n, p = _nb_shape(mu, fit.alpha)
    return stats.nbinom.logpmf(y, n, p)


def assumption_scorecard(
    fit: ExposureFit, mean_variance: pl.DataFrame, alpha: float = DEFAULT_ALPHA
) -> pl.DataFrame:
    """One row per model-choice check (mean_variance_table,
    overdispersion_tests, model_checks): the evidence in words and which
    model it favours ("poisson", "negbin" or "inconclusive")."""
    tests = dict(zip(*overdispersion_tests(fit).select("test", "p_value")))
    checks = {
        check: (p, nb)
        for check, p, nb in model_checks(fit)
        .select("check", "poisson", "negbin")
        .rows()
    }
    ratio = mean_variance["var_to_mean"].drop_nulls()
    lr_p = tests["likelihood ratio (alpha = 0)"]
    ct_p = tests["Cameron-Trivedi"]
    aic_p, aic_nb = checks["AIC"]
    disp_p, disp_nb = checks["Pearson dispersion"]
    observed_zeros = checks["share of zero countries, observed"][0]
    zeros_p, zeros_nb = checks["share of zero countries, expected"]

    def p_text(p: float) -> str:
        # Below double precision a p-value prints as 0, which reads as
        # "impossible" rather than "vanishingly small".
        return "p < 1e-16" if p < 1e-16 else f"p = {p:.2g}"

    def closer(poisson_value: float, negbin_value: float, target: float) -> str:
        return (
            "negbin"
            if abs(negbin_value - target) < abs(poisson_value - target)
            else "poisson"
        )

    rows = [
        (
            "variance vs mean (binned, model-free)",
            f"var/mean runs {ratio.min():.1f} to {ratio.max():.1f}; "
            f"moment alpha {moment_alpha(mean_variance):.2f}",
            "negbin" if ratio.max() > 2 else "poisson",
        ),
        (
            "likelihood ratio, alpha = 0",
            p_text(lr_p),
            "negbin" if lr_p < alpha else "poisson",
        ),
        (
            "Cameron-Trivedi",
            f"{p_text(ct_p)} (low power, see overdispersion_tests)",
            "negbin" if ct_p < alpha else "inconclusive",
        ),
        (
            "AIC (lower is better)",
            f"Poisson {aic_p:,.0f} vs NB {aic_nb:,.0f}",
            "negbin" if aic_nb < aic_p else "poisson",
        ),
        (
            "share of zero countries",
            f"observed {observed_zeros:.0%}; Poisson expects {zeros_p:.0%}, "
            f"NB {zeros_nb:.0%}",
            closer(zeros_p, zeros_nb, observed_zeros),
        ),
        (
            "Pearson dispersion (~1 if variance model fits)",
            f"Poisson {disp_p:.1f} vs NB {disp_nb:.1f}",
            closer(disp_p, disp_nb, 1.0),
        ),
    ]
    return pl.DataFrame(rows, schema=["check", "evidence", "favours"], orient="row")
