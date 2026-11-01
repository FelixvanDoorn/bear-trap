# honeypot/analysis/geo/covariates.py
#
# Beyond address space: does adding internet users and GDP per capita
# (World Bank, common/worldbank.py) explain more of the country counts?
#
#   log E[count] = b0 + beta * log(addresses)
#                  + g1 * log(internet users) + g2 * log(GDP per capita)
#
# Every slope is an elasticity (the % change in expected attackers per 1%
# change in the predictor). Address space and internet users are strongly
# correlated -- both grow with how much internet a country has -- so
# their separate slopes are less certain than either alone, and only the
# pair is well identified. log_correlations shows how strong that is.
#
# Specifications are compared on the same countries (those with every
# covariate), in sample (compare_specifications: AIC, beta, alpha) and
# out of sample (validation.cross_validate with COVARIATE_SPECS). A
# covariate earns its place only if held-out log-likelihood improves.
import numpy as np
import polars as pl

from honeypot.analysis.geo.regression import (
    DEFAULT_RESPONSE,
    coefficients,
    fit_exposure_model,
    slope,
)
from honeypot.analysis.geo.validation import ModelSpec

COVARIATES = ("internet_users", "gdp_per_capita")

COVARIATE_SPECS: dict[str, ModelSpec] = {
    "address space": ("negbin", False, ()),
    "+ internet users": ("negbin", False, ("internet_users",)),
    "+ GDP per capita": ("negbin", False, ("gdp_per_capita",)),
    "+ both": ("negbin", False, COVARIATES),
}


def with_indicators(counts: pl.DataFrame, indicators: pl.DataFrame) -> pl.DataFrame:
    """country_counts output with the COVARIATES columns of
    common.worldbank.load_indicators joined on (null where the World Bank
    has no value)."""
    return counts.join(
        indicators.select("country", *COVARIATES), on="country", how="left"
    )


def covered(
    counts: pl.DataFrame, covariates: tuple[str, ...] = COVARIATES
) -> pl.DataFrame:
    """The rows every specification can be fitted on: all `covariates`
    present and positive (they enter on the log scale)."""
    return counts.filter(
        pl.all_horizontal(
            [pl.col(c).is_not_null() & (pl.col(c) > 0) for c in covariates]
        )
    )


def log_correlations(
    counts: pl.DataFrame, columns: tuple[str, ...] = ("addresses", *COVARIATES)
) -> pl.DataFrame:
    """Pearson correlation matrix of log(column) over the covered
    countries, plus each predictor's variance inflation factor (VIF: how
    much its slope's variance grows from sharing information with the
    others; above ~5 the separate slopes are poorly determined)."""
    logs = covered(counts, columns).select(
        [pl.col(c).cast(pl.Float64).log().alias(c) for c in columns]
    )
    matrix = np.corrcoef(logs.to_numpy(), rowvar=False)
    vif = np.diag(np.linalg.inv(matrix))
    return pl.DataFrame(
        {"variable": list(columns)}
        | {column: matrix[:, i] for i, column in enumerate(columns)}
        | {"vif": vif}
    )


def compare_specifications(
    counts: pl.DataFrame,
    specs: dict[str, ModelSpec] = COVARIATE_SPECS,
    response: str = DEFAULT_RESPONSE,
) -> pl.DataFrame:
    """Each NB specification fitted on the covered countries: number of
    countries, log-likelihood, AIC (lower is better), beta on address
    space with its CI, and alpha. Alpha falling as covariates are added
    means they explain part of the between-country variance the baseline
    left as noise."""
    data = covered(counts)
    rows = []
    for name, (_, quadratic, spec_covariates) in specs.items():
        fit = fit_exposure_model(
            data, response, quadratic=quadratic, covariates=spec_covariates
        )
        s = slope(fit)
        rows.append(
            {
                "spec": name,
                "countries": fit.data.height,
                "loglik": float(fit.negbin.llf),
                "aic": float(fit.negbin.aic),
                "beta": s["beta"],
                "beta_ci_low": s["ci_low"],
                "beta_ci_high": s["ci_high"],
                "alpha": s["alpha"],
            }
        )
    return pl.DataFrame(rows)


def full_model_coefficients(
    counts: pl.DataFrame, response: str = DEFAULT_RESPONSE
) -> pl.DataFrame:
    """coefficients() of the "+ both" specification on the covered
    countries."""
    return coefficients(
        fit_exposure_model(covered(counts), response, covariates=COVARIATES)
    )
