# honeypot/analysis/geo/validation.py
#
# Validation for the exposure model in geo/regression.py. It's an
# explanatory model fitted once on ~250 countries, so "does it generalize"
# means "do its conclusions survive", not a train/test score:
#
# - Split stability (compare_split_methods): split the attacking IPs into
#   parts, refit each part, and check whether the same countries come out
#   over/under-represented (Spearman rank correlation of their log ratios,
#   and overlap of the BH-flagged "over" sets). How the split is made
#   matters, which is why several are compared:
#     random_ip   -- each IP a coin flip. The naive split: one campaign's
#                    IPs land on both sides, so agreement is inflated.
#     random_asn  -- each ASN a coin flip (grouped split). A campaign or an
#                    infected ISP stays on one side, like GroupKFold.
#     time_blocks -- IPs by first sighting, in equal-size consecutive
#                    blocks (disjoint). Asks: would next month look the
#                    same? A country carried by one burst fails this.
#     node_role   -- IPs seen by each honeypot node (an IP on both nodes
#                    is in both parts). Two vantage points, but with
#                    different configs (AWS ssh only, Azure ssh + telnet).
#     protocol    -- ssh-only vs telnet-only IPs. Not a replication: two
#                    attacker populations, compared on the same footing.
# - Response robustness: refit with /24 networks and ASNs as the count,
#   so an ISP cycling one device through many addresses counts less.
# - Influence: Cook's distance at the NB fit, and beta after dropping the
#   most influential countries.
# - Bootstrap over countries: percentile CI for beta and alpha that
#   doesn't lean on the model's own standard errors.
# - Cross-validation over countries: held-out log-likelihood of Poisson,
#   NB and NB with a quadratic term. Only for choosing between model
#   specifications, not as a performance claim.
from itertools import combinations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

from honeypot.analysis.common.stats import DEFAULT_ALPHA
from honeypot.analysis.geo.exposure import RESPONSES, country_counts
from honeypot.analysis.geo.regression import (
    DEFAULT_RESPONSE,
    ExposureFit,
    country_residuals,
    fit_exposure_model,
    heldout_loglik,
    slope,
)

SPLIT_METHODS = ["random_ip", "random_asn", "time_blocks", "node_role", "protocol"]

# Countries need this many IPs in the full data before their rank enters a
# stability correlation; below it, the log ratio is mostly Poisson noise.
DEFAULT_MIN_IPS = 10
DEFAULT_SEED = 0
DEFAULT_REPEATS = 5
DEFAULT_N_BLOCKS = 2


def _coin_flip(keys: pl.Series, seed: int) -> pl.DataFrame:
    """Each distinct key -> part "A" or "B" with probability 1/2,
    reproducible for a given seed (keys sorted first)."""
    distinct = keys.unique().sort()
    flips = np.random.default_rng(seed).integers(0, 2, distinct.len())
    return pl.DataFrame({keys.name: distinct, "part": np.where(flips == 0, "A", "B")})


def split_ips(
    attackers: pl.DataFrame,
    method: str,
    seed: int = DEFAULT_SEED,
    n_blocks: int = DEFAULT_N_BLOCKS,
) -> pl.DataFrame:
    """(src_ip, part) rows assigning geo.exposure.attacker_ips rows to
    the parts of one SPLIT_METHODS split. node_role can put an IP in
    several parts; the protocol split keeps only single-protocol IPs."""
    if method == "random_ip":
        return _coin_flip(attackers["src_ip"], seed)
    if method == "random_asn":
        # An IP without an ASN is its own group.
        keyed = attackers.with_columns(
            pl.coalesce(pl.col("asn").cast(pl.Utf8), pl.col("src_ip")).alias("group")
        )
        return keyed.join(_coin_flip(keyed["group"], seed), on="group").select(
            "src_ip", "part"
        )
    if method == "time_blocks":
        ordered = attackers.sort("first_seen", "src_ip").with_row_index("order")
        size = ordered.height / n_blocks
        blocks = ordered.with_columns(
            (pl.col("order") / size)
            .floor()
            .cast(pl.Int64)
            .clip(0, n_blocks - 1)
            .alias("block")
        )
        labels = blocks.group_by("block").agg(
            pl.format(
                "{} to {}",
                pl.col("first_seen").min().dt.strftime("%m-%d"),
                pl.col("first_seen").max().dt.strftime("%m-%d"),
            ).alias("part")
        )
        return blocks.join(labels, on="block").select("src_ip", "part")
    if method == "node_role":
        return attackers.select(
            "src_ip", pl.col("node_roles").str.split("+").alias("part")
        ).explode("part", empty_as_null=True)
    if method == "protocol":
        return attackers.filter(pl.col("protocols").is_in(["ssh", "telnet"])).select(
            "src_ip", pl.col("protocols").alias("part")
        )
    raise ValueError(f"Unknown split method: {method!r}")


def part_counts(
    attackers: pl.DataFrame, parts: pl.DataFrame, space: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """country_counts per part of a split_ips assignment, every country
    present in each. Ordered by part label."""
    joined = attackers.join(parts, on="src_ip")
    return {
        part: country_counts(joined.filter(pl.col("part") == part), space)
        for part in sorted(joined["part"].unique())
    }


def _part_stability(
    fits: dict[str, ExposureFit],
    countries: pl.Series,
    alpha: float,
) -> list[dict]:
    """Per pair of parts: Spearman correlation of country log ratios
    (over `countries` only), both slopes, the Jaccard overlap of the
    countries each part flags as significantly over-represented, and
    whether both NB fits converged."""
    residuals = {part: country_residuals(fit, alpha) for part, fit in fits.items()}
    rows = []
    for a, b in combinations(fits, 2):
        paired = (
            residuals[a]
            .select("country", pl.col("log_ratio").alias("ratio_a"))
            .join(
                residuals[b].select("country", pl.col("log_ratio").alias("ratio_b")),
                on="country",
            )
            .filter(pl.col("country").is_in(countries.implode()))
        )
        flagged = [
            set(
                residuals[part].filter(
                    pl.col("significant") & (pl.col("direction") == "over")
                )["country"]
            )
            for part in (a, b)
        ]
        union = flagged[0] | flagged[1]
        rows.append(
            {
                "part_a": a,
                "part_b": b,
                "countries": paired.height,
                "spearman": float(
                    stats.spearmanr(paired["ratio_a"], paired["ratio_b"]).statistic
                ),
                "beta_a": slope(fits[a])["beta"],
                "beta_b": slope(fits[b])["beta"],
                "flagged_a": len(flagged[0]),
                "flagged_b": len(flagged[1]),
                "flag_jaccard": (
                    len(flagged[0] & flagged[1]) / len(union) if union else None
                ),
                "converged": fits[a].converged and fits[b].converged,
            }
        )
    return rows


def _rated_countries(attackers: pl.DataFrame, min_ips: int) -> pl.Series:
    return (
        attackers.group_by("country").len().filter(pl.col("len") >= min_ips)["country"]
    )


def fit_parts(
    attackers: pl.DataFrame,
    space: pl.DataFrame,
    method: str,
    response: str = DEFAULT_RESPONSE,
    seed: int = DEFAULT_SEED,
    n_blocks: int = DEFAULT_N_BLOCKS,
) -> dict[str, ExposureFit]:
    """The exposure model refitted on each part of one split_ips split,
    keyed by part label."""
    parts = split_ips(attackers, method, seed=seed, n_blocks=n_blocks)
    return {
        part: fit_exposure_model(counts, response)
        for part, counts in part_counts(attackers, parts, space).items()
    }


def split_stability(
    attackers: pl.DataFrame,
    space: pl.DataFrame,
    method: str,
    response: str = DEFAULT_RESPONSE,
    min_ips: int = DEFAULT_MIN_IPS,
    seed: int = DEFAULT_SEED,
    n_blocks: int = DEFAULT_N_BLOCKS,
    alpha: float = DEFAULT_ALPHA,
) -> pl.DataFrame:
    """One split_ips split, every part refitted: one row per pair of parts
    (see _part_stability). Countries enter the rank correlation only with
    >= `min_ips` IPs in the full data."""
    fits = fit_parts(attackers, space, method, response, seed, n_blocks)
    rows = _part_stability(fits, _rated_countries(attackers, min_ips), alpha)
    return pl.DataFrame(rows).with_columns(
        pl.lit(method).alias("method"), pl.lit(seed).alias("seed")
    )


def split_log_ratios(
    attackers: pl.DataFrame,
    space: pl.DataFrame,
    method: str,
    response: str = DEFAULT_RESPONSE,
    seed: int = DEFAULT_SEED,
    n_blocks: int = DEFAULT_N_BLOCKS,
) -> pl.DataFrame:
    """Long table (country, part, observed, log_ratio, significant,
    direction) of each part's refit: the data behind a part-vs-part
    scatter."""
    fits = fit_parts(attackers, space, method, response, seed, n_blocks)
    return pl.concat(
        [
            country_residuals(fit)
            .select("country", "observed", "log_ratio", "significant", "direction")
            .with_columns(pl.lit(part).alias("part"))
            for part, fit in fits.items()
        ]
    )


def compare_slopes(
    fits: dict[str, ExposureFit], alpha: float = DEFAULT_ALPHA
) -> pl.DataFrame:
    """Slope and CI per fit, then for each pair a z-test of equal slopes:
    (beta_a - beta_b) / sqrt(se_a^2 + se_b^2). Valid when the fits use
    disjoint data (e.g. the ssh-only and telnet-only populations of the
    protocol split), so their estimates are independent."""
    slopes = {label: slope(fit, alpha) for label, fit in fits.items()}
    rows = [
        {
            "comparison": label,
            "beta": s["beta"],
            "ci_low": s["ci_low"],
            "ci_high": s["ci_high"],
            "p_value": s["p_beta_eq_1"],
            "test": "beta = 1",
        }
        for label, s in slopes.items()
    ]
    for a, b in combinations(slopes, 2):
        diff = slopes[a]["beta"] - slopes[b]["beta"]
        se = float(np.hypot(slopes[a]["se"], slopes[b]["se"]))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        rows.append(
            {
                "comparison": f"{a} - {b}",
                "beta": diff,
                "ci_low": diff - z_crit * se,
                "ci_high": diff + z_crit * se,
                "p_value": float(2 * stats.norm.sf(abs(diff) / se)),
                "test": "equal slopes",
            }
        )
    return pl.DataFrame(rows)


def compare_country_residuals(
    fits: dict[str, ExposureFit],
    alpha: float = DEFAULT_ALPHA,
    min_observed: int = DEFAULT_MIN_IPS,
) -> pl.DataFrame:
    """Country residuals of several fits side by side (observed, ratio
    and q_bh per fit label), keeping countries with >= `min_observed` in
    at least one fit. Answers "does the same country stand out under
    another response / in another population?". Sorted by the first
    fit's ratio, descending."""
    labels = list(fits)
    wide = None
    for label, fit in fits.items():
        part = country_residuals(fit, alpha).select(
            "country",
            pl.col("observed").alias(f"observed_{label}"),
            pl.col("ratio").alias(f"ratio_{label}"),
            pl.col("q_bh").alias(f"q_{label}"),
        )
        wide = part if wide is None else wide.join(part, on="country")
    return wide.filter(
        pl.max_horizontal([pl.col(f"observed_{label}") for label in labels])
        >= min_observed
    ).sort(f"ratio_{labels[0]}", descending=True)


def compare_split_methods(
    attackers: pl.DataFrame,
    space: pl.DataFrame,
    methods: list[str] = SPLIT_METHODS,
    response: str = DEFAULT_RESPONSE,
    min_ips: int = DEFAULT_MIN_IPS,
    repeats: int = DEFAULT_REPEATS,
    n_blocks: int = DEFAULT_N_BLOCKS,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """split_stability for every method; the random ones repeated over
    `repeats` seeds, since one coin flip can be lucky. Returns (summary
    with one row per method, every pair row). Summary: number of part
    pairs, median and min Spearman, slope range across parts, mean flag
    overlap, and whether every fit converged."""
    pairs = pl.concat(
        [
            split_stability(
                attackers,
                space,
                method,
                response=response,
                min_ips=min_ips,
                seed=seed,
                n_blocks=n_blocks,
                alpha=alpha,
            )
            for method in methods
            for seed in (range(repeats) if method.startswith("random") else [0])
        ],
        how="diagonal_relaxed",
    )
    summary = pairs.group_by("method", maintain_order=True).agg(
        pl.len().alias("pairs"),
        pl.col("spearman").median().alias("median_spearman"),
        pl.col("spearman").min().alias("min_spearman"),
        pl.min_horizontal("beta_a", "beta_b").min().alias("beta_min"),
        pl.max_horizontal("beta_a", "beta_b").max().alias("beta_max"),
        pl.col("flag_jaccard").mean().alias("mean_flag_jaccard"),
        pl.col("converged").all().alias("all_converged"),
    )
    return summary, pairs


def response_robustness(
    counts: pl.DataFrame,
    responses: list[str] = RESPONSES,
    min_ips: int = DEFAULT_MIN_IPS,
    alpha: float = DEFAULT_ALPHA,
) -> pl.DataFrame:
    """The model refitted with each count in `responses`: slope, its CI,
    alpha, how many countries are flagged over/under, and the Spearman
    correlation of country log ratios with the first response's (over
    countries with >= `min_ips` IPs)."""
    rated = counts.filter(pl.col("ips") >= min_ips)["country"]
    residuals = {}
    rows = []
    for response in responses:
        fit = fit_exposure_model(counts, response)
        residuals[response] = country_residuals(fit, alpha).filter(
            pl.col("country").is_in(rated.implode())
        )
        flagged = residuals[response].filter("significant")
        reference = residuals[responses[0]].select(
            "country", pl.col("log_ratio").alias("reference")
        )
        paired = residuals[response].join(reference, on="country")
        rows.append(
            {
                "response": response,
                "total": int(counts[response].sum()),
                **{
                    k: v
                    for k, v in slope(fit).items()
                    if k in ("beta", "ci_low", "ci_high", "alpha")
                },
                "flagged_over": flagged.filter(pl.col("direction") == "over").height,
                "flagged_under": flagged.filter(pl.col("direction") == "under").height,
                f"spearman_vs_{responses[0]}": float(
                    stats.spearmanr(paired["log_ratio"], paired["reference"]).statistic
                ),
            }
        )
    return pl.DataFrame(rows)


def influence_table(fit: ExposureFit) -> pl.DataFrame:
    """Per country: leverage (hat value) and Cook's distance from a GLM
    with the NB family at the fitted alpha. Sorted by Cook's distance,
    descending."""
    glm = sm.GLM(
        fit.y,
        fit.negbin.model.exog,
        family=sm.families.NegativeBinomial(alpha=fit.alpha),
    ).fit()
    influence = glm.get_influence()
    return (
        fit.data.rename({fit.response: "observed"})
        .with_columns(
            pl.Series("leverage", influence.hat_matrix_diag),
            pl.Series("cooks_d", influence.cooks_distance[0]),
        )
        .sort("cooks_d", descending=True)
    )


def drop_influential(
    counts: pl.DataFrame,
    fit: ExposureFit,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> pl.DataFrame:
    """beta and its CI refitted without the k most influential countries
    (influence_table order), for each k; k = 0 is the full fit."""
    order = influence_table(fit)["country"]
    rows = []
    for k in (0, *k_values):
        dropped = order.head(k)
        refit = fit_exposure_model(
            counts.filter(~pl.col("country").is_in(dropped.implode())), fit.response
        )
        s = slope(refit)
        rows.append(
            {
                "dropped": k,
                "countries_dropped": ", ".join(dropped.to_list()),
                "beta": s["beta"],
                "ci_low": s["ci_low"],
                "ci_high": s["ci_high"],
                "alpha": s["alpha"],
            }
        )
    return pl.DataFrame(rows)


def bootstrap_slope(
    counts: pl.DataFrame,
    response: str = DEFAULT_RESPONSE,
    n_boot: int = 500,
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    """beta and alpha refitted on `n_boot` resamples of the countries
    (with replacement). Draws whose fit fails to converge are dropped, so
    the result can have fewer than n_boot rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(n_boot):
        sample = counts[rng.integers(0, counts.height, counts.height)]
        fit = fit_exposure_model(sample, response)
        if not fit.converged:
            continue
        rows.append({"draw": draw, "beta": slope(fit)["beta"], "alpha": fit.alpha})
    return pl.DataFrame(
        rows, schema={"draw": pl.Int64, "beta": pl.Float64, "alpha": pl.Float64}
    )


def bootstrap_summary(draws: pl.DataFrame, level: float = 0.95) -> pl.DataFrame:
    """Median and percentile CI per parameter of bootstrap_slope draws."""
    tail = (1 - level) / 2
    return pl.DataFrame(
        [
            {
                "parameter": parameter,
                "draws": draws.height,
                "median": draws[parameter].median(),
                "ci_low": draws[parameter].quantile(tail, "linear"),
                "ci_high": draws[parameter].quantile(1 - tail, "linear"),
            }
            for parameter in ("beta", "alpha")
        ]
    )


CV_MODELS = {
    "poisson": ("poisson", False),
    "negbin": ("negbin", False),
    "negbin_quadratic": ("negbin", True),
}


def cross_validate(
    counts: pl.DataFrame,
    response: str = DEFAULT_RESPONSE,
    k: int = 5,
    repeats: int = 10,
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    """Repeated k-fold cross-validation over countries for each CV_MODELS
    entry: mean held-out log-likelihood per country (higher is better),
    its standard deviation across the k * repeats folds, and the paired
    per-fold difference from plain NB with the share of folds where the
    model beat it. Compare models on the paired difference: the fold-to-
    fold spread mostly reflects which countries landed in a fold, and it
    is the same for every model."""
    rng = np.random.default_rng(seed)
    scores: dict[str, list[float]] = {name: [] for name in CV_MODELS}
    for _ in range(repeats):
        folds = rng.permutation(counts.height) % k
        for fold in range(k):
            train = counts.filter(pl.Series(folds != fold))
            test = counts.filter(pl.Series(folds == fold))
            fits = {
                quadratic: fit_exposure_model(train, response, quadratic=quadratic)
                for quadratic in (False, True)
            }
            for name, (model, quadratic) in CV_MODELS.items():
                scores[name].append(
                    float(heldout_loglik(fits[quadratic], test, model).mean())
                )
    baseline = np.array(scores["negbin"])
    rows = []
    for name, values in scores.items():
        diff = np.array(values) - baseline
        rows.append(
            {
                "model": name,
                "folds": len(values),
                "mean_heldout_loglik": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "diff_vs_negbin": float(diff.mean()),
                "sd_diff": float(np.std(diff, ddof=1)),
                "share_folds_better": float(np.mean(diff > 0)),
            }
        )
    return pl.DataFrame(rows).sort("mean_heldout_loglik", descending=True)
