# honeypot/analysis/geo/pipeline.py
import argparse
from pathlib import Path

import polars as pl

from honeypot.analysis.common.enrich import (
    ENRICHMENT_DIR,
    enrich_ips,
    fetch_dbip,
    read_ranges,
)
from honeypot.analysis.common.extract import get_sessions
from honeypot.analysis.geo.charts import (
    plot_bootstrap_slope,
    plot_country_ratios,
    plot_exposure_fit,
    plot_mean_variance,
    plot_residuals_vs_fitted,
    plot_rootogram,
    plot_split_methods,
    plot_split_scatter,
)
from honeypot.analysis.geo.exposure import (
    address_space,
    attacker_ips,
    country_counts,
)
from honeypot.analysis.geo.regression import (
    country_residuals,
    fit_exposure_model,
    frequency_table,
    mean_variance_table,
    model_checks,
    overdispersion_tests,
    prediction_band,
    quantile_residuals,
    slope,
    variance_by_fitted,
)
from honeypot.analysis.geo.validation import (
    DEFAULT_MIN_IPS,
    DEFAULT_REPEATS,
    bootstrap_slope,
    bootstrap_summary,
    compare_split_methods,
    cross_validate,
    drop_influential,
    influence_table,
    response_robustness,
    split_log_ratios,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_N_BOOT = 500


def run(
    mode: str,
    snapshot_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    enrichment_dir: Path = ENRICHMENT_DIR,
    n_boot: int = DEFAULT_N_BOOT,
    repeats: int = DEFAULT_REPEATS,
    min_ips: int = DEFAULT_MIN_IPS,
) -> pl.DataFrame:
    """Regenerate the exposure-regression tables and charts from scratch.
    Returns country_residuals of the main fit."""
    events = get_sessions(mode=mode, snapshot_path=snapshot_path)
    country_ranges = read_ranges(fetch_dbip("country", enrichment_dir), "country")
    enrichment = enrich_ips(
        events, country_ranges, read_ranges(fetch_dbip("asn", enrichment_dir), "asn")
    )
    space = address_space(country_ranges)
    attackers = attacker_ips(events, enrichment)
    counts = country_counts(attackers, space)

    fit = fit_exposure_model(counts)
    mean_variance = mean_variance_table(counts)
    frequencies = frequency_table(fit)
    residuals = country_residuals(fit)
    split_summary, split_pairs = compare_split_methods(
        attackers, space, repeats=repeats, min_ips=min_ips
    )
    draws = bootstrap_slope(counts, n_boot=n_boot)

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "country_counts": counts,
        "mean_variance": mean_variance,
        "overdispersion_tests": overdispersion_tests(fit),
        "model_checks": model_checks(fit),
        "frequency_table": frequencies,
        "variance_by_fitted": variance_by_fitted(fit),
        "slope": pl.DataFrame([slope(fit)]),
        "country_residuals": residuals,
    }
    validation_tables = {
        "split_summary": split_summary,
        "split_pairs": split_pairs,
        "response_robustness": response_robustness(counts, min_ips=min_ips),
        "influence": influence_table(fit),
        "drop_influential": drop_influential(counts, fit),
        "bootstrap_draws": draws,
        "bootstrap_summary": bootstrap_summary(draws),
        "cross_validation": cross_validate(counts),
    }
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for directory, group in ((model_dir, tables), (validation_dir, validation_tables)):
        for name, table in group.items():
            table.write_parquet(directory / f"{name}.parquet")

    charts_dir = output_dir / "charts"
    plot_mean_variance(mean_variance, fit.alpha, output_dir=charts_dir)
    plot_rootogram(frequencies, output_dir=charts_dir)
    plot_residuals_vs_fitted(quantile_residuals(fit), output_dir=charts_dir)
    plot_exposure_fit(fit, residuals, prediction_band(fit), output_dir=charts_dir)
    plot_country_ratios(residuals, output_dir=charts_dir)
    plot_split_methods(split_summary, output_dir=charts_dir)
    for method, title in (
        ("time_blocks", "First vs second half"),
        ("random_asn", "Random ASN halves"),
    ):
        plot_split_scatter(
            split_log_ratios(attackers, space, method), title, output_dir=charts_dir
        )
    plot_bootstrap_slope(draws, slope(fit)["beta"], output_dir=charts_dir)

    return residuals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the attackers-vs-address-space regression."
    )
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Explicit snapshot parquet to use (offline mode).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Bootstrap resamples of the countries (default: %(default)s).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Seeds per random split method (default: %(default)s).",
    )
    parser.add_argument(
        "--min-ips",
        type=int,
        default=DEFAULT_MIN_IPS,
        help=(
            "Countries need this many attacking IPs to enter a stability "
            "correlation (default: %(default)s)."
        ),
    )
    args = parser.parse_args()
    run(
        args.mode,
        snapshot_path=args.snapshot,
        n_boot=args.n_boot,
        repeats=args.repeats,
        min_ips=args.min_ips,
    )


if __name__ == "__main__":
    main()
