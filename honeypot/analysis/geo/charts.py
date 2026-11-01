# honeypot/analysis/geo/charts.py
#
# Charts for the exposure regression. Seaborn on matplotlib's headless
# Agg backend, styled per chart via _style() like overview/charts.py, so
# importing this module never restyles anything else.
#
# Color by job: model identity uses fixed categorical slots (NB blue,
# Poisson orange, observed data in ink); over / under-representation is a
# diverging pair (orange over, blue under) with gray for "not
# significant". Every chart with two or more series has a legend.
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported, so a headless run never tries to
# initialize a GUI backend.
matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
)
from scipy import stats  # noqa: E402

from honeypot.analysis.geo.regression import ExposureFit  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs" / "charts"

_NEGBIN_COLOR = "#2a78d6"
_POISSON_COLOR = "#eb6834"
_OVER_COLOR = "#eb6834"
_UNDER_COLOR = "#2a78d6"
_NEUTRAL_COLOR = "#a9a7a1"
_INK = "#52514e"


@contextmanager
def _style() -> Iterator[None]:
    """Seaborn's whitegrid look for one chart: light recessive gridlines,
    no heavy frame."""
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        yield


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _significance_colors(residuals: pl.DataFrame) -> list[str]:
    return [
        (_OVER_COLOR if direction == "over" else _UNDER_COLOR)
        if significant
        else _NEUTRAL_COLOR
        for direction, significant in residuals.select(
            "direction", "significant"
        ).rows()
    ]


def plot_mean_variance(
    table: pl.DataFrame, alpha: float, output_dir: Path = OUTPUT_DIR
) -> Path:
    """geo.regression.mean_variance_table on log-log axes: each bin's mean
    against its variance, with the Poisson line (variance = mean) and the
    NB2 curve at the fitted alpha (variance = mean + alpha * mean^2)."""
    rows = table.filter(pl.col("mean") > 0)
    mean = rows["mean"].to_numpy()
    grid = np.geomspace(mean.min() / 2, mean.max() * 2, 100)
    with _style():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grid, grid, color=_POISSON_COLOR, lw=2, label="Poisson: var = mean")
        ax.plot(
            grid,
            grid + alpha * grid**2,
            color=_NEGBIN_COLOR,
            lw=2,
            label=f"NB2: var = mean + {alpha:.2f}·mean²",
        )
        ax.scatter(
            mean,
            rows["variance"].to_numpy(),
            s=60,
            color=_INK,
            edgecolor="white",
            linewidth=2,
            zorder=3,
            label="address-space bin",
        )
        # Bins with identical mean and variance (the all-zero smallest
        # countries) share one label instead of overprinting.
        labels = rows.group_by("mean", "variance", maintain_order=True).agg(
            pl.col("bin").cast(pl.Utf8).str.join(", ")
        )
        for m, v, b in labels.rows():
            ax.annotate(
                b,
                (m, v),
                xytext=(6, -10),
                textcoords="offset points",
                color=_INK,
                fontsize=9,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="mean attacking IPs per country (bin)",
            ylabel="variance across the bin's countries",
            title="Variance grows like mean², not like the mean",
        )
        ax.legend(frameon=False)
        return _save(fig, output_dir, "mean_variance.png")


def plot_rootogram(freq: pl.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """geo.regression.frequency_table as bars on a square-root axis
    (a rootogram: sqrt evens out the visual weight of large and small
    bands). Observed countries per band in ink, each model's expected
    count as a marker; a marker far from its bar top is a misfit."""
    bands = freq["band"].to_list()
    x = np.arange(len(bands))
    with _style():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x, freq["observed"], color="#d8d6d0", width=0.6, label="observed")
        for column, color, marker, label in (
            ("poisson_expected", _POISSON_COLOR, "D", "Poisson expects"),
            ("negbin_expected", _NEGBIN_COLOR, "o", "NB expects"),
        ):
            ax.plot(
                x,
                freq[column],
                color=color,
                marker=marker,
                markersize=9,
                # Markers only: the bands are categories, not a continuous
                # axis a line could interpolate along.
                ls="none",
                markeredgecolor="white",
                markeredgewidth=2,
                label=label,
            )
        ax.set_yscale("function", functions=(np.sqrt, np.square))
        ax.set_xticks(x, bands)
        ax.set(
            xlabel="attacking IPs per country",
            ylabel="countries (square-root scale)",
            title="How many countries fall in each count band",
        )
        ax.legend(frameon=False)
        return _save(fig, output_dir, "rootogram.png")


def plot_residuals_vs_fitted(
    quantile: pl.DataFrame, output_dir: Path = OUTPUT_DIR
) -> Path:
    """geo.regression.quantile_residuals against the fitted value, one
    panel per model, plus a normal QQ plot of both. Under the right model
    the residuals are N(0, 1) everywhere: an even band around 0 with ~5%
    outside +-2, and QQ points on the diagonal. A band that widens with
    the fitted value, or QQ points peeling away, means the variance model
    is wrong."""
    with _style():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, model, color, title in (
            (axes[0], "poisson", _POISSON_COLOR, "Poisson"),
            (axes[1], "negbin", _NEGBIN_COLOR, "Negative binomial"),
        ):
            residuals = quantile[f"qresid_{model}"].to_numpy()
            outside = float(np.mean(np.abs(residuals) > 2))
            ax.scatter(
                quantile[f"fitted_{model}"],
                residuals,
                s=24,
                color=color,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axhline(0, color=_INK, lw=1)
            for y in (-2, 2):
                ax.axhline(y, color=_INK, lw=1, ls=":")
            ax.set(
                xscale="log",
                xlabel="fitted attacking IPs",
                ylabel="quantile residual",
                title=f"{title}: {outside:.0%} outside ±2 (N(0,1): 5%)",
            )
        qq = axes[2]
        for model, color, label in (
            ("poisson", _POISSON_COLOR, "Poisson"),
            ("negbin", _NEGBIN_COLOR, "Negative binomial"),
        ):
            residuals = np.sort(quantile[f"qresid_{model}"].to_numpy())
            theoretical = stats.norm.ppf(
                (np.arange(1, len(residuals) + 1) - 0.5) / len(residuals)
            )
            qq.scatter(
                theoretical,
                residuals,
                s=18,
                color=color,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
                label=label,
            )
        limit = 4
        qq.plot([-limit, limit], [-limit, limit], color=_INK, lw=1, ls=":")
        qq.set(
            ylim=(-limit * 2, limit * 2),
            xlabel="standard normal quantile",
            ylabel="quantile residual",
            title="Normal QQ plot",
        )
        qq.legend(frameon=False, loc="upper left")
        fig.suptitle(
            "Randomized quantile residuals (dotted: ±2 and y = x; "
            "beyond ±7 drawn at ±7)"
        )
        return _save(fig, output_dir, "residuals_vs_fitted.png")


def plot_exposure_fit(
    fit: ExposureFit,
    residuals: pl.DataFrame,
    band: pl.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    label_top: int = 6,
) -> Path:
    """Attacking IPs against address space, log-log (zero counts drawn at
    0.5 so they stay on the chart), with the NB fit and its 95% predictive
    band. Significant over / under countries are colored and the
    `label_top` largest deviations in each direction are labelled."""
    plotted = residuals.with_columns(
        pl.when(pl.col("observed") > 0)
        .then(pl.col("observed").cast(pl.Float64))
        .otherwise(0.5)
        .alias("y")
    )
    with _style():
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.fill_between(
            band["addresses"],
            np.maximum(band["low"], 0.5),
            np.maximum(band["high"], 0.5),
            color=_NEGBIN_COLOR,
            alpha=0.12,
            label="95% predictive band (NB)",
        )
        ax.plot(
            band["addresses"],
            band["expected"],
            color=_NEGBIN_COLOR,
            lw=2,
            label=f"NB fit, β = {fit.negbin.params[1]:.2f}",
        )
        ax.scatter(
            plotted["addresses"],
            plotted["y"],
            s=30,
            c=_significance_colors(plotted),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        labelled = pl.concat(
            [
                plotted.filter(pl.col("observed") >= 1).head(label_top),
                plotted.filter(pl.col("observed") >= 1).tail(label_top),
            ]
        )
        for country, x, y in labelled.select("country", "addresses", "y").rows():
            ax.annotate(
                country,
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                color=_INK,
            )
        for color, label in (
            (_OVER_COLOR, "more than expected (q < 0.05)"),
            (_UNDER_COLOR, "fewer than expected (q < 0.05)"),
            (_NEUTRAL_COLOR, "within expectation"),
        ):
            ax.scatter([], [], s=30, color=color, label=label)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="IPv4 addresses in the country (DB-IP)",
            ylabel="attacking IPs (0 drawn at 0.5)",
            title="Attacking IPs vs address space",
        )
        ax.legend(frameon=False, loc="upper left")
        return _save(fig, output_dir, "exposure_fit.png")


def plot_country_ratios(
    residuals: pl.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    min_observed: int = 10,
    top_n: int = 12,
) -> Path:
    """Observed / expected ratio (log axis) for the `top_n` most over-
    and under-represented countries with at least `min_observed`
    attacking IPs, colored by BH significance."""
    rated = residuals.filter(pl.col("observed") >= min_observed)
    shown = (
        pl.concat([rated.head(top_n), rated.tail(top_n)])
        .unique("country", maintain_order=True)
        .sort("log_ratio")
    )
    with _style():
        fig, ax = plt.subplots(figsize=(7, 0.3 * shown.height + 1.5))
        y = np.arange(shown.height)
        ax.hlines(y, 1, shown["ratio"], color="#d8d6d0", lw=2)
        ax.scatter(
            shown["ratio"],
            y,
            s=60,
            c=_significance_colors(shown),
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.axvline(1, color=_INK, lw=1)
        ax.set_yticks(
            y,
            [f"{c} ({o})" for c, o in shown.select("country", "observed").rows()],
        )
        for color, label in (
            (_OVER_COLOR, "more than expected (q < 0.05)"),
            (_UNDER_COLOR, "fewer than expected (q < 0.05)"),
            (_NEUTRAL_COLOR, "not significant"),
        ):
            ax.scatter([], [], s=60, color=color, label=label)
        ax.set(
            xscale="log",
            xlabel="observed / expected attacking IPs (log scale)",
            title=f"Countries furthest from the fit (≥ {min_observed} IPs)",
        )
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        return _save(fig, output_dir, "country_ratios.png")


def plot_split_methods(summary: pl.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """geo.validation.compare_split_methods summary: median Spearman
    correlation of country log ratios between parts, per split method,
    with the minimum as a whisker."""
    with _style():
        fig, ax = plt.subplots(figsize=(7, 0.5 * summary.height + 1.5))
        y = np.arange(summary.height)
        ax.hlines(
            y,
            summary["min_spearman"],
            summary["median_spearman"],
            color="#d8d6d0",
            lw=2,
            label="lowest pair",
        )
        ax.scatter(
            summary["median_spearman"],
            y,
            s=70,
            color=_NEGBIN_COLOR,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
            label="median pair",
        )
        for yi, value in zip(y, summary["median_spearman"]):
            ax.annotate(
                f"{value:.2f}",
                (value, yi),
                xytext=(8, -4),
                textcoords="offset points",
                fontsize=9,
                color=_INK,
            )
        ax.set_yticks(y, summary["method"].to_list())
        ax.set(
            xlim=(min(0.0, float(summary["min_spearman"].min())) - 0.05, 1.05),
            xlabel="Spearman ρ of country log ratios between parts",
            title="Do the same countries stand out in every part?",
        )
        ax.invert_yaxis()
        ax.legend(frameon=False, loc="lower left", fontsize=9)
        return _save(fig, output_dir, "split_methods.png")


def plot_split_scatter(
    log_ratios: pl.DataFrame,
    title: str,
    output_dir: Path = OUTPUT_DIR,
    min_observed: int = 10,
    label_top: int = 5,
) -> Path:
    """geo.validation.split_log_ratios for a two-part split: each
    country's log ratio in the first part against the second, countries
    with >= `min_observed` IPs in both parts combined. Points on the
    diagonal stand out equally in both parts."""
    parts = sorted(log_ratios["part"].unique())[:2]
    wide = (
        log_ratios.filter(pl.col("part") == parts[0])
        .select(
            "country", pl.col("log_ratio").alias("a"), pl.col("observed").alias("n_a")
        )
        .join(
            log_ratios.filter(pl.col("part") == parts[1]).select(
                "country",
                pl.col("log_ratio").alias("b"),
                pl.col("observed").alias("n_b"),
            ),
            on="country",
        )
        .filter((pl.col("n_a") + pl.col("n_b")) >= min_observed)
        .with_columns((pl.col("a") + pl.col("b")).alias("sum"))
        .sort("sum")
    )
    with _style():
        fig, ax = plt.subplots(figsize=(6, 6))
        limits = (
            float(min(wide["a"].min(), wide["b"].min())) - 0.3,
            float(max(wide["a"].max(), wide["b"].max())) + 0.3,
        )
        ax.plot(limits, limits, color=_INK, lw=1, ls=":")
        ax.axhline(0, color="#d8d6d0", lw=1)
        ax.axvline(0, color="#d8d6d0", lw=1)
        ax.scatter(
            wide["a"],
            wide["b"],
            s=36,
            color=_NEGBIN_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for country, a, b in (
            pl.concat([wide.head(label_top), wide.tail(label_top)])
            .select("country", "a", "b")
            .rows()
        ):
            ax.annotate(
                country,
                (a, b),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
                color=_INK,
            )
        ax.set(
            xlim=limits,
            ylim=limits,
            xlabel=f"log(observed / expected), {parts[0]}",
            ylabel=f"log(observed / expected), {parts[1]}",
            title=title,
        )
        safe = title.lower().replace(" ", "_").replace("/", "_")[:40]
        return _save(fig, output_dir, f"split_scatter_{safe}.png")


def plot_bootstrap_slope(
    draws: pl.DataFrame, estimate: float, output_dir: Path = OUTPUT_DIR
) -> Path:
    """Histogram of bootstrap beta draws, with the full-data estimate and
    beta = 1 (attackers proportional to address space) marked."""
    with _style():
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            x=draws["beta"].to_numpy(),
            bins=30,
            color=_NEGBIN_COLOR,
            ax=ax,
            edgecolor="white",
            linewidth=1,
        )
        ax.axvline(estimate, color=_INK, lw=2, label=f"full data β = {estimate:.2f}")
        ax.axvline(1, color=_POISSON_COLOR, lw=2, ls="--", label="β = 1 (proportional)")
        ax.set(
            xlabel="β (slope on log address space)",
            ylabel="bootstrap draws",
            title="Slope across country resamples",
        )
        ax.legend(frameon=False)
        return _save(fig, output_dir, "bootstrap_slope.png")


# Sequential blue ramp (light -> dark) for magnitudes, and a diverging
# blue <- neutral -> orange pair for under / over expectation.
_SEQUENTIAL_BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
_DIVERGING = [
    "#104281",
    "#3987e5",
    "#9ec5f4",
    "#f0efec",
    "#f7c3a6",
    "#eb6834",
    "#a8401a",
]
_NO_DATA = "#e4e2dc"
_RATIO_TICKS = [1 / 20, 1 / 5, 1 / 2, 1, 2, 5, 20]


def _ratio_label(ratio: float) -> str:
    return f"1/{round(1 / ratio)}" if ratio < 1 else f"{ratio:g}"


def plot_choropleth_maps(
    shapes: gpd.GeoDataFrame,
    counts: pl.DataFrame,
    residual_sets: dict[str, pl.DataFrame],
    output_dir: Path = OUTPUT_DIR,
    max_ratio: float = 20.0,
) -> Path:
    """Data-Driven Security ch. 5 in one figure: a map of raw attacking
    IPs per country (sequential, log scale), then one map per entry of
    `residual_sets` (title -> country_residuals output) of observed /
    expected on a diverging log scale clipped at 1/`max_ratio` and
    `max_ratio`. Significant countries (q < 0.05) get a dark outline;
    countries with no value are drawn in light gray."""
    sequential = LinearSegmentedColormap.from_list("blues", _SEQUENTIAL_BLUES)
    diverging = LinearSegmentedColormap.from_list("over_under", _DIVERGING)
    limit = float(np.log(max_ratio))
    panels = 1 + len(residual_sets)
    with _style():
        fig, axes = plt.subplots(panels, 1, figsize=(12, 4.8 * panels))
        axes = np.atleast_1d(axes)

        raw = shapes.merge(
            counts.select("country", "ips").to_pandas(), on="country", how="left"
        )
        has_ips = raw["ips"] > 0
        if (~has_ips).any():
            raw[~has_ips].plot(ax=axes[0], color=_NO_DATA, edgecolor="white", lw=0.3)
        raw[has_ips].plot(
            ax=axes[0],
            column="ips",
            cmap=sequential,
            norm=LogNorm(vmin=1, vmax=float(raw["ips"].max())),
            edgecolor="white",
            lw=0.3,
            legend=True,
            legend_kwds={"label": "attacking IPs (log scale)", "shrink": 0.6},
        )
        axes[0].set_title("Raw count: mostly a map of address space")

        for ax, (title, residuals) in zip(axes[1:], residual_sets.items()):
            merged = shapes.merge(
                residuals.select("country", "log_ratio", "significant").to_pandas(),
                on="country",
                how="left",
            )
            known = merged["log_ratio"].notna()
            if (~known).any():
                merged[~known].plot(ax=ax, color=_NO_DATA, edgecolor="white", lw=0.3)
            merged.loc[known, "clipped"] = merged.loc[known, "log_ratio"].clip(
                -limit, limit
            )
            merged[known].plot(
                ax=ax,
                column="clipped",
                cmap=diverging,
                norm=Normalize(-limit, limit),
                edgecolor="white",
                lw=0.3,
            )
            flagged = merged[known & merged["significant"].fillna(False).astype(bool)]
            if len(flagged):
                flagged.boundary.plot(ax=ax, color=_INK, lw=1.2)
            colorbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap=diverging),
                ax=ax,
                shrink=0.6,
                ticks=np.log(_RATIO_TICKS),
            )
            colorbar.ax.set_yticklabels([_ratio_label(r) for r in _RATIO_TICKS])
            colorbar.set_label("observed / expected (outlined: q < 0.05)")
            ax.set_title(title)

        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        return _save(fig, output_dir, "choropleth_maps.png")
