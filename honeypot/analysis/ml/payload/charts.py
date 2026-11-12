# honeypot/analysis/ml/payload/charts.py
#
# Charts for the payload-drop models, with ml/plotting.py's styling
# (headless Agg, per-chart style, byte-stable PNGs). Each function takes
# a run table, writes one PNG into `output_dir` and returns its path. Every
# chart is faceted by horizon (k = commands seen), the experiment's main
# axis.
#
# Color by job: models are named on the axis, so marks are ink, with the
# chosen candidate in the first categorical slot and baselines gray;
# series that need a legend (imbalance strategies, CV vs test curves)
# take the categorical slots in order; signed weights use slot 2 (orange,
# toward a drop) against slot 1 (blue, against). Values sit in the run's
# tables, the table view behind every chart.
from collections.abc import Sequence
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported, so a headless run never tries to
# initialize a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, NullFormatter  # noqa: E402

from honeypot.analysis.ml.payload.models import (  # noqa: E402
    BASELINES,
    MODEL_ORDER,
    STRATEGIES,
)
from honeypot.analysis.ml.plotting import (  # noqa: E402
    CATEGORICAL,
    INK,
    NEUTRAL,
    save,
    style,
)

_CHOSEN = CATEGORICAL[0]
_TOWARD, _AGAINST = CATEGORICAL[1], CATEGORICAL[0]


def _horizons(table: pl.DataFrame) -> list[int]:
    return sorted(table["horizon"].unique().to_list())


def _facets(
    n: int, width: float = 4.2, height: float = 4.2, sharey: bool = True
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        1, n, figsize=(width * n, height), sharey=sharey, squeeze=False
    )
    return fig, axes[0]


def _legend_below(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(labels),
        frameon=False,
    )


def plot_model_comparison(
    cv_summary: pl.DataFrame, chosen: pl.DataFrame, output_dir: Path
) -> Path:
    """Per horizon: each model's best candidate, CV PR-AUC mean +- one fold
    SD, the chosen candidate highlighted, baselines gray."""
    horizons = _horizons(cv_summary)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    with style():
        fig, axes = _facets(len(horizons), height=4.6)
        for ax, horizon in zip(axes, horizons, strict=True):
            rows = cv_summary.filter(pl.col("horizon") == horizon)
            picked = chosen.filter(pl.col("horizon") == horizon)["candidate"][0]
            # Each model's best candidate -- or the chosen one, for its model.
            best = (
                rows.with_columns((pl.col("candidate") == picked).alias("is_chosen"))
                .sort(["is_chosen", "pr_auc_mean"], descending=[True, True])
                .group_by("model", maintain_order=True)
                .first()
                .with_columns(pl.col("model").replace_strict(order).alias("order"))
                .sort("order", descending=True)
            )
            y = np.arange(best.height)
            for i, row in enumerate(best.iter_rows(named=True)):
                is_chosen = row["candidate"] == picked
                color = (
                    _CHOSEN
                    if is_chosen
                    else NEUTRAL
                    if row["model"] in BASELINES
                    else INK
                )
                ax.errorbar(
                    row["pr_auc_mean"],
                    y[i],
                    xerr=row["pr_auc_sd"],
                    fmt="o",
                    ms=8 if is_chosen else 6,
                    color=color,
                    ecolor=color,
                    elinewidth=2,
                    capsize=0,
                    label="chosen" if is_chosen else None,
                )
            ax.set_yticks(y, [m.replace("_", " ") for m in best["model"]])
            ax.set_xlim(0, 1)
            ax.set_xlabel("PR-AUC (CV mean ± fold SD)")
            ax.set_title(f"k = {horizon} commands seen")
        fig.suptitle(
            "Model comparison: grouped 5-fold CV on the training weeks", fontsize=13
        )
        fig.tight_layout()
        _legend_below(fig, axes)
        return save(fig, output_dir, "model_comparison.png")


def plot_imbalance_strategies(cv_summary: pl.DataFrame, output_dir: Path) -> Path:
    """Per model's best parameters: PR-AUC (ranking) and Brier score
    (calibration) under each imbalance strategy, first horizon."""
    horizon = _horizons(cv_summary)[0]
    rows = cv_summary.filter(
        (pl.col("horizon") == horizon) & ~pl.col("model").is_in(BASELINES)
    )
    best_params = (
        rows.sort("pr_auc_mean", descending=True)
        .group_by("model", maintain_order=True)
        .agg(pl.col("params").first())
    )
    rows = rows.join(best_params, on=["model", "params"])
    models = [m for m in MODEL_ORDER if m in rows["model"].to_list()][::-1]
    y = {m: i for i, m in enumerate(models)}
    offsets = np.linspace(-0.27, 0.27, len(STRATEGIES))
    with style():
        fig, axes = _facets(2, width=5.2, height=4.6)
        for ax, (metric, label) in zip(
            axes,
            (
                ("pr_auc_mean", "PR-AUC (higher is better)"),
                ("brier_mean", "Brier score (lower is better)"),
            ),
            strict=True,
        ):
            for slot, strategy in enumerate(STRATEGIES):
                points = rows.filter(pl.col("strategy") == strategy)
                ax.scatter(
                    points[metric],
                    [y[m] + offsets[slot] for m in points["model"]],
                    s=46,
                    color=CATEGORICAL[slot],
                    edgecolor="white",
                    linewidth=0.8,
                    label=strategy.replace("_", " "),
                    zorder=3,
                )
            ax.set_yticks(range(len(models)), [m.replace("_", " ") for m in models])
            ax.set_xlabel(label)
        axes[1].set_xscale("log")
        axes[1].set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        axes[1].xaxis.set_major_formatter(FormatStrFormatter("%g"))
        axes[1].xaxis.set_minor_formatter(NullFormatter())
        fig.suptitle(
            f"Imbalance strategies, k = {horizon} (CV on the training weeks)",
            fontsize=13,
        )
        fig.tight_layout()
        _legend_below(fig, axes)
        return save(fig, output_dir, "imbalance_strategies.png")


_CURVES = {
    "cv": ("CV (out-of-fold)", CATEGORICAL[0]),
    "test": ("test week", CATEGORICAL[1]),
    "test_cold_start": ("test week, unseen networks", CATEGORICAL[2]),
}


def plot_pr_curves(
    curves: pl.DataFrame, base_rates: pl.DataFrame, output_dir: Path
) -> Path:
    """Per horizon: the chosen model's precision-recall curve in CV, on
    the test week and on its cold-start subset; dashed line = the test
    week's base rate (what a random ranking scores)."""
    horizons = _horizons(curves)
    with style():
        fig, axes = _facets(len(horizons))
        for ax, horizon in zip(axes, horizons, strict=True):
            for curve, (label, color) in _CURVES.items():
                points = curves.filter(
                    (pl.col("horizon") == horizon) & (pl.col("curve") == curve)
                )
                if points.height:
                    ax.plot(
                        points["recall"],
                        points["precision"],
                        lw=2,
                        color=color,
                        label=label,
                    )
            rate = base_rates.filter(pl.col("horizon") == horizon)["base_rate"][0]
            ax.axhline(rate, color=INK, lw=1, ls="--", label="test base rate")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("recall")
            ax.set_title(f"k = {horizon}")
        axes[0].set_ylabel("precision")
        fig.suptitle("Precision-recall: CV vs the held-out test week", fontsize=13)
        fig.tight_layout()
        _legend_below(fig, axes)
        return save(fig, output_dir, "pr_curves.png")


def plot_calibration(calibration: pl.DataFrame, output_dir: Path) -> Path:
    """Per horizon: mean predicted vs observed drop share per score bin on
    the test week (marker area ~ sessions); on the diagonal = calibrated."""
    horizons = _horizons(calibration)
    with style():
        fig, axes = _facets(len(horizons))
        for ax, horizon in zip(axes, horizons, strict=True):
            bins = calibration.filter(pl.col("horizon") == horizon)
            ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1.5, zorder=1)
            ax.plot(bins["predicted"], bins["observed"], color=_CHOSEN, lw=2, zorder=2)
            ax.scatter(
                bins["predicted"],
                bins["observed"],
                s=20 + 180 * np.sqrt(bins["sessions"] / bins["sessions"].max()),
                color=_CHOSEN,
                edgecolor="white",
                linewidth=1,
                zorder=3,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("predicted probability")
            ax.set_title(f"k = {horizon}")
        axes[0].set_ylabel("observed share that dropped")
        fig.suptitle("Calibration on the test week", fontsize=13)
        fig.tight_layout()
        return save(fig, output_dir, "calibration.png")


def plot_ablation(ablation: pl.DataFrame, output_dir: Path) -> Path:
    """Per horizon: change in CV PR-AUC when one feature block is left
    out of the chosen model (more negative = the block matters more)."""
    horizons = _horizons(ablation)
    with style():
        fig, axes = _facets(len(horizons), height=4.0)
        for ax, horizon in zip(axes, horizons, strict=True):
            rows = ablation.filter(
                (pl.col("horizon") == horizon) & (pl.col("without") != "(none)")
            ).sort("change", descending=True)
            ax.barh(
                rows["without"],
                rows["change"],
                height=0.6,
                color=[INK if c < 0 else NEUTRAL for c in rows["change"]],
            )
            ax.axvline(0, color=INK, lw=1)
            ax.set_xlabel("PR-AUC change without the block")
            ax.set_title(f"k = {horizon}")
        fig.suptitle("Which features carry the signal (block ablation)", fontsize=13)
        fig.tight_layout()
        return save(fig, output_dir, "ablation.png")


def plot_horizon_curve(test_metrics: pl.DataFrame, output_dir: Path) -> Path:
    """PR-AUC of the chosen model by horizon: CV mean +- fold SD, test week
    with its 95% group-bootstrap interval, and the test base rate."""
    rows = test_metrics.filter(pl.col("view") == "all").sort("horizon")
    k = rows["horizon"].to_numpy()
    with style():
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.errorbar(
            k - 0.05,
            rows["cv_pr_auc"],
            yerr=rows["cv_pr_auc_sd"],
            fmt="o-",
            lw=2,
            ms=7,
            capsize=0,
            color=CATEGORICAL[0],
            label="CV (mean ± fold SD)",
        )
        ax.errorbar(
            k + 0.05,
            rows["pr_auc"],
            yerr=[
                rows["pr_auc"] - rows["pr_auc_low"],
                rows["pr_auc_high"] - rows["pr_auc"],
            ],
            fmt="o-",
            lw=2,
            ms=7,
            capsize=0,
            color=CATEGORICAL[1],
            label="test week (95% CI)",
        )
        ax.plot(
            k,
            rows["positives"] / rows["sessions"],
            ls="--",
            lw=1.5,
            color=INK,
            label="test base rate",
        )
        ax.set_xticks(k, [f"k = {h}" for h in k])
        ax.set_ylim(0, 1)
        ax.set_ylabel("PR-AUC")
        ax.set_title("How early can a drop be predicted?")
        ax.legend(loc="center right", frameon=False)
        fig.tight_layout()
        return save(fig, output_dir, "horizon_curve.png")


def plot_errors_by_campaign(errors: pl.DataFrame, output_dir: Path) -> Path:
    """Per horizon, test week: drops caught and missed, and false alarms,
    per campaign (the 8 with the most drops)."""
    horizons = _horizons(errors)
    parts = (
        ("caught", CATEGORICAL[0]),
        ("missed", CATEGORICAL[1]),
        ("false_alarms", NEUTRAL),
    )
    with style():
        fig, axes = _facets(len(horizons), height=4.2, sharey=False)
        for ax, horizon in zip(axes, horizons, strict=True):
            rows = (
                errors.filter(pl.col("horizon") == horizon)
                .filter((pl.col("positives") > 0) | (pl.col("false_alarms") > 0))
                .sort(["positives", "false_alarms"], descending=True)
                .head(8)
                .reverse()
            )
            left = np.zeros(rows.height)
            for name, color in parts:
                ax.barh(
                    rows["campaign"],
                    rows[name],
                    left=left,
                    height=0.6,
                    color=color,
                    edgecolor="white",
                    linewidth=2,
                    label=name.replace("_", " "),
                )
                left += rows[name].to_numpy()
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("test-week sessions")
            ax.set_title(f"k = {horizon}")
        axes[0].set_ylabel("campaign")
        fig.suptitle(
            "Test-week errors by campaign (at the max-F1 threshold)", fontsize=13
        )
        fig.tight_layout()
        _legend_below(fig, axes)
        return save(fig, output_dir, "errors_by_campaign.png")


def plot_feature_weights(weights: pl.DataFrame, output_dir: Path) -> Path:
    """Per horizon whose chosen model is readable: its largest weights
    (orange pushes toward a drop, blue against; importances are unsigned)."""
    horizons = _horizons(weights)
    with style():
        fig, axes = plt.subplots(
            max(len(horizons), 1), 1, figsize=(9, 4.4 * max(len(horizons), 1))
        )
        axes = np.atleast_1d(axes)
        for ax, horizon in zip(axes, horizons, strict=False):
            rows = weights.filter(pl.col("horizon") == horizon).reverse()
            labels = [f if len(f) <= 60 else f[:59] + "…" for f in rows["feature"]]
            ax.barh(
                labels,
                rows["weight"],
                height=0.6,
                color=[_TOWARD if w > 0 else _AGAINST for w in rows["weight"]],
            )
            ax.axvline(0, color=INK, lw=1)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(rows["kind"][0].replace("_", " ") if rows.height else "")
            ax.set_title(f"k = {horizon}: {rows['model'][0].replace('_', ' ')}")
        if not horizons:
            axes[0].text(0.5, 0.5, "no readable model chosen", ha="center")
            axes[0].set_axis_off()
        fig.suptitle("What the chosen models learned", fontsize=13)
        fig.tight_layout()
        return save(fig, output_dir, "feature_weights.png")
