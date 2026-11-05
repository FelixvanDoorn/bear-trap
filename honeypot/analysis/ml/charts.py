# honeypot/analysis/ml/charts.py
#
# Charts for the campaign clustering. Seaborn on matplotlib's headless Agg
# backend, styled per chart via _style() like geo/charts.py, so importing
# this module never restyles anything else. Each function writes one PNG
# into `output_dir` (a run's charts/ directory) and returns its path;
# with the same inputs the PNG is byte-identical, so a rerun of the same
# run id reproduces its charts exactly.
#
# Color by job: campaign identity uses the fixed categorical slots in
# order (campaign 0 = slot 1, ..., the 8 largest), everything else is
# neutral gray "other"; three of the slots are below 3:1 contrast on white,
# so the map labels every colored campaign directly and the catalog table
# is the table view. Magnitudes (active IPs, shares) use one sequential
# blue ramp. Hyperparameter series (SVD dimensions) take the first three
# slots, with a legend.
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported, so a headless run never tries to
# initialize a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from honeypot.analysis.ml.campaigns import ALGORITHMS, NOISE  # noqa: E402

CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_NEUTRAL = "#c9c7c1"
_INK = "#52514e"
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)
_PARAM_LABELS = {
    "kmeans": "k",
    "ward": "k",
    "hdbscan": "min cluster size",
}


@contextmanager
def _style() -> Iterator[None]:
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        yield


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    # No "Software" tag: the PNG depends only on what's drawn.
    fig.savefig(path, bbox_inches="tight", dpi=110, metadata={"Software": None})
    plt.close(fig)
    return path


def campaign_color(campaign: int) -> str:
    """Slot color for the 8 largest campaigns (ids 0-7), gray otherwise."""
    return CATEGORICAL[campaign] if 0 <= campaign < len(CATEGORICAL) else _NEUTRAL


def plot_tuning_curves(
    tuning: pl.DataFrame, min_prediction_strength: float, output_dir: Path
) -> Path:
    """One column per algorithm: weighted validation prediction strength
    (top) and validation silhouette (bottom) against the algorithm's
    parameter, one line per SVD dimension count. The chosen config is
    ringed; the dashed line is the eligibility threshold."""
    dims = sorted(tuning["n_components"].unique().to_list())
    with _style():
        fig, axes = plt.subplots(
            2, len(ALGORITHMS), figsize=(12, 6.5), sharey="row", squeeze=False
        )
        for column, algorithm in enumerate(ALGORITHMS):
            rows = tuning.filter(pl.col("algorithm") == algorithm).sort("param")
            for row, (metric, label) in enumerate(
                (
                    ("val_prediction_strength_weighted", "prediction strength (val)"),
                    ("val_silhouette", "silhouette (val)"),
                )
            ):
                ax = axes[row][column]
                for slot, n in enumerate(dims):
                    line = rows.filter(pl.col("n_components") == n)
                    ax.plot(
                        line["param"],
                        line[metric],
                        marker="o",
                        ms=6,
                        lw=2,
                        color=CATEGORICAL[slot],
                        markeredgecolor="white",
                        label=f"{n} SVD dims",
                    )
                chosen = rows.filter("chosen")
                if chosen.height:
                    ax.scatter(
                        chosen["param"],
                        chosen[metric],
                        s=220,
                        facecolor="none",
                        edgecolor=_INK,
                        linewidth=2,
                        zorder=4,
                        label="chosen",
                    )
                if row == 0:
                    ax.axhline(
                        min_prediction_strength, color=_INK, lw=1, ls="--", zorder=1
                    )
                    ax.set_title(algorithm)
                ax.set_xlabel(_PARAM_LABELS[algorithm] if row == 1 else "")
                if column == 0:
                    ax.set_ylabel(label)
        handles, labels = axes[0][0].get_legend_handles_labels()
        for ax in axes.ravel():
            for handle, name in zip(*ax.get_legend_handles_labels()):
                if name not in labels:
                    handles.append(handle)
                    labels.append(name)
        fig.suptitle("Hyperparameter tuning on the validation split", fontsize=13)
        fig.tight_layout()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(labels),
            frameon=False,
        )
        return _save(fig, output_dir, "tuning_curves.png")


def plot_campaign_map(
    coordinates: np.ndarray, campaigns: np.ndarray, output_dir: Path
) -> Path:
    """t-SNE map of every clustered IP, the 8 largest campaigns colored
    and labeled at their median position, the rest gray."""
    campaigns = np.asarray(campaigns)
    with _style():
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = [campaign_color(int(c)) for c in campaigns]
        order = np.argsort([c < len(CATEGORICAL) and c != NOISE for c in campaigns])
        ax.scatter(
            coordinates[order, 0],
            coordinates[order, 1],
            s=14,
            c=[colors[i] for i in order],
            edgecolor="white",
            linewidth=0.4,
        )
        for campaign in range(len(CATEGORICAL)):
            members = coordinates[campaigns == campaign]
            if not len(members):
                continue
            x, y = np.median(members, axis=0)
            ax.annotate(
                str(campaign),
                (x, y),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=_INK,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": campaign_color(campaign),
                    "linewidth": 2,
                },
            )
        ax.scatter([], [], s=30, color=_NEUTRAL, label="smaller campaigns / noise")
        for campaign in range(min(len(CATEGORICAL), int(campaigns.max()) + 1)):
            ax.scatter(
                [],
                [],
                s=30,
                color=campaign_color(campaign),
                label=f"campaign {campaign}",
            )
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set(
            xticks=[],
            yticks=[],
            title="Attacking IPs by campaign (t-SNE map; distances only local)",
        )
        ax.grid(False)
        return _save(fig, output_dir, "campaign_map.png")


def plot_campaign_sizes(catalog: pl.DataFrame, output_dir: Path) -> Path:
    """IPs per campaign (largest first), annotated with the campaign's
    held-out test reproducibility; campaigns below 0.8 are flagged in the
    label and drawn gray."""
    rows = catalog.filter(pl.col("campaign") != NOISE).sort("campaign")
    with _style():
        fig, ax = plt.subplots(figsize=(8, 0.42 * rows.height + 1.5))
        y = np.arange(rows.height)
        scores = rows["test_reproducibility"].to_list()
        flagged = [score is not None and score < 0.8 for score in scores]
        ax.barh(
            y,
            rows["ips"],
            color=[_NEUTRAL if flag else CATEGORICAL[0] for flag in flagged],
            height=0.6,
        )
        for yi, ips, score, flag in zip(y, rows["ips"], scores, flagged):
            text = "test n/a" if score is None else f"test {score:.2f}"
            if flag:
                text += " (doesn't reproduce)"
            ax.annotate(
                text,
                (ips, yi),
                xytext=(6, -4),
                textcoords="offset points",
                fontsize=9,
                color=_INK,
            )
        ax.set_yticks(y, [f"campaign {c}" for c in rows["campaign"]])
        ax.invert_yaxis()
        ax.set_xlim(0, float(rows["ips"].max()) * 1.45)
        ax.set(xlabel="attacking IPs", title="Campaign sizes and test reproducibility")
        return _save(fig, output_dir, "campaign_sizes.png")


def plot_campaign_activity(activity: pl.DataFrame, output_dir: Path) -> Path:
    """Heatmap of active IPs per campaign per day (campaigns.daily_activity),
    log color scale, days without activity left blank."""
    pivot = (
        activity.filter(pl.col("campaign") != NOISE)
        .pivot(on="day", index="campaign", values="ips", sort_columns=True)
        .sort("campaign")
        .fill_null(0)
    )
    days = pivot.columns[1:]
    with _style():
        fig, ax = plt.subplots(figsize=(12, 0.4 * pivot.height + 2))
        values = pivot.select(days).to_numpy().astype(float)
        # Log scale: one campaign's burst (hundreds of IPs a day) would
        # otherwise wash out every steady campaign; empty days stay blank.
        sns.heatmap(
            values,
            ax=ax,
            cmap=_SEQUENTIAL,
            norm=LogNorm(vmin=1, vmax=max(values.max(), 2)),
            mask=values == 0,
            linewidths=1,
            linecolor="white",
            cbar_kws={"label": "active IPs (log scale)"},
        )
        ax.set_yticks(
            np.arange(pivot.height) + 0.5,
            [f"campaign {c}" for c in pivot["campaign"]],
            rotation=0,
        )
        ax.grid(False)
        step = max(1, len(days) // 10)
        ax.set_xticks(
            np.arange(0, len(days), step) + 0.5, days[::step], rotation=45, ha="right"
        )
        ax.set(xlabel="", title="When each campaign was active")
        return _save(fig, output_dir, "campaign_activity.png")


_REFERENCE_LABELS = {
    "credential_clusters": "credential clusters (ch. 4)",
    "command_sequence": "dominant command sequence",
}
_PROFILE_COLUMNS = {
    "telnet_share": "telnet sessions",
    "command_ip_share": "IPs running commands",
    "download_ip_share": "IPs downloading",
    "ai_username_share": "IPs trying AI-tooling usernames",
    "blocklisted_share": "on a blocklist",
}


def plot_campaign_profiles(catalog: pl.DataFrame, output_dir: Path) -> Path:
    """Heatmap of behavioural shares per campaign, annotated with values."""
    columns = [c for c in _PROFILE_COLUMNS if c in catalog.columns]
    rows = catalog.filter(pl.col("campaign") != NOISE).sort("campaign")
    with _style():
        fig, ax = plt.subplots(figsize=(1.4 * len(columns) + 3, 0.4 * rows.height + 2))
        sns.heatmap(
            rows.select(columns).to_numpy(),
            ax=ax,
            cmap=_SEQUENTIAL,
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 8},
            linewidths=1,
            linecolor="white",
            cbar_kws={"label": "share"},
        )
        ax.set_xticks(
            np.arange(len(columns)) + 0.5,
            [_PROFILE_COLUMNS[c] for c in columns],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(
            np.arange(rows.height) + 0.5,
            [f"campaign {c}" for c in rows["campaign"]],
            rotation=0,
        )
        ax.set_title("What each campaign does")
        return _save(fig, output_dir, "campaign_profiles.png")


def plot_external_agreement(agreement: pl.DataFrame, output_dir: Path) -> Path:
    """Per reference labelling, on the test split: the share of IPs it
    links (same label) that also share a campaign, and the ARI between the
    two partitions."""
    rows = agreement.drop_nulls("linked_kept_together")
    with _style():
        fig, ax = plt.subplots(figsize=(8, 0.8 * max(rows.height, 1) + 1.8))
        y = np.arange(rows.height)
        for offset, (metric, label, slot) in zip(
            (-0.15, 0.15),
            (
                ("linked_kept_together", "linked IPs in one campaign", 0),
                ("ari", "adjusted Rand index", 1),
            ),
        ):
            ax.barh(
                y + offset,
                rows[metric],
                height=0.28,
                color=CATEGORICAL[slot],
                label=label,
            )
            for yi, value in zip(y, rows[metric]):
                ax.annotate(
                    f"{value:.2f}",
                    (value, yi + offset),
                    xytext=(5, -4),
                    textcoords="offset points",
                    fontsize=9,
                    color=_INK,
                )
        ax.set_yticks(
            y,
            [
                f"{_REFERENCE_LABELS.get(r, r)}\n({linked} of {n} IPs linked)"
                for r, n, linked in zip(
                    rows["reference"], rows["ips"], rows["linked_ips"]
                )
            ],
        )
        ax.invert_yaxis()
        ax.set(
            xlim=(0, 1.1),
            xlabel="test split (1 = perfect agreement)",
            title="Do campaigns match structure found independently?",
        )
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            fontsize=9,
        )
        return _save(fig, output_dir, "external_agreement.png")
