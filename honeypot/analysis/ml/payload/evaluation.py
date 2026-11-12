# honeypot/analysis/ml/payload/evaluation.py
#
# The test week, touched once per horizon by the chosen model (refit on
# all training weeks). Two views:
#
# - all: every test-week session -- how the model would have done in
#   deployment, returning networks included;
# - cold_start: test-week sessions from ASN groups never seen in training
#   -- networks it has never seen.
#
# Uncertainty: the test week holds a few hundred positives, concentrated
# in few networks, so every metric gets a 95% bootstrap interval that
# resamples ASN groups (whole networks), not sessions -- resampling
# sessions would treat a botnet's thousand near-identical sessions as a
# thousand independent draws.
#
# Decisions use the threshold chosen on the training folds' out-of-fold
# predictions; the test week never moves it.
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from honeypot.analysis.ml.payload.models import recall_at_precision

DEFAULT_RESAMPLES = 1000
CALIBRATION_BINS = 10


def _decisions(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    fn = int((~flagged & y).sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "flagged": int(flagged.sum()),
    }


def _metric_functions(
    target_precision: float,
) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "pr_auc": average_precision_score,
        "roc_auc": roc_auc_score,
        "brier": brier_score_loss,
        "recall_at_precision": lambda y, s: recall_at_precision(y, s, target_precision),
    }


def bootstrap_intervals(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metrics: Mapping[str, Callable[[np.ndarray, np.ndarray], float]],
    resamples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """95% percentile interval per metric, resampling whole groups with
    replacement (resamples without both classes are skipped)."""
    rng = np.random.default_rng(seed)
    unique, index = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(index == g) for g in range(len(unique))]
    values: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(resamples):
        picked = rng.integers(len(unique), size=len(unique))
        rows = np.concatenate([members[g] for g in picked])
        if y[rows].all() or not y[rows].any():
            continue
        for name, metric in metrics.items():
            values[name].append(float(metric(y[rows], scores[rows])))
    return {
        name: (
            (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            if v
            else (float("nan"), float("nan"))
        )
        for name, v in values.items()
    }


def evaluate(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: Mapping[str, float | None],
    target_precision: float,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Metrics, bootstrap intervals and decisions at each threshold for
    one view of the test week."""
    y = y.astype(bool)
    result: dict[str, Any] = {
        "sessions": int(len(y)),
        "positives": int(y.sum()),
        "groups": int(len(np.unique(groups))),
    }
    if y.all() or not y.any():
        return result | {"note": "needs both classes"}
    metrics = _metric_functions(target_precision)
    intervals = bootstrap_intervals(y, scores, groups, metrics, resamples, seed)
    for name, metric in metrics.items():
        low, high = intervals[name]
        result[name] = float(metric(y, scores))
        result[f"{name}_low"] = low
        result[f"{name}_high"] = high
    for name, threshold in thresholds.items():
        if threshold is not None:
            decisions = _decisions(y, scores >= threshold)
            result |= {f"{name}_{key}": value for key, value in decisions.items()}
    return result


def pr_curve(y: np.ndarray, scores: np.ndarray, label: str) -> pl.DataFrame:
    """Precision-recall points (at most ~200, evenly over recall)."""
    precision, recall, _ = precision_recall_curve(y, scores)
    keep = np.unique(np.linspace(0, len(recall) - 1, 200).round().astype(int))
    return pl.DataFrame(
        {
            "curve": label,
            "recall": recall[keep],
            "precision": precision[keep],
        }
    )


def calibration(
    y: np.ndarray, scores: np.ndarray, bins: int = CALIBRATION_BINS
) -> pl.DataFrame:
    """Per equal-width score bin: sessions, mean predicted, observed
    share of positives."""
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
    return (
        pl.DataFrame({"bin": index, "predicted": scores, "observed": y.astype(float)})
        .group_by("bin")
        .agg(
            pl.len().alias("sessions"),
            pl.col("predicted").mean(),
            pl.col("observed").mean(),
        )
        .sort("bin")
    )


def errors_by_campaign(
    rows: pl.DataFrame, flagged: np.ndarray, campaigns: pl.DataFrame | None
) -> pl.DataFrame:
    """Test-week sessions per campaign (from a campaigns run's
    assignments; "unclustered" if the IP has none): positives, caught,
    missed, false alarms."""
    frame = rows.select("src_ip", "dropped").with_columns(pl.Series("flagged", flagged))
    if campaigns is None:
        frame = frame.with_columns(pl.lit("unclustered").alias("campaign"))
    else:
        frame = frame.join(
            campaigns.select("src_ip", pl.col("campaign").cast(pl.Utf8)),
            on="src_ip",
            how="left",
            maintain_order="left",
        ).with_columns(pl.col("campaign").fill_null("unclustered"))
    return (
        frame.group_by("campaign")
        .agg(
            pl.len().alias("sessions"),
            pl.col("dropped").sum().alias("positives"),
            (pl.col("dropped") & pl.col("flagged")).sum().alias("caught"),
            (pl.col("dropped") & ~pl.col("flagged")).sum().alias("missed"),
            (~pl.col("dropped") & pl.col("flagged")).sum().alias("false_alarms"),
        )
        .sort(["positives", "sessions", "campaign"], descending=[True, True, False])
    )
