import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from honeypot.analysis.ml.payload.evaluation import (
    bootstrap_intervals,
    calibration,
    errors_by_campaign,
    evaluate,
    pr_curve,
)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 60, 3000)
    y = rng.random(3000) < 0.1
    scores = np.clip(0.6 * y + rng.normal(0.2, 0.15, 3000), 0, 1)
    return y, scores, groups


def test_evaluate_reports_metrics_intervals_and_decisions(data) -> None:
    y, scores, groups = data
    result = evaluate(
        y,
        scores,
        groups,
        {"threshold_f1": 0.5, "threshold_precision": None},
        target_precision=0.8,
        resamples=200,
    )
    assert result["positives"] == y.sum()
    assert result["pr_auc"] == pytest.approx(average_precision_score(y, scores))
    assert result["pr_auc_low"] <= result["pr_auc"] <= result["pr_auc_high"]
    assert 0 < result["threshold_f1_precision"] <= 1
    assert result["threshold_f1_flagged"] == int((scores >= 0.5).sum())
    assert "threshold_precision_recall" not in result


def test_one_class_view_is_reported_not_scored(data) -> None:
    y, scores, groups = data
    result = evaluate(np.zeros_like(y), scores, groups, {}, 0.8, resamples=10)
    assert result["note"] == "needs both classes"
    assert "pr_auc" not in result


def test_bootstrap_resamples_whole_groups() -> None:
    # Two groups, perfectly ranked inside each, but group "b" is all
    # negatives: group resampling sees draws without any positive.
    y = np.array([1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4])
    groups = np.array(["a", "a", "a", "a", "b", "b"])
    intervals = bootstrap_intervals(
        y.astype(bool), scores, groups, {"ap": average_precision_score}, 200, 0
    )
    assert intervals["ap"][1] == pytest.approx(1.0)


def test_curves_and_calibration(data) -> None:
    y, scores, _ = data
    curve = pr_curve(y, scores, "test")
    assert curve.height <= 200
    assert curve["recall"].max() == 1.0
    table = calibration(y, scores)
    assert table["sessions"].sum() == len(y)
    assert table["bin"].is_sorted()


def test_errors_by_campaign() -> None:
    rows = pl.DataFrame(
        {"src_ip": ["a", "a", "b", "c"], "dropped": [True, False, True, False]}
    )
    flagged = np.array([True, True, False, False])
    campaigns = pl.DataFrame({"src_ip": ["a", "b"], "campaign": [3, 5]})
    table = errors_by_campaign(rows, flagged, campaigns)
    by = {row["campaign"]: row for row in table.iter_rows(named=True)}
    assert (by["3"]["caught"], by["3"]["false_alarms"]) == (1, 1)
    assert by["5"]["missed"] == 1
    assert by["unclustered"]["sessions"] == 1
    assert errors_by_campaign(rows, flagged, None)["campaign"].to_list() == [
        "unclustered"
    ]
