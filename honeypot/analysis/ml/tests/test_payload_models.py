import json

import numpy as np
import polars as pl
import pytest

from honeypot.analysis.ml.payload.models import (
    Folds,
    PayloadConfig,
    ablation,
    feature_weights,
    final_pipeline,
    make_pipeline,
    positive_scores,
    select,
    thresholds,
    tune_horizon,
)
from honeypot.analysis.ml.payload.sessions import horizon_rows, session_table
from honeypot.analysis.ml.splits import group_folds
from honeypot.analysis.ml.tests.fixtures import campaign_events

SMALL = PayloadConfig(
    n_folds=3,
    grid={
        "majority": [{}],
        "one_rule": [{}],
        "naive_bayes": [{"alpha": 1.0}],
        "logistic": [{"C": 1.0}],
    },
    strategies=("none", "class_weight", "undersample"),
    min_df=2,
)


@pytest.fixture(scope="module")
def rows() -> pl.DataFrame:
    """Horizon-0 rows of the planted campaigns: the SSH dropper campaign is
    the positive class. Group = the IP's /16 (the fixture's ASN)."""
    events, _ = campaign_events(ips_per_campaign=30)
    frame = horizon_rows(events, session_table(events), 0)
    return frame.with_columns(
        pl.col("src_ip").str.extract(r"^(\d+\.\d+)").alias("group")
    )


@pytest.fixture(scope="module")
def fold(rows) -> np.ndarray:
    return group_folds(rows["group"], rows["dropped"], SMALL.n_folds, SMALL.seed)


@pytest.fixture(scope="module")
def tuned(rows, fold) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    return tune_horizon(0, Folds.encode(rows, fold, SMALL), SMALL)


def test_tuning_beats_the_baselines_on_a_planted_signal(tuned) -> None:
    summary, per_fold, oof = tuned
    by_model = {
        row["model"]: row["pr_auc_mean"]
        for row in summary.filter(pl.col("stage") == "A").iter_rows(named=True)
    }
    assert by_model["logistic"] > by_model["majority"] + 0.3
    chosen = select(summary, per_fold)
    # The dropper's client fingerprint alone separates the classes here, so
    # the one-rule stump may tie the best model and win as the simplest.
    assert chosen["model"] != "majority"
    assert chosen["pr_auc_mean"] > 0.9
    assert set(oof) == set(summary["candidate"].to_list())


def test_stage_b_tries_the_other_strategies_for_each_model(tuned) -> None:
    summary, _, _ = tuned
    strategies = {
        model: set(group["strategy"].to_list())
        for (model,), group in summary.group_by("model")
    }
    assert strategies["logistic"] == {"none", "class_weight", "undersample"}
    assert strategies["naive_bayes"] == {"none", "undersample"}
    assert strategies["majority"] == {"none"}


def test_fold_encoders_learn_from_training_folds_only(rows, fold) -> None:
    marked = rows.with_columns(
        pl.when(pl.Series(fold == 0))
        .then(pl.col("credentials").list.concat(pl.lit(["pair=only:fold0"])))
        .otherwise(pl.col("credentials"))
        .alias("credentials")
    )
    folds = Folds.encode(marked, fold, SMALL)
    names = [set(encoder.feature_names()) for encoder in folds.encoders]
    assert "credentials:pair=only:fold0" not in names[0]
    assert "credentials:pair=only:fold0" in names[1]


def test_resampling_happens_in_fit_only(rows, fold) -> None:
    folds = Folds.encode(rows, fold, SMALL)
    x_train, x_held_out = folds.matrices[0]
    y_train = folds.y[fold != 0]
    pipeline = make_pipeline("naive_bayes", {"alpha": 1.0}, "undersample", 0)
    pipeline.fit(x_train, y_train)
    counts = pipeline.named_steps["model"].class_count_
    assert counts[0] == counts[1] == y_train.sum()
    assert len(positive_scores(pipeline, x_held_out)) == x_held_out.shape[0]


def test_select_judges_ties_on_paired_fold_differences() -> None:
    folds = [0.9, 0.5, 0.8, 0.6, 0.7]
    noisy = [0.95, 0.42, 0.83, 0.55, 0.68]  # mean -0.006, noisy: tied
    worse = [f - 0.02 for f in folds]  # consistently worse: not tied
    per_fold = pl.DataFrame(
        [
            {"candidate": c, "fold": i, "pr_auc": v}
            for c, values in (("best", folds), ("noisy", noisy), ("worse", worse))
            for i, v in enumerate(values)
        ]
    )
    summary = pl.DataFrame(
        [
            {"candidate": "best", "model": "gradient_boosting", "pr_auc_mean": 0.7},
            {"candidate": "noisy", "model": "logistic", "pr_auc_mean": 0.694},
            {"candidate": "worse", "model": "naive_bayes", "pr_auc_mean": 0.68},
        ]
    )
    chosen = select(summary, per_fold)
    assert chosen["candidate"] == "noisy"
    assert chosen["best_candidate"] == "best"
    assert chosen["tied_candidates"] == 2


def test_thresholds() -> None:
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cut = thresholds(y, scores, target_precision=1.0)
    assert cut["threshold_precision"] == pytest.approx(0.6)
    assert cut["threshold_f1"] == pytest.approx(0.4)
    assert thresholds(y, 1 - scores, 1.0)["threshold_precision"] is None


def test_final_pipeline_takes_frames_and_matches_the_cv_encoding(rows) -> None:
    candidate = {
        "model": "logistic",
        "params": json.dumps({"C": 1.0}),
        "strategy": "none",
    }
    pipeline = final_pipeline(candidate, SMALL).fit(rows, rows["dropped"].to_numpy())
    encoder = SMALL.encoder().fit(rows)
    manual = make_pipeline("logistic", {"C": 1.0}, "none", SMALL.seed).fit(
        encoder.transform(rows), rows["dropped"].to_numpy()
    )
    assert np.allclose(
        positive_scores(pipeline, rows),
        positive_scores(manual, encoder.transform(rows)),
    )


def test_ablation_reports_each_present_block(rows, fold, tuned) -> None:
    summary, per_fold, _ = tuned
    table = ablation(rows, fold, select(summary, per_fold), SMALL)
    assert table.filter(pl.col("without") == "(none)")["change"][0] == 0
    assert {"credentials", "client", "timing"} <= set(table["without"].to_list())


@pytest.mark.parametrize(
    ("model", "params", "kind"),
    [
        ("logistic", {"C": 1.0}, "coefficient"),
        ("svm", {"C": 1.0}, "coefficient"),
        ("naive_bayes", {"alpha": 1.0}, "log_probability_ratio"),
        ("tree", {"max_depth": 3}, "importance"),
        ("knn", {"k": 5}, None),
    ],
)
def test_feature_weights_for_readable_models(rows, model, params, kind) -> None:
    candidate = {"model": model, "params": json.dumps(params), "strategy": "none"}
    pipeline = final_pipeline(candidate, SMALL).fit(rows, rows["dropped"].to_numpy())
    weights = feature_weights(pipeline, top=5)
    if kind is None:
        assert weights.is_empty()
        return
    assert weights.height == 5
    assert set(weights["kind"].to_list()) == {kind}
    assert weights["weight"].abs().is_sorted(descending=True)
