from pathlib import Path

import numpy as np
import polars as pl
import pytest

from honeypot.analysis.ml.campaigns.pipeline import run as run_campaigns
from honeypot.analysis.ml.payload.models import PayloadConfig, positive_scores
from honeypot.analysis.ml.payload.pipeline import load_config, run
from honeypot.analysis.ml.payload.sessions import horizon_rows, session_table
from honeypot.analysis.ml.runs import file_sha256, load_model, load_run
from honeypot.analysis.ml.tests.test_campaigns_pipeline import (
    SMALL as SMALL_CAMPAIGNS,
)
from honeypot.analysis.ml.tests.test_campaigns_pipeline import (
    write_campaign_inputs,
)

SMALL = PayloadConfig(
    n_folds=3,
    min_df=2,
    grid={
        "majority": [{}],
        "one_rule": [{}],
        "logistic": [{"C": 1.0}],
        "knn": [{"k": 5}],
        "gradient_boosting": [{"learning_rate": 0.1, "max_leaf_nodes": 15}],
    },
    strategies=("none", "class_weight", "undersample"),
)
TABLES = {
    "label_summary",
    "population",
    "cv_summary",
    "cv_folds",
    "chosen",
    "test_metrics",
    "pr_curves",
    "calibration",
    "errors_by_campaign",
    "feature_weights",
    "ablation",
}
CHARTS = {
    "model_comparison.png",
    "imbalance_strategies.png",
    "pr_curves.png",
    "calibration.png",
    "horizon_curve.png",
    "errors_by_campaign.png",
    "feature_weights.png",
    "ablation.png",
}


def _hashes(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): file_sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and p.name not in {"manifest.json", "run.log"}
        and p.parent.name != "models"
    }


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("payload")
    dirs = write_campaign_inputs(root)
    kwargs = {
        "snapshot_dir": dirs["snapshot_dir"],
        "enrichment_dir": dirs["enrichment_dir"],
        "campaigns_dir": root / "runs" / "campaigns",
        "runs_dir": root / "runs" / "payload_drop",
        "config": SMALL,
    }
    return {"result": run(**kwargs), "kwargs": kwargs, "dirs": dirs}


def test_run_writes_tables_charts_and_one_model_per_horizon(first_run) -> None:
    path = first_run["result"].path
    assert {p.stem for p in (path / "tables").glob("*.parquet")} == TABLES
    assert {p.name for p in (path / "charts").glob("*.png")} == CHARTS
    manifest = load_run(path)["manifest"]
    assert set(manifest["models"]) == {"horizon_0", "horizon_1", "horizon_2"}
    metrics = load_run(path)["metrics"]
    assert metrics["data"]["campaign_breakdown"] is False
    assert metrics["horizon_0"]["chosen"]["pr_auc_mean"] > 0.9


def test_saved_model_scores_horizon_rows(first_run) -> None:
    path = first_run["result"].path
    events = pl.read_parquet(next(first_run["dirs"]["snapshot_dir"].glob("*.parquet")))
    sessions = session_table(events).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("country"),
        pl.lit(None, dtype=pl.Utf8).alias("as_label"),
    )
    model, meta = load_model(path, "horizon_1")
    rows = horizon_rows(events, sessions, 1)
    scores = positive_scores(model, rows)
    assert scores.shape == (rows.height,)
    dropped = rows["dropped"].to_numpy()
    assert scores[dropped].mean() > scores[~dropped].mean()
    assert meta["thresholds"]["threshold_f1"] is not None


def test_model_never_trained_on_the_test_week(first_run) -> None:
    path = first_run["result"].path
    _, meta = load_model(path, "horizon_0")
    population = pl.read_parquet(path / "tables" / "population.parquet")
    row = population.filter(pl.col("horizon") == 0).row(0, named=True)
    assert meta["training_sessions"] == row["train_sessions"]
    test_start = load_run(path)["metrics"]["data"]["test_start"]
    assert meta["trained_before"] == test_start
    assert row["train_sessions"] + row["test_sessions"] > row["train_sessions"]


def test_rerun_is_skipped_and_forced_rerun_is_byte_identical(first_run) -> None:
    again = run(**first_run["kwargs"])
    assert again.skipped
    assert again.run_id == first_run["result"].run_id
    before = _hashes(first_run["result"].path)
    forced = run(**first_run["kwargs"], force=True)
    assert not forced.skipped
    assert _hashes(forced.path) == before


def test_campaign_breakdown_uses_a_campaigns_run_on_the_same_snapshot(
    first_run,
) -> None:
    kwargs = first_run["kwargs"]
    campaigns = run_campaigns(
        snapshot_dir=kwargs["snapshot_dir"],
        enrichment_dir=kwargs["enrichment_dir"],
        threatintel_dir=first_run["dirs"]["threatintel_dir"],
        runs_dir=kwargs["campaigns_dir"],
        config=SMALL_CAMPAIGNS,
    )
    result = run(**kwargs)
    assert result.run_id != first_run["result"].run_id
    manifest = load_run(result.path)["manifest"]
    assignments = manifest["inputs"]["campaign_assignments"]
    assert assignments["path"].startswith(str(campaigns.path))
    errors = pl.read_parquet(result.path / "tables" / "errors_by_campaign.parquet")
    assert errors["campaign"].is_in(["unclustered"]).not_().any()
    assert load_run(result.path)["metrics"]["data"]["campaign_breakdown"] is True


def test_load_config_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"horizons": [0], "n_folds": 4}')
    config = load_config(path, seed=7)
    assert (config.horizons, config.n_folds, config.seed) == ((0,), 4, 7)
    assert np.isclose(config.target_precision, 0.8)
