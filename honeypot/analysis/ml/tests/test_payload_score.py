import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from honeypot.analysis.ml.payload.pipeline import run as run_training
from honeypot.analysis.ml.payload.score import ScoringConfig, run
from honeypot.analysis.ml.runs import file_sha256, load_run
from honeypot.analysis.ml.tests.test_campaigns_pipeline import write_campaign_inputs
from honeypot.analysis.ml.tests.test_payload_pipeline import SMALL

COMPARED = ("sessions", "positives", "groups", "pr_auc", "pr_auc_low", "brier")


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("scoring")
    dirs = write_campaign_inputs(root)
    training = run_training(
        snapshot_dir=dirs["snapshot_dir"],
        enrichment_dir=dirs["enrichment_dir"],
        campaigns_dir=root / "runs" / "campaigns",
        runs_dir=root / "runs" / "payload_drop",
        config=SMALL,
    )
    kwargs = {
        "snapshot_dir": dirs["snapshot_dir"],
        "enrichment_dir": dirs["enrichment_dir"],
        "model_runs_dir": root / "runs" / "payload_drop",
        "runs_dir": root / "runs" / "payload_scoring",
    }
    return {"training": training, "kwargs": kwargs}


def _score(trained: dict, **config) -> Path:
    config = ScoringConfig(model_run=trained["training"].run_id, **config)
    return run(config, **trained["kwargs"]).path


def test_scoring_the_training_snapshot_reproduces_the_test_week(trained) -> None:
    path = _score(trained)
    test = pl.read_parquet(trained["training"].path / "tables" / "test_metrics.parquet")
    metrics = pl.read_parquet(path / "tables" / "metrics.parquet")
    assert metrics["horizon"].to_list() == [0, 1, 2]
    for row in metrics.iter_rows(named=True):
        expected = test.filter(
            (pl.col("horizon") == row["horizon"]) & (pl.col("view") == "all")
        ).row(0, named=True)
        assert {k: row[k] for k in COMPARED} == {k: expected[k] for k in COMPARED}
    scores = pl.read_parquet(path / "tables" / "scores.parquet")
    assert scores.height == metrics["sessions"].sum()
    assert scores["score"].is_between(0, 1).all()
    assert load_run(path)["metrics"]["model_run"] == trained["training"].run_id


def test_inputs_pin_the_scored_model_files(trained) -> None:
    path = _score(trained, horizons=(1,))
    manifest = load_run(path)["manifest"]
    assert set(manifest["inputs"]) == {
        "snapshot",
        "dbip_country",
        "dbip_asn",
        "horizon_1",
    }
    model = trained["training"].path / "models" / "horizon_1.joblib"
    assert manifest["inputs"]["horizon_1"]["sha256"] == file_sha256(model)
    metrics = pl.read_parquet(path / "tables" / "metrics.parquet")
    assert metrics["horizon"].to_list() == [1]


def test_since_widens_the_window_and_drift_is_counted(trained) -> None:
    default = pl.read_parquet(_score(trained) / "tables" / "scores.parquet")
    earlier = default["start"].min() - dt.timedelta(days=7)
    path = _score(trained, since=str(earlier))
    wider = pl.read_parquet(path / "tables" / "scores.parquet")
    assert wider.height > default.height
    drift = pl.read_parquet(path / "tables" / "vocabulary_drift.parquet")
    assert set(drift["block"]) == {
        "credentials",
        "client",
        "commands",
        "context",
        "network",
    }
    assert (drift["unknown_tokens"] <= drift["tokens"]).all()


def test_rerun_is_skipped_and_errors_are_explicit(trained) -> None:
    config = ScoringConfig(model_run=trained["training"].run_id)
    first = run(config, **trained["kwargs"])
    again = run(config, **trained["kwargs"])
    assert again.skipped and again.run_id == first.run_id
    with pytest.raises(ValueError, match="saved no model"):
        _score(trained, horizons=(5,))
    with pytest.raises(FileNotFoundError):
        run(ScoringConfig(model_run="nope"), **trained["kwargs"])
    with pytest.raises(ValueError, match="no sessions"):
        _score(trained, since="2100-01-01T00:00:00+00:00")


def test_config_round_trip() -> None:
    config = ScoringConfig(model_run="abc", horizons=(0, 2), since="2026-09-10")
    assert ScoringConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
