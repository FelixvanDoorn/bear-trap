from pathlib import Path

import polars as pl
import pytest

from honeypot.analysis.ml.campaigns import CampaignConfig
from honeypot.analysis.ml.pipeline import load_config, run
from honeypot.analysis.ml.runs import file_sha256, load_run
from honeypot.analysis.ml.tests.fixtures import campaign_events, write_campaign_dbip

SMALL = CampaignConfig(
    n_components=(5,),
    kmeans_k=(3, 4, 6),
    ward_k=(4,),
    hdbscan_min_cluster_size=(5,),
    stability_resamples=2,
)
CHARTS = {
    "tuning_curves.png",
    "campaign_map.png",
    "campaign_sizes.png",
    "campaign_activity.png",
    "campaign_profiles.png",
    "external_agreement.png",
}
TABLES = {
    "split_summary",
    "tuning",
    "assignments",
    "external_agreement",
    "catalog",
    "activity",
    "map_coordinates",
}


def write_campaign_inputs(root: Path) -> dict[str, Path]:
    """Snapshot, DB-IP and blocklist fixture files under `root`."""
    events, truth = campaign_events(ips_per_campaign=40)
    snapshot_dir = root / "snapshots"
    snapshot_dir.mkdir(parents=True)
    events.write_parquet(snapshot_dir / "logs_20260901T000000Z.parquet")
    write_campaign_dbip(root / "enrichment")
    threatintel = root / "threatintel"
    threatintel.mkdir()
    listed = truth.filter(pl.col("campaign") == "ssh_sprayer")["src_ip"].head(10)
    (threatintel / "blocklist_de_ssh-2026-09-01.txt").write_text(
        "\n".join(listed) + "\n"
    )
    (threatintel / "firehol_level1-2026-09-01.txt").write_text("# empty\n")
    return {
        "snapshot_dir": snapshot_dir,
        "enrichment_dir": root / "enrichment",
        "threatintel_dir": threatintel,
    }


def _hashes(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): file_sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", "run.log"}
    }


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("ml")
    dirs = write_campaign_inputs(root)
    kwargs = dirs | {"config": SMALL, "runs_dir": root / "runs"}
    return {"kwargs": kwargs, "result": run(**kwargs)}


def test_run_writes_tables_charts_and_metrics(first_run: dict) -> None:
    result = first_run["result"]
    assert not result.skipped
    assert {p.stem for p in (result.path / "tables").glob("*.parquet")} == TABLES
    assert {p.name for p in (result.path / "charts").glob("*.png")} == CHARTS

    loaded = load_run(result.path)
    assert loaded["config"] == SMALL.to_dict()
    metrics = loaded["metrics"]
    assert metrics["data"]["clustered_ips"] == 160
    assert metrics["data"]["connect_only_ips"] == 20
    assert metrics["chosen"]["train_n_clusters"] == 4
    assert metrics["test"]["prediction_strength_weighted"] > 0.9
    assert set(loaded["manifest"]["inputs"]) == {
        "snapshot",
        "dbip_country",
        "dbip_asn",
        "blocklist_de_ssh",
        "firehol_level1",
    }

    catalog = pl.read_parquet(result.path / "tables" / "catalog.parquet")
    assert catalog["ips"].sum() == 160
    assert catalog["blocklisted_share"].max() > 0
    assert "test_reproducibility" in catalog.columns
    assignments = pl.read_parquet(result.path / "tables" / "assignments.parquet")
    assert set(assignments["split"]) == {"train", "validation", "test"}

    index = pl.read_parquet(first_run["kwargs"]["runs_dir"] / "index.parquet")
    assert index["run_id"].to_list() == [result.run_id]
    assert index["metrics.campaigns.count"].to_list() == [4]


def test_rerun_skips_and_forced_rerun_is_byte_identical(first_run: dict) -> None:
    result = first_run["result"]
    before = _hashes(result.path)

    again = run(**first_run["kwargs"])
    assert again.skipped and again.run_id == result.run_id

    forced = run(force=True, **first_run["kwargs"])
    assert not forced.skipped and forced.run_id == result.run_id
    assert _hashes(forced.path) == before


def test_load_config_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"kmeans_k": [3, 5], "min_df": 1}')
    config = load_config(path, seed=7)
    assert config.kmeans_k == (3, 5)
    assert config.min_df == 1
    assert config.seed == 7
    assert config.ward_k == CampaignConfig().ward_k
