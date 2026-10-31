# honeypot/analysis/geo/tests/test_pipeline.py
from pathlib import Path

from honeypot.analysis.geo.pipeline import run
from honeypot.analysis.geo.tests.fixtures import world_events, write_world_dbip


def test_run_writes_every_table_and_chart(tmp_path: Path) -> None:
    snapshot = tmp_path / "logs_20260101T000000Z.parquet"
    world_events().write_parquet(snapshot)
    enrichment_dir = tmp_path / "enrichment"
    write_world_dbip(enrichment_dir)
    output_dir = tmp_path / "out"

    residuals = run(
        "offline",
        snapshot_path=snapshot,
        output_dir=output_dir,
        enrichment_dir=enrichment_dir,
        n_boot=10,
        repeats=1,
    )

    assert residuals.height == 40
    assert {p.name for p in (output_dir / "model").iterdir()} == {
        "country_counts.parquet",
        "mean_variance.parquet",
        "overdispersion_tests.parquet",
        "model_checks.parquet",
        "frequency_table.parquet",
        "variance_by_fitted.parquet",
        "slope.parquet",
        "country_residuals.parquet",
    }
    assert {p.name for p in (output_dir / "validation").iterdir()} == {
        "split_summary.parquet",
        "split_pairs.parquet",
        "response_robustness.parquet",
        "influence.parquet",
        "drop_influential.parquet",
        "bootstrap_draws.parquet",
        "bootstrap_summary.parquet",
        "cross_validation.parquet",
    }
    assert len(list((output_dir / "charts").glob("*.png"))) == 9
