# honeypot/analysis/geo/tests/test_notebook.py
#
# Executes the real regression.ipynb end to end against the synthetic
# world (fixtures.py), so a change that breaks the notebook fails the test
# suite. Same subprocess + ipykernel approach as the network notebook
# tests.
from pathlib import Path

from honeypot.analysis.geo.tests.fixtures import (
    world_events,
    write_world_dbip,
    write_world_indicators,
    write_world_shapes,
)
from honeypot.analysis.network.tests.test_notebook import _errors, _run_notebook

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "regression.ipynb"


def test_regression_notebook_runs_end_to_end(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    world_events().write_parquet(snapshot_dir / "logs_20260101T000000Z.parquet")
    enrichment_dir = tmp_path / "enrichment"
    write_world_dbip(enrichment_dir)
    write_world_indicators(tmp_path / "worldbank")
    write_world_shapes(tmp_path / "shapes")

    executed = _run_notebook(
        snapshot_dir,
        tmp_path,
        notebook_path=NOTEBOOK_PATH,
        # Fixture DB-IP, World Bank and shape files: the notebook never
        # downloads.
        extra_env={
            "BEARTRAP_ENRICHMENT_DIR": str(enrichment_dir),
            "BEARTRAP_WORLDBANK_DIR": str(tmp_path / "worldbank"),
            "BEARTRAP_SHAPES_DIR": str(tmp_path / "shapes"),
        },
    )

    assert _errors(executed) == []
