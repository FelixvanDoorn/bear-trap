# honeypot/analysis/ml/tests/test_notebook.py
#
# Executes the real campaigns.ipynb end to end against the planted-campaign
# fixture, so a change that breaks the notebook fails the test suite. Same
# subprocess + ipykernel approach as the other notebook tests; the run is
# written to a temp runs dir, never the real run history.
from pathlib import Path

from honeypot.analysis.ml.tests.test_campaigns_pipeline import write_campaign_inputs
from honeypot.analysis.network.tests.test_notebook import _errors, _run_notebook

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "campaigns.ipynb"


def test_campaigns_notebook_runs_end_to_end(tmp_path: Path) -> None:
    dirs = write_campaign_inputs(tmp_path / "inputs")
    executed = _run_notebook(
        dirs["snapshot_dir"],
        tmp_path,
        notebook_path=NOTEBOOK_PATH,
        extra_env={
            "BEARTRAP_ENRICHMENT_DIR": str(dirs["enrichment_dir"]),
            "BEARTRAP_THREATINTEL_DIR": str(dirs["threatintel_dir"]),
            "BEARTRAP_ML_RUNS_DIR": str(tmp_path / "runs"),
        },
    )

    assert _errors(executed) == []
    assert len(list((tmp_path / "runs").glob("*/manifest.json"))) == 1
