# honeypot/analysis/ml/tests/test_payload_notebook.py
#
# Executes the real payload_drop.ipynb end to end against the planted-
# campaign fixture, so a change that breaks the notebook fails the test
# suite. A small model grid (via BEARTRAP_ML_PAYLOAD_CONFIG) keeps it
# fast; the run is written to a temp runs dir, never the real run history.
import json
from pathlib import Path

from honeypot.analysis.ml.tests.test_campaigns_pipeline import write_campaign_inputs
from honeypot.analysis.network.tests.test_notebook import _errors, _run_notebook

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "payload_drop.ipynb"


def test_payload_notebook_runs_end_to_end(tmp_path: Path) -> None:
    dirs = write_campaign_inputs(tmp_path / "inputs")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "n_folds": 3,
                "grid": {
                    "majority": [{}],
                    "logistic": [{"C": 1.0}],
                    "tree": [{"max_depth": 3}],
                },
                "strategies": ["none", "class_weight", "undersample"],
            }
        )
    )
    executed = _run_notebook(
        dirs["snapshot_dir"],
        tmp_path,
        notebook_path=NOTEBOOK_PATH,
        extra_env={
            "BEARTRAP_ENRICHMENT_DIR": str(dirs["enrichment_dir"]),
            "BEARTRAP_THREATINTEL_DIR": str(dirs["threatintel_dir"]),
            "BEARTRAP_ML_RUNS_DIR": str(tmp_path / "runs"),
            "BEARTRAP_ML_PAYLOAD_CONFIG": str(config),
        },
    )

    assert _errors(executed) == []
    runs = list((tmp_path / "runs" / "payload_drop").glob("*/manifest.json"))
    assert len(runs) == 1
