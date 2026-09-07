# honeypot/analysis/fingerprinting/tests/test_notebook.py
#
# Executes the real eda.ipynb end to end against a synthetic snapshot, so a
# code change that silently breaks the notebook (e.g. renaming a function
# it imports) fails the test suite instead of only being noticed the next
# time someone happens to open Jupyter. Runs via subprocess + a real
# ipykernel rather than importing notebook internals, since that's the
# same execution path a person opening the notebook actually exercises.
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

FINGERPRINTING_DIR = Path(__file__).parent.parent
REPO_ROOT = FINGERPRINTING_DIR.parent.parent.parent
NOTEBOOK_PATH = FINGERPRINTING_DIR / "notebooks" / "eda.ipynb"


def _fixture_snapshot(path: Path) -> None:
    pl.DataFrame(
        [
            {
                "session": "s1",
                "eventid": "cowrie.client.version",
                "event_timestamp": datetime.fromisoformat("2026-08-19T10:00:00"),
                "sensor": "sensor-01",
                "src_ip": "1.2.3.4",
                "cloud_provider": "aws",
                "protocol": "ssh",
                "command_input": None,
                "duration_ms": None,
                "username": None,
                "password": None,
                "agent_type": None,
                "raw": json.dumps({"version": "SSH-2.0-Go"}),
            },
            {
                "session": "s1",
                "eventid": "cowrie.command.input",
                "event_timestamp": datetime.fromisoformat("2026-08-19T10:00:05"),
                "sensor": "sensor-01",
                "src_ip": "1.2.3.4",
                "cloud_provider": "aws",
                "protocol": "ssh",
                "command_input": "whoami",
                "duration_ms": None,
                "username": None,
                "password": None,
                "agent_type": None,
                "raw": None,
            },
        ]
    ).write_parquet(path)


class TestNotebookExecutesCleanly:
    def test_eda_notebook_runs_end_to_end(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        _fixture_snapshot(snapshot_dir / "logs_20260101T000000Z.parquet")

        env = {
            **os.environ,
            "BEARTRAP_SNAPSHOT_DIR": str(snapshot_dir),
            "PYTHONPATH": str(REPO_ROOT),
        }

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--output-dir",
                str(tmp_path),
                "--output",
                "executed.ipynb",
                str(NOTEBOOK_PATH),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stderr

        executed = json.loads((tmp_path / "executed.ipynb").read_text())
        errors = [
            (i, out.get("ename"), out.get("evalue"))
            for i, cell in enumerate(executed["cells"])
            for out in cell.get("outputs", [])
            if out.get("output_type") == "error"
        ]
        assert errors == []
