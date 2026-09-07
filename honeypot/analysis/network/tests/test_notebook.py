# honeypot/analysis/network/tests/test_notebook.py
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
from pathlib import Path

import polars as pl

NETWORK_DIR = Path(__file__).parent.parent
REPO_ROOT = NETWORK_DIR.parent.parent.parent
NOTEBOOK_PATH = NETWORK_DIR / "notebooks" / "eda.ipynb"


def _fixture_snapshot(path: Path) -> None:
    pl.DataFrame(
        [
            {"src_ip": "1.1.1.1", "username": "root", "password": "toor"},
            {"src_ip": "2.2.2.2", "username": "root", "password": "toor"},
            {"src_ip": "3.3.3.3", "username": "svc-deploy", "password": "Xk9mP2vQ7z"},
        ]
    ).write_parquet(path)


def _oversized_cluster_snapshot(path: Path) -> None:
    # 200 distinct IPs sharing one (non-generic) credential pair -> a
    # single 201-node cluster (200 ip nodes + 1 cred node), comfortably
    # past plot_top_clusters_subgraph's default max_nodes=150. Exercises
    # the None-return path the notebook's own cell has to handle -- the
    # exact case that crashed with `Image(None)` before that cell checked
    # for None.
    pl.DataFrame(
        [
            {
                "src_ip": f"10.0.{i // 256}.{i % 256}",
                "username": "botnet-op",
                "password": "c2-relay-9931",
            }
            for i in range(200)
        ]
    ).write_parquet(path)


def _run_notebook(snapshot_dir: Path, tmp_path: Path) -> dict:
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

    return json.loads((tmp_path / "executed.ipynb").read_text())


def _errors(executed: dict) -> list[tuple[int, str, str]]:
    return [
        (i, out.get("ename"), out.get("evalue"))
        for i, cell in enumerate(executed["cells"])
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]


class TestNotebookExecutesCleanly:
    def test_eda_notebook_runs_end_to_end(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        _fixture_snapshot(snapshot_dir / "logs_20260101T000000Z.parquet")

        executed = _run_notebook(snapshot_dir, tmp_path)

        assert _errors(executed) == []

    def test_eda_notebook_handles_oversized_subgraph_without_raising(
        self, tmp_path: Path
    ) -> None:
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        _oversized_cluster_snapshot(snapshot_dir / "logs_20260101T000000Z.parquet")

        executed = _run_notebook(snapshot_dir, tmp_path)

        assert _errors(executed) == []

        subgraph_cells = [
            cell
            for cell in executed["cells"]
            if "plot_top_clusters_subgraph(graph)" in "".join(cell.get("source", []))
        ]
        assert len(subgraph_cells) == 1
        stream_text = "".join(
            "".join(out.get("text", []))
            for out in subgraph_cells[0].get("outputs", [])
            if out.get("output_type") == "stream"
        )
        assert "Skipped" in stream_text
