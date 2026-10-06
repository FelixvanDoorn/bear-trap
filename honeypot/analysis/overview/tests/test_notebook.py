# honeypot/analysis/overview/tests/test_notebook.py
#
# Executes the real eda.ipynb end to end against a synthetic snapshot -- same
# approach and rationale as network/tests/test_notebook.py: a code change
# that silently breaks the notebook fails the suite instead of only being
# noticed the next time someone opens Jupyter.
import json
import os
import subprocess
import sys
from pathlib import Path

from honeypot.analysis.overview.tests.fixtures import make_event, make_events

OVERVIEW_DIR = Path(__file__).parent.parent
REPO_ROOT = OVERVIEW_DIR.parent.parent.parent
NOTEBOOK_PATH = OVERVIEW_DIR / "notebooks" / "eda.ipynb"


def _fixture_snapshot(path: Path) -> None:
    # Four days (two survive daily_activity's partial-day trim), both roles,
    # both protocols, and at least one session reaching every funnel stage.
    make_events(
        [
            *[
                make_event(f"scan{day}", "cowrie.session.connect", day=day)
                for day in (1, 2, 3, 4)
            ],
            make_event("s1", "cowrie.login.success", day=2, src_ip="2.2.2.2"),
            make_event("s1", "cowrie.command.input", day=2, src_ip="2.2.2.2"),
            make_event("s1", "cowrie.session.file_download", day=2, src_ip="2.2.2.2"),
            make_event(
                "s1",
                "cowrie.session.closed",
                day=2,
                src_ip="2.2.2.2",
                duration_ms=900,
            ),
            make_event(
                "s2", "cowrie.login.failed", day=3, protocol="telnet", src_ip="3.3.3.3"
            ),
            make_event(
                "s3",
                "cowrie.direct-tcpip.request",
                day=3,
                node_role="control_proxy",
                dst_ip="mx.example",
                dst_port=25,
            ),
        ]
    ).write_parquet(path)


def _no_tunnels_two_days_snapshot(path: Path) -> None:
    # Only two dates (so nothing gets trimmed) and no tunnel requests at
    # all -- exercises the notebook against an empty tunnel_targets table.
    make_events(
        [
            make_event("s1", "cowrie.session.connect", day=1),
            make_event("s2", "cowrie.login.success", day=2, src_ip="2.2.2.2"),
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
            # Override the notebook's "bear-trap-analysis" kernelspec with
            # ipykernel's built-in python3 kernel, which runs in this same
            # environment (imports resolve via PYTHONPATH above) and exists on
            # any machine -- the custom kernel only exists where someone ran
            # the one-time `ipykernel install`, so CI would fail NoSuchKernel.
            "--ExecutePreprocessor.kernel_name=python3",
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
        _fixture_snapshot(snapshot_dir / "logs_20260801T000000Z.parquet")

        executed = _run_notebook(snapshot_dir, tmp_path)

        assert _errors(executed) == []

    def test_eda_notebook_handles_empty_tunnel_table(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        _no_tunnels_two_days_snapshot(snapshot_dir / "logs_20260801T000000Z.parquet")

        executed = _run_notebook(snapshot_dir, tmp_path)

        assert _errors(executed) == []
