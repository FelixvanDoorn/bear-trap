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

from honeypot.analysis.common.tests.test_enrich import write_dbip_fixtures
from honeypot.analysis.common.tests.test_threatintel import (
    write_blocklist_fixtures,
)
from honeypot.analysis.network.tests.test_ai_contrast import _events, _ip_events

NETWORK_DIR = Path(__file__).parent.parent
REPO_ROOT = NETWORK_DIR.parent.parent.parent
NOTEBOOK_PATH = NETWORK_DIR / "notebooks" / "eda.ipynb"
AI_CONTRAST_NOTEBOOK_PATH = NETWORK_DIR / "notebooks" / "ai_contrast.ipynb"


def _fixture_snapshot(path: Path) -> None:
    pl.DataFrame(
        [
            {"src_ip": "1.1.1.1", "username": "root", "password": "toor"},
            {"src_ip": "2.2.2.2", "username": "root", "password": "toor"},
            {"src_ip": "3.3.3.3", "username": "svc-deploy", "password": "Xk9mP2vQ7z"},
        ]
    ).write_parquet(path)


def _oversized_cluster_snapshot(path: Path) -> None:
    # 9 groups of 19 IPs, each group sharing two group-specific credential
    # pairs. 19 stays under build_credential_graph's default min_ips_to_
    # exclude=20, so the pairs survive into the bipartite graph (a single
    # pair shared by 200+ IPs, as an earlier version of this fixture used,
    # would just get excluded entirely and never reach the similarity
    # graph at all). Within a group, identical TF-IDF vectors give cosine
    # similarity 1.0, so each group becomes its own 19-IP cluster; the 9
    # clusters combined (171 IPs) comfortably exceed plot_top_clusters_
    # subgraph's default max_nodes=150 once it unions its top_n=10 largest
    # clusters. One extra, unrelated IP keeps each group's df (19) below
    # the total ip_count, so the shared pairs aren't *literally* ubiquitous
    # -- that would zero out their idf (see graph._pair_idf). Exercises
    # the None-return path the notebook's own cell has to handle -- the
    # exact case that crashed with `Image(None)` before that cell checked
    # for None.
    pl.DataFrame(
        [
            row
            for group in range(9)
            for i in range(19)
            for row in (
                {
                    "src_ip": f"10.{group}.{i // 256}.{i % 256}",
                    "username": f"botnet-op-{group}",
                    "password": f"c2-relay-{group}",
                },
                {
                    "src_ip": f"10.{group}.{i // 256}.{i % 256}",
                    "username": f"botnet-op2-{group}",
                    "password": f"c2-relay2-{group}",
                },
            )
        ]
        + [{"src_ip": "203.0.113.1", "username": "unrelated", "password": "unrelated"}]
    ).write_parquet(path)


def _ai_contrast_snapshot(path: Path) -> None:
    # Fixture IPs land in test_enrich's DB-IP fixture ranges: 1.0.0.x =
    # AS13335 (AU), 1.0.4.0-1.0.7.255 = AS38803 (CN). AS13335 is the
    # over-represented "campaign" ASN among AI probes (20 of 30 vs 5 of
    # 30), so without_top_ai_level has a real level to exclude and both
    # groups still have IPs afterwards. Dictionary sizes vary so the
    # numeric tests aren't all untestable.
    rows: list[dict] = []
    for i in range(30):
        ai_ip = f"1.0.0.{i}" if i < 20 else f"1.0.5.{i}"
        other_ip = f"1.0.0.{100 + i}" if i < 5 else f"1.0.4.{i}"
        dictionary = [f"user{j}" for j in range(10 + i % 4)]
        hassh = "campaign" if i < 20 else "other_tool"
        rows += _ip_events(ai_ip, dictionary + ["claude"], f"a{i}", hassh=hassh)
        rows += _ip_events(other_ip, dictionary, f"b{i}", hassh="other_tool")
    _events(rows).write_parquet(path)


def _run_notebook(
    snapshot_dir: Path,
    tmp_path: Path,
    notebook_path: Path = NOTEBOOK_PATH,
    extra_env: dict[str, str] | None = None,
) -> dict:
    env = {
        **os.environ,
        "BEARTRAP_SNAPSHOT_DIR": str(snapshot_dir),
        "PYTHONPATH": str(REPO_ROOT),
        **(extra_env or {}),
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
            str(notebook_path),
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
            if "plot_top_clusters_subgraph(similarity_graph)"
            in "".join(cell.get("source", []))
        ]
        assert len(subgraph_cells) == 1
        stream_text = "".join(
            "".join(out.get("text", []))
            for out in subgraph_cells[0].get("outputs", [])
            if out.get("output_type") == "stream"
        )
        assert "Skipped" in stream_text


class TestAiContrastNotebookExecutesCleanly:
    def test_ai_contrast_notebook_runs_end_to_end(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        _ai_contrast_snapshot(snapshot_dir / "logs_20260101T000000Z.parquet")
        enrichment_dir = tmp_path / "enrichment"
        write_dbip_fixtures(enrichment_dir)
        threatintel_dir = tmp_path / "threatintel"
        write_blocklist_fixtures(threatintel_dir)

        executed = _run_notebook(
            snapshot_dir,
            tmp_path,
            notebook_path=AI_CONTRAST_NOTEBOOK_PATH,
            # Fixture DB-IP files and blocklists, so the notebook never
            # downloads.
            extra_env={
                "BEARTRAP_ENRICHMENT_DIR": str(enrichment_dir),
                "BEARTRAP_THREATINTEL_DIR": str(threatintel_dir),
            },
        )

        assert _errors(executed) == []
        stream_text = "".join(
            "".join(out.get("text", []))
            for cell in executed["cells"]
            for out in cell.get("outputs", [])
            if out.get("output_type") == "stream"
        )
        assert "Excluded 25 IPs in AS13335 Cloudflare, Inc." in stream_text
