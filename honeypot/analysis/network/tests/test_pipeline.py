# honeypot/analysis/network/tests/test_pipeline.py
from pathlib import Path

import polars as pl

from honeypot.analysis.network.pipeline import run


def _attempt(src_ip: str, username: str | None, password: str | None) -> dict:
    return {"src_ip": src_ip, "username": username, "password": password}


def _fixture_snapshot(path: Path) -> None:
    pl.DataFrame(
        [
            # A 2-IP cluster sharing one credential pair.
            _attempt("1.1.1.1", "root", "toor"),
            _attempt("2.2.2.2", "root", "toor"),
            # A singleton with a unique (non-generic) pair.
            _attempt("3.3.3.3", "svc-deploy", "Xk9mP2vQ7z"),
            # A generic pair, excluded from the graph but not from the
            # separate generic-attempts breakdown.
            _attempt("4.4.4.4", "admin", "admin"),
        ]
    ).write_parquet(path)


class TestEndToEndOfflinePipeline:
    def test_produces_clusters_and_chart_files(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "logs_20260819T100000Z.parquet"
        _fixture_snapshot(snapshot)
        output_dir = tmp_path / "outputs"

        clusters = run(mode="offline", snapshot_path=snapshot, output_dir=output_dir)

        assert clusters.height == 3
        assert set(clusters["cluster_size"].to_list()) == {1, 2}
        assert (output_dir / "clusters" / "credential_clusters.parquet").exists()
        assert (
            output_dir / "clusters" / "generic_credential_attempts.parquet"
        ).exists()
        assert (output_dir / "charts" / "credential_cluster_sizes.png").exists()
        assert (output_dir / "charts" / "top_clusters_subgraph.png").exists()
        assert (output_dir / "charts" / "generic_credential_breakdown.png").exists()
