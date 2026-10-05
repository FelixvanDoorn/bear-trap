# honeypot/analysis/network/tests/test_pipeline.py
from pathlib import Path

import polars as pl

from honeypot.analysis.network.pipeline import run


def _attempt(src_ip: str, username: str | None, password: str | None) -> dict:
    return {"src_ip": src_ip, "username": username, "password": password}


def _fixture_snapshot(path: Path) -> None:
    pl.DataFrame(
        [
            # A 2-IP cluster sharing two pairs below the exclusion threshold
            # (3) -- both stay in the graph, and two shared pairs meet the
            # default min_shared_pairs.
            _attempt("1.1.1.1", "root", "toor"),
            _attempt("2.2.2.2", "root", "toor"),
            _attempt("1.1.1.1", "deploy", "d3pl0y!"),
            _attempt("2.2.2.2", "deploy", "d3pl0y!"),
            # A singleton with a unique pair.
            _attempt("3.3.3.3", "svc-deploy", "Xk9mP2vQ7z"),
            # A pair at the exclusion threshold -- excluded from the graph,
            # but still surfaced by the separate generic-attempts breakdown.
            _attempt("4.4.4.4", "admin", "admin"),
            _attempt("5.5.5.5", "admin", "admin"),
            _attempt("6.6.6.6", "admin", "admin"),
            # One username, two different passwords -- no single pair
            # crosses the threshold, but should still show up (as two
            # singleton IPs) in the username-targeting breakdown.
            _attempt("7.7.7.7", "postgres", "postgres123"),
            _attempt("8.8.8.8", "postgres", "pgpass1"),
            # An AI-tooling username -- surfaced by the AI-targeting views.
            _attempt("9.9.9.9", "claude", "claude"),
        ]
    ).write_parquet(path)


class TestEndToEndOfflinePipeline:
    def test_produces_clusters_and_chart_files(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "logs_20260819T100000Z.parquet"
        _fixture_snapshot(snapshot)
        output_dir = tmp_path / "outputs"

        clusters = run(
            mode="offline",
            snapshot_path=snapshot,
            output_dir=output_dir,
            min_ips_to_exclude=3,
            # 0.0 so the two-shared-pair cluster above still forms
            # regardless of its exact cosine similarity -- min_similarity's
            # and min_shared_pairs' own threshold behavior is covered by
            # test_graph.py, not this wiring test.
            min_similarity=0.0,
        )

        assert clusters.height == 6
        assert set(clusters["cluster_size"].to_list()) == {1, 2}
        assert (output_dir / "clusters" / "credential_clusters.parquet").exists()
        assert (
            output_dir / "clusters" / "generic_credential_attempts.parquet"
        ).exists()
        assert (
            output_dir / "clusters" / "username_targeting_breakdown.parquet"
        ).exists()
        assert (output_dir / "charts" / "credential_cluster_sizes.png").exists()
        assert (output_dir / "charts" / "top_clusters_subgraph.png").exists()
        assert (output_dir / "charts" / "generic_credential_breakdown.png").exists()
        assert (output_dir / "charts" / "username_targeting_breakdown.png").exists()
        assert (output_dir / "clusters" / "ai_username_targeting.parquet").exists()
        assert (output_dir / "clusters" / "ai_targeting_ips.parquet").exists()
        assert (output_dir / "charts" / "ai_username_targeting.png").exists()
