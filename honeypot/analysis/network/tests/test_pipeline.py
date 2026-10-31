# honeypot/analysis/network/tests/test_pipeline.py
import datetime as dt
from pathlib import Path

import polars as pl

from honeypot.analysis.common.tests.test_enrich import write_dbip_fixtures
from honeypot.analysis.common.tests.test_threatintel import (
    write_blocklist_fixtures,
)
from honeypot.analysis.network.pipeline import run


def _attempt(src_ip: str, username: str | None, password: str | None) -> dict:
    # Session-level columns too: the AI-probe profiles build on
    # overview.summary.session_table. One session per IP.
    return {
        "src_ip": src_ip,
        "username": username,
        "password": password,
        "session": f"s-{src_ip}",
        "eventid": "cowrie.login.failed",
        "node_role": "agent_detector",
        "protocol": "ssh",
        "event_timestamp": dt.datetime(2026, 8, 19, tzinfo=dt.timezone.utc),
        "duration_ms": None,
        "command_input": None,
        "raw": None,
    }


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
        ],
        # All-None columns would otherwise infer as pl.Null, not the
        # String / Int64 a real snapshot has.
        schema_overrides={
            "raw": pl.Utf8,
            "duration_ms": pl.Int64,
            "command_input": pl.Utf8,
        },
    ).write_parquet(path)


class TestEndToEndOfflinePipeline:
    def test_produces_clusters_and_chart_files(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "logs_20260819T100000Z.parquet"
        _fixture_snapshot(snapshot)
        output_dir = tmp_path / "outputs"
        enrichment_dir = tmp_path / "enrichment"
        write_dbip_fixtures(enrichment_dir)
        threatintel_dir = tmp_path / "threatintel"
        write_blocklist_fixtures(threatintel_dir)

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
            # Every fixture IP tried one username; 1 keeps them all in the
            # size-matched AI-probe comparison.
            min_usernames=1,
            enrichment_dir=enrichment_dir,
            # 1 so every fixture IP (one attempt each) is reportable.
            min_login_attempts=1,
            exclude_ips=["9.9.9.9"],
            threatintel_dir=threatintel_dir,
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
        contrast_dir = output_dir / "ai_contrast"
        assert (contrast_dir / "ip_enrichment.parquet").exists()
        profiles = pl.read_parquet(contrast_dir / "ai_probe_profiles.parquet")
        assert profiles.filter(pl.col("is_ai_probe"))["src_ip"].to_list() == ["9.9.9.9"]
        assert (contrast_dir / "ai_probe_comparison.parquet").exists()
        assert (contrast_dir / "ai_probe_comparison_by_cluster.parquet").exists()
        report_dir = output_dir / "report"
        reportable = pl.read_csv(report_dir / "reportable_ips.csv")
        # Every fixture IP except the excluded one.
        assert reportable.height == 8
        assert "9.9.9.9" not in reportable["src_ip"].to_list()
        bulk = pl.read_csv(report_dir / "abuseipdb_bulk.csv")
        assert bulk.columns == ["IP", "Categories", "ReportDate", "Comment"]
        assert bulk.height == 8
        # 1.1.1.1 is on neither fixture list; the column is still there.
        assert reportable.filter(pl.col("src_ip") == "1.1.1.1")["listed"].to_list() == [
            False
        ]
        threatintel_out = output_dir / "threatintel"
        assert (threatintel_out / "ip_listings.parquet").exists()
        assert (threatintel_out / "coverage_by_report_tier.parquet").exists()
        assert (threatintel_out / "coverage_by_ai_probe.parquet").exists()
