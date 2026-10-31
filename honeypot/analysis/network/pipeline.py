# honeypot/analysis/network/pipeline.py
import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from honeypot.analysis.common.enrich import ENRICHMENT_DIR, load_enrichment
from honeypot.analysis.common.extract import get_sessions
from honeypot.analysis.common.threatintel import (
    THREATINTEL_DIR,
    listing_coverage,
    load_listings,
)
from honeypot.analysis.network.ai_contrast import (
    DEFAULT_MIN_USERNAMES,
    ai_probe_profiles,
    compare_ai_probes,
    one_ip_per_cluster,
)
from honeypot.analysis.network.ai_targeting import (
    ai_targeting_ips,
    ai_username_targeting,
)
from honeypot.analysis.network.charts import (
    plot_ai_username_targeting,
    plot_cluster_size_distribution,
    plot_generic_credential_breakdown,
    plot_top_clusters_subgraph,
    plot_username_targeting_breakdown,
)
from honeypot.analysis.network.graph import (
    DEFAULT_MIN_IPS_TO_EXCLUDE,
    DEFAULT_MIN_SHARED_PAIRS,
    DEFAULT_MIN_SIMILARITY,
    assign_clusters,
    build_credential_graph,
    build_ip_similarity_graph,
    generic_credential_attempts,
    username_targeting_breakdown,
)
from honeypot.analysis.network.report import (
    DEFAULT_MIN_LOGIN_ATTEMPTS,
    abuseipdb_bulk,
    reportable_ips,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def run(
    mode: str,
    snapshot_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    min_ips_to_exclude: int = DEFAULT_MIN_IPS_TO_EXCLUDE,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_shared_pairs: int = DEFAULT_MIN_SHARED_PAIRS,
    min_usernames: int = DEFAULT_MIN_USERNAMES,
    enrichment_dir: Path = ENRICHMENT_DIR,
    min_login_attempts: int = DEFAULT_MIN_LOGIN_ATTEMPTS,
    exclude_ips: Sequence[str] = (),
    threatintel_dir: Path = THREATINTEL_DIR,
) -> pl.DataFrame:
    """Regenerate the credential-sharing cluster dataset and chart set from
    scratch."""
    events = get_sessions(mode=mode, snapshot_path=snapshot_path)
    graph = build_credential_graph(events, min_ips_to_exclude=min_ips_to_exclude)
    similarity_graph = build_ip_similarity_graph(
        graph, min_similarity=min_similarity, min_shared_pairs=min_shared_pairs
    )
    clusters = assign_clusters(similarity_graph)
    generic_attempts = generic_credential_attempts(
        events, min_ips_to_exclude=min_ips_to_exclude
    )
    username_breakdown = username_targeting_breakdown(events)
    ai_usernames = ai_username_targeting(events)
    ai_ips = ai_targeting_ips(events, clusters=clusters)
    enrichment = load_enrichment(events, enrichment_dir=enrichment_dir)
    listings = load_listings(events, threatintel_dir=threatintel_dir)
    probe_profiles = ai_probe_profiles(
        events,
        enrichment=enrichment,
        min_usernames=min_usernames,
        listings=listings,
    )
    probe_comparison = compare_ai_probes(probe_profiles)
    probe_comparison_by_cluster = compare_ai_probes(
        one_ip_per_cluster(probe_profiles, clusters)
    )
    reportable = reportable_ips(
        events,
        min_login_attempts=min_login_attempts,
        exclude_ips=exclude_ips,
        enrichment=enrichment,
    ).join(listings, on="src_ip", how="left")

    clusters_dir = output_dir / "clusters"
    charts_dir = output_dir / "charts"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    clusters.write_parquet(clusters_dir / "credential_clusters.parquet")
    generic_attempts.write_parquet(clusters_dir / "generic_credential_attempts.parquet")
    username_breakdown.write_parquet(
        clusters_dir / "username_targeting_breakdown.parquet"
    )
    ai_usernames.write_parquet(clusters_dir / "ai_username_targeting.parquet")
    ai_ips.write_parquet(clusters_dir / "ai_targeting_ips.parquet")

    contrast_dir = output_dir / "ai_contrast"
    contrast_dir.mkdir(parents=True, exist_ok=True)
    enrichment.write_parquet(contrast_dir / "ip_enrichment.parquet")
    probe_profiles.write_parquet(contrast_dir / "ai_probe_profiles.parquet")
    probe_comparison.write_parquet(contrast_dir / "ai_probe_comparison.parquet")
    probe_comparison_by_cluster.write_parquet(
        contrast_dir / "ai_probe_comparison_by_cluster.parquet"
    )

    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    reportable.write_csv(report_dir / "reportable_ips.csv")
    abuseipdb_bulk(reportable).write_csv(report_dir / "abuseipdb_bulk.csv")

    threatintel_out = output_dir / "threatintel"
    threatintel_out.mkdir(parents=True, exist_ok=True)
    listings.write_parquet(threatintel_out / "ip_listings.parquet")
    listing_coverage(reportable, "tier").write_parquet(
        threatintel_out / "coverage_by_report_tier.parquet"
    )
    listing_coverage(
        probe_profiles.select("src_ip", "is_ai_probe").join(listings, on="src_ip"),
        "is_ai_probe",
    ).write_parquet(threatintel_out / "coverage_by_ai_probe.parquet")

    plot_cluster_size_distribution(clusters, output_dir=charts_dir)
    plot_top_clusters_subgraph(similarity_graph, output_dir=charts_dir)
    plot_generic_credential_breakdown(generic_attempts, output_dir=charts_dir)
    plot_username_targeting_breakdown(username_breakdown, output_dir=charts_dir)
    plot_ai_username_targeting(ai_usernames, output_dir=charts_dir)

    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the credential-sharing cluster dataset and charts."
    )
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Explicit snapshot parquet to use (offline mode).",
    )
    parser.add_argument(
        "--min-ips-to-exclude",
        type=int,
        default=DEFAULT_MIN_IPS_TO_EXCLUDE,
        help=(
            "Exclude credential pairs attempted by at least this many "
            "distinct IPs from the graph (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help=(
            "Link two IPs in the similarity graph only if their TF-IDF "
            "cosine similarity reaches this (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-shared-pairs",
        type=int,
        default=DEFAULT_MIN_SHARED_PAIRS,
        help=(
            "Link two IPs in the similarity graph only if they also share at "
            "least this many credential pairs (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-usernames",
        type=int,
        default=DEFAULT_MIN_USERNAMES,
        help=(
            "Size-match the AI-probe comparison: only compare IPs that tried "
            "at least this many distinct usernames (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-login-attempts",
        type=int,
        default=DEFAULT_MIN_LOGIN_ATTEMPTS,
        help=(
            "Report an IP for brute force once it made at least this many "
            "login attempts (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--exclude-ip",
        action="append",
        default=[],
        dest="exclude_ips",
        help="Never report this IP (repeatable), e.g. your own scanners.",
    )
    args = parser.parse_args()

    clusters = run(
        mode=args.mode,
        snapshot_path=args.snapshot,
        min_ips_to_exclude=args.min_ips_to_exclude,
        min_similarity=args.min_similarity,
        min_shared_pairs=args.min_shared_pairs,
        min_usernames=args.min_usernames,
        min_login_attempts=args.min_login_attempts,
        exclude_ips=args.exclude_ips,
    )
    print(f"Wrote {len(clusters)} cluster-assignment rows and charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
