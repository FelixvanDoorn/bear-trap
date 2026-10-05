# honeypot/analysis/network/pipeline.py
import argparse
from pathlib import Path

import polars as pl

from honeypot.analysis.common.extract import get_sessions
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

OUTPUT_DIR = Path(__file__).parent / "outputs"


def run(
    mode: str,
    snapshot_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    min_ips_to_exclude: int = DEFAULT_MIN_IPS_TO_EXCLUDE,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_shared_pairs: int = DEFAULT_MIN_SHARED_PAIRS,
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
    args = parser.parse_args()

    clusters = run(
        mode=args.mode,
        snapshot_path=args.snapshot,
        min_ips_to_exclude=args.min_ips_to_exclude,
        min_similarity=args.min_similarity,
        min_shared_pairs=args.min_shared_pairs,
    )
    print(f"Wrote {len(clusters)} cluster-assignment rows and charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
