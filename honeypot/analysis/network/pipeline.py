# honeypot/analysis/network/pipeline.py
import argparse
from pathlib import Path

import polars as pl

from honeypot.analysis.common.extract import get_sessions
from honeypot.analysis.network.charts import (
    plot_cluster_size_distribution,
    plot_generic_credential_breakdown,
    plot_top_clusters_subgraph,
)
from honeypot.analysis.network.graph import (
    assign_clusters,
    build_credential_graph,
    generic_credential_attempts,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def run(
    mode: str,
    snapshot_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> pl.DataFrame:
    """Regenerate the credential-sharing cluster dataset and chart set from
    scratch."""
    events = get_sessions(mode=mode, snapshot_path=snapshot_path)
    graph = build_credential_graph(events)
    clusters = assign_clusters(graph)
    generic_attempts = generic_credential_attempts(events)

    clusters_dir = output_dir / "clusters"
    charts_dir = output_dir / "charts"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    clusters.write_parquet(clusters_dir / "credential_clusters.parquet")
    generic_attempts.write_parquet(clusters_dir / "generic_credential_attempts.parquet")

    plot_cluster_size_distribution(clusters, output_dir=charts_dir)
    plot_top_clusters_subgraph(graph, output_dir=charts_dir)
    plot_generic_credential_breakdown(generic_attempts, output_dir=charts_dir)

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
    args = parser.parse_args()

    clusters = run(mode=args.mode, snapshot_path=args.snapshot)
    print(f"Wrote {len(clusters)} cluster-assignment rows and charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
