# honeypot/analysis/fingerprinting/pipeline.py
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from charts import (
    plot_client_fingerprint_breakdown,
    plot_command_sequence_clusters,
    plot_inter_command_timing,
)
from extract import get_sessions
from features import extract_session_features

OUTPUT_DIR = Path(__file__).parent / "outputs"


def run(
    mode: str,
    snapshot_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> pl.DataFrame:
    """Regenerate the session-features dataset and chart set from scratch."""
    events = get_sessions(mode=mode, snapshot_path=snapshot_path)
    features = extract_session_features(events)

    features_dir = output_dir / "features"
    charts_dir = output_dir / "charts"
    features_dir.mkdir(parents=True, exist_ok=True)
    features.write_parquet(features_dir / "session_features.parquet")

    plot_client_fingerprint_breakdown(features, output_dir=charts_dir)
    plot_inter_command_timing(features, output_dir=charts_dir)
    plot_command_sequence_clusters(features, output_dir=charts_dir)

    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the fingerprinting dataset and charts."
    )
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Explicit snapshot parquet to use (offline mode).",
    )
    args = parser.parse_args()

    features = run(mode=args.mode, snapshot_path=args.snapshot)
    print(f"Wrote {len(features)} session feature rows and charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
