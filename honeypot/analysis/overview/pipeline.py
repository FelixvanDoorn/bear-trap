# honeypot/analysis/overview/pipeline.py
import argparse
from pathlib import Path

import polars as pl

from honeypot.analysis.common.extract import get_sessions
from honeypot.analysis.overview.charts import (
    plot_daily_activity,
    plot_session_funnel,
    plot_tunnel_targets,
)
from honeypot.analysis.overview.summary import (
    coverage,
    daily_activity,
    session_funnel,
    tunnel_targets,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def run(
    mode: str,
    snapshot_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pl.DataFrame]:
    """Regenerate the overview tables and chart set from scratch. Returns
    the tables by name, in the order they're meant to be read."""
    events = get_sessions(mode=mode, snapshot_path=snapshot_path)
    tables = {
        "coverage": coverage(events),
        "session_funnel": session_funnel(events),
        "daily_activity": daily_activity(events),
        "tunnel_targets": tunnel_targets(events),
    }

    tables_dir = output_dir / "tables"
    charts_dir = output_dir / "charts"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(tables_dir / f"{name}.parquet")

    plot_session_funnel(tables["session_funnel"], output_dir=charts_dir)
    plot_daily_activity(tables["daily_activity"], output_dir=charts_dir)
    plot_tunnel_targets(tables["tunnel_targets"], output_dir=charts_dir)

    return tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the honeypot overview tables and charts."
    )
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Explicit snapshot parquet to use (offline mode).",
    )
    args = parser.parse_args()

    tables = run(mode=args.mode, snapshot_path=args.snapshot)
    print(f"Wrote {len(tables)} tables and charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
