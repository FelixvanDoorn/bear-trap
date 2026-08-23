# honeypot/analysis/fingerprinting/extract.py
#
# live mode always writes a snapshot as a side effect, so every live pull
# doubles as a fresh offline snapshot for later, credential-free reruns.
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Literal

import polars as pl

PROJECT_ID = "mineral-droplet-160709"
DATASET = "bear_trap_logs"
VIEW = "logs"

# Overridable via env var (set before this module is imported) so the
# notebook -- which calls get_sessions() with no explicit snapshot_dir --
# can be pointed at a synthetic fixture for hermetic testing, without
# touching the real local snapshots.
SNAPSHOT_DIR = Path(
    os.environ.get("BEARTRAP_SNAPSHOT_DIR")
    or Path(__file__).parent / "outputs" / "snapshots"
)

QUERY = f"""
SELECT
  publish_time,
  event_timestamp,
  eventid,
  cloud_provider,
  node_role,
  protocol,
  session,
  sensor,
  src_ip,
  src_port,
  dst_ip,
  dst_port,
  command_input,
  username,
  password,
  duration_ms,
  agent_type,
  agent_framework,
  agent_primary_objective,
  agent_operator_identity,
  raw
FROM `{PROJECT_ID}.{DATASET}.{VIEW}`
"""


def _query_bigquery() -> pl.DataFrame:
    # Imported lazily so offline-only usage never needs google-cloud-bigquery
    # importable.
    from google.cloud import bigquery

    client = bigquery.Client(project=PROJECT_ID)
    arrow_table = client.query(QUERY).to_arrow()
    return pl.from_arrow(arrow_table)


def _write_snapshot(df: pl.DataFrame, snapshot_dir: Path) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = snapshot_dir / f"logs_{timestamp}.parquet"
    df.write_parquet(path)
    return path


def _latest_snapshot(snapshot_dir: Path) -> Path:
    # Filenames are lexically sortable since the timestamp format is
    # zero-padded and fixed-width.
    snapshots = sorted(snapshot_dir.glob("logs_*.parquet"))
    if not snapshots:
        raise FileNotFoundError(
            f"No snapshots found in {snapshot_dir}. Run with mode='live' "
            "first, or pass an explicit snapshot_path."
        )
    return snapshots[-1]


def get_sessions(
    mode: Literal["live", "offline"],
    snapshot_path: Path | None = None,
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> pl.DataFrame:
    """Return the flat event-level DataFrame backing session feature extraction."""
    if mode == "live":
        df = _query_bigquery()
        _write_snapshot(df, snapshot_dir)
        return df

    if mode == "offline":
        path = snapshot_path or _latest_snapshot(snapshot_dir)
        return pl.read_parquet(path)

    raise ValueError(f"Unknown mode: {mode!r}")
