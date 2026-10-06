# honeypot/analysis/overview/tests/fixtures.py
#
# Synthetic Cowrie event builder shared by this module's tests (summary,
# pipeline and notebook execution), so every fixture snapshot carries the
# same column set summary.py reads from a real BigQuery snapshot.
import datetime as dt

import polars as pl

EVENT_SCHEMA = {
    "event_timestamp": pl.Datetime("us", "UTC"),
    "eventid": pl.Utf8,
    "node_role": pl.Utf8,
    "protocol": pl.Utf8,
    "session": pl.Utf8,
    "sensor": pl.Utf8,
    "src_ip": pl.Utf8,
    "dst_ip": pl.Utf8,
    "dst_port": pl.Int64,
    "duration_ms": pl.Int64,
}


def make_event(
    session: str,
    eventid: str,
    *,
    day: int = 1,
    hour: int = 12,
    src_ip: str = "1.1.1.1",
    node_role: str = "agent_detector",
    protocol: str = "ssh",
    sensor: str = "sensor-a",
    dst_ip: str = "10.0.0.1",
    dst_port: int = 2222,
    duration_ms: int | None = None,
) -> dict:
    return {
        "event_timestamp": dt.datetime(2026, 8, day, hour, tzinfo=dt.UTC),
        "eventid": eventid,
        "node_role": node_role,
        "protocol": protocol,
        "session": session,
        "sensor": sensor,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "duration_ms": duration_ms,
    }


def make_events(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=EVENT_SCHEMA)
