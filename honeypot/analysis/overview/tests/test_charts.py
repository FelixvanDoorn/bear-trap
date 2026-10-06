# honeypot/analysis/overview/tests/test_charts.py
import datetime as dt
from pathlib import Path

import polars as pl

from honeypot.analysis.overview.charts import (
    plot_daily_activity,
    plot_session_funnel,
    plot_tunnel_targets,
)


def _funnel_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "node_role": ["agent_detector", "agent_detector", "control_proxy"],
            "protocol": ["ssh", "telnet", "ssh"],
            "sessions": [100, 50, 10],
            "tried_login": [90, 25, 7],
            "authenticated": [30, 20, 7],
            "ran_commands": [28, 19, 7],
            "file_transfer": [1, 2, 0],
            # telnet never tunnels -- zero share must not break the log axis.
            "tunnel_request": [3, 0, 0],
            "median_duration_s": [1.8, 37.0, 1.5],
        }
    )


class TestPlotSessionFunnel:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        path = plot_session_funnel(_funnel_df(), output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_unknown_role_with_fallback_color(self, tmp_path: Path) -> None:
        funnel = _funnel_df().with_columns(pl.lit("new_role").alias("node_role"))

        assert plot_session_funnel(funnel, output_dir=tmp_path).exists()

    def test_handles_empty_input_without_raising(self, tmp_path: Path) -> None:
        path = plot_session_funnel(_funnel_df().clear(), output_dir=tmp_path)

        assert path.exists()


class TestPlotDailyActivity:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        daily = pl.DataFrame(
            {
                "date": [dt.date(2026, 8, 2), dt.date(2026, 8, 2), dt.date(2026, 8, 3)],
                "node_role": ["agent_detector", "control_proxy", "agent_detector"],
                "sessions": [10, 3, 12],
                "distinct_ips": [5, 2, 6],
                "new_ips": [4, 2, 1],
                "returning_ips": [1, 0, 5],
            }
        )

        path = plot_daily_activity(daily, output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_empty_input_without_raising(self, tmp_path: Path) -> None:
        daily = pl.DataFrame(
            schema={
                "date": pl.Date,
                "node_role": pl.Utf8,
                "sessions": pl.UInt32,
                "distinct_ips": pl.UInt32,
                "new_ips": pl.UInt32,
                "returning_ips": pl.UInt32,
            }
        )

        assert plot_daily_activity(daily, output_dir=tmp_path).exists()


class TestPlotTunnelTargets:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        targets = pl.DataFrame(
            {
                "dst_ip": ["77.88.21.158", "google.com"],
                "dst_port": [25, 443],
                "requests": [1203, 29],
                "sessions": [1203, 29],
                "distinct_ips": [732, 22],
            }
        )

        path = plot_tunnel_targets(targets, output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_empty_input_without_raising(self, tmp_path: Path) -> None:
        targets = pl.DataFrame(
            schema={
                "dst_ip": pl.Utf8,
                "dst_port": pl.Int64,
                "requests": pl.UInt32,
                "sessions": pl.UInt32,
                "distinct_ips": pl.UInt32,
            }
        )

        assert plot_tunnel_targets(targets, output_dir=tmp_path).exists()
