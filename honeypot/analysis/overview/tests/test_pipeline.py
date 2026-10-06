# honeypot/analysis/overview/tests/test_pipeline.py
from pathlib import Path

from honeypot.analysis.overview.pipeline import run
from honeypot.analysis.overview.tests.fixtures import make_event, make_events


def _fixture_snapshot(path: Path) -> None:
    make_events(
        [
            # Three days so daily_activity's partial-day trim keeps day 2.
            *[
                make_event(f"scan{day}", "cowrie.session.connect", day=day)
                for day in (1, 2, 3)
            ],
            make_event("s1", "cowrie.login.success", day=2, src_ip="2.2.2.2"),
            make_event("s1", "cowrie.command.input", day=2, src_ip="2.2.2.2"),
            make_event(
                "s1", "cowrie.session.closed", day=2, src_ip="2.2.2.2", duration_ms=900
            ),
            make_event(
                "s2",
                "cowrie.direct-tcpip.request",
                day=2,
                node_role="control_proxy",
                dst_ip="mx.example",
                dst_port=25,
            ),
        ]
    ).write_parquet(path)


class TestEndToEndOfflinePipeline:
    def test_produces_tables_and_chart_files(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "logs_20260801T000000Z.parquet"
        _fixture_snapshot(snapshot)
        output_dir = tmp_path / "outputs"

        tables = run(mode="offline", snapshot_path=snapshot, output_dir=output_dir)

        assert list(tables) == [
            "coverage",
            "session_funnel",
            "daily_activity",
            "tunnel_targets",
        ]
        assert tables["tunnel_targets"].height == 1
        for name in tables:
            assert (output_dir / "tables" / f"{name}.parquet").exists()
        for chart in ("session_funnel", "daily_activity", "tunnel_targets"):
            assert (output_dir / "charts" / f"{chart}.png").exists()
