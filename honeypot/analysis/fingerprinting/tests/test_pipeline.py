# honeypot/analysis/fingerprinting/tests/test_pipeline.py
import json
from datetime import datetime
from pathlib import Path

import polars as pl
from pipeline import run


def _fixture_snapshot(path: Path) -> None:
    pl.DataFrame(
        [
            {
                "session": "s1",
                "eventid": "cowrie.client.version",
                "event_timestamp": datetime.fromisoformat("2026-08-19T10:00:00"),
                "sensor": "sensor-01",
                "src_ip": "1.2.3.4",
                "cloud_provider": "aws",
                "command_input": None,
                "duration_ms": None,
                "username": None,
                "password": None,
                "agent_type": None,
                "raw": json.dumps({"version": "SSH-2.0-Go"}),
            },
            {
                "session": "s1",
                "eventid": "cowrie.command.input",
                "event_timestamp": datetime.fromisoformat("2026-08-19T10:00:05"),
                "sensor": "sensor-01",
                "src_ip": "1.2.3.4",
                "cloud_provider": "aws",
                "command_input": "whoami",
                "duration_ms": None,
                "username": None,
                "password": None,
                "agent_type": None,
                "raw": None,
            },
        ]
    ).write_parquet(path)


class TestEndToEndOfflinePipeline:
    def test_produces_features_and_chart_files(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "logs_20260819T100000Z.parquet"
        _fixture_snapshot(snapshot)
        output_dir = tmp_path / "outputs"

        features = run(mode="offline", snapshot_path=snapshot, output_dir=output_dir)

        assert features.height == 1
        assert features.row(0, named=True)["client_version"] == "SSH-2.0-Go"
        assert (output_dir / "features" / "session_features.parquet").exists()
        assert (output_dir / "charts" / "client_fingerprint_breakdown.png").exists()
        assert (output_dir / "charts" / "inter_command_timing.png").exists()
        assert (output_dir / "charts" / "command_sequence_clusters.png").exists()
