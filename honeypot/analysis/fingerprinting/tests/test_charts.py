# honeypot/analysis/fingerprinting/tests/test_charts.py
from pathlib import Path

import polars as pl
from charts import (
    _NO_BANNER_LABEL,
    _NON_SSH_LABEL,
    _bucket_client_versions,
    _cluster_command_sequences,
    plot_client_fingerprint_breakdown,
    plot_command_sequence_clusters,
    plot_inter_command_timing,
)


def _features_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "client_version": ["SSH-2.0-Go", "SSH-2.0-OpenSSH_8.9", None],
            "is_ssh_client": [True, True, False],
            "inter_command_mean_s": [1.5, None, 12.0],
        }
    )


class TestPlotClientFingerprintBreakdown:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        path = plot_client_fingerprint_breakdown(_features_df(), output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_buckets_non_ssh_and_missing_banners_separately(self) -> None:
        df = pl.DataFrame(
            {
                "client_version": [
                    "SSH-2.0-OpenSSH_8.9",
                    "SSH-2.0-OpenSSH_8.9",
                    "GET / HTTP/1.1",
                    "\x16\x03\x01\x00{\x01\x00\x00w\x03\x03",
                    None,
                ],
                "is_ssh_client": [True, True, False, False, False],
            }
        )

        counts = _bucket_client_versions(df)
        result = dict(
            zip(counts["client_version"].to_list(), counts["count"].to_list())
        )

        # Two distinct non-SSH values collapse into one bucket of 2, not
        # two separate one-off bars.
        assert result == {
            "SSH-2.0-OpenSSH_8.9": 2,
            _NON_SSH_LABEL: 2,
            _NO_BANNER_LABEL: 1,
        }

    def test_handles_adversarial_client_version_values(self, tmp_path: Path) -> None:
        # Real honeypot traffic includes non-SSH protocol noise (TLS
        # ClientHellos, RDP scan cookies, ...) landing in this field
        # verbatim as raw bytes -- this exact shape (control bytes plus a
        # literal '$') previously crashed matplotlib's mathtext parser.
        adversarial_df = pl.DataFrame(
            {
                "client_version": [
                    "\x16\x03\x01\x00{\x01\x00\x00w\x03\x03$+d\xa9\x00OA",
                    "x" * 200,
                ],
                "is_ssh_client": [False, False],
                "inter_command_mean_s": [1.0, 2.0],
            }
        )

        path = plot_client_fingerprint_breakdown(adversarial_df, output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestClusterCommandSequences:
    def test_groups_identical_sequences_and_sorts_by_size(self) -> None:
        df = pl.DataFrame(
            {
                "command_sequence_hash": ["a", "a", "a", "b", "b", None],
                "commands": [
                    ["wget x", "chmod +x x"],
                    ["wget x", "chmod +x x"],
                    ["wget x", "chmod +x x"],
                    ["ls"],
                    ["ls"],
                    [],
                ],
            }
        )

        clusters = _cluster_command_sequences(df)

        assert clusters.height == 2  # sessions with no commands excluded
        top = clusters.row(0, named=True)
        assert top["command_sequence_hash"] == "a"
        assert top["count"] == 3
        assert top["commands"] == ["wget x", "chmod +x x"]

    def test_excludes_sessions_with_no_commands(self) -> None:
        df = pl.DataFrame(
            {"command_sequence_hash": [None, None], "commands": [[], []]},
            schema={"command_sequence_hash": pl.Utf8, "commands": pl.List(pl.Utf8)},
        )

        clusters = _cluster_command_sequences(df)

        assert clusters.height == 0


class TestPlotCommandSequenceClusters:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "command_sequence_hash": ["a", "a", "b", None],
                "commands": [
                    ["wget x", "chmod +x x", "./x"],
                    ["wget x", "chmod +x x", "./x"],
                    ["uname -a"],
                    [],
                ],
            }
        )

        path = plot_command_sequence_clusters(df, output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_no_command_sessions_without_raising(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {"command_sequence_hash": [None, None], "commands": [[], []]},
            schema={"command_sequence_hash": pl.Utf8, "commands": pl.List(pl.Utf8)},
        )

        path = plot_command_sequence_clusters(df, output_dir=tmp_path)

        assert path.exists()

    def test_handles_adversarial_command_content(self, tmp_path: Path) -> None:
        # Commands are attacker-typed input -- same crash risk as
        # client_version if not sanitized before hitting matplotlib.
        df = pl.DataFrame(
            {
                "command_sequence_hash": ["a"],
                "commands": [
                    ["\x16\x03\x01\x00{\x01\x00\x00w\x03\x03$+d\xa9\x00OA", "x" * 200]
                ],
            }
        )

        path = plot_command_sequence_clusters(df, output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotInterCommandTiming:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        path = plot_inter_command_timing(_features_df(), output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_no_data_without_raising(self, tmp_path: Path) -> None:
        empty_df = pl.DataFrame(
            {"client_version": [], "inter_command_mean_s": []},
            schema={"client_version": pl.Utf8, "inter_command_mean_s": pl.Float64},
        )

        path = plot_inter_command_timing(empty_df, output_dir=tmp_path)

        assert path.exists()
