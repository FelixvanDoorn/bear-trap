# honeypot/analysis/common/tests/test_extract.py
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal
from pytest_mock import MockerFixture

from honeypot.analysis.common.extract import get_sessions


class TestOfflineMode:
    def test_reads_given_snapshot_path(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "logs_20260101T000000Z.parquet"
        pl.DataFrame({"session": ["s1"]}).write_parquet(snapshot)

        result = get_sessions("offline", snapshot_path=snapshot)

        assert result["session"].to_list() == ["s1"]

    def test_picks_most_recent_snapshot_by_default(self, tmp_path: Path) -> None:
        pl.DataFrame({"session": ["old"]}).write_parquet(
            tmp_path / "logs_20260101T000000Z.parquet"
        )
        pl.DataFrame({"session": ["new"]}).write_parquet(
            tmp_path / "logs_20260102T000000Z.parquet"
        )

        result = get_sessions("offline", snapshot_dir=tmp_path)

        assert result["session"].to_list() == ["new"]

    def test_raises_when_no_snapshots_exist(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_sessions("offline", snapshot_dir=tmp_path)


class TestLiveMode:
    def test_queries_bigquery_and_writes_snapshot(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        fake_df = pl.DataFrame({"session": ["s1"]})
        mock_client = mocker.MagicMock()
        mock_client.query.return_value.to_arrow.return_value = pa.table(
            fake_df.to_dict(as_series=False)
        )
        mocker.patch("google.cloud.bigquery.Client", return_value=mock_client)

        result = get_sessions("live", snapshot_dir=tmp_path)

        assert_frame_equal(result, fake_df)
        mock_client.query.assert_called_once()
        assert len(list(tmp_path.glob("logs_*.parquet"))) == 1


class TestInvalidMode:
    def test_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            get_sessions("bogus")  # type: ignore[arg-type]
