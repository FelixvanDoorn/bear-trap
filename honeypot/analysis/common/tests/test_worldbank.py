# honeypot/analysis/common/tests/test_worldbank.py
import datetime as dt
import json
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from honeypot.analysis.common.worldbank import (
    fetch_indicator,
    load_indicators,
    read_indicator_json,
)


def _payload(rows: list[dict], pages: int = 1) -> str:
    return json.dumps([{"page": 1, "pages": pages}, rows])


def _row(country: str, value: float | None, date: str = "2025") -> dict:
    return {"country": {"id": country, "value": country}, "date": date, "value": value}


class TestReadIndicatorJson:
    def test_rows_with_values(self) -> None:
        result = read_indicator_json(_payload([_row("AR", 1.5), _row("XK", None)]))

        assert result.rows() == [("AR", 2025, 1.5)]

    @pytest.mark.parametrize(
        "body",
        [
            json.dumps([{"message": [{"id": "120", "value": "Invalid value"}]}]),
            json.dumps([{"page": 1, "pages": 1}, None]),
            _payload([_row("AR", 1.0)], pages=2),
        ],
    )
    def test_rejects_error_and_partial_responses(self, body: str) -> None:
        with pytest.raises(ValueError):
            read_indicator_json(body)


class TestFetchIndicator:
    def test_uses_newest_cached_file(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        download = mocker.patch("urllib.request.urlopen")
        (tmp_path / "SP.POP.TOTL-2026-01-01.json").write_text("")
        (tmp_path / "SP.POP.TOTL-2026-02-01.json").write_text("")

        assert fetch_indicator("SP.POP.TOTL", tmp_path).name == (
            "SP.POP.TOTL-2026-02-01.json"
        )
        download.assert_not_called()

    def test_downloads_and_caches_a_valid_response(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        download = mocker.patch("urllib.request.urlopen")
        body = _payload([_row("AR", 1.0)]).encode()
        download.return_value.__enter__.return_value.read.return_value = body

        path = fetch_indicator("SP.POP.TOTL", tmp_path, today=dt.date(2026, 10, 31))

        assert path.name == "SP.POP.TOTL-2026-10-31.json"
        assert path.read_bytes() == body
        assert "mrnev=1" in download.call_args.args[0].full_url

    def test_error_response_is_not_cached(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        download = mocker.patch("urllib.request.urlopen")
        download.return_value.__enter__.return_value.read.return_value = (
            b'[{"message": []}]'
        )

        with pytest.raises(ValueError):
            fetch_indicator("SP.POP.TOTL", tmp_path)

        assert list(tmp_path.glob("*")) == []


class TestLoadIndicators:
    def test_joins_indicators_derives_users_and_drops_stale(
        self, tmp_path: Path
    ) -> None:
        files = {
            "NY.GDP.PCAP.CD": [_row("AR", 15000.0), _row("JP", 36000.0)],
            "IT.NET.USER.ZS": [_row("AR", 90.0), _row("JP", 80.0, date="1990")],
            "SP.POP.TOTL": [_row("AR", 1e7), _row("JP", 1e8)],
        }
        for code, rows in files.items():
            (tmp_path / f"{code}-2026-01-01.json").write_text(_payload(rows))

        result = load_indicators(tmp_path)

        ar, jp = result.rows(named=True)
        assert ar["internet_users"] == pytest.approx(9e6)
        assert ar["internet_share_year"] == 2025
        # 1990 is before DEFAULT_MIN_YEAR: treated as missing.
        assert jp["internet_share"] is None
        assert jp["internet_users"] is None
        assert jp["gdp_per_capita"] == 36000.0
