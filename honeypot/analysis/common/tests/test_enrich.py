# honeypot/analysis/common/tests/test_enrich.py
import datetime as dt
import gzip
import urllib.error
from pathlib import Path

import polars as pl
import pytest
from pytest_mock import MockerFixture

from honeypot.analysis.common.enrich import (
    enrich_ips,
    fetch_dbip,
    ipv4_to_int,
    load_enrichment,
    read_ranges,
)

COUNTRY_CSV = """0.0.0.0,0.255.255.255,ZZ
1.0.0.0,1.0.0.255,AU
1.0.4.0,1.0.7.255,CN
2001:200::,2001:200:ffff:ffff:ffff:ffff:ffff:ffff,JP
"""

ASN_CSV = """1.0.0.0,1.0.0.255,13335,"Cloudflare, Inc."
1.0.4.0,1.0.7.255,38803,"Gtelecom Pty Ltd"
2001:200::,2001:200:ffff:ffff:ffff:ffff:ffff:ffff,2500,"WIDE Project"
"""


def write_dbip_fixtures(enrichment_dir: Path, month: str = "2026-01") -> None:
    """Gzipped DB-IP Lite CSVs in the published filename format, so
    fetch_dbip finds them cached and never downloads. Shared with the
    network pipeline test."""
    enrichment_dir.mkdir(parents=True, exist_ok=True)
    for kind, body in (("country", COUNTRY_CSV), ("asn", ASN_CSV)):
        path = enrichment_dir / f"dbip-{kind}-lite-{month}.csv.gz"
        path.write_bytes(gzip.compress(body.encode()))


class TestIpv4ToInt:
    @pytest.mark.parametrize(
        ("ip", "expected"),
        [
            ("0.0.0.0", 0),
            ("1.0.0.1", 16777217),
            ("255.255.255.255", 2**32 - 1),
            ("2001:db8::1", None),
            ("256.0.0.1", None),
            ("1.2.3", None),
            ("not-an-ip", None),
            (None, None),
        ],
    )
    def test_converts_dotted_quads_only(self, ip: str | None, expected: int) -> None:
        result = pl.DataFrame({"ip": [ip]}, schema={"ip": pl.Utf8}).select(
            ipv4_to_int(pl.col("ip"))
        )
        assert result.item() == expected


class TestReadRanges:
    def test_country_drops_ipv6_and_nulls_unknown(self, tmp_path: Path) -> None:
        write_dbip_fixtures(tmp_path)

        ranges = read_ranges(tmp_path / "dbip-country-lite-2026-01.csv.gz", "country")

        assert ranges["country"].to_list() == [None, "AU", "CN"]
        assert ranges["start"].to_list() == sorted(ranges["start"].to_list())

    def test_asn_keeps_quoted_commas_in_org(self, tmp_path: Path) -> None:
        write_dbip_fixtures(tmp_path)

        ranges = read_ranges(tmp_path / "dbip-asn-lite-2026-01.csv.gz", "asn")

        assert ranges["asn"].to_list() == [13335, 38803]
        assert ranges["as_org"].to_list()[0] == "Cloudflare, Inc."


class TestEnrichIps:
    def test_looks_up_inside_ranges_and_nulls_gaps(self, tmp_path: Path) -> None:
        write_dbip_fixtures(tmp_path)
        country = read_ranges(tmp_path / "dbip-country-lite-2026-01.csv.gz", "country")
        asn = read_ranges(tmp_path / "dbip-asn-lite-2026-01.csv.gz", "asn")
        ips = pl.DataFrame(
            {
                "src_ip": [
                    "1.0.0.7",
                    "1.0.0.7",  # duplicate -> one output row
                    "1.0.5.1",
                    "1.0.2.1",  # gap between ranges
                    "2001:200::1",  # IPv6 -> not looked up
                ]
            }
        )

        result = enrich_ips(ips, country, asn).sort("src_ip")

        assert result.height == 4
        rows = {row["src_ip"]: row for row in result.to_dicts()}
        assert rows["1.0.0.7"]["country"] == "AU"
        assert rows["1.0.0.7"]["as_label"] == "AS13335 Cloudflare, Inc."
        assert rows["1.0.5.1"]["asn"] == 38803
        assert rows["1.0.2.1"]["country"] is None
        assert rows["1.0.2.1"]["as_label"] is None
        assert rows["2001:200::1"]["asn"] is None

    def test_load_enrichment_uses_cached_files(self, tmp_path: Path) -> None:
        write_dbip_fixtures(tmp_path)

        result = load_enrichment(pl.DataFrame({"src_ip": ["1.0.4.4"]}), tmp_path)

        assert result.row(0, named=True)["country"] == "CN"


class TestFetchDbip:
    def test_returns_newest_cached_without_downloading(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        write_dbip_fixtures(tmp_path, month="2026-01")
        write_dbip_fixtures(tmp_path, month="2026-02")
        download = mocker.patch("urllib.request.urlretrieve")

        path = fetch_dbip("asn", tmp_path)

        assert path.name == "dbip-asn-lite-2026-02.csv.gz"
        download.assert_not_called()

    def test_falls_back_to_previous_month_on_404(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        not_found = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        download = mocker.patch(
            "urllib.request.urlretrieve", side_effect=[not_found, None]
        )

        path = fetch_dbip("country", tmp_path, today=dt.date(2026, 1, 2))

        assert path.name == "dbip-country-lite-2025-12.csv.gz"
        assert download.call_args_list[0].args[0].endswith("lite-2026-01.csv.gz")
        assert download.call_args_list[1].args[0].endswith("lite-2025-12.csv.gz")
