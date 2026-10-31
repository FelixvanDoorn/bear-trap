# honeypot/analysis/common/tests/test_threatintel.py
import datetime as dt
import urllib.error
from pathlib import Path

import polars as pl
import pytest
from pytest_mock import MockerFixture

from honeypot.analysis.common.threatintel import (
    LISTED_COLUMN,
    fetch_blocklist,
    listing_coverage,
    load_listings,
    match_blocklists,
    read_blocklist,
)

BLOCKLIST_DE_SSH = """1.0.0.1
1.0.4.9
2001:db8::1
"""

FIREHOL_LEVEL1 = """#
# firehol_level1
#
0.0.0.0/8
1.0.4.0/22
"""


def write_blocklist_fixtures(threatintel_dir: Path, date: str = "2026-01-01") -> None:
    """Blocklist files in the cached filename format, so fetch_blocklist
    finds them and never downloads. Shared with the network pipeline and
    notebook tests."""
    threatintel_dir.mkdir(parents=True, exist_ok=True)
    for name, body in (
        ("blocklist_de_ssh", BLOCKLIST_DE_SSH),
        ("firehol_level1", FIREHOL_LEVEL1),
    ):
        (threatintel_dir / f"{name}-{date}.txt").write_text(body)


def _ranges(path: Path, body: str) -> pl.DataFrame:
    path.write_text(body)
    return read_blocklist(path)


class TestReadBlocklist:
    def test_parses_ips_and_cidrs_skipping_comments_and_ipv6(
        self, tmp_path: Path
    ) -> None:
        ranges = _ranges(
            tmp_path / "list.txt",
            "# header\n\n10.0.0.0/24\n1.2.3.4  # inline comment\n2001:db8::/32\nbad\n",
        )

        assert ranges.rows() == [
            (16909060, 16909060),  # 1.2.3.4
            (167772160, 167772415),  # 10.0.0.0 - 10.0.0.255
        ]

    def test_merges_overlapping_and_adjacent_ranges(self, tmp_path: Path) -> None:
        ranges = _ranges(
            tmp_path / "list.txt",
            # /24 containing a single listed IP, then the next /24.
            "10.0.0.0/24\n10.0.0.7\n10.0.1.0/24\n",
        )

        assert ranges.rows() == [(167772160, 167772671)]

    def test_empty_list(self, tmp_path: Path) -> None:
        assert _ranges(tmp_path / "list.txt", "# nothing\n").height == 0


class TestMatchBlocklists:
    def test_flags_per_list_and_any(self, tmp_path: Path) -> None:
        ips = pl.DataFrame(
            {"src_ip": ["1.0.0.1", "1.0.4.9", "1.0.5.1", "9.9.9.9", "2001:db8::1"]}
        )
        blocklists = {
            "single": _ranges(tmp_path / "a.txt", "1.0.0.1\n1.0.4.9\n2001:db8::1\n"),
            "networks": _ranges(tmp_path / "b.txt", "1.0.4.0/22\n"),
        }

        result = match_blocklists(ips, blocklists)

        assert result.columns == ["src_ip", "on_single", "on_networks", LISTED_COLUMN]
        assert {row[0]: row[1:] for row in result.rows()} == {
            "1.0.0.1": (True, False, True),
            "1.0.4.9": (True, True, True),
            "1.0.5.1": (False, True, True),
            "9.9.9.9": (False, False, False),
            # IPv6 is never matched, even when the list contains it.
            "2001:db8::1": (False, False, False),
        }


class TestFetchBlocklist:
    def test_uses_newest_cached_file_without_downloading(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        download = mocker.patch("urllib.request.urlopen")
        (tmp_path / "firehol_level1-2026-01-01.txt").write_text("")
        (tmp_path / "firehol_level1-2026-02-01.txt").write_text("")

        path = fetch_blocklist("firehol_level1", tmp_path)

        assert path.name == "firehol_level1-2026-02-01.txt"
        download.assert_not_called()

    def test_downloads_with_todays_date_when_nothing_cached(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        download = mocker.patch("urllib.request.urlopen")
        download.return_value.__enter__.return_value.read.return_value = b"1.2.3.4\n"

        path = fetch_blocklist(
            "blocklist_de_ssh", tmp_path / "ti", today=dt.date(2026, 10, 27)
        )

        assert path == tmp_path / "ti" / "blocklist_de_ssh-2026-10-27.txt"
        assert path.read_text() == "1.2.3.4\n"
        request = download.call_args.args[0]
        assert request.full_url == "https://lists.blocklist.de/lists/ssh.txt"
        assert request.get_header("User-agent").startswith("bear-trap-analysis")

    def test_failed_download_leaves_nothing_cached(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("u", 403, "Forbidden", {}, None),
        )

        with pytest.raises(urllib.error.HTTPError):
            fetch_blocklist("blocklist_de_ssh", tmp_path)

        assert list(tmp_path.glob("*")) == []


class TestLoadListings:
    def test_against_fixture_files(self, tmp_path: Path) -> None:
        write_blocklist_fixtures(tmp_path)
        ips = pl.DataFrame({"src_ip": ["1.0.0.1", "1.0.5.1", "8.8.8.8"]})

        result = load_listings(ips, tmp_path)

        assert dict(result.select("src_ip", LISTED_COLUMN).rows()) == {
            "1.0.0.1": True,
            "1.0.5.1": True,
            "8.8.8.8": False,
        }


class TestListingCoverage:
    def test_shares_per_level(self) -> None:
        listed = pl.DataFrame(
            {
                "src_ip": ["a", "b", "c", "d"],
                "tier": [1, 1, 2, 2],
                "on_x": [True, False, False, False],
                "on_y": [True, True, False, False],
                LISTED_COLUMN: [True, True, False, False],
            }
        )

        result = listing_coverage(listed, "tier")

        assert result.columns == [
            "tier",
            "ips",
            "share_on_x",
            "share_on_y",
            f"share_{LISTED_COLUMN}",
        ]
        assert result.rows() == [(1, 2, 0.5, 1.0, 1.0), (2, 2, 0.0, 0.0, 0.0)]
