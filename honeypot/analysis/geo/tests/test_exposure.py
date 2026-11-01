# honeypot/analysis/geo/tests/test_exposure.py
import datetime as dt

import polars as pl
import pytest

from honeypot.analysis.geo.exposure import (
    RESPONSES,
    address_space,
    attacker_ips,
    country_counts,
    country_networks,
)
from honeypot.analysis.overview.tests.fixtures import make_event, make_events


def _ranges() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start": [0, 256, 1024, 4096],
            "end": [255, 511, 2047, 4351],
            "country": [None, "AA", "BB", "AA"],
        }
    )


class TestAddressSpace:
    def test_sums_ranges_per_country_and_drops_unknown(self) -> None:
        assert address_space(_ranges()).rows() == [("AA", 512), ("BB", 1024)]


class TestAttackerIps:
    def test_one_row_per_placed_ip(self) -> None:
        events = make_events(
            [
                make_event("s1", "cowrie.session.connect", src_ip="1.1.1.1", day=2),
                make_event(
                    "s2",
                    "cowrie.session.connect",
                    src_ip="1.1.1.1",
                    day=1,
                    node_role="control_proxy",
                    protocol="telnet",
                ),
                make_event("s3", "cowrie.session.connect", src_ip="1.1.1.200"),
                # Not placed by DB-IP -> dropped.
                make_event("s4", "cowrie.session.connect", src_ip="9.9.9.9"),
            ]
        )
        enrichment = pl.DataFrame(
            {
                "src_ip": ["1.1.1.1", "1.1.1.200", "9.9.9.9"],
                "country": ["AA", "AA", None],
                "asn": [1, 2, None],
                "as_label": ["AS1 One", "AS2 Two", None],
            }
        )

        result = attacker_ips(events, enrichment)

        assert result["src_ip"].to_list() == ["1.1.1.1", "1.1.1.200"]
        first = result.row(0, named=True)
        assert first["protocols"] == "ssh+telnet"
        assert first["node_roles"] == "agent_detector+control_proxy"
        assert first["first_seen"] == dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC)
        # Same /24 for both IPs.
        assert result["slash24"].n_unique() == 1


class TestCountryCounts:
    def test_every_country_with_zeros_and_all_responses(self) -> None:
        attackers = pl.DataFrame(
            {
                "src_ip": ["1.0.0.1", "1.0.0.2", "1.0.1.1"],
                "country": ["AA", "AA", "AA"],
                "asn": [1, 1, None],
                "slash24": [10, 10, 11],
            }
        )

        result = country_counts(attackers, address_space(_ranges()))

        assert result.columns == ["country", "addresses", *RESPONSES]
        assert result.rows() == [("AA", 512, 3, 2, 1), ("BB", 1024, 0, 0, 0)]


class TestCountryNetworks:
    def test_concentration_per_country(self) -> None:
        attackers = pl.DataFrame(
            {
                "src_ip": ["a", "b", "c", "d", "e"],
                "country": ["AA", "AA", "AA", "AA", "BB"],
                "asn": [1, 1, 1, 2, 3],
                "as_label": ["AS1 Host", "AS1 Host", "AS1 Host", "AS2 Isp", "AS3"],
                "slash24": [10, 10, 11, 12, 20],
            }
        )

        result = country_networks(attackers)

        assert result["country"].to_list() == ["AA", "BB"]
        aa = result.row(0, named=True)
        assert aa["ips"] == 4
        assert aa["asns"] == 2
        assert aa["slash24s"] == 3
        assert aa["top_network"] == "AS1 Host"
        assert aa["top_share"] == 0.75
        assert aa["ips_per_slash24"] == pytest.approx(4 / 3)
