# honeypot/analysis/network/tests/test_ai_contrast.py
import datetime as dt
import json

import polars as pl

from honeypot.analysis.network.ai_contrast import (
    CATEGORICAL_PROPERTIES,
    GROUP_COLUMN,
    NUMERIC_PROPERTIES,
    ai_probe_profiles,
    compare_ai_probes,
    ip_profiles,
    one_ip_per_cluster,
    without_top_ai_level,
)

START = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


def _event(
    src_ip: str,
    session: str,
    eventid: str,
    username: str | None = None,
    password: str | None = None,
    raw: dict | None = None,
    duration_ms: int | None = None,
    protocol: str = "ssh",
    day: int = 0,
) -> dict:
    return {
        "src_ip": src_ip,
        "session": session,
        "eventid": eventid,
        "node_role": "agent_detector",
        "protocol": protocol,
        "event_timestamp": START + dt.timedelta(days=day),
        "username": username,
        "password": password,
        "duration_ms": duration_ms,
        "raw": json.dumps(raw) if raw is not None else None,
    }


def _ip_events(
    src_ip: str,
    usernames: list[str],
    session: str,
    version: str = "SSH-2.0-Go",
    hassh: str = "h1",
    day: int = 0,
) -> list[dict]:
    """One SSH session: client banner + kex, one failed login per username,
    then a close."""
    return [
        _event(
            src_ip, session, "cowrie.client.version", raw={"version": version}, day=day
        ),
        _event(src_ip, session, "cowrie.client.kex", raw={"hassh": hassh}, day=day),
        *[
            _event(src_ip, session, "cowrie.login.failed", u, "123456", day=day)
            for u in usernames
        ],
        _event(src_ip, session, "cowrie.session.closed", duration_ms=4000, day=day),
    ]


def _events(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


class TestIpProfiles:
    def test_one_row_per_login_ip_with_all_properties(self) -> None:
        events = _events(
            _ip_events("1.1.1.1", ["root", "admin"], "s1", day=0)
            + _ip_events("1.1.1.1", ["root"], "s2", version="SSH-2.0-Go", day=1)
            # Connect-only IP: no credentials -> not profiled.
            + [_event("9.9.9.9", "s9", "cowrie.session.connect")]
        )
        enrichment = pl.DataFrame(
            {"src_ip": ["1.1.1.1"], "country": ["NL"], "as_label": ["AS1 Test"]}
        )

        profiles = ip_profiles(events, enrichment)

        assert profiles["src_ip"].to_list() == ["1.1.1.1"]
        row = profiles.row(0, named=True)
        assert set(CATEGORICAL_PROPERTIES + NUMERIC_PROPERTIES) <= set(row)
        assert row["distinct_usernames"] == 2
        assert row["credential_attempts"] == 2  # distinct (user, pass) pairs
        assert row["sessions"] == 2
        assert row["active_days"] == 2
        assert row["median_session_s"] == 4.0
        assert row["protocols"] == "ssh"
        assert row["ssh_client"] == "SSH-2.0-Go"
        assert row["hassh"] == "h1"
        assert row["country"] == "NL"
        assert row["authenticated"] is False

    def test_mixed_protocols_and_modal_client(self) -> None:
        events = _events(
            _ip_events("1.1.1.1", ["root"], "s1", version="SSH-2.0-A")
            + _ip_events("1.1.1.1", ["root"], "s2", version="SSH-2.0-B")
            + _ip_events("1.1.1.1", ["root"], "s3", version="SSH-2.0-B")
            + [
                _event(
                    "1.1.1.1",
                    "t1",
                    "cowrie.login.failed",
                    "admin",
                    "x",
                    protocol="telnet",
                )
            ]
        )

        row = ip_profiles(events).row(0, named=True)

        assert row["protocols"] == "ssh+telnet"
        assert row["ssh_client"] == "SSH-2.0-B"
        assert row["country"] is None  # no enrichment passed


class TestAiProbeProfiles:
    def test_size_matching_and_group_flag(self) -> None:
        dictionary = [f"user{i}" for i in range(10)]
        events = _events(
            # AI probe with a big dictionary.
            _ip_events("1.1.1.1", dictionary + ["claude"], "s1")
            # Non-AI big dictionary -- the comparison group.
            + _ip_events("2.2.2.2", dictionary, "s2")
            # AI probe but below min_usernames -> excluded.
            + _ip_events("3.3.3.3", ["ollama"], "s3")
        )

        profiles = ai_probe_profiles(events, min_usernames=10)

        assert dict(zip(profiles["src_ip"], profiles[GROUP_COLUMN])) == {
            "1.1.1.1": True,
            "2.2.2.2": False,
        }


class TestCompareAiProbes:
    def test_flags_a_client_difference(self) -> None:
        dictionary = [f"user{i}" for i in range(10)]
        rows: list[dict] = []
        for i in range(30):
            rows += _ip_events(
                f"10.0.0.{i}", dictionary + ["claude"], f"a{i}", hassh="ai_tool"
            )
            rows += _ip_events(f"10.0.1.{i}", dictionary, f"b{i}", hassh="other_tool")

        result = compare_ai_probes(ai_probe_profiles(_events(rows), min_usernames=10))

        assert set(result["property"]) == set(
            CATEGORICAL_PROPERTIES + NUMERIC_PROPERTIES
        )
        hassh = result.filter(pl.col("property") == "hassh").row(0, named=True)
        assert hassh["significant_bh"] is True
        assert hassh["effect_size"] == 1.0
        # Every IP has exactly one session -> nothing to test.
        sessions = result.filter(pl.col("property") == "sessions").row(0, named=True)
        assert sessions["test"] == "untestable"


class TestWithoutTopAiLevel:
    def _profiles(self, rows: list[tuple[bool, str | None]]) -> pl.DataFrame:
        return pl.DataFrame(
            rows, schema={GROUP_COLUMN: pl.Boolean, "as_label": pl.Utf8}, orient="row"
        )

    def test_drops_the_most_over_represented_level_from_both_groups(self) -> None:
        profiles = self._profiles(
            [(True, "AS1 Campaign")] * 30
            + [(True, "AS2 Mixed")] * 10
            + [(False, "AS1 Campaign")] * 10
            + [(False, "AS2 Mixed")] * 90
            + [(False, None)]
        )

        level, remaining = without_top_ai_level(profiles)

        assert level == "AS1 Campaign"
        assert "AS1 Campaign" not in remaining["as_label"].to_list()
        # Null-ASN IPs are kept, not silently dropped by the filter.
        assert remaining.height == 10 + 90 + 1

    def test_no_over_represented_level_returns_input(self) -> None:
        same = [(True, "AS1 A")] * 30 + [(False, "AS1 A")] * 30
        profiles = self._profiles(same)

        level, remaining = without_top_ai_level(profiles)

        assert level is None
        assert remaining.height == profiles.height


class TestOneIpPerCluster:
    def test_keeps_the_busiest_ip_per_cluster_and_group(self) -> None:
        profiles = pl.DataFrame(
            {
                "src_ip": ["1.1.1.1", "1.1.1.2", "1.1.1.3", "2.2.2.2", "3.3.3.3"],
                GROUP_COLUMN: [True, True, False, True, False],
                "credential_attempts": [50, 80, 10, 5, 7],
            }
        )
        clusters = pl.DataFrame(
            {
                "src_ip": ["1.1.1.1", "1.1.1.2", "1.1.1.3", "2.2.2.2"],
                "cluster_id": [0, 0, 0, 1],
                "cluster_size": [3, 3, 3, 1],
            }
        )

        result = one_ip_per_cluster(profiles, clusters)

        # Cluster 0 keeps its busiest AI probe and its only non-AI IP;
        # 3.3.3.3 isn't in the graph at all and stays as a singleton.
        assert result["src_ip"].to_list() == [
            "1.1.1.2",
            "1.1.1.3",
            "2.2.2.2",
            "3.3.3.3",
        ]
        assert result.columns == profiles.columns

    def test_ties_broken_by_src_ip(self) -> None:
        profiles = pl.DataFrame(
            {
                "src_ip": ["1.1.1.9", "1.1.1.1"],
                GROUP_COLUMN: [True, True],
                "credential_attempts": [10, 10],
            }
        )
        clusters = pl.DataFrame(
            {
                "src_ip": ["1.1.1.9", "1.1.1.1"],
                "cluster_id": [0, 0],
                "cluster_size": [2, 2],
            }
        )

        assert one_ip_per_cluster(profiles, clusters)["src_ip"].to_list() == ["1.1.1.1"]
