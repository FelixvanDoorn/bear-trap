# honeypot/analysis/overview/tests/test_summary.py
import datetime as dt

import polars as pl

from honeypot.analysis.overview.summary import (
    coverage,
    daily_activity,
    session_funnel,
    session_table,
    tunnel_targets,
)
from honeypot.analysis.overview.tests.fixtures import make_event, make_events


class TestCoverage:
    def test_counts_per_role_and_protocol(self) -> None:
        events = make_events(
            [
                make_event("s1", "cowrie.session.connect", sensor="old-container"),
                make_event("s1", "cowrie.session.closed", sensor="old-container"),
                make_event("s2", "cowrie.session.connect", src_ip="2.2.2.2"),
                make_event("s3", "cowrie.session.connect", protocol="telnet"),
            ]
        )
        result = coverage(events)

        ssh = result.filter(pl.col("protocol") == "ssh").row(0, named=True)
        assert ssh["sensors"] == 2
        assert ssh["events"] == 3
        assert ssh["sessions"] == 2
        assert ssh["distinct_ips"] == 2
        assert result["protocol"].to_list() == ["ssh", "telnet"]


class TestSessionTable:
    def test_flags_each_stage_and_takes_duration_from_closed_event(self) -> None:
        events = make_events(
            [
                make_event("s1", "cowrie.session.connect"),
                make_event("s1", "cowrie.login.failed"),
                make_event("s1", "cowrie.login.success"),
                make_event("s1", "cowrie.command.input"),
                make_event("s1", "cowrie.session.file_download"),
                make_event("s1", "cowrie.session.closed", duration_ms=4500),
            ]
        )
        row = session_table(events).row(0, named=True)

        assert row["tried_login"]
        assert row["authenticated"]
        assert row["ran_commands"]
        assert row["file_transfer"]
        assert not row["tunnel_request"]
        assert row["duration_ms"] == 4500

    def test_unclosed_session_has_null_duration(self) -> None:
        events = make_events([make_event("s1", "cowrie.session.connect")])

        assert session_table(events)["duration_ms"].to_list() == [None]


class TestSessionFunnel:
    def test_counts_sessions_reaching_each_stage(self) -> None:
        events = make_events(
            [
                # Scanner: connects, never logs in.
                make_event("s1", "cowrie.session.connect"),
                make_event("s1", "cowrie.session.closed", duration_ms=1000),
                # Failed login only.
                make_event("s2", "cowrie.login.failed"),
                make_event("s2", "cowrie.session.closed", duration_ms=2000),
                # Authenticated, ran commands.
                make_event("s3", "cowrie.login.success"),
                make_event("s3", "cowrie.command.input"),
                make_event("s3", "cowrie.session.closed", duration_ms=9000),
                # Authenticated tunnel, no commands -- stages aren't nested.
                make_event("s4", "cowrie.login.success"),
                make_event("s4", "cowrie.direct-tcpip.request", dst_port=25),
                make_event("s4", "cowrie.session.closed", duration_ms=3000),
            ]
        )
        row = session_funnel(events).row(0, named=True)

        assert row["sessions"] == 4
        assert row["tried_login"] == 3
        assert row["authenticated"] == 2
        assert row["ran_commands"] == 1
        assert row["file_transfer"] == 0
        assert row["tunnel_request"] == 1
        assert row["median_duration_s"] == 2.5

    def test_one_row_per_role_and_protocol(self) -> None:
        events = make_events(
            [
                make_event("s1", "cowrie.session.connect"),
                make_event("s2", "cowrie.session.connect", protocol="telnet"),
                make_event("s3", "cowrie.session.connect", node_role="control_proxy"),
            ]
        )
        result = session_funnel(events)

        assert result.select("node_role", "protocol").rows() == [
            ("agent_detector", "ssh"),
            ("agent_detector", "telnet"),
            ("control_proxy", "ssh"),
        ]


class TestDailyActivity:
    def test_splits_new_and_returning_ips(self) -> None:
        events = make_events(
            [
                make_event("s1", "cowrie.session.connect", day=1, src_ip="1.1.1.1"),
                make_event("s2", "cowrie.session.connect", day=2, src_ip="1.1.1.1"),
                make_event("s3", "cowrie.session.connect", day=2, src_ip="2.2.2.2"),
                make_event("s4", "cowrie.session.connect", day=2, src_ip="2.2.2.2"),
            ]
        )
        result = daily_activity(events, trim_partial_days=False)

        day2 = result.filter(pl.col("date") == dt.date(2026, 8, 2)).row(0, named=True)
        assert day2["sessions"] == 3
        assert day2["distinct_ips"] == 2
        assert day2["new_ips"] == 1
        assert day2["returning_ips"] == 1

    def test_first_seen_is_global_across_roles(self) -> None:
        # 1.1.1.1 first appears on the agent sensor; its first visit to the
        # control proxy a day later is still "returning".
        events = make_events(
            [
                make_event("s1", "cowrie.session.connect", day=1),
                make_event(
                    "s2", "cowrie.session.connect", day=2, node_role="control_proxy"
                ),
            ]
        )
        result = daily_activity(events, trim_partial_days=False)

        control = result.filter(pl.col("node_role") == "control_proxy").row(
            0, named=True
        )
        assert control["new_ips"] == 0
        assert control["returning_ips"] == 1

    def test_trims_first_and_last_date(self) -> None:
        events = make_events(
            [
                make_event(f"s{day}", "cowrie.session.connect", day=day)
                for day in (1, 2, 3)
            ]
        )

        assert daily_activity(events)["date"].to_list() == [dt.date(2026, 8, 2)]

    def test_trim_is_a_no_op_with_fewer_than_three_dates(self) -> None:
        events = make_events(
            [make_event(f"s{day}", "cowrie.session.connect", day=day) for day in (1, 2)]
        )

        assert daily_activity(events).height == 2

    def test_session_spanning_midnight_counts_on_its_start_date(self) -> None:
        events = make_events(
            [
                make_event("s1", "cowrie.session.connect", day=1, hour=23),
                make_event("s1", "cowrie.command.input", day=2, hour=0),
            ]
        )
        result = daily_activity(events, trim_partial_days=False)

        assert result["date"].to_list() == [dt.date(2026, 8, 1)]


class TestTunnelTargets:
    def test_counts_requests_sessions_and_source_ips_per_target(self) -> None:
        events = make_events(
            [
                make_event(
                    "s1",
                    "cowrie.direct-tcpip.request",
                    dst_ip="mx.example",
                    dst_port=25,
                ),
                make_event(
                    "s2",
                    "cowrie.direct-tcpip.request",
                    src_ip="2.2.2.2",
                    dst_ip="mx.example",
                    dst_port=25,
                ),
                make_event(
                    "s3",
                    "cowrie.direct-tcpip.request",
                    dst_ip="google.com",
                    dst_port=443,
                ),
                make_event(
                    "s3",
                    "cowrie.direct-tcpip.request",
                    dst_ip="google.com",
                    dst_port=443,
                ),
                # Not a tunnel request -- the honeypot's own port, ignored.
                make_event("s4", "cowrie.session.connect", dst_port=2222),
            ]
        )
        result = tunnel_targets(events)

        assert result.rows() == [
            ("google.com", 443, 2, 1, 1),
            ("mx.example", 25, 2, 2, 2),
        ]
