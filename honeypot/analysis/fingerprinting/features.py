# honeypot/analysis/fingerprinting/features.py
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

import polars as pl


@dataclass
class SessionFeatures:
    session: str
    sensor: str | None
    src_ip: str | None
    cloud_provider: str | None
    protocol: str | None
    event_count: int
    command_count: int
    commands: list[str] = field(default_factory=list)
    command_sequence_hash: str | None = None
    session_duration_ms: float | None = None
    inter_command_mean_s: float | None = None
    inter_command_std_s: float | None = None
    inter_command_min_s: float | None = None
    client_version: str | None = None
    is_ssh_client: bool = False
    hassh: str | None = None
    telnet_options: list[str] = field(default_factory=list)
    telnet_option_hash: str | None = None
    has_exploit_attempt: bool = False
    exploit_cve: str | None = None
    username: str | None = None
    password: str | None = None
    has_agent_report: bool = False


def _raw_field(raw_json: object, field_name: str) -> str | None:
    if not isinstance(raw_json, str) or not raw_json:
        return None
    try:
        return json.loads(raw_json).get(field_name)
    except json.JSONDecodeError:
        return None


def _parse_raw(raw_json: object) -> dict:
    if not isinstance(raw_json, str) or not raw_json:
        return {}
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        return {}


def _telnet_option_label(raw_json: object) -> str | None:
    """`cowrie.telnet.option` -> "<IAC command>:<option name>", e.g.
    "WILL:NAWS". Telnet has no version banner or kex exchange the way SSH
    does, so this ordered option-negotiation sequence is the closest
    per-client analogue -- see _hash_telnet_options."""
    parsed = _parse_raw(raw_json)
    command = parsed.get("command")
    if command is None:
        return None
    option_name = parsed.get("option_name") or f"OPT{parsed.get('option_byte')}"
    return f"{command}:{option_name}"


def _first_non_null(series: pl.Series) -> object | None:
    non_null = series.drop_nulls()
    return non_null[0] if non_null.len() > 0 else None


_IPV4 = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
# Matches only a *single* trailing token after busybox (e.g. "/bin/busybox
# V3G4"), not multi-arg invocations like "/bin/busybox hostname slur" --
# those are a different applet call, not a verification-tag probe.
_BUSYBOX_VERIFICATION_TAG = re.compile(r"^(?:/bin/busybox|busybox)\s+\S+$")


def _normalize_command(command: str) -> str:
    """Strip known per-connection noise that doesn't reflect a behaviorally
    different command, found by inspecting real singleton clusters:
    botnets append a random verification tag after `busybox` to confirm
    they're talking to a real shell (not a honeypot), and the same
    download-and-run playbook gets reused across campaigns with only the
    C2/payload IP differing."""
    if _BUSYBOX_VERIFICATION_TAG.match(command):
        return command.rsplit(" ", 1)[0]
    return _IPV4.sub("<IP>", command)


def _dedupe_consecutive(commands: list[str]) -> list[str]:
    """Collapse immediately-repeated commands (observed as pairs/runs of
    the same probe logged back to back, likely a retry/reconnect artifact)
    so that noise alone doesn't split an otherwise-identical sequence into
    its own cluster."""
    deduped: list[str] = []
    for command in commands:
        if not deduped or deduped[-1] != command:
            deduped.append(command)
    return deduped


def _hash_commands(commands: list[str]) -> str | None:
    """Exact-match fingerprint of a session's *normalized* command
    sequence, so sessions replaying an identical playbook (same commands,
    same order, modulo the noise _normalize_command/_dedupe_consecutive
    strip) group under one hash. \x1f (ASCII unit separator) joins
    commands rather than a printable character, since attacker input could
    legitimately contain any printable separator and create false
    collisions. Returns None for zero commands -- "ran nothing" is a
    distinct case from any real sequence, not just another hash value."""
    if not commands:
        return None
    normalized = _dedupe_consecutive([_normalize_command(c) for c in commands])
    return hashlib.sha256("\x1f".join(normalized).encode()).hexdigest()[:16]


def _hash_telnet_options(options: list[str]) -> str | None:
    """Fingerprint of a session's ordered telnet option-negotiation
    sequence -- the telnet equivalent of _hash_commands/hassh. Consecutive
    repeats (e.g. NAWS renegotiated on a terminal resize) are collapsed
    first, same reasoning as _dedupe_consecutive for commands: that's a
    renegotiation artifact, not a distinct client. Returns None for zero
    options, same "nothing observed" vs. "real empty sequence" distinction
    as _hash_commands."""
    if not options:
        return None
    deduped = _dedupe_consecutive(options)
    return hashlib.sha256("\x1f".join(deduped).encode()).hexdigest()[:16]


def _session_features(session_id: str, events: pl.DataFrame) -> SessionFeatures:
    events = events.sort("event_timestamp")

    client_version_raw = _first_non_null(
        events.filter(pl.col("eventid") == "cowrie.client.version")["raw"]
    )
    kex_raw = _first_non_null(
        events.filter(pl.col("eventid") == "cowrie.client.kex")["raw"]
    )

    command_rows = events.filter(pl.col("command_input").is_not_null())
    commands = command_rows["command_input"].to_list()

    inter_command_mean_s = inter_command_std_s = inter_command_min_s = None
    if command_rows.height > 1:
        # dt.total_seconds() truncates to whole seconds -- go via
        # total_microseconds() to keep sub-second precision, since that's
        # exactly the kind of signal that separates scripted/agent timing
        # from human typing.
        intervals = (
            command_rows["event_timestamp"].diff().dt.total_microseconds() / 1_000_000
        ).drop_nulls()
        inter_command_mean_s = float(intervals.mean())
        # ddof=0 (population std): with exactly 2 commands there's only one
        # interval, and the default ddof=1 sample std is undefined (null)
        # for a single observation.
        inter_command_std_s = float(intervals.std(ddof=0))
        inter_command_min_s = float(intervals.min())

    client_version = _raw_field(client_version_raw, "version")

    telnet_option_rows = events.filter(pl.col("eventid") == "cowrie.telnet.option")
    telnet_options = [
        label
        for raw in telnet_option_rows["raw"].to_list()
        if (label := _telnet_option_label(raw)) is not None
    ]

    exploit_rows = events.filter(pl.col("eventid") == "cowrie.telnet.exploit_attempt")
    has_exploit_attempt = exploit_rows.height > 0
    exploit_cve = (
        _raw_field(_first_non_null(exploit_rows["raw"]), "cve")
        if has_exploit_attempt
        else None
    )

    return SessionFeatures(
        session=session_id,
        sensor=_first_non_null(events["sensor"]),
        src_ip=_first_non_null(events["src_ip"]),
        cloud_provider=_first_non_null(events["cloud_provider"]),
        protocol=_first_non_null(events["protocol"]),
        event_count=events.height,
        command_count=len(commands),
        commands=commands,
        command_sequence_hash=_hash_commands(commands),
        session_duration_ms=_first_non_null(events["duration_ms"]),
        inter_command_mean_s=inter_command_mean_s,
        inter_command_std_s=inter_command_std_s,
        inter_command_min_s=inter_command_min_s,
        client_version=client_version,
        # RFC 4253 mandates real SSH version-exchange strings start with
        # "SSH-" -- anything else here is protocol-blind noise (TLS/HTTP/FTP
        # probes, scanner payloads, ...) that never actually spoke SSH.
        is_ssh_client=client_version is not None and client_version.startswith("SSH-"),
        hassh=_raw_field(kex_raw, "hassh"),
        telnet_options=telnet_options,
        telnet_option_hash=_hash_telnet_options(telnet_options),
        has_exploit_attempt=has_exploit_attempt,
        exploit_cve=exploit_cve,
        username=_first_non_null(events["username"]),
        password=_first_non_null(events["password"]),
        has_agent_report=bool(events["agent_type"].is_not_null().any()),
    )


def extract_session_features(events: pl.DataFrame) -> pl.DataFrame:
    """Collapse a flat event-level DataFrame (see extract.get_sessions) into
    one row per session."""
    rows = [
        _session_features(session_id, group)
        for (session_id,), group in events.partition_by("session", as_dict=True).items()
    ]
    # infer_schema_length=None: sparse optional fields (e.g. exploit_cve) are
    # None for virtually every session, so the default 100-row inference
    # window can lock in the wrong dtype and then choke on the first real
    # value it meets further down -- seen live once exploit_cve is involved.
    return pl.DataFrame([asdict(row) for row in rows], infer_schema_length=None)
