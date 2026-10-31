# honeypot/analysis/network/report.py
#
# Which source IPs are worth reporting for abuse, with the evidence for
# each, in two formats: a plain CSV to read or share, and AbuseIPDB's
# bulk-report CSV (https://www.abuseipdb.com/bulk-report). Only writes
# files: uploading them stays a manual step, since abuse reports are
# public and hard to retract.
#
# A honeypot has no legitimate users, so every login attempt is
# unsolicited -- but one stray attempt is weak evidence. An IP is
# reportable once it reaches any tier:
#   1 brute force:        >= min_login_attempts login attempts
#   2 post-login activity: ran commands or asked for a TCP tunnel
#   3 payload delivery:   a download command (wget/curl/tftp/ftpget) or a
#                         Cowrie file download/upload event
# `tier` is the highest one reached. Comments state counts and behaviour
# only -- never the honeypot's own address, credentials or command text.
import ipaddress
from collections.abc import Iterable

import polars as pl

from honeypot.analysis.overview.summary import LOGIN_EVENTS, session_table

DEFAULT_MIN_LOGIN_ATTEMPTS = 10

PAYLOAD_COMMAND_PATTERN = r"(?i)\b(?:wget|curl|tftp|ftpget)\b"
FILE_EVENT_PREFIX = "cowrie.session.file_"

# AbuseIPDB category ids: https://www.abuseipdb.com/categories
CATEGORY_HACKING = 15
CATEGORY_BRUTE_FORCE = 18
CATEGORY_EXPLOITED_HOST = 20
CATEGORY_SSH = 22
CATEGORY_IOT_TARGETED = 23  # AbuseIPDB has no telnet category

_PROTOCOL_LABELS = {"ssh": "SSH", "telnet": "Telnet", "ssh+telnet": "SSH and Telnet"}


def _is_global(ip: str) -> bool:
    """False for private, loopback, link-local, reserved and unparsable
    addresses -- nothing worth (or possible) to report."""
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False


def ip_evidence(
    events: pl.DataFrame, enrichment: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per src_ip with at least one session: first/last event,
    sessions, protocols ("ssh", "telnet" or "ssh+telnet"), login_attempts
    (Cowrie login events, not distinct pairs), distinct_usernames, and
    booleans authenticated / ran_commands / tunnel_request /
    payload_download. With `enrichment` (common.enrich.enrich_ips output),
    also country and as_label. Sorted by src_ip."""
    per_event = events.group_by("src_ip").agg(
        pl.col("event_timestamp").min().alias("first_seen"),
        pl.col("event_timestamp").max().alias("last_seen"),
        pl.col("eventid").is_in(LOGIN_EVENTS).sum().alias("login_attempts"),
        pl.col("username")
        .filter(pl.col("eventid").is_in(LOGIN_EVENTS))
        .n_unique()
        .alias("distinct_usernames"),
        (
            pl.col("command_input").str.contains(PAYLOAD_COMMAND_PATTERN)
            | pl.col("eventid").str.starts_with(FILE_EVENT_PREFIX)
        )
        .any()
        .alias("payload_download"),
    )
    per_session = (
        session_table(events)
        .group_by("src_ip")
        .agg(
            pl.len().alias("sessions"),
            pl.col("protocol")
            .drop_nulls()
            .unique()
            .sort()
            .str.join("+")
            .alias("protocols"),
            pl.col("authenticated").any(),
            pl.col("ran_commands").any(),
            pl.col("tunnel_request").any(),
        )
    )
    evidence = per_session.join(per_event, on="src_ip", how="left").with_columns(
        pl.col("payload_download").fill_null(False)
    )
    if enrichment is not None:
        evidence = evidence.join(
            enrichment.select("src_ip", "country", "as_label"), on="src_ip", how="left"
        )
    return evidence.sort("src_ip")


def _categories() -> pl.Expr:
    """AbuseIPDB category ids for one row, as the comma-separated string
    the bulk format expects."""
    ids = [
        pl.when(pl.col("login_attempts") >= pl.col("_min_attempts")).then(
            pl.lit(CATEGORY_BRUTE_FORCE)
        ),
        pl.when(pl.col("protocols").str.contains("ssh")).then(pl.lit(CATEGORY_SSH)),
        pl.when(pl.col("protocols").str.contains("telnet")).then(
            pl.lit(CATEGORY_IOT_TARGETED)
        ),
        pl.when(pl.col("tier") >= 2).then(pl.lit(CATEGORY_HACKING)),
        pl.when(pl.col("tier") == 3).then(pl.lit(CATEGORY_EXPLOITED_HOST)),
    ]
    return (
        pl.concat_list(ids)
        .list.drop_nulls()
        .list.sort()
        .cast(pl.List(pl.Utf8))
        .list.join(",")
        .alias("abuseipdb_categories")
    )


def reportable_ips(
    events: pl.DataFrame,
    min_login_attempts: int = DEFAULT_MIN_LOGIN_ATTEMPTS,
    exclude_ips: Iterable[str] = (),
    enrichment: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """ip_evidence rows that reach at least one tier (see module comment),
    with `tier` (1-3, highest reached) and `abuseipdb_categories`. Drops
    non-global addresses and anything in `exclude_ips` (e.g. your own
    scanners or monitoring). Sorted by tier descending, then
    login_attempts descending, then src_ip."""
    excluded = list(exclude_ips)
    return (
        ip_evidence(events, enrichment)
        .filter(
            pl.col("src_ip").map_elements(_is_global, return_dtype=pl.Boolean)
            & ~pl.col("src_ip").is_in(pl.Series(excluded, dtype=pl.Utf8).implode())
        )
        .with_columns(
            pl.when(pl.col("payload_download"))
            .then(3)
            .when(pl.col("ran_commands") | pl.col("tunnel_request"))
            .then(2)
            .when(pl.col("login_attempts") >= min_login_attempts)
            .then(1)
            .alias("tier"),
            pl.lit(min_login_attempts).alias("_min_attempts"),
        )
        .filter(pl.col("tier").is_not_null())
        .with_columns(_categories())
        .drop("_min_attempts")
        .sort(["tier", "login_attempts", "src_ip"], descending=[True, True, False])
    )


def _comment(row: dict) -> str:
    protocols = _PROTOCOL_LABELS.get(row["protocols"], row["protocols"])
    actions = []
    if row["login_attempts"]:
        actions.append(f"{row['login_attempts']} {protocols} login attempts")
    if row["ran_commands"]:
        actions.append("ran shell commands")
    if row["payload_download"]:
        actions.append("tried to download or upload a payload")
    if row["tunnel_request"]:
        actions.append("requested a TCP tunnel")
    first = row["first_seen"].strftime("%Y-%m-%d %H:%M")
    last = row["last_seen"].strftime("%Y-%m-%d %H:%M")
    return f"Honeypot: {', '.join(actions)}. Seen {first} to {last} UTC."


def abuseipdb_bulk(reportable: pl.DataFrame) -> pl.DataFrame:
    """reportable_ips output in AbuseIPDB's bulk-report columns: IP,
    Categories, ReportDate (the IP's last event, ISO 8601 UTC), Comment."""
    return pl.DataFrame(
        {
            "IP": reportable["src_ip"],
            "Categories": reportable["abuseipdb_categories"],
            "ReportDate": reportable["last_seen"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Comment": [_comment(row) for row in reportable.iter_rows(named=True)],
        },
        schema={
            "IP": pl.Utf8,
            "Categories": pl.Utf8,
            "ReportDate": pl.Utf8,
            "Comment": pl.Utf8,
        },
    )
