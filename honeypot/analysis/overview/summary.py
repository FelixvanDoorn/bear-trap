# honeypot/analysis/overview/summary.py
#
# Descriptive "what did we collect" tables -- the Data-Driven Security
# ch. 3 baseline -- shaped around how Cowrie and this deployment actually
# work rather than generic security-event counting:
#
# - node_role, not cloud_provider or sensor, is the unit of comparison.
#   Each cloud runs a different role with a different config (AWS =
#   control_proxy, SSH only; Azure = agent_detector, SSH + telnet + agent
#   prompt handlers), so a per-cloud comparison is really a per-config one.
#   `sensor` is Cowrie's container id and changes on every redeploy -- the
#   Azure VM shows up as two "sensors" because its container was replaced
#   on 2026-08-17.
# - Authentication outcomes are set by userdb.txt (root/admin accept any
#   password), so "login success rate" measures which usernames attackers
#   try, not how good they are. session_funnel tracks what happens *after*
#   authentication instead: commands, file transfers, tunnel requests.
# - dst_port is the honeypot's own listening port (2222/2223) for every
#   event except cowrie.direct-tcpip.request, where it's the target of an
#   attacker asking the honeypot to relay traffic -- so destination ports
#   only appear in tunnel_targets.
#
# Client versions, command sequences and telnet options live in
# fingerprinting/; credentials in network/.
import polars as pl

LOGIN_EVENTS = ["cowrie.login.success", "cowrie.login.failed"]
FILE_TRANSFER_EVENTS = ["cowrie.session.file_download", "cowrie.session.file_upload"]
TUNNEL_REQUEST_EVENT = "cowrie.direct-tcpip.request"

FUNNEL_STAGES = [
    "tried_login",
    "authenticated",
    "ran_commands",
    "file_transfer",
    "tunnel_request",
]


def coverage(events: pl.DataFrame) -> pl.DataFrame:
    """One row per (node_role, protocol): how many sensor containers
    reported for it, first/last event timestamp, and event, session and
    distinct-IP counts. The "what does this dataset actually contain"
    table to read before any other number here. Sorted by node_role, then
    protocol."""
    return (
        events.group_by("node_role", "protocol")
        .agg(
            pl.col("sensor").n_unique().alias("sensors"),
            pl.col("event_timestamp").min().alias("first_seen"),
            pl.col("event_timestamp").max().alias("last_seen"),
            pl.len().alias("events"),
            pl.col("session").n_unique().alias("sessions"),
            pl.col("src_ip").n_unique().alias("distinct_ips"),
        )
        .sort("node_role", "protocol")
    )


def session_table(events: pl.DataFrame) -> pl.DataFrame:
    """One row per session: node_role, protocol, src_ip, start time, a
    boolean per FUNNEL_STAGES entry, and duration_ms from the session's
    cowrie.session.closed event (null if it never closed within the
    dataset). The shared base for session_funnel and daily_activity."""
    return events.group_by("session").agg(
        pl.col("node_role").first(),
        pl.col("protocol").first(),
        pl.col("src_ip").first(),
        pl.col("event_timestamp").min().alias("start"),
        pl.col("eventid").is_in(LOGIN_EVENTS).any().alias("tried_login"),
        (pl.col("eventid") == "cowrie.login.success").any().alias("authenticated"),
        (pl.col("eventid") == "cowrie.command.input").any().alias("ran_commands"),
        pl.col("eventid").is_in(FILE_TRANSFER_EVENTS).any().alias("file_transfer"),
        (pl.col("eventid") == TUNNEL_REQUEST_EVENT).any().alias("tunnel_request"),
        pl.col("duration_ms")
        .filter(pl.col("eventid") == "cowrie.session.closed")
        .max()
        .alias("duration_ms"),
    )


def session_funnel(events: pl.DataFrame) -> pl.DataFrame:
    """One row per (node_role, protocol): total sessions, how many reached
    each FUNNEL_STAGES step, and the median session duration in seconds.
    Stages aren't strictly nested -- a tunnel request needs authentication
    but no commands -- so each count is "sessions that did X", not "sessions
    that survived to step X". authenticated mostly reflects userdb.txt (see
    module comment); the stages after it are the attacker-driven ones.
    Sorted by node_role, then protocol."""
    return (
        session_table(events)
        .group_by("node_role", "protocol")
        .agg(
            pl.len().alias("sessions"),
            *[pl.col(stage).sum() for stage in FUNNEL_STAGES],
            (pl.col("duration_ms").median() / 1000).alias("median_duration_s"),
        )
        .sort("node_role", "protocol")
    )


def daily_activity(
    events: pl.DataFrame, trim_partial_days: bool = True
) -> pl.DataFrame:
    """One row per (UTC date, node_role): sessions, distinct_ips, and how
    many of those IPs were new (first seen anywhere in the dataset, on any
    role, that day) vs returning. Counting IPs and sessions rather than
    events keeps a handful of very chatty sessions from dominating the
    series. With trim_partial_days, the dataset's first and last dates are
    dropped -- collection starts and stops mid-day, so those two days
    undercount by construction (a no-op with fewer than three dates).
    Note that the dataset start makes every IP "new" on the first full day
    or two -- read new_ips as meaningful only once the series has warmed
    up. Sorted by date, then node_role."""
    sessions = session_table(events).with_columns(
        pl.col("start").dt.date().alias("date")
    )
    first_seen = sessions.group_by("src_ip").agg(
        pl.col("date").min().alias("first_date")
    )

    daily = (
        sessions.join(first_seen, on="src_ip")
        .group_by("date", "node_role")
        .agg(
            pl.len().alias("sessions"),
            pl.col("src_ip").n_unique().alias("distinct_ips"),
            pl.col("src_ip")
            .filter(pl.col("date") == pl.col("first_date"))
            .n_unique()
            .alias("new_ips"),
        )
        .with_columns(
            (pl.col("distinct_ips") - pl.col("new_ips")).alias("returning_ips")
        )
        .sort("date", "node_role")
    )

    if trim_partial_days:
        dates = daily["date"].unique()
        if dates.len() >= 3:
            daily = daily.filter(
                (pl.col("date") != dates.min()) & (pl.col("date") != dates.max())
            )

    return daily


def tunnel_targets(events: pl.DataFrame) -> pl.DataFrame:
    """One row per (dst_ip, dst_port) attackers asked the honeypot to relay
    a connection to (cowrie.direct-tcpip.request), with request, session
    and distinct-source-IP counts. A single target requested by hundreds
    of unrelated IPs (e.g. one mail server on port 25) is a relay-probing
    campaign; 443/80 targets like ip-who.com or google.com are usually
    connectivity checks. dst_ip may be a hostname -- Cowrie logs whatever
    the client asked for. Sorted by requests descending, then dst_ip."""
    return (
        events.filter(pl.col("eventid") == TUNNEL_REQUEST_EVENT)
        .group_by("dst_ip", "dst_port")
        .agg(
            pl.len().alias("requests"),
            pl.col("session").n_unique().alias("sessions"),
            pl.col("src_ip").n_unique().alias("distinct_ips"),
        )
        .sort(["requests", "dst_ip"], descending=[True, False])
    )
