# honeypot/analysis/ml/campaigns/profiles.py
#
# One row per attacking IP (any IP with at least one login attempt), in two
# stages so nothing learned from the data leaks across splits:
#
# 1. ip_profiles(events): raw per-IP material, computed from that IP's own
#    events only -- token lists (credentials, commands, client
#    fingerprints) and behaviour counts. Safe to compute on all IPs at
#    once.
# 2. campaign_encoder(): the shared FeatureEncoder (ml/encoding.py) over
#    these blocks -- everything *learned* (TF-IDF vocabularies and IDF
#    weights, numeric scaling) is fit on the training split and only
#    applied to validation/test, like any sklearn transformer.
#
# Why IPs and not sessions: command sessions are near-duplicates (247
# distinct sequences across 55k sessions in the 2026-09-16 snapshot), so a
# session clustering would just rediscover exact sequence hashes. A
# campaign is a set of IPs running the same tooling: the same credential
# dictionary, playbook and SSH/telnet client.
#
# Feature blocks, each scaled to the same total weight (BLOCK_WEIGHTS) so
# a block with many columns (credentials: thousands of pairs) doesn't
# drown out a small one (behaviour: a dozen numbers):
#
# - credentials, commands, client: every token of the IP's sessions
#   (ml/tokens.py), pooled per IP;
# - behaviour: log counts and rates (BEHAVIOUR_FEATURES).
from collections.abc import Mapping

import polars as pl

from honeypot.analysis.ml.encoding import DEFAULT_MIN_DF, FeatureEncoder
from honeypot.analysis.ml.tokens import (
    LOGIN_EVENTS,
    client_tokens,
    command_tokens,
    credential_tokens,
)

TOKEN_BLOCKS = ("credentials", "commands", "client")
BEHAVIOUR_FEATURES = (
    "log_sessions",
    "log_login_attempts",
    "log_distinct_pairs",
    "login_success_rate",
    "log_span_hours",
    "active_days",
    "log_sessions_per_active_hour",
    "command_session_share",
    "log_commands_per_session",
    "telnet_share",
    "agent_detector_share",
    "downloaded",
)
BLOCKS = (*TOKEN_BLOCKS, "behaviour")
BLOCK_WEIGHTS = {block: 1.0 for block in BLOCKS}


def campaign_encoder(
    weights: Mapping[str, float] = BLOCK_WEIGHTS, min_df: int = DEFAULT_MIN_DF
) -> FeatureEncoder:
    """FeatureEncoder over ip_profiles' blocks (unfitted)."""
    return FeatureEncoder(
        token_blocks=TOKEN_BLOCKS,
        numeric_blocks={"behaviour": BEHAVIOUR_FEATURES},
        weights=dict(weights),
        min_df=min_df,
    )


def _token_lists(frame: pl.DataFrame, token: str, name: str) -> pl.DataFrame:
    return frame.group_by("src_ip").agg(pl.col(token).alias(name))


def _ip_commands(events: pl.DataFrame) -> pl.DataFrame:
    """Per IP: every session's normalized commands, concatenated."""
    return (
        command_tokens(events)
        .group_by("src_ip")
        .agg(pl.col("commands").list.explode(keep_nulls=False, empty_as_null=False))
    )


def _ip_clients(events: pl.DataFrame) -> pl.DataFrame:
    """Per IP: distinct "version=", "hassh=" and "telnet=" tokens."""
    tokens = client_tokens(events).select("src_ip", "token").unique()
    return _token_lists(tokens.sort("src_ip", "token"), "token", "client")


def ip_profiles(events: pl.DataFrame) -> pl.DataFrame:
    """One row per IP with a login attempt, sorted by src_ip: the token
    lists `credentials`, `commands`, `client` (empty list when none),
    BEHAVIOUR_FEATURES, and descriptive columns that aren't features
    (sessions, first_seen, last_seen, protocols, node_roles)."""
    login_ips = (
        events.filter(pl.col("eventid").is_in(LOGIN_EVENTS)).select("src_ip").unique()
    )
    events = events.join(login_ips, on="src_ip", how="semi")

    logins = events.filter(pl.col("eventid").is_in(LOGIN_EVENTS)).with_columns(
        pl.col("username").fill_null(""), pl.col("password").fill_null("")
    )
    login_stats = logins.group_by("src_ip").agg(
        pl.len().alias("login_attempts"),
        pl.struct("username", "password").n_unique().alias("distinct_pairs"),
        (pl.col("eventid") == "cowrie.login.success")
        .mean()
        .alias("login_success_rate"),
    )

    sessions = events.group_by("src_ip", "session").agg(
        pl.col("event_timestamp").min().alias("start"),
        pl.col("protocol").first(),
        pl.col("node_role").first(),
        (pl.col("eventid") == "cowrie.command.input").sum().alias("commands"),
        (pl.col("eventid") == "cowrie.session.file_download").any().alias("download"),
    )
    session_stats = sessions.group_by("src_ip").agg(
        pl.len().alias("sessions"),
        pl.col("start").min().alias("first_seen"),
        pl.col("start").max().alias("last_seen"),
        pl.col("start").dt.truncate("1h").n_unique().alias("active_hours"),
        pl.col("start").dt.date().n_unique().alias("active_days"),
        (pl.col("commands") > 0).mean().alias("command_session_share"),
        pl.col("commands").mean().alias("commands_per_session"),
        (pl.col("protocol") == "telnet").mean().alias("telnet_share"),
        (pl.col("node_role") == "agent_detector").mean().alias("agent_detector_share"),
        pl.col("download").any().cast(pl.Float64).alias("downloaded"),
        pl.col("protocol").drop_nulls().unique().sort().alias("protocols"),
        pl.col("node_role").drop_nulls().unique().sort().alias("node_roles"),
    )

    profiles = (
        session_stats.join(login_stats, on="src_ip", how="left")
        .join(
            _token_lists(credential_tokens(events), "token", "credentials"),
            on="src_ip",
        )
        .join(_ip_commands(events), on="src_ip", how="left")
        .join(_ip_clients(events), on="src_ip", how="left")
    )
    span_hours = (pl.col("last_seen") - pl.col("first_seen")).dt.total_seconds() / 3600
    return (
        profiles.with_columns(
            [
                pl.col(block).fill_null(pl.lit([], dtype=pl.List(pl.Utf8)))
                for block in TOKEN_BLOCKS
            ]
        )
        .with_columns(
            pl.col("sessions").log1p().alias("log_sessions"),
            pl.col("login_attempts").log1p().alias("log_login_attempts"),
            pl.col("distinct_pairs").log1p().alias("log_distinct_pairs"),
            span_hours.log1p().alias("log_span_hours"),
            pl.col("active_days").cast(pl.Float64),
            (pl.col("sessions") / pl.col("active_hours"))
            .log1p()
            .alias("log_sessions_per_active_hour"),
            pl.col("commands_per_session").log1p().alias("log_commands_per_session"),
        )
        .select(
            "src_ip",
            *TOKEN_BLOCKS,
            *BEHAVIOUR_FEATURES,
            "sessions",
            "login_attempts",
            "distinct_pairs",
            "first_seen",
            "last_seen",
            "protocols",
            "node_roles",
        )
        .sort("src_ip")
    )
