# honeypot/analysis/ml/features.py
#
# One row per attacking IP (any IP with at least one login attempt), in two
# stages so nothing learned from the data leaks across splits:
#
# 1. ip_profiles(events): raw per-IP material, computed from that IP's own
#    events only -- token lists (credentials, commands, client
#    fingerprints) and behaviour counts. Safe to compute on all IPs at
#    once.
# 2. FeatureEncoder: everything *learned* -- TF-IDF vocabularies and IDF
#    weights, numeric scaling -- is fit on the training split and only
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
# - credentials: "user:password" pairs and bare usernames tried;
# - commands: normalized commands run (fingerprinting's normalization:
#   C2 IPs and busybox verification tags stripped);
# - client: SSH client version, hassh, telnet option-sequence hash;
# - behaviour: log counts and rates (BEHAVIOUR_FEATURES).
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from honeypot.analysis.fingerprinting.features import (
    _dedupe_consecutive,
    _hash_telnet_options,
    _normalize_command,
    _telnet_option_label,
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
DEFAULT_MIN_DF = 2

_LOGIN_EVENTS = ("cowrie.login.success", "cowrie.login.failed")


def _json_field(field_name: str) -> pl.Expr:
    return pl.col("raw").str.json_path_match(f"$.{field_name}")


def _token_lists(frame: pl.DataFrame, token: str, name: str) -> pl.DataFrame:
    return frame.group_by("src_ip").agg(pl.col(token).alias(name))


def _command_tokens(events: pl.DataFrame) -> pl.DataFrame:
    """Per IP: every normalized command of every session, consecutive
    repeats within a session collapsed (same noise filter as
    fingerprinting's sequence hash)."""
    commands = (
        events.filter(
            (pl.col("eventid") == "cowrie.command.input")
            & pl.col("command_input").is_not_null()
        )
        .sort("event_timestamp")
        .group_by("src_ip", "session", maintain_order=True)
        .agg("command_input")
    )
    rows = [
        {
            "src_ip": src_ip,
            "commands": _dedupe_consecutive([_normalize_command(c) for c in inputs]),
        }
        for src_ip, inputs in commands.select("src_ip", "command_input").iter_rows()
    ]
    tokens = pl.DataFrame(
        rows, schema={"src_ip": pl.Utf8, "commands": pl.List(pl.Utf8)}
    )
    return tokens.group_by("src_ip").agg(
        pl.col("commands").list.explode(keep_nulls=False, empty_as_null=False)
    )


def _client_tokens(events: pl.DataFrame) -> pl.DataFrame:
    """Per IP: distinct "version=", "hassh=" and "telnet=" tokens."""
    ssh = events.filter(
        pl.col("eventid").is_in(["cowrie.client.version", "cowrie.client.kex"])
    ).select(
        "src_ip",
        pl.when(pl.col("eventid") == "cowrie.client.version")
        .then(pl.lit("version=") + _json_field("version"))
        .otherwise(pl.lit("hassh=") + _json_field("hassh"))
        .alias("token"),
    )
    options = (
        events.filter(pl.col("eventid") == "cowrie.telnet.option")
        .sort("event_timestamp")
        .group_by("src_ip", "session", maintain_order=True)
        .agg("raw")
    )
    telnet_rows = []
    for src_ip, raws in options.select("src_ip", "raw").iter_rows():
        labels = [label for raw in raws if (label := _telnet_option_label(raw))]
        if digest := _hash_telnet_options(labels):
            telnet_rows.append({"src_ip": src_ip, "token": f"telnet={digest}"})
    telnet = pl.DataFrame(telnet_rows, schema={"src_ip": pl.Utf8, "token": pl.Utf8})
    tokens = pl.concat([ssh, telnet]).drop_nulls().unique().sort("src_ip", "token")
    return _token_lists(tokens, "token", "client")


def ip_profiles(events: pl.DataFrame) -> pl.DataFrame:
    """One row per IP with a login attempt, sorted by src_ip: the token
    lists `credentials`, `commands`, `client` (empty list when none),
    BEHAVIOUR_FEATURES, and descriptive columns that aren't features
    (sessions, first_seen, last_seen, protocols, node_roles)."""
    login_ips = (
        events.filter(pl.col("eventid").is_in(_LOGIN_EVENTS)).select("src_ip").unique()
    )
    events = events.join(login_ips, on="src_ip", how="semi")

    logins = events.filter(pl.col("eventid").is_in(_LOGIN_EVENTS)).with_columns(
        pl.col("username").fill_null(""), pl.col("password").fill_null("")
    )
    credential_tokens = pl.concat(
        [
            logins.select(
                "src_ip",
                (pl.lit("pair=") + pl.col("username") + ":" + pl.col("password")).alias(
                    "token"
                ),
            ),
            logins.select(
                "src_ip", (pl.lit("user=") + pl.col("username")).alias("token")
            ),
        ]
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
        .join(_token_lists(credential_tokens, "token", "credentials"), on="src_ip")
        .join(_command_tokens(events), on="src_ip", how="left")
        .join(_client_tokens(events), on="src_ip", how="left")
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


def _identity(tokens: list[str]) -> list[str]:
    return tokens


@dataclass
class FeatureEncoder:
    """Profiles -> one sparse matrix (rows = IPs). fit() learns the TF-IDF
    vocabularies (tokens in >= min_df training IPs) and the behaviour
    scaling on the training profiles only.

    Each block's rows are L2-normalized and multiplied by
    sqrt(weight / total weight), so every IP's full vector has unit
    length and each block contributes its weight's share of it (an IP
    with no tokens in a block gets zeros there)."""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(BLOCK_WEIGHTS))
    min_df: int = DEFAULT_MIN_DF
    vectorizers: dict[str, TfidfVectorizer] = field(default_factory=dict, init=False)
    scaler: StandardScaler | None = field(default=None, init=False)

    def fit(self, profiles: pl.DataFrame) -> "FeatureEncoder":
        for block in TOKEN_BLOCKS:
            if self.weights.get(block, 0) <= 0:
                continue
            vectorizer = TfidfVectorizer(
                analyzer=_identity,
                lowercase=False,
                min_df=self.min_df,
                sublinear_tf=True,
            )
            try:
                vectorizer.fit(profiles[block].to_list())
            except ValueError:
                # No token reaches min_df: the block is empty for everyone.
                continue
            self.vectorizers[block] = vectorizer
        if self.weights.get("behaviour", 0) > 0:
            self.scaler = StandardScaler().fit(self._behaviour(profiles))
        return self

    @staticmethod
    def _behaviour(profiles: pl.DataFrame) -> np.ndarray:
        return profiles.select(BEHAVIOUR_FEATURES).fill_null(0).to_numpy()

    def _scale(self, block: str) -> float:
        total = sum(w for w in self.weights.values() if w > 0)
        return float(np.sqrt(self.weights[block] / total))

    def transform(self, profiles: pl.DataFrame) -> sp.csr_matrix:
        parts = []
        for block, vectorizer in self.vectorizers.items():
            parts.append(
                vectorizer.transform(profiles[block].to_list()) * self._scale(block)
            )
        if self.scaler is not None:
            scaled = self.scaler.transform(self._behaviour(profiles))
            norms = np.linalg.norm(scaled, axis=1, keepdims=True)
            scaled = np.divide(
                scaled, norms, out=np.zeros_like(scaled), where=norms > 0
            )
            parts.append(sp.csr_matrix(scaled * self._scale("behaviour")))
        return sp.hstack(parts, format="csr")

    def feature_names(self) -> list[str]:
        names = [
            f"{block}:{token}"
            for block, vectorizer in self.vectorizers.items()
            for token in vectorizer.get_feature_names_out()
        ]
        if self.scaler is not None:
            names += [f"behaviour:{name}" for name in BEHAVIOUR_FEATURES]
        return names

    def block_slices(self) -> dict[str, slice]:
        """Column range of each block in transform()'s output."""
        slices, start = {}, 0
        for block, vectorizer in self.vectorizers.items():
            width = len(vectorizer.vocabulary_)
            slices[block] = slice(start, start + width)
            start += width
        if self.scaler is not None:
            slices["behaviour"] = slice(start, start + len(BEHAVIOUR_FEATURES))
        return slices
