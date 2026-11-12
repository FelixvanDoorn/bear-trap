# honeypot/analysis/ml/payload/sessions.py
#
# The supervised question: from what a session shows at login (and its
# first k commands), will it try to drop a payload? One row per session
# with a successful login.
#
# Label (`dropped`): the session tries to get a file onto the box --
#
# - a fetch tool pointed at a network location (DROP_PATTERNS["fetch"]):
#   wget / curl / tftp / ftpget / lwp-download plus a URL or IPv4. Without
#   a location it's a probe ("/bin/busybox wget; /bin/busybox STC" only
#   checks that wget exists);
# - scp from a remote host;
# - an echo dropper: a binary written with echo, >= 32 hex-escaped bytes
#   in one command (short ones like echo -e "\x6F\x6B" just print "ok", a
#   liveness probe);
# - or a cowrie transfer event (download, failed download, upload), which
#   also catches SFTP/SCP uploads that run no command.
#
# `drop_position`: the number of commands the session ran before its first
# drop attempt (the fetch command's index, or the commands before the
# first transfer event). The model must decide before that.
#
# Horizon k (horizon_rows): the model sees the session up to its login
# (k = 0) or up to its k-th command (k >= 1), nothing after. Population at
# horizon k: sessions with at least k commands that haven't dropped within
# the first k. A session that already dropped is decided; one that ended
# before k commands is left out too, else "fewer than k commands" would
# give away "ended early, so negative". Session end, duration and
# anything after the cutoff (closing events, later commands, downloads)
# never reach the features.
#
# Feature blocks (`token_blocks` / `numeric_blocks` for FeatureEncoder):
#
# - credentials: pair= / user= tokens of the attempts so far;
# - client: SSH version, hassh, telnet option hash (ml/tokens.py);
# - commands: the first k normalized commands (empty at k = 0);
# - context: protocol and sensor role;
# - network: country and the words of the AS name ("digitalocean",
#   "telecom"): an ASN number never generalizes under an ASN-grouped
#   split, words of its name can;
# - login (numeric): failed attempts before the success, distinct
#   usernames tried, first-attempt success;
# - timing (numeric): log seconds connect -> login and login -> cutoff,
#   hour of day (UTC) on the circle.
#
# No IP history (an IP's earlier sessions): the label is near-constant per
# IP (28 of 620 dropping IPs ever had a non-dropping session in the
# 2026-09-16 snapshot), so history would make it a lookup, and a new IP
# has none. Blocklist membership is out too: the lists are fetched after
# the fact and partly built from exactly these attacks.
import polars as pl

from honeypot.analysis.ml.tokens import (
    LOGIN_EVENTS,
    client_tokens,
    command_tokens,
    credential_tokens,
)

DROP_PATTERNS = {
    "fetch": (
        r"(?i)\b(wget|curl|tftp|ftpget|lwp-download)\b"
        r".*(https?://|ftp://|\b\d{1,3}(\.\d{1,3}){3}\b)"
    ),
    "scp": r"(?i)\bscp\b.*\S@[^\s:]+:",
    "echo_dropper": r"(?i)\becho\b.*(\\x[0-9a-f]{2}){32}",
}
TRANSFER_EVENTS = (
    "cowrie.session.file_download",
    "cowrie.session.file_download.failed",
    "cowrie.session.file_upload",
)
HORIZONS = (0, 1, 2)
TOKEN_BLOCKS = ("credentials", "client", "commands", "context", "network")
NUMERIC_BLOCKS = {
    "login": ("log_failed_before_login", "log_usernames_tried", "first_try_login"),
    "timing": (
        "log_seconds_to_login",
        "log_seconds_since_login",
        "hour_sin",
        "hour_cos",
    ),
}
BLOCKS = (*TOKEN_BLOCKS, *NUMERIC_BLOCKS)


def _ordered(events: pl.DataFrame) -> pl.DataFrame:
    """Events in session order with `rank` (0-based position in the
    session; ties keep snapshot order)."""
    return events.sort("session", "event_timestamp", maintain_order=True).with_columns(
        pl.int_range(pl.len()).over("session").alias("rank")
    )


def _commands(ordered: pl.DataFrame) -> pl.DataFrame:
    """Command inputs with their 0-based `position` and the drop pattern
    they match (null if none)."""
    kind = pl.lit(None, dtype=pl.Utf8)
    for name, pattern in reversed(DROP_PATTERNS.items()):
        kind = (
            pl.when(pl.col("command_input").str.contains(pattern))
            .then(pl.lit(name))
            .otherwise(kind)
        )
    return (
        ordered.filter(
            (pl.col("eventid") == "cowrie.command.input")
            & pl.col("command_input").is_not_null()
        )
        .with_columns(pl.int_range(pl.len()).over("session").alias("position"))
        .with_columns(kind.alias("drop_kind"))
        .select("session", "rank", "position", "drop_kind")
    )


def session_table(events: pl.DataFrame) -> pl.DataFrame:
    """One row per session with a successful login, sorted by start:
    session, src_ip, start, protocol, node_role, n_commands, login_rank,
    dropped, drop_position, drop_kind."""
    ordered = _ordered(events)
    commands = _commands(ordered)
    sessions = ordered.group_by("session").agg(
        pl.col("src_ip").first(),
        pl.col("event_timestamp").min().alias("start"),
        pl.col("protocol").drop_nulls().first(),
        pl.col("node_role").drop_nulls().first(),
        pl.col("rank")
        .filter(pl.col("eventid") == "cowrie.login.success")
        .min()
        .alias("login_rank"),
        pl.col("rank")
        .filter(pl.col("eventid").is_in(TRANSFER_EVENTS))
        .min()
        .alias("transfer_rank"),
    )
    command_stats = commands.group_by("session").agg(
        pl.len().alias("n_commands"),
        pl.col("position")
        .filter(pl.col("drop_kind").is_not_null())
        .min()
        .alias("fetch_position"),
        pl.col("drop_kind").drop_nulls().first().alias("command_drop_kind"),
    )
    with_commands = sessions.filter(pl.col("login_rank").is_not_null()).join(
        command_stats, on="session", how="left"
    )
    before_transfer = (
        commands.join(with_commands.select("session", "transfer_rank"), on="session")
        .filter(pl.col("rank") < pl.col("transfer_rank"))
        .group_by("session")
        .agg(pl.len().alias("commands_before_transfer"))
    )
    table = (
        with_commands.join(before_transfer, on="session", how="left")
        .with_columns(
            pl.col("n_commands").fill_null(0),
            pl.when(pl.col("transfer_rank").is_not_null())
            .then(pl.col("commands_before_transfer").fill_null(0))
            .alias("commands_before_transfer"),
        )
        .with_columns(
            pl.min_horizontal("fetch_position", "commands_before_transfer").alias(
                "drop_position"
            ),
        )
        .with_columns(
            pl.col("drop_position").is_not_null().alias("dropped"),
            pl.when(
                pl.col("fetch_position").is_not_null()
                & (pl.col("fetch_position") == pl.col("drop_position"))
            )
            .then(pl.col("command_drop_kind"))
            .when(pl.col("drop_position").is_not_null())
            .then(pl.lit("transfer"))
            .alias("drop_kind"),
        )
    )
    return table.select(
        "session",
        "src_ip",
        "start",
        "protocol",
        "node_role",
        "n_commands",
        "login_rank",
        "dropped",
        "drop_position",
        "drop_kind",
    ).sort("start", "session")


def population(sessions: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Sessions still undecided after `horizon` commands (see header)."""
    return sessions.filter(
        (pl.col("n_commands") >= horizon)
        & (~pl.col("dropped") | (pl.col("drop_position") >= horizon))
    )


def _cutoffs(
    ordered: pl.DataFrame, sessions: pl.DataFrame, horizon: int
) -> pl.DataFrame:
    """session -> rank of the last event the model may see: the login
    success at horizon 0, else the horizon-th command."""
    if horizon == 0:
        return sessions.select("session", pl.col("login_rank").alias("cutoff_rank"))
    return (
        _commands(ordered)
        .filter(pl.col("position") == horizon - 1)
        .select("session", pl.col("rank").alias("cutoff_rank"))
        .join(sessions.select("session"), on="session", how="semi")
    )


def _session_lists(tokens: pl.DataFrame, column: str, name: str) -> pl.DataFrame:
    return tokens.group_by("session", maintain_order=True).agg(
        pl.col(column).alias(name)
    )


def _as_words(as_label: pl.Expr) -> pl.Expr:
    return (
        as_label.fill_null("")
        .str.to_lowercase()
        .str.extract_all(r"[a-z]{3,}")
        .list.eval(pl.lit("as=") + pl.element())
    )


def horizon_rows(
    events: pl.DataFrame, sessions: pl.DataFrame, horizon: int
) -> pl.DataFrame:
    """Feature rows at `horizon` for the sessions in population(): the
    columns of `sessions` (country / as_label optional, for the network
    block) plus every token list in TOKEN_BLOCKS and every NUMERIC_BLOCKS
    column, computed from events up to the cutoff only."""
    rows = population(sessions, horizon)
    ordered = _ordered(events.join(rows.select("session"), on="session", how="semi"))
    cutoffs = _cutoffs(ordered, rows, horizon)
    cut = ordered.join(cutoffs, on="session").filter(
        pl.col("rank") <= pl.col("cutoff_rank")
    )

    logins = cut.filter(pl.col("eventid").is_in(LOGIN_EVENTS))
    login_stats = logins.group_by("session").agg(
        (pl.col("eventid") == "cowrie.login.failed").sum().alias("failed_before_login"),
        pl.col("username").n_unique().alias("usernames_tried"),
    )
    times = cut.group_by("session").agg(
        pl.col("event_timestamp").min().alias("connect_time"),
        pl.col("event_timestamp")
        .filter(pl.col("eventid") == "cowrie.login.success")
        .min()
        .alias("login_time"),
        pl.col("event_timestamp").max().alias("cutoff_time"),
    )
    commands = (
        command_tokens(cut).select("session", "commands")
        if horizon > 0
        else pl.DataFrame(schema={"session": pl.Utf8, "commands": pl.List(pl.Utf8)})
    )
    client = client_tokens(cut).unique(["session", "token"]).sort("session", "token")
    empty = pl.lit([], dtype=pl.List(pl.Utf8))
    has_network = {"country", "as_label"} <= set(rows.columns)
    hour = pl.col("start").dt.hour().cast(pl.Float64) * (2 * 3.141592653589793 / 24)
    return (
        rows.join(
            _session_lists(credential_tokens(cut), "token", "credentials"),
            on="session",
            how="left",
        )
        .join(_session_lists(client, "token", "client"), on="session", how="left")
        .join(commands, on="session", how="left")
        .join(login_stats, on="session", how="left")
        .join(times, on="session", how="left")
        .with_columns(
            pl.col("credentials").fill_null(empty),
            pl.col("client").fill_null(empty),
            pl.col("commands").fill_null(empty),
            pl.concat_list(
                pl.lit("protocol=") + pl.col("protocol").fill_null("?"),
                pl.lit("node=") + pl.col("node_role").fill_null("?"),
            ).alias("context"),
            (
                pl.concat_list(
                    pl.lit("country=") + pl.col("country").fill_null("?"),
                    _as_words(pl.col("as_label")),
                )
                if has_network
                else empty
            ).alias("network"),
            pl.col("failed_before_login")
            .fill_null(0)
            .log1p()
            .alias("log_failed_before_login"),
            pl.col("usernames_tried").fill_null(0).log1p().alias("log_usernames_tried"),
            (pl.col("failed_before_login").fill_null(0) == 0)
            .cast(pl.Float64)
            .alias("first_try_login"),
            (pl.col("login_time") - pl.col("connect_time"))
            .dt.total_milliseconds()
            .truediv(1000)
            .log1p()
            .alias("log_seconds_to_login"),
            (pl.col("cutoff_time") - pl.col("login_time"))
            .dt.total_milliseconds()
            .truediv(1000)
            .log1p()
            .alias("log_seconds_since_login"),
            hour.sin().alias("hour_sin"),
            hour.cos().alias("hour_cos"),
        )
        .drop(
            "failed_before_login",
            "usernames_tried",
            "connect_time",
            "login_time",
            "cutoff_time",
        )
        .with_columns(pl.lit(horizon).alias("horizon"))
        .sort("start", "session")
    )
