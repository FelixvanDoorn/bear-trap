# honeypot/analysis/ml/tokens.py
#
# Per-session tokens shared by the ml experiments: what an attacker
# typed and which client it used, as strings a TF-IDF vocabulary can
# count. One row per (src_ip, session) or per token, never aggregated, so
# each experiment picks its own unit -- the campaign clustering pools them
# per IP, the payload model keeps sessions and cuts commands off at its
# horizon (filter `events` first to see only part of a session).
#
# - credentials: "pair=user:password" and "user=user" per login attempt;
# - commands: normalized commands (fingerprinting's normalization: C2 IPs
#   and busybox verification tags stripped), consecutive repeats
#   collapsed, in order;
# - client: "version=" (SSH client version), "hassh=" (key exchange
#   fingerprint) and "telnet=" (hash of the telnet option negotiation).
import polars as pl

from honeypot.analysis.fingerprinting.features import (
    dedupe_consecutive,
    hash_telnet_options,
    normalize_command,
    telnet_option_label,
)

LOGIN_EVENTS = ("cowrie.login.success", "cowrie.login.failed")


def json_field(field_name: str) -> pl.Expr:
    return pl.col("raw").str.json_path_match(f"$.{field_name}")


def credential_tokens(events: pl.DataFrame) -> pl.DataFrame:
    """src_ip, session, token: every login attempt's "pair=" token, then
    every attempt's "user=" token (missing user/password -> "")."""
    logins = events.filter(pl.col("eventid").is_in(LOGIN_EVENTS)).with_columns(
        pl.col("username").fill_null(""), pl.col("password").fill_null("")
    )
    return pl.concat(
        [
            logins.select(
                "src_ip",
                "session",
                (pl.lit("pair=") + pl.col("username") + ":" + pl.col("password")).alias(
                    "token"
                ),
            ),
            logins.select(
                "src_ip",
                "session",
                (pl.lit("user=") + pl.col("username")).alias("token"),
            ),
        ]
    )


def command_tokens(events: pl.DataFrame) -> pl.DataFrame:
    """src_ip, session, commands: each session's normalized commands in
    order, consecutive repeats collapsed (same noise filter as
    fingerprinting's sequence hash). Sessions without commands are
    absent."""
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
            "session": session,
            "commands": dedupe_consecutive([normalize_command(c) for c in inputs]),
        }
        for src_ip, session, inputs in commands.select(
            "src_ip", "session", "command_input"
        ).iter_rows()
    ]
    return pl.DataFrame(
        rows,
        schema={"src_ip": pl.Utf8, "session": pl.Utf8, "commands": pl.List(pl.Utf8)},
    )


def client_tokens(events: pl.DataFrame) -> pl.DataFrame:
    """src_ip, session, token: "version=" / "hassh=" per SSH client event
    and one "telnet=" per session with option negotiation; nulls dropped,
    duplicates kept (callers dedupe at their own unit)."""
    ssh = events.filter(
        pl.col("eventid").is_in(["cowrie.client.version", "cowrie.client.kex"])
    ).select(
        "src_ip",
        "session",
        pl.when(pl.col("eventid") == "cowrie.client.version")
        .then(pl.lit("version=") + json_field("version"))
        .otherwise(pl.lit("hassh=") + json_field("hassh"))
        .alias("token"),
    )
    options = (
        events.filter(pl.col("eventid") == "cowrie.telnet.option")
        .sort("event_timestamp")
        .group_by("src_ip", "session", maintain_order=True)
        .agg("raw")
    )
    telnet_rows = []
    for src_ip, session, raws in options.select("src_ip", "session", "raw").iter_rows():
        labels = [label for raw in raws if (label := telnet_option_label(raw))]
        if digest := hash_telnet_options(labels):
            telnet_rows.append(
                {"src_ip": src_ip, "session": session, "token": f"telnet={digest}"}
            )
    telnet = pl.DataFrame(
        telnet_rows,
        schema={"src_ip": pl.Utf8, "session": pl.Utf8, "token": pl.Utf8},
    )
    return pl.concat([ssh, telnet]).drop_nulls(["src_ip", "token"])
