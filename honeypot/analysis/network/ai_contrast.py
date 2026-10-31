# honeypot/analysis/network/ai_contrast.py
#
# Do IPs that probe AI-tooling usernames differ from other attackers --
# in where they come from (country, ASN), what client they run (SSH
# version, hassh), whether public blocklists already know them, what they
# do after logging in, and how they behave over time? Screens a fixed set
# of per-IP properties with common/stats.py's compare_groups (chi-square
# / Fisher / Mann-Whitney + BH correction).
#
# Size-matched by design: in the 2026-09-16 snapshot only 118 of 4,843
# login-attempting IPs tried an AI username, and they tried a median of
# 31.5 distinct usernames vs 1 for everyone else -- most are ~240-name
# dictionaries where AI names are ~4% of the list. Against all IPs, every
# test would mostly measure "big dictionary vs one-shot" traffic. So both
# groups are restricted to IPs that tried >= min_usernames distinct
# usernames; distinct_usernames stays in the screen as a balance check on
# how well that matching worked.
#
# Unit is the src_ip, never the session or attempt: one IP's sessions
# aren't independent observations. IPs within one credential cluster
# (network/graph.py) aren't fully independent either: one_ip_per_cluster
# reruns the screen counting each cluster once.
import polars as pl

from honeypot.analysis.common.stats import (
    DEFAULT_ALPHA,
    OTHER_LEVEL,
    category_breakdown,
    compare_groups,
)
from honeypot.analysis.common.threatintel import LISTED_COLUMN
from honeypot.analysis.network.ai_targeting import ai_targeting_ips
from honeypot.analysis.network.graph import _pair_attempts
from honeypot.analysis.overview.summary import session_table

DEFAULT_MIN_USERNAMES = 10

GROUP_COLUMN = "is_ai_probe"

CATEGORICAL_PROPERTIES = [
    "country",
    "as_label",
    "protocols",
    "node_roles",
    "ssh_client",
    "hassh",
    "authenticated",
    "ran_commands",
    "file_transfer",
    "tunnel_request",
    "listed",
]
NUMERIC_PROPERTIES = [
    "distinct_usernames",
    "distinct_passwords",
    "credential_attempts",
    "sessions",
    "active_days",
    "median_session_s",
]


def _modal_raw_field(events: pl.DataFrame, eventid: str, field: str) -> pl.DataFrame:
    """Per src_ip, the most common value of `field` in the raw JSON of its
    `eventid` events (ties broken alphabetically, for determinism)."""
    return (
        events.filter(pl.col("eventid") == eventid)
        .select("src_ip", pl.col("raw").str.json_path_match(f"$.{field}").alias(field))
        .filter(pl.col(field).is_not_null())
        .group_by("src_ip")
        .agg(pl.col(field).mode().sort().first())
    )


def _joined_values(column: str) -> pl.Expr:
    """Sorted distinct values joined with "+": a categorical for "which of
    ssh/telnet" (or which node roles) an IP used."""
    return pl.col(column).drop_nulls().unique().sort().str.join("+").alias(f"{column}s")


def ip_profiles(
    events: pl.DataFrame,
    enrichment: pl.DataFrame | None = None,
    listings: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per src_ip that attempted at least one (username, password)
    pair, with every CATEGORICAL_PROPERTIES / NUMERIC_PROPERTIES column.
    Session outcomes come from overview.summary.session_table (any session
    of the IP authenticated / ran commands / ...), ssh_client and hassh are
    the IP's most common values. `enrichment` (common.enrich.enrich_ips
    output) supplies country and as_label, and `listings`
    (common.threatintel.load_listings output) supplies `listed`; without
    them those are null. Sorted by src_ip."""
    attempts = _pair_attempts(events)
    credentials = attempts.group_by("src_ip").agg(
        pl.col("username").n_unique().alias("distinct_usernames"),
        pl.col("password").n_unique().alias("distinct_passwords"),
        pl.len().alias("credential_attempts"),
    )
    sessions = (
        session_table(events)
        .group_by("src_ip")
        .agg(
            pl.len().alias("sessions"),
            pl.col("start").dt.date().n_unique().alias("active_days"),
            (pl.col("duration_ms").median() / 1000).alias("median_session_s"),
            _joined_values("protocol"),
            _joined_values("node_role"),
            pl.col("authenticated").any(),
            pl.col("ran_commands").any(),
            pl.col("file_transfer").any(),
            pl.col("tunnel_request").any(),
        )
    )
    if enrichment is None:
        enrichment = credentials.select("src_ip").with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("country"),
            pl.lit(None, dtype=pl.Utf8).alias("as_label"),
        )
    if listings is None:
        listings = credentials.select("src_ip").with_columns(
            pl.lit(None, dtype=pl.Boolean).alias(LISTED_COLUMN)
        )

    return (
        credentials.join(sessions, on="src_ip", how="left")
        .join(
            _modal_raw_field(events, "cowrie.client.version", "version").rename(
                {"version": "ssh_client"}
            ),
            on="src_ip",
            how="left",
        )
        .join(
            _modal_raw_field(events, "cowrie.client.kex", "hassh"),
            on="src_ip",
            how="left",
        )
        .join(
            enrichment.select("src_ip", "country", "as_label"),
            on="src_ip",
            how="left",
        )
        .join(listings.select("src_ip", LISTED_COLUMN), on="src_ip", how="left")
        .sort("src_ip")
    )


def ai_probe_profiles(
    events: pl.DataFrame,
    enrichment: pl.DataFrame | None = None,
    min_usernames: int = DEFAULT_MIN_USERNAMES,
    listings: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """ip_profiles restricted to IPs with distinct_usernames >=
    `min_usernames`, plus a boolean GROUP_COLUMN: True when the IP tried
    at least one AI-tooling username (ai_targeting.classify_ai_username).
    The size-matched population compare_ai_probes tests -- also the input
    for stats.category_breakdown drill-downs."""
    ai_ips = ai_targeting_ips(events)["src_ip"]
    return (
        ip_profiles(events, enrichment, listings)
        .filter(pl.col("distinct_usernames") >= min_usernames)
        .with_columns(pl.col("src_ip").is_in(ai_ips.implode()).alias(GROUP_COLUMN))
    )


def compare_ai_probes(
    profiles: pl.DataFrame, alpha: float = DEFAULT_ALPHA
) -> pl.DataFrame:
    """stats.compare_groups over ai_probe_profiles output: group a (n_a) =
    AI-probe IPs, group b (n_b) = size-matched other IPs. One row per
    property, BH-adjusted across all of them."""
    return compare_groups(
        profiles,
        GROUP_COLUMN,
        categorical=CATEGORICAL_PROPERTIES,
        numeric=NUMERIC_PROPERTIES,
        alpha=alpha,
    )


def without_top_ai_level(
    profiles: pl.DataFrame, column: str = "as_label"
) -> tuple[str | None, pl.DataFrame]:
    """Sensitivity check against one campaign dominating the AI-probe
    group: the level of `column` most over-represented among AI probes
    (stats.category_breakdown's top row, never the pooled "other"), and
    `profiles` with every IP at that level -- from both groups -- removed.
    (None, profiles unchanged) when no real level is over-represented.
    In the 2026-09-16 snapshot this drops AS197170 TechTies (NL): 44 AI
    probes + 13 others sharing one hassh."""
    over_represented = category_breakdown(profiles, GROUP_COLUMN, column).filter(
        (pl.col("level") != OTHER_LEVEL) & (pl.col("share_diff") > 0)
    )
    if over_represented.height == 0:
        return None, profiles
    level = over_represented["level"][0]
    return level, profiles.filter(pl.col(column).fill_null("") != level)


def one_ip_per_cluster(profiles: pl.DataFrame, clusters: pl.DataFrame) -> pl.DataFrame:
    """Independence check: `profiles` (ai_probe_profiles output) cut down
    to one IP per (credential cluster, group), so a campaign spread over
    many IPs that share a wordlist counts once per group rather than once
    per IP. Keeps the IP with the most credential_attempts (ties by
    src_ip). `clusters` is graph.assign_clusters output; an IP missing from
    it (every pair it tried was excluded from the graph) is its own
    singleton. Kept per group rather than per cluster, so a cluster
    containing both AI probes and others still contributes to both sides.
    Sorted by src_ip."""
    return (
        profiles.join(clusters.select("src_ip", "cluster_id"), on="src_ip", how="left")
        .with_columns(pl.col("cluster_id").cast(pl.Utf8).fill_null(pl.col("src_ip")))
        .sort(["credential_attempts", "src_ip"], descending=[True, False])
        .unique(["cluster_id", GROUP_COLUMN], keep="first", maintain_order=True)
        .drop("cluster_id")
        .sort("src_ip")
    )
