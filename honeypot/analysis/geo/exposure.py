# honeypot/analysis/geo/exposure.py
#
# Per-country counts of attacking IPs next to each country's IPv4 address
# space (Data-Driven Security ch. 5): the exposure a count model needs
# before "country X sends many attackers" means anything. Raw counts mostly
# show where the address space is.
#
# - Exposure: IPv4 addresses per country, summed from the cached DB-IP
#   Lite country ranges (common/enrich.py) -- no extra download. This is
#   allocation, not use: legacy blocks (much of the US) count in full even
#   when they're mostly dark.
# - Counts: every src_ip seen at all (any event), placed by the same DB-IP
#   release. Three responses, because one attacker isn't one IP:
#   `ips`, distinct `/24` networks (`slash24s`, so an ISP that rotates a
#   device through many addresses in one subnet counts less) and distinct
#   ASNs (`asns`).
# - Every country in the database is a row, zeros included. Dropping the
#   countries that sent nothing would bias the slope (only the "lucky"
#   small countries would remain).
import polars as pl

from honeypot.analysis.common.enrich import ipv4_to_int

RESPONSES = ["ips", "slash24s", "asns"]


def address_space(country_ranges: pl.DataFrame) -> pl.DataFrame:
    """IPv4 addresses per country from enrich.read_ranges(..., "country")
    output. Ranges with no country (DB-IP's "ZZ" reserved / unallocated
    placeholder) are dropped. Sorted by country."""
    return (
        country_ranges.filter(pl.col("country").is_not_null())
        .group_by("country")
        .agg((pl.col("end") - pl.col("start") + 1).sum().alias("addresses"))
        .sort("country")
    )


def _protocol_group(protocols: pl.Expr) -> pl.Expr:
    """Which protocols an IP used: "ssh", "telnet" or "ssh+telnet"."""
    return protocols.drop_nulls().unique().sort().str.join("+")


def attacker_ips(events: pl.DataFrame, enrichment: pl.DataFrame) -> pl.DataFrame:
    """One row per src_ip in `events` that DB-IP places in a country:
    country, asn, as_label, slash24 (the /24 network as an integer), protocols,
    node_roles and first_seen (the IP's first event_timestamp). The unit
    every count and split in geo/ is built from. Sorted by src_ip."""
    per_ip = events.group_by("src_ip").agg(
        _protocol_group(pl.col("protocol")).alias("protocols"),
        _protocol_group(pl.col("node_role")).alias("node_roles"),
        pl.col("event_timestamp").min().alias("first_seen"),
    )
    return (
        per_ip.join(
            enrichment.select("src_ip", "country", "asn", "as_label"),
            on="src_ip",
            how="left",
        )
        .filter(pl.col("country").is_not_null())
        .with_columns((ipv4_to_int(pl.col("src_ip")) // 256).alias("slash24"))
        .select(
            "src_ip",
            "country",
            "asn",
            "as_label",
            "slash24",
            "protocols",
            "node_roles",
            "first_seen",
        )
        .sort("src_ip")
    )


def country_counts(attackers: pl.DataFrame, space: pl.DataFrame) -> pl.DataFrame:
    """One row per country in `space` (address_space output): addresses
    plus every RESPONSES count over `attackers` (attacker_ips output, or
    any subset of it), 0 where a country sent nothing. Sorted by
    country."""
    counts = attackers.group_by("country").agg(
        pl.col("src_ip").n_unique().alias("ips"),
        pl.col("slash24").n_unique().alias("slash24s"),
        pl.col("asn").drop_nulls().n_unique().alias("asns"),
    )
    return (
        space.join(counts, on="country", how="left")
        .with_columns(pl.col(RESPONSES).fill_null(0).cast(pl.Int64))
        .sort("country")
    )


def country_networks(attackers: pl.DataFrame) -> pl.DataFrame:
    """How concentrated each country's attacking IPs are: per country with
    any attackers, ips, distinct ASNs and /24s, ips_per_slash24, the
    network (as_label) with the most IPs and its share. A country whose
    excess sits in one network (top_share near 1) is a statement about
    that network -- a hosting provider, one ISP's infected customers --
    more than about the country. Sorted by ips, descending."""
    per_network = attackers.group_by("country", "as_label").agg(
        pl.len().alias("network_ips")
    )
    top = (
        per_network.sort(["network_ips", "as_label"], descending=[True, False])
        .group_by("country", maintain_order=True)
        .first()
        .select(
            "country",
            pl.col("as_label").alias("top_network"),
            pl.col("network_ips").alias("top_network_ips"),
        )
    )
    return (
        attackers.group_by("country")
        .agg(
            pl.len().alias("ips"),
            pl.col("asn").drop_nulls().n_unique().alias("asns"),
            pl.col("slash24").n_unique().alias("slash24s"),
        )
        .join(top, on="country")
        .with_columns(
            (pl.col("ips") / pl.col("slash24s")).alias("ips_per_slash24"),
            (pl.col("top_network_ips") / pl.col("ips")).alias("top_share"),
        )
        .drop("top_network_ips")
        .sort(["ips", "country"], descending=[True, False])
    )
