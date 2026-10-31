# honeypot/analysis/common/enrich.py
#
# IP -> ASN / AS org / country enrichment (Data-Driven Security ch. 4)
# from DB-IP's free "Lite" databases: CC BY 4.0, no account or license key
# needed, published monthly as CSVs of (start_ip, end_ip, ...) ranges.
# Attribution required -- see network/README.md.
#
# Lookups are a sorted range join: each IPv4 address becomes an integer,
# join_asof finds the last range starting at or below it, and a range-end
# check drops addresses that fall in a gap between ranges. IPv4 only: the
# collected data has no IPv6 sources, so IPv6 rows are dropped at load
# time and any IPv6 src_ip simply gets null enrichment.
#
# Caveat: this is who owns / where DB-IP places an address *now* (the
# database month), not at attack time. Over a few months of collection
# that drift is small, but it's why the month is kept in the filename.
import datetime as dt
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

import polars as pl

DbipKind = Literal["country", "asn"]

DBIP_URL = "https://download.db-ip.com/free/dbip-{kind}-lite-{month}.csv.gz"

# Overridable via env var for the same reason as extract.SNAPSHOT_DIR: the
# notebook calls load_enrichment() with no explicit directory, and its
# execution test points it at fixture CSVs instead of downloading.
ENRICHMENT_DIR = Path(
    os.environ.get("BEARTRAP_ENRICHMENT_DIR")
    or Path(__file__).parent / "outputs" / "enrichment"
)

# DB-IP's placeholder for reserved / unallocated space (0.0.0.0/8,
# multicast, ...) -- not a real country.
_UNKNOWN_COUNTRY = "ZZ"

_RANGE_COLUMNS: dict[DbipKind, list[str]] = {
    "country": ["start", "end", "country"],
    "asn": ["start", "end", "asn", "as_org"],
}


def ipv4_to_int(ip: pl.Expr) -> pl.Expr:
    """Dotted-quad IPv4 string -> Int64 (a.b.c.d -> a*2^24 + b*2^16 + c*2^8
    + d). Null for anything that isn't exactly four 0-255 octets (IPv6,
    malformed input, null)."""
    octets = ip.str.extract_groups(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
    parts = [octets.struct.field(str(i)).cast(pl.Int64) for i in range(1, 5)]
    value = parts[0] * 2**24 + parts[1] * 2**16 + parts[2] * 2**8 + parts[3]
    in_range = pl.all_horizontal([part <= 255 for part in parts])
    return pl.when(in_range).then(value).otherwise(None)


def _cached(kind: DbipKind, enrichment_dir: Path) -> Path | None:
    # Month is zero-padded YYYY-MM, so lexical order is chronological.
    files = sorted(enrichment_dir.glob(f"dbip-{kind}-lite-*.csv.gz"))
    return files[-1] if files else None


def _months_back(today: dt.date, count: int) -> list[str]:
    """`count` YYYY-MM strings, starting at `today`'s month and going back."""
    months = []
    year, month = today.year, today.month
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year, month - 1) if month > 1 else (year - 1, 12)
    return months


def fetch_dbip(
    kind: DbipKind,
    enrichment_dir: Path = ENRICHMENT_DIR,
    today: dt.date | None = None,
) -> Path:
    """Path to the newest cached DB-IP Lite CSV of `kind`, downloading the
    current month's release first if nothing is cached. Falls back to the
    previous month when the current one isn't published yet (early in a
    month). Never re-downloads once a file is cached -- delete it to
    refresh."""
    cached = _cached(kind, enrichment_dir)
    if cached is not None:
        return cached

    enrichment_dir.mkdir(parents=True, exist_ok=True)
    for month in _months_back(today or dt.date.today(), 2):
        path = enrichment_dir / f"dbip-{kind}-lite-{month}.csv.gz"
        try:
            urllib.request.urlretrieve(DBIP_URL.format(kind=kind, month=month), path)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                continue
            raise
        return path
    raise FileNotFoundError(f"No DB-IP {kind} release found for the last 2 months.")


def read_ranges(path: Path, kind: DbipKind) -> pl.DataFrame:
    """A DB-IP Lite CSV as IPv4 ranges: Int64 `start`/`end` plus `country`
    or `asn`/`as_org`, sorted by start (as join_asof requires). IPv6 rows
    are dropped, and the "ZZ" unknown-country placeholder becomes null."""
    columns = _RANGE_COLUMNS[kind]
    ranges = (
        pl.read_csv(
            path,
            has_header=False,
            new_columns=columns,
            schema_overrides={column: pl.Utf8 for column in columns},
        )
        .with_columns(
            ipv4_to_int(pl.col("start")).alias("start"),
            ipv4_to_int(pl.col("end")).alias("end"),
        )
        .filter(pl.col("start").is_not_null() & pl.col("end").is_not_null())
    )
    if kind == "country":
        ranges = ranges.with_columns(
            pl.when(pl.col("country") != _UNKNOWN_COUNTRY).then(pl.col("country"))
        )
    else:
        ranges = ranges.with_columns(pl.col("asn").cast(pl.Int64))
    return ranges.sort("start")


def _range_lookup(ips: pl.DataFrame, ranges: pl.DataFrame) -> pl.DataFrame:
    """Left-join `ranges`' value columns onto `ips` (which carries an
    `ip_int` column) for every address inside a range; null otherwise."""
    value_columns = [c for c in ranges.columns if c not in ("start", "end")]
    matched = (
        ips.filter(pl.col("ip_int").is_not_null())
        .sort("ip_int")
        .join_asof(ranges, left_on="ip_int", right_on="start", strategy="backward")
        .with_columns(
            pl.when(pl.col("ip_int") <= pl.col("end")).then(pl.col(column))
            for column in value_columns
        )
        .select("ip_int", *value_columns)
    )
    return ips.join(matched, on="ip_int", how="left")


def enrich_ips(
    ips: pl.DataFrame, country_ranges: pl.DataFrame, asn_ranges: pl.DataFrame
) -> pl.DataFrame:
    """One row per distinct `src_ip` in `ips`, with `country` (ISO 3166-1
    alpha-2), `asn`, `as_org` and `as_label` ("AS13335 Cloudflare, Inc.",
    a readable categorical) -- all null when the address isn't IPv4 or
    isn't covered by the database."""
    distinct = (
        ips.select("src_ip")
        .unique()
        .with_columns(ipv4_to_int(pl.col("src_ip")).alias("ip_int"))
    )
    enriched = _range_lookup(_range_lookup(distinct, country_ranges), asn_ranges)
    return enriched.drop("ip_int").with_columns(
        pl.when(pl.col("asn").is_not_null())
        .then(pl.format("AS{} {}", pl.col("asn"), pl.col("as_org")))
        .alias("as_label")
    )


def load_enrichment(
    ips: pl.DataFrame, enrichment_dir: Path = ENRICHMENT_DIR
) -> pl.DataFrame:
    """enrich_ips against the cached (or freshly downloaded) DB-IP Lite
    country and ASN databases."""
    return enrich_ips(
        ips,
        read_ranges(fetch_dbip("country", enrichment_dir), "country"),
        read_ranges(fetch_dbip("asn", enrichment_dir), "asn"),
    )
