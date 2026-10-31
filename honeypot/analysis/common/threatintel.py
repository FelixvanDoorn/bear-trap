# honeypot/analysis/common/threatintel.py
#
# Cross-references source IPs against public blocklists (Data-Driven
# Security ch. 4: does our data agree with, or add to, existing threat
# intel?). Free, no account:
#
# - blocklist_de_ssh: IPs reported to blocklist.de for SSH attacks. The
#   closest match to this honeypot's traffic -- and partly fed by other
#   honeypots, so a high hit rate is partly circular.
# - firehol_level1: FireHOL's low-false-positive aggregate (DShield,
#   Feodo, Spamhaus DROP, full bogons), as CIDR ranges. Mostly whole
#   criminal or unallocated networks, so few hits are expected.
#
# Same cache-once pattern and range join as enrich.py: each list is
# downloaded once into THREATINTEL_DIR with the download date in its
# filename, and the newest cached file is reused until deleted. IPv4 only.
#
# Caveat: a list shows what's listed on the day it was downloaded, not at
# attack time. Lists with short retention (blocklist.de drops IPs that
# stop being reported) measure "still active then" for older data.
import datetime as dt
import ipaddress
import os
import urllib.request
from pathlib import Path

import polars as pl

from honeypot.analysis.common.enrich import _range_lookup, ipv4_to_int

BLOCKLISTS = {
    "blocklist_de_ssh": "https://lists.blocklist.de/lists/ssh.txt",
    "firehol_level1": (
        "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/"
        "firehol_level1.netset"
    ),
}

# Overridable for the same reason as enrich.ENRICHMENT_DIR: the notebook
# calls load_listings() with no explicit directory, and its execution test
# points it at fixture lists instead of downloading.
THREATINTEL_DIR = Path(
    os.environ.get("BEARTRAP_THREATINTEL_DIR")
    or Path(__file__).parent / "outputs" / "threatintel"
)

LISTED_COLUMN = "listed"

# blocklist.de answers 403 to urllib's default User-Agent.
_USER_AGENT = "bear-trap-analysis (+https://github.com/FelixvanDoorn/bear-trap)"


def _on(name: str) -> str:
    return f"on_{name}"


def fetch_blocklist(
    name: str, threatintel_dir: Path = THREATINTEL_DIR, today: dt.date | None = None
) -> Path:
    """Path to the newest cached copy of BLOCKLISTS[name], downloading
    today's first if nothing is cached. Never re-downloads once a file is
    cached -- delete it to refresh."""
    # ISO date suffix, so lexical order is chronological.
    cached = sorted(threatintel_dir.glob(f"{name}-*.txt"))
    if cached:
        return cached[-1]

    request = urllib.request.Request(
        BLOCKLISTS[name], headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    # Written only after a complete download: a partial file would
    # otherwise be cached as the list from then on.
    threatintel_dir.mkdir(parents=True, exist_ok=True)
    path = threatintel_dir / f"{name}-{(today or dt.date.today()).isoformat()}.txt"
    path.write_bytes(body)
    return path


def read_blocklist(path: Path) -> pl.DataFrame:
    """A blocklist file (one IPv4 address or CIDR per line; `#` comments,
    blank lines and IPv6 entries skipped) as non-overlapping Int64
    `start`/`end` ranges sorted by start, as enrich._range_lookup needs --
    overlapping or adjacent entries are merged, since join_asof only ever
    sees the last range starting at or below an address."""
    ranges = []
    for line in path.read_text().splitlines():
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        try:
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if network.version == 4:
            ranges.append(
                (int(network.network_address), int(network.broadcast_address))
            )

    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return pl.DataFrame(
        merged, schema={"start": pl.Int64, "end": pl.Int64}, orient="row"
    )


def match_blocklists(
    ips: pl.DataFrame, blocklists: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """One row per distinct `src_ip` in `ips`, with a boolean `on_<name>`
    per blocklist (read_blocklist output) and LISTED_COLUMN = on any of
    them. Non-IPv4 addresses are never listed. Sorted by src_ip."""
    matched = (
        ips.select("src_ip")
        .unique()
        .with_columns(ipv4_to_int(pl.col("src_ip")).alias("ip_int"))
    )
    for name, ranges in blocklists.items():
        hit = ranges.with_columns(pl.lit(True).alias(_on(name)))
        matched = _range_lookup(matched, hit).with_columns(
            pl.col(_on(name)).fill_null(False)
        )
    columns = [_on(name) for name in blocklists]
    return (
        matched.drop("ip_int")
        .with_columns(pl.any_horizontal(columns).alias(LISTED_COLUMN))
        .sort("src_ip")
    )


def load_listings(
    ips: pl.DataFrame, threatintel_dir: Path = THREATINTEL_DIR
) -> pl.DataFrame:
    """match_blocklists against the cached (or freshly downloaded) copy of
    every list in BLOCKLISTS."""
    return match_blocklists(
        ips,
        {
            name: read_blocklist(fetch_blocklist(name, threatintel_dir))
            for name in BLOCKLISTS
        },
    )


def listing_coverage(listed: pl.DataFrame, by: str) -> pl.DataFrame:
    """Per level of `by`, how many IPs there are and what share is on each
    list and on any list. `listed` is match_blocklists output joined onto
    a frame carrying `by` (one row per IP). Sorted by `by`."""
    flags = [c for c in listed.columns if c.startswith("on_")] + [LISTED_COLUMN]
    return (
        listed.group_by(by)
        .agg(
            pl.len().alias("ips"),
            *[pl.col(flag).mean().alias(f"share_{flag}") for flag in flags],
        )
        .sort(by)
    )
