# honeypot/analysis/common/worldbank.py
#
# Country indicators from the World Bank's open data API (CC BY 4.0, no
# account needed), as covariates for geo/'s exposure regression:
#
# - gdp_per_capita: GDP per capita, current US$ (NY.GDP.PCAP.CD)
# - internet_share: individuals using the internet, % of population
#   (IT.NET.USER.ZS)
# - population: total population (SP.POP.TOTL)
# - internet_users: internet_share / 100 * population, derived here
#
# Each indicator is requested with mrnev=1 (the most recent non-empty
# value per country), so the year can differ between countries and
# indicators; it's kept per indicator, and values older than
# DEFAULT_MIN_YEAR are treated as missing (the "most recent" internet
# share for a few territories is from 1990). The World Bank uses ISO 3166-1
# alpha-2 country codes, the same as DB-IP. It also returns regional and
# income aggregates ("ZH" Africa Eastern and Southern, "1A" Arab World,
# ...) under codes no country uses; they simply never match a DB-IP
# country. Taiwan isn't covered.
#
# Same cache-once pattern as enrich.py and threatintel.py: each indicator
# is downloaded once into WORLDBANK_DIR with the download date in its
# filename, and the newest cached file is reused until deleted.
import datetime as dt
import json
import os
import urllib.request
from pathlib import Path

import polars as pl

INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "internet_share": "IT.NET.USER.ZS",
    "population": "SP.POP.TOTL",
}

API_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/{code}"
    "?format=json&mrnev=1&per_page=20000"
)

# Overridable for the same reason as enrich.ENRICHMENT_DIR: the notebook
# calls load_indicators() with no explicit directory, and its execution
# test points it at fixture files instead of downloading.
WORLDBANK_DIR = Path(
    os.environ.get("BEARTRAP_WORLDBANK_DIR")
    or Path(__file__).parent / "outputs" / "worldbank"
)

DEFAULT_MIN_YEAR = 2015

_USER_AGENT = "bear-trap-analysis (+https://github.com/FelixvanDoorn/bear-trap)"


def fetch_indicator(
    code: str, worldbank_dir: Path = WORLDBANK_DIR, today: dt.date | None = None
) -> Path:
    """Path to the newest cached JSON response for indicator `code`,
    downloading today's first if nothing is cached. Never re-downloads
    once a file is cached -- delete it to refresh."""
    # ISO date suffix, so lexical order is chronological.
    cached = sorted(worldbank_dir.glob(f"{code}-*.json"))
    if cached:
        return cached[-1]

    request = urllib.request.Request(
        API_URL.format(code=code), headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    # Parsed before writing: a truncated or error response must not be
    # cached as the indicator from then on.
    read_indicator_json(body)
    worldbank_dir.mkdir(parents=True, exist_ok=True)
    path = worldbank_dir / f"{code}-{(today or dt.date.today()).isoformat()}.json"
    path.write_bytes(body)
    return path


def read_indicator_json(body: bytes | str) -> pl.DataFrame:
    """One World Bank API response ([metadata, rows]) as country, year,
    value, dropping rows without a value. Raises ValueError on anything
    that isn't a single complete page of rows (an API error message, or a
    response split over several pages)."""
    payload = json.loads(body)
    if not isinstance(payload, list) or len(payload) != 2 or payload[1] is None:
        raise ValueError(f"Unexpected World Bank response: {str(payload)[:200]}")
    metadata, rows = payload
    if metadata.get("pages", 1) != 1:
        raise ValueError("World Bank response spans several pages.")
    return pl.DataFrame(
        [
            {
                "country": row["country"]["id"],
                "year": int(row["date"]),
                "value": float(row["value"]),
            }
            for row in rows
            if row.get("value") is not None
        ],
        schema={"country": pl.Utf8, "year": pl.Int64, "value": pl.Float64},
    )


def load_indicators(
    worldbank_dir: Path = WORLDBANK_DIR, min_year: int = DEFAULT_MIN_YEAR
) -> pl.DataFrame:
    """One row per country with any INDICATORS value from `min_year` on:
    each indicator and its year (`<name>_year`), plus internet_users =
    internet_share / 100 * population (null unless both are present).
    Sorted by country."""
    frames = [
        read_indicator_json(fetch_indicator(code, worldbank_dir).read_bytes())
        .filter(pl.col("year") >= min_year)
        .rename({"value": name, "year": f"{name}_year"})
        for name, code in INDICATORS.items()
    ]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, on="country", how="full", coalesce=True)
    return merged.with_columns(
        (pl.col("internet_share") / 100 * pl.col("population")).alias("internet_users")
    ).sort("country")
