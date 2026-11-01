# honeypot/analysis/geo/maps.py
#
# Country shapes for choropleth maps (Data-Driven Security ch. 5: a raw
# count map mostly shows where the address space is; mapping the model's
# residual shows who really stands out).
#
# Natural Earth admin-0 countries at 1:50m (public domain, ~800 KB zip),
# cached once in SHAPES_DIR like the other external data in common/.
# Countries are keyed by ISO_A2_EH rather than ISO_A2: the plain column
# has "-99" for France, Norway and Kosovo, and "CN-TW" for Taiwan. A few
# small DB-IP territories (Réunion, Gibraltar, ...) have no shape at this
# scale and simply aren't drawn. Projected to Equal Earth (EPSG:8857), an
# equal-area projection, so a country's area on the map isn't inflated
# by latitude as on Mercator.
import os
import urllib.request
from pathlib import Path

import geopandas as gpd

NATURAL_EARTH_URL = (
    "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
)
SHAPES_BASENAME = "ne_50m_admin_0_countries"
EQUAL_EARTH = "EPSG:8857"

# Overridable for the same reason as enrich.ENRICHMENT_DIR: the notebook
# loads shapes with no explicit directory, and its execution test points
# it at a fixture file instead of downloading.
SHAPES_DIR = Path(
    os.environ.get("BEARTRAP_SHAPES_DIR")
    or Path(__file__).parent.parent / "common" / "outputs" / "naturalearth"
)

_USER_AGENT = "bear-trap-analysis (+https://github.com/FelixvanDoorn/bear-trap)"
# Not a country, and its area would dominate an equal-area map.
_DROPPED = {"-99", "AQ"}


def fetch_country_shapes(shapes_dir: Path = SHAPES_DIR) -> Path:
    """Path to the cached Natural Earth countries file (the downloaded
    .zip, or any other format geopandas reads, e.g. a fixture .geojson),
    downloading the zip first if nothing is cached."""
    cached = sorted(shapes_dir.glob(f"{SHAPES_BASENAME}.*"))
    if cached:
        return cached[0]
    request = urllib.request.Request(
        NATURAL_EARTH_URL, headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    # Written only after a complete download, as in threatintel.py.
    shapes_dir.mkdir(parents=True, exist_ok=True)
    path = shapes_dir / f"{SHAPES_BASENAME}.zip"
    path.write_bytes(body)
    return path


def load_country_shapes(shapes_dir: Path = SHAPES_DIR) -> gpd.GeoDataFrame:
    """One (multi)polygon per ISO 3166-1 alpha-2 `country`, in Equal Earth
    coordinates. Overseas pieces that Natural Earth files under another
    country's code are merged into it."""
    shapes = gpd.read_file(fetch_country_shapes(shapes_dir))
    shapes = shapes.rename(columns={"ISO_A2_EH": "country"})[["country", "geometry"]]
    shapes = shapes[~shapes["country"].isin(_DROPPED)]
    return shapes.dissolve(by="country", as_index=False).to_crs(EQUAL_EARTH)
