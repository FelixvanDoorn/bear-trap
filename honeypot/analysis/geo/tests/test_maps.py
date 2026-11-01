# honeypot/analysis/geo/tests/test_maps.py
from pathlib import Path

from pytest_mock import MockerFixture

from honeypot.analysis.geo.maps import (
    EQUAL_EARTH,
    fetch_country_shapes,
    load_country_shapes,
)
from honeypot.analysis.geo.tests.fixtures import WORLD_COUNTRIES, write_world_shapes


def test_cached_shapes_are_used_without_downloading(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    download = mocker.patch("urllib.request.urlopen")
    write_world_shapes(tmp_path)

    assert fetch_country_shapes(tmp_path).suffix == ".geojson"
    download.assert_not_called()


def test_load_drops_placeholder_codes_and_projects(tmp_path: Path) -> None:
    write_world_shapes(tmp_path)

    shapes = load_country_shapes(tmp_path)

    # The synthetic world's 17th country happens to be coded AQ, which is
    # dropped as Antarctica along with the "-99" placeholder.
    assert len(shapes) == WORLD_COUNTRIES - 1
    assert not {"-99", "AQ"} & set(shapes["country"])
    assert shapes.crs.to_string() == EQUAL_EARTH
