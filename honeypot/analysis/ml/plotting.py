# honeypot/analysis/ml/plotting.py
#
# Chart styling shared by the ml experiments. Seaborn on matplotlib's
# headless Agg backend, styled per chart via style() like geo/charts.py,
# so importing this never restyles anything else. save() writes PNGs
# without a "Software" tag, so the same figure always gives the same
# bytes and a rerun of a run id reproduces its charts exactly.
#
# Color by job: identity (campaigns, models, ...) uses the fixed
# categorical slots in order, never cycled -- a ninth series is NEUTRAL
# "other". Three of the slots are below 3:1 contrast on white, so charts
# using them label marks directly or come with a table view. Magnitudes
# use the one sequential blue ramp.
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported, so a headless run never tries to
# initialize a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
NEUTRAL = "#c9c7c1"
INK = "#52514e"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)


@contextmanager
def style() -> Iterator[None]:
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        yield


def save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    # No "Software" tag: the PNG depends only on what's drawn.
    fig.savefig(path, bbox_inches="tight", dpi=110, metadata={"Software": None})
    plt.close(fig)
    return path
