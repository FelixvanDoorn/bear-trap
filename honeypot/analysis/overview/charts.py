# honeypot/analysis/overview/charts.py
#
# Seaborn on top of matplotlib's headless Agg backend. Styling is scoped
# per chart (`with _style():`) rather than set globally with
# sns.set_theme() at import, so importing this module never restyles
# another module's charts. Seaborn takes pandas, so polars frames are
# converted at the plotting boundary only.
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported, so a headless run never tries to
# initialize a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from honeypot.analysis.overview.summary import FUNNEL_STAGES  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs" / "charts"

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MAX_LABEL_LEN = 60

# Fixed series -> color order (categorical slots of the dataviz reference
# palette), so a role/protocol keeps its color across charts and runs.
# Anything unlisted (a future role) falls back to neutral gray rather than
# a generated hue.
_SERIES_COLORS: dict[str, str] = {
    "agent_detector/ssh": "#2a78d6",
    "agent_detector/telnet": "#eb6834",
    "control_proxy/ssh": "#1baf7a",
}
_FALLBACK_COLOR = "#888888"
_NEW_IPS_COLOR = "#eb6834"
_RETURNING_IPS_COLOR = "#2a78d6"


@contextmanager
def _style() -> Iterator[None]:
    """Seaborn's whitegrid look for one chart: light recessive gridlines,
    no heavy frame."""
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        yield


def _sanitize_label(value: str, max_len: int = _MAX_LABEL_LEN) -> str:
    """dst_ip is attacker-controlled (Cowrie logs whatever host the client
    asked to tunnel to), same reasoning as network.charts._sanitize_label
    -- strip control bytes and truncate rather than trust it as plain
    text."""
    cleaned = _CONTROL_CHARS.sub("�", value)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "…"
    return cleaned


def _funnel_long(funnel: pl.DataFrame) -> pl.DataFrame:
    """summary.session_funnel's wide table -> one row per (series, stage)
    with that stage's share of the series' sessions. Zero shares are
    dropped: they can't be drawn on a log axis, and "never happened" reads
    better as an absent bar than as a sliver."""
    return (
        funnel.with_columns(
            pl.concat_str("node_role", "protocol", separator="/").alias("series")
        )
        .unpivot(
            index=["series", "sessions"],
            on=FUNNEL_STAGES,
            variable_name="stage",
            value_name="count",
        )
        .with_columns((pl.col("count") / pl.col("sessions")).alias("share"))
        .filter(pl.col("share") > 0)
    )


def plot_session_funnel(funnel: pl.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """Grouped bars: per FUNNEL_STAGES step, the share of each
    (node_role, protocol)'s sessions that reached it -- see
    summary.session_funnel. Log-scaled, since file transfers and tunnel
    requests sit two to three orders of magnitude below logins and would
    otherwise be invisible. A zero share (e.g. telnet has no tunnels) is
    simply not drawn."""
    output_dir.mkdir(parents=True, exist_ok=True)

    long = _funnel_long(funnel)
    series_order = funnel.select(
        pl.concat_str("node_role", "protocol", separator="/")
    ).to_series()
    sessions = dict(
        zip(series_order.to_list(), funnel["sessions"].to_list(), strict=True)
    )

    with _style():
        fig, ax = plt.subplots(figsize=(10, 6))
        if long.height:
            sns.barplot(
                data=long.to_pandas(),
                saturation=1,
                x="share",
                y="stage",
                hue="series",
                order=FUNNEL_STAGES,
                hue_order=series_order.to_list(),
                palette={s: _SERIES_COLORS.get(s, _FALLBACK_COLOR) for s in sessions},
                orient="h",
                edgecolor="white",
                linewidth=1,
                ax=ax,
            )
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                [f"{label} ({sessions[label]:,} sessions)" for label in labels],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=len(labels),
                frameon=False,
            )
        ax.set_yticks(
            range(len(FUNNEL_STAGES)),
            [stage.replace("_", " ") for stage in FUNNEL_STAGES],
        )
        ax.set_xlabel("Share of sessions (log scale)")
        ax.set_ylabel("")
        ax.set_title("What sessions did, by sensor role and protocol")
        fig.tight_layout()

        path = output_dir / "session_funnel.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_daily_activity(daily: pl.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """Small multiples, one panel per node_role (each on its own y-scale --
    the roles differ ~10x in volume): stacked bars of distinct IPs per day,
    split into new (first seen anywhere in the dataset that day) vs
    returning. See summary.daily_activity. Stacked the usual seaborn way:
    the full-height bar (all IPs) in the "new" color, then the returning
    count drawn over it from the baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    roles = sorted(daily["node_role"].unique().to_list())
    n_panels = max(len(roles), 1)

    with _style():
        fig, grid = plt.subplots(
            n_panels, 1, figsize=(12, 3.5 * n_panels), squeeze=False
        )
        axes = grid[:, 0]

        for ax, role in zip(axes, roles):
            rows = (
                daily.filter(pl.col("node_role") == role)
                .sort("date")
                .with_columns(pl.col("date").dt.strftime("%m-%d").alias("day"))
                .to_pandas()
            )
            sns.barplot(
                data=rows,
                saturation=1,
                x="day",
                y="distinct_ips",
                color=_NEW_IPS_COLOR,
                ax=ax,
            )
            sns.barplot(
                data=rows,
                saturation=1,
                x="day",
                y="returning_ips",
                color=_RETURNING_IPS_COLOR,
                ax=ax,
            )
            ax.set_title(role)
            ax.set_xlabel("")
            ax.set_ylabel("Distinct IPs")
            ax.tick_params(axis="x", labelrotation=90, labelsize=8)
            ax.legend(
                handles=[
                    Patch(color=_RETURNING_IPS_COLOR, label="returning IPs"),
                    Patch(color=_NEW_IPS_COLOR, label="new IPs"),
                ],
                loc="upper left",
            )

        if not roles:
            axes[0].set_title("No full days of activity")
        fig.suptitle("Distinct source IPs per day (UTC, MM-DD), new vs returning")
        fig.tight_layout()

        path = output_dir / "daily_activity.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_tunnel_targets(
    targets: pl.DataFrame, output_dir: Path = OUTPUT_DIR, top_n: int = 15
) -> Path:
    """Bar chart of the most-requested tunnel (direct-tcpip) targets by
    request count, each labeled with how many distinct source IPs asked for
    it -- many IPs on one target reads as a campaign, many requests from
    one or two IPs as a single persistent actor. Expects `targets` from
    summary.tunnel_targets, already sorted descending."""
    output_dir.mkdir(parents=True, exist_ok=True)

    top = targets.head(top_n).with_columns(
        pl.concat_str("dst_ip", pl.col("dst_port").cast(pl.Utf8), separator=":")
        .map_elements(_sanitize_label, return_dtype=pl.Utf8)
        .alias("target")
    )

    with _style():
        fig, ax = plt.subplots(figsize=(10, 7))
        if top.height:
            sns.barplot(
                data=top.to_pandas(),
                saturation=1,
                x="requests",
                y="target",
                color=_SERIES_COLORS["agent_detector/ssh"],
                orient="h",
                ax=ax,
            )
            for label in ax.get_yticklabels():
                label.set_parse_math(False)
            ax.bar_label(
                ax.containers[0],
                labels=[f"{n} IPs" for n in top["distinct_ips"].to_list()],
                padding=3,
                fontsize=8,
            )
        ax.set_xlabel("Tunnel requests")
        ax.set_ylabel("")
        ax.set_title(
            f"Top {top.height} of {targets.height} tunnel targets "
            "(label: distinct source IPs)"
        )
        fig.tight_layout()

        path = output_dir / "tunnel_targets.png"
        fig.savefig(path)
        plt.close(fig)
    return path
