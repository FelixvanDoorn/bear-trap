# honeypot/analysis/network/charts.py
#
# The older charts here are plain matplotlib; the AI-probe contrast charts
# (bottom of the file) use seaborn, styled per chart via _style() like
# overview/charts.py, so importing this module never restyles anything
# else.
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported, so a headless run never tries to
# initialize a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import polars as pl  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from honeypot.analysis.network.ai_contrast import GROUP_COLUMN  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs" / "charts"

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MAX_LABEL_LEN = 60


def _sanitize_label(value: str, max_len: int = _MAX_LABEL_LEN) -> str:
    """src_ip/credential values are attacker-controlled, same reasoning as
    fingerprinting.charts._sanitize_label -- strip control bytes and
    truncate rather than trust them as plain text."""
    cleaned = _CONTROL_CHARS.sub("�", value)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "…"
    return cleaned


def plot_cluster_size_distribution(
    clusters: pl.DataFrame, output_dir: Path = OUTPUT_DIR, top_n: int = 20
) -> Path:
    """Bar chart of the largest clusters by IP count; singleton-cluster
    count is summarized in the title rather than plotted individually --
    same shape as fingerprinting.charts.plot_command_sequence_clusters."""
    output_dir.mkdir(parents=True, exist_ok=True)

    by_cluster = (
        clusters.group_by("cluster_id")
        .agg(pl.len().alias("count"), pl.col("cluster_size").first())
        .sort("cluster_size", descending=True)
    )

    n_clusters = by_cluster.height
    n_singletons = by_cluster.filter(pl.col("cluster_size") == 1).height
    n_ips = int(by_cluster["count"].sum()) if n_clusters else 0

    top = by_cluster.head(top_n)
    labels = [f"cluster {cid}" for cid in top["cluster_id"].to_list()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, top["cluster_size"].to_list())
    ax.invert_yaxis()
    ax.set_xlabel("IP count")
    ax.set_title(
        f"Top {top.height} credential-sharing clusters of {n_clusters} total "
        f"across {n_ips} IPs ({n_singletons} singletons)"
    )
    fig.tight_layout()

    path = output_dir / "credential_cluster_sizes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_generic_credential_breakdown(
    attempts: pl.DataFrame, output_dir: Path = OUTPUT_DIR, top_n: int = 20
) -> Path:
    """Bar chart of the denylisted credential pairs actually attempted (see
    graph.generic_credential_attempts), by distinct-IP count -- surfaces
    what build_credential_graph excludes from the graph rather than
    silently dropping it. Expects `attempts` with `username`/`password`/
    `distinct_ips` columns, already sorted descending."""
    output_dir.mkdir(parents=True, exist_ok=True)

    top = attempts.head(top_n)
    labels = [
        _sanitize_label(f"{username}/{password}")
        for username, password in zip(
            top["username"].to_list(), top["password"].to_list()
        )
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels:
        ax.barh(labels, top["distinct_ips"].to_list())
        for label in ax.get_yticklabels():
            label.set_parse_math(False)
        ax.invert_yaxis()
    ax.set_xlabel("Distinct IP count")
    ax.set_title(
        f"Top {top.height} of {attempts.height} generic credential pairs "
        "excluded from the graph"
    )
    fig.tight_layout()

    path = output_dir / "generic_credential_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_username_targeting_breakdown(
    breakdown: pl.DataFrame, output_dir: Path = OUTPUT_DIR, top_n: int = 20
) -> Path:
    """Bar chart of the most-attempted usernames by distinct IP count --
    independent of the credential-sharing graph, see
    graph.username_targeting_breakdown. Expects `breakdown` with
    username/distinct_ips columns, already sorted descending."""
    output_dir.mkdir(parents=True, exist_ok=True)

    top = breakdown.head(top_n)
    labels = [_sanitize_label(u) for u in top["username"].to_list()]

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels:
        ax.barh(labels, top["distinct_ips"].to_list())
        for label in ax.get_yticklabels():
            label.set_parse_math(False)
        ax.invert_yaxis()
    ax.set_xlabel("Distinct IP count")
    ax.set_title(f"Top {top.height} of {breakdown.height} usernames attempted")
    fig.tight_layout()

    path = output_dir / "username_targeting_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# Fixed category -> color order (categorical slots 1-5 of the dataviz
# reference palette), so a category keeps its color across runs no matter
# which categories a given snapshot happens to contain.
_AI_CATEGORY_COLORS: dict[str, str] = {
    "agent_tooling": "#2a78d6",
    "model_vendor": "#eb6834",
    "llm_runtime": "#1baf7a",
    "ml_infra": "#eda100",
    "generic_ai": "#e87ba4",
}


def plot_ai_username_targeting(
    breakdown: pl.DataFrame, output_dir: Path = OUTPUT_DIR, top_n: int = 25
) -> Path:
    """Bar chart of the most-attempted AI-tooling usernames by distinct IP
    count, colored by category -- see ai_targeting.ai_username_targeting.
    Expects `breakdown` with username/category/distinct_ips columns,
    already sorted descending."""
    output_dir.mkdir(parents=True, exist_ok=True)

    top = breakdown.head(top_n)
    labels = [_sanitize_label(u) for u in top["username"].to_list()]
    categories = top["category"].to_list()

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels:
        ax.barh(
            labels,
            top["distinct_ips"].to_list(),
            color=[_AI_CATEGORY_COLORS.get(c, "#888888") for c in categories],
        )
        for label in ax.get_yticklabels():
            label.set_parse_math(False)
        ax.invert_yaxis()
        present = [c for c in _AI_CATEGORY_COLORS if c in set(categories)]
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=_AI_CATEGORY_COLORS[c]) for c in present
        ]
        ax.legend(handles, present, title="Category", loc="lower right")
    ax.set_xlabel("Distinct IP count")
    ax.set_title(
        f"Top {top.height} of {breakdown.height} AI-tooling usernames attempted"
    )
    fig.tight_layout()

    path = output_dir / "ai_username_targeting.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_top_clusters_subgraph(
    graph: nx.Graph,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = 10,
    max_nodes: int = 150,
) -> Path | None:
    """Node-link plot of ONLY the induced subgraph for the top_n largest
    connected components -- never the full graph, which could be thousands
    of nodes. Normally passed graph.build_ip_similarity_graph's IP-only
    output (which pipeline.py and the notebook do), so every node is an
    IP; also labels credential-pair nodes if given the raw bipartite graph.
    Re-derives components from `graph` directly rather than taking
    assign_clusters' output, since it needs the edges, not just
    membership. If the induced subgraph still exceeds max_nodes, no file
    is written and a warning is printed instead of forcing a hairball onto
    disk. On the 2026-09-16 snapshot the default top 10 clusters span 148
    nodes -- just under max_nodes, so a modestly bigger snapshot may tip
    this into skipping again."""
    components = sorted(
        nx.connected_components(graph), key=lambda c: len(c), reverse=True
    )
    top_nodes: set = set()
    for component in components[:top_n]:
        top_nodes |= component

    if not top_nodes:
        print("plot_top_clusters_subgraph: no clusters to plot, skipping.")
        return None

    if len(top_nodes) > max_nodes:
        print(
            f"plot_top_clusters_subgraph: top {top_n} clusters span "
            f"{len(top_nodes)} nodes, exceeding max_nodes={max_nodes}. "
            "Skipping to avoid an unreadable plot."
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    subgraph = graph.subgraph(top_nodes)

    colors = ["#4C72B0" if node[0] == "ip" else "#DD8452" for node in subgraph.nodes]
    labels = {
        node: _sanitize_label(node[1] if node[0] == "ip" else f"{node[1]}:{node[2]}")
        for node in subgraph.nodes
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    layout = nx.spring_layout(subgraph, seed=0)
    nx.draw_networkx(
        subgraph,
        pos=layout,
        labels=labels,
        node_color=colors,
        font_size=7,
        node_size=200,
        ax=ax,
    )
    ax.set_title(f"Top {min(top_n, len(components))} credential-sharing clusters")
    ax.axis("off")
    fig.tight_layout()

    path = output_dir / "top_clusters_subgraph.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --- AI-probe contrast (ai_contrast.py) -----------------------------------

# Fixed group -> color (categorical slots 1-2 of the dataviz reference
# palette). Also the scenario order for plot_ai_probe_effect_sizes: the
# first comparison passed gets slot 1, the second slot 2.
_GROUP_LABELS = {True: "AI probe", False: "matched other"}
_GROUP_COLORS = {"AI probe": "#2a78d6", "matched other": "#eb6834"}
_SCENARIO_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
_SIGNIFICANCE_MARKERS = {"q < alpha (BH)": "o", "not significant": "X"}
_EFFECT_PANELS = [
    ("categorical", "Cramér's V (0 = no association)"),
    ("numeric", "rank-biserial r (> 0: AI probes higher)"),
]


@contextmanager
def _style() -> Iterator[None]:
    """Seaborn's whitegrid look for one chart: light recessive gridlines,
    no heavy frame."""
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        yield


def _effect_long(comparisons: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """stats.compare_groups outputs, one per scenario label -> one row per
    (scenario, property) with a significance label. Untestable rows have
    no effect size to draw and are dropped."""
    return pl.concat(
        [
            comparison.filter(pl.col("effect_size").is_not_null())
            .select("property", "kind", "effect_size", "significant_bh")
            .with_columns(
                pl.lit(label).alias("scenario"),
                pl.when(pl.col("significant_bh"))
                .then(pl.lit("q < alpha (BH)"))
                .otherwise(pl.lit("not significant"))
                .alias("significance"),
            )
            for label, comparison in comparisons.items()
        ]
    )


def plot_ai_probe_effect_sizes(
    comparisons: dict[str, pl.DataFrame],
    output_dir: Path = OUTPUT_DIR,
    filename: str = "ai_probe_effect_sizes.png",
) -> Path:
    """Dot plot of every property's effect size from one or more
    ai_contrast.compare_ai_probes results (scenario label -> result, e.g.
    {"all IPs": ..., "excluding AS197170": ...}). Two panels, since the
    measures differ: Cramér's V for categoricals, signed rank-biserial r
    for numerics. Color = scenario, marker = BH significance (so it never
    rests on color alone), and a thin line joins one property's scenarios
    so a collapsing effect is visible as a long bar. Properties are
    ordered by the first scenario's |effect size|."""
    output_dir.mkdir(parents=True, exist_ok=True)
    long = _effect_long(comparisons)
    scenarios = list(comparisons)
    palette = dict(zip(scenarios, _SCENARIO_COLORS))
    first = long.filter(pl.col("scenario") == scenarios[0])

    panels = [
        (kind, xlabel, long.filter(pl.col("kind") == kind))
        for kind, xlabel in _EFFECT_PANELS
    ]
    panels = [panel for panel in panels if panel[2].height]
    heights = [max(panel[2]["property"].n_unique(), 1) for panel in panels] or [1]

    with _style():
        fig, axes = plt.subplots(
            len(heights),
            1,
            figsize=(9, 1.5 + 0.42 * sum(heights)),
            gridspec_kw={"height_ratios": heights},
            squeeze=False,
        )
        for ax, (kind, xlabel, data) in zip(axes[:, 0], panels):
            order = (
                first.filter(pl.col("kind") == kind)
                .sort(pl.col("effect_size").abs(), descending=True)["property"]
                .to_list()
            )
            order += [p for p in data["property"].unique().sort() if p not in order]
            rank = {prop: i for i, prop in enumerate(order)}
            spans = data.group_by("property").agg(
                pl.col("effect_size").min().alias("lo"),
                pl.col("effect_size").max().alias("hi"),
            )
            ax.hlines(
                [rank[p] for p in spans["property"]],
                spans["lo"],
                spans["hi"],
                color="#c3c2b7",
                linewidth=2,
                zorder=1,
            )
            sns.scatterplot(
                data=data.to_pandas(),
                x="effect_size",
                y=data["property"].replace_strict(rank).to_list(),
                hue="scenario",
                hue_order=scenarios,
                palette=palette,
                style="significance",
                style_order=list(_SIGNIFICANCE_MARKERS),
                markers=_SIGNIFICANCE_MARKERS,
                s=90,
                edgecolor="white",
                linewidth=1,
                zorder=2,
                ax=ax,
                legend="auto" if ax is axes[0, 0] else False,
            )
            ax.set_yticks(range(len(order)), order)
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.set_xlim((0, 1) if kind == "categorical" else (-1, 1))
            if kind == "numeric":
                ax.axvline(0, color="#52514e", linewidth=1, zorder=0)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("")
        if axes[0, 0].get_legend() is not None:
            sns.move_legend(
                axes[0, 0], "upper left", bbox_to_anchor=(1.01, 1), frameon=False
            )
        axes[0, 0].set_title("AI probes vs size-matched IPs: effect size per property")
        fig.tight_layout()

        path = output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def plot_ai_probe_breakdown(
    breakdown: pl.DataFrame,
    column: str,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = 12,
) -> Path:
    """Grouped bars of the share of each group's IPs per level of one
    categorical property -- stats.category_breakdown output (already
    collapsed and sorted most-over-represented-in-AI-probes first), so the
    levels drawn are exactly the ones compare_ai_probes tested. Shares, not
    counts: the groups differ ~5x in size."""
    output_dir.mkdir(parents=True, exist_ok=True)
    top = breakdown.head(top_n).with_columns(
        pl.col("level").map_elements(_sanitize_label, return_dtype=pl.Utf8)
    )
    long = top.unpivot(
        index="level",
        on=["share_a", "share_b"],
        variable_name="group",
        value_name="share",
    ).with_columns(
        pl.col("group").replace_strict(
            {"share_a": _GROUP_LABELS[True], "share_b": _GROUP_LABELS[False]}
        )
    )

    with _style():
        fig, ax = plt.subplots(figsize=(10, 1.5 + 0.55 * max(top.height, 1)))
        if top.height:
            sns.barplot(
                data=long.to_pandas(),
                x="share",
                y="level",
                hue="group",
                hue_order=list(_GROUP_COLORS),
                palette=_GROUP_COLORS,
                order=top["level"].to_list(),
                edgecolor="white",
                linewidth=2,
                ax=ax,
            )
            for label in ax.get_yticklabels():
                label.set_parse_math(False)
            sns.move_legend(
                ax, "upper left", bbox_to_anchor=(1.01, 1), title=None, frameon=False
            )
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Share of the group's IPs")
        ax.set_ylabel("")
        ax.set_title(f"{column}: AI probes vs size-matched IPs")
        fig.tight_layout()

        path = output_dir / f"ai_probe_breakdown_{column}.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_ai_probe_distributions(
    profiles: pl.DataFrame,
    columns: list[str],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """One box plot per numeric property (small multiples, each on its own
    symlog x-axis -- these are heavy-tailed counts with zeros), AI probes
    vs size-matched IPs. ai_contrast.ai_probe_profiles output; nulls are
    left out per property, same as the Mann-Whitney tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = profiles.with_columns(
        pl.col(GROUP_COLUMN).replace_strict(_GROUP_LABELS).alias("group")
    )

    with _style():
        fig, axes = plt.subplots(
            len(columns), 1, figsize=(9, 1.2 + 1.3 * len(columns)), squeeze=False
        )
        for ax, column in zip(axes[:, 0], columns):
            values = data.filter(pl.col(column).is_not_null())
            if values.height:
                sns.boxplot(
                    data=values.select("group", column).to_pandas(),
                    x=column,
                    y="group",
                    hue="group",
                    order=list(_GROUP_COLORS),
                    hue_order=list(_GROUP_COLORS),
                    palette=_GROUP_COLORS,
                    width=0.6,
                    linewidth=1,
                    fliersize=3,
                    ax=ax,
                )
                ax.set_xscale("symlog", linthresh=1)
                # Every property is >= 0; symlog would otherwise autoscale
                # a mirrored, meaningless negative half.
                ax.set_xlim(left=0)
            ax.set_ylabel("")
            ax.set_xlabel(column)
        axes[0, 0].set_title("Numeric properties: AI probes vs size-matched IPs")
        fig.tight_layout()

        path = output_dir / "ai_probe_distributions.png"
        fig.savefig(path)
        plt.close(fig)
    return path
