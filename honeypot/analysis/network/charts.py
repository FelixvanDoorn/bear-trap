# honeypot/analysis/network/charts.py
import re
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported, so a headless run never tries to
# initialize a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import polars as pl  # noqa: E402

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
