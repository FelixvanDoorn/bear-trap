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


def plot_top_clusters_subgraph(
    graph: nx.Graph,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = 10,
    max_nodes: int = 150,
) -> Path | None:
    """Node-link plot of ONLY the induced subgraph for the top_n largest
    connected components -- never the full graph, which could be thousands
    of nodes. Re-derives components from `graph` directly since credential-
    pair node membership isn't captured by assign_clusters' IP-only output.
    If the induced subgraph itself still exceeds max_nodes, no file is
    written and a warning is printed instead of forcing a hairball onto
    disk -- max_nodes is a placeholder pending real cluster-size numbers
    from the notebook, not a load-bearing threshold."""
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
