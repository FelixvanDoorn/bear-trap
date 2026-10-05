# honeypot/analysis/network/graph.py
#
# Builds a bipartite graph of src_ip <-> (username, password) credential
# pairs from raw session events, projects it onto an IP-only similarity
# graph (TF-IDF cosine over each IP's attempted pairs, plus a minimum
# shared-pair count), then clusters IPs via connected components over that
# projection -- a first-pass signal for shared campaign/botnet
# infrastructure. See README.md "Filtering" for why each stage exists.
import math
from collections import Counter
from itertools import combinations

import networkx as nx
import polars as pl

# A pair attempted by this many distinct src_ips or more is excluded from
# the graph -- it acts as a hub node that connected_components merges every
# attempting IP through, regardless of whether those IPs share anything
# else. Computed fresh from whatever `events` is passed in (see
# _credential_pair_frequency) rather than a fixed list of literal pairs: a
# prior version hardcoded a denylist derived from one snapshot, and it went
# stale the moment new data arrived -- pairs like (root, admin)/132 IPs and
# (support, support)/120 IPs were never added to it, leaving a 2,906-of
# -3,761-IP mega-cluster in a later pull despite the list "working" when it
# was written. 20 is carried over from that same original snapshot's
# frequency cliff as a starting default, not re-derived here.
DEFAULT_MIN_IPS_TO_EXCLUDE = 20


def _pair_attempts(events: pl.DataFrame) -> pl.DataFrame:
    """One row per distinct (src_ip, username, password) triple actually
    attempted -- rows missing either username or password (a partial
    attempt) excluded. Shared base for the graph itself, credential-pair
    frequency, and the username-targeting breakdown."""
    return (
        events.filter(
            pl.col("username").is_not_null() & pl.col("password").is_not_null()
        )
        .select("src_ip", "username", "password")
        .unique()
    )


def _credential_pair_frequency(attempts: pl.DataFrame) -> pl.DataFrame:
    """One row per distinct (username, password) pair, with distinct_ips =
    how many distinct src_ips attempted it. Sorted by distinct_ips
    descending."""
    return (
        attempts.group_by("username", "password")
        .agg(pl.col("src_ip").n_unique().alias("distinct_ips"))
        .sort("distinct_ips", descending=True)
    )


def build_credential_graph(
    events: pl.DataFrame,
    min_ips_to_exclude: int | None = DEFAULT_MIN_IPS_TO_EXCLUDE,
) -> nx.Graph:
    """Bipartite graph: one node per distinct src_ip, one node per distinct
    (username, password) pair, an edge between them for every src_ip that
    attempted that pair. Node ids are typed tuples -- ("ip", src_ip) /
    ("cred", username, password) -- rather than string-prefixed labels, so
    an attacker-controlled username/password value can never collide with a
    real IP string. Rows missing either username or password (a partial
    attempt) contribute no node or edge, and so does any pair attempted by
    >= min_ips_to_exclude distinct IPs (see DEFAULT_MIN_IPS_TO_EXCLUDE) --
    an IP whose every attempt is excluded or partial ends up absent from
    the graph entirely, not an isolated node. Pass min_ips_to_exclude=None
    to disable filtering."""
    attempts = _pair_attempts(events)

    exclude_pairs: set[tuple[str, str]] = set()
    if min_ips_to_exclude is not None:
        frequency = _credential_pair_frequency(attempts)
        excluded = frequency.filter(pl.col("distinct_ips") >= min_ips_to_exclude)
        exclude_pairs = set(
            zip(excluded["username"].to_list(), excluded["password"].to_list())
        )

    graph = nx.Graph()
    for src_ip, username, password in attempts.iter_rows():
        if (username, password) in exclude_pairs:
            continue
        ip_node = ("ip", src_ip)
        cred_node = ("cred", username, password)
        graph.add_edge(ip_node, cred_node)

    return graph


def generic_credential_attempts(
    events: pl.DataFrame,
    min_ips_to_exclude: int = DEFAULT_MIN_IPS_TO_EXCLUDE,
) -> pl.DataFrame:
    """The mirror image of build_credential_graph's filtering: one row per
    (username, password) pair attempted by >= min_ips_to_exclude distinct
    IPs, i.e. exactly what that call would exclude from the graph at the
    same threshold -- surfaced separately for review, not silently
    dropped. Sorted by distinct_ips descending."""
    attempts = _pair_attempts(events)
    frequency = _credential_pair_frequency(attempts)
    return frequency.filter(pl.col("distinct_ips") >= min_ips_to_exclude)


def username_targeting_breakdown(events: pl.DataFrame) -> pl.DataFrame:
    """One row per distinct username attempted alongside a non-null
    password, with distinct_ips = how many distinct src_ips tried it --
    independent of the credential-sharing graph entirely. A widely-known
    service (postgres, mysql, ...) gets probed with the same handful of
    obvious passwords by many unrelated actors, so at the pair level it
    looks like noise to connected_components for the same reason
    root/admin does (see build_credential_graph's exclusion) -- this view
    answers "who is looking for which app" directly, a question pair-level
    clustering can't answer for exactly that reason. Sorted by
    distinct_ips descending."""
    attempts = _pair_attempts(events)
    return (
        attempts.group_by("username")
        .agg(pl.col("src_ip").n_unique().alias("distinct_ips"))
        .sort("distinct_ips", descending=True)
    )


def _pair_idf(graph: nx.Graph) -> dict[tuple, float]:
    """Inverse-document-frequency weight per credential-pair node, treating
    each IP as a "document" and each pair it attempted as a "term": idf(p)
    = log(N / df(p)), N = distinct IPs in `graph`, df(p) = the pair's
    degree (how many of them attempted it). A pair every IP attempted gets
    idf 0 (contributes nothing); a pair only one or two IPs attempted gets
    a large positive weight. Every credential-pair node has degree >= 1 by
    construction (build_credential_graph only adds nodes with an edge), so
    df is never 0 and this never divides by zero."""
    ip_count = sum(1 for node in graph.nodes if node[0] == "ip")
    return {
        node: math.log(ip_count / graph.degree[node])
        for node in graph.nodes
        if node[0] == "cred"
    }


# build_ip_similarity_graph links two IPs only when BOTH thresholds below
# hold -- each alone fails in a different, measured way (2026-09-16
# snapshot, 3,401 clustered IPs):
#
# - DEFAULT_MIN_SIMILARITY: TF-IDF cosine similarity of the two IPs'
#   attempted-pair sets. Scores *how distinctive* the overlap is: a pair
#   shared by 19 IPs (barely under DEFAULT_MIN_IPS_TO_EXCLUDE) counts for
#   far less than one shared by just 2, and cosine normalizes away IPs that
#   simply try huge dictionaries. Used alone, though, 88% of its edges at
#   0.1 rested on a single shared pair -- two IPs with tiny dictionaries
#   that happen to share one rare pair score near 1.0 -- and those chained
#   into a 1,185-IP component. Raising the threshold doesn't help, since
#   those single-pair edges are exactly the highest-scoring ones.
# - DEFAULT_MIN_SHARED_PAIRS: a minimum *amount* of evidence. Used alone
#   (the earlier approach), it gave a 273-IP largest component -- but one
#   with density 0.022 and a median edge cosine of 0.048: big-dictionary
#   IPs with marginal overlap chaining a real core (top-decile edges share
#   75+ pairs at cosine > 0.7) to everything around it.
#
# Combined, the largest component drops to 23 IPs (94% singletons), and
# the result is much less sensitive to the cosine threshold (0.1 -> 0.5
# moves the largest component 23 -> 19, vs 1,185 -> 321 for cosine alone).
# Trade-off: connected_components is still single-linkage, and this may
# over-split some real campaigns -- community detection (e.g. Louvain) on
# the weighted graph is the natural next step if that shows up.
DEFAULT_MIN_SIMILARITY = 0.1
DEFAULT_MIN_SHARED_PAIRS = 2


def build_ip_similarity_graph(
    graph: nx.Graph,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_shared_pairs: int = DEFAULT_MIN_SHARED_PAIRS,
) -> nx.Graph:
    """Projects the bipartite credential-sharing graph (see
    build_credential_graph) onto IPs alone: one node per distinct src_ip,
    an edge between two IPs weighted by the TF-IDF cosine similarity of
    their attempted credential-pair sets (see _pair_idf), kept only when
    that similarity is >= min_similarity AND the two IPs share >=
    min_shared_pairs credential pairs (see DEFAULT_MIN_SHARED_PAIRS for why
    both). Each edge also carries the raw count as `shared_pairs`. Every IP
    present in `graph` appears here too, even with no qualifying edges (an
    isolated node, not absent) -- so assign_clusters still reports it as
    its own singleton cluster. Accumulates each credential-pair node's
    contribution across its neighbor pairs rather than comparing every IP
    pair directly, since most credential-pair nodes have very few
    neighbors -- cheap in practice even though a handful of higher-degree
    nodes exist."""
    idf = _pair_idf(graph)
    ip_nodes = [node for node in graph.nodes if node[0] == "ip"]
    cred_nodes = [node for node in graph.nodes if node[0] == "cred"]

    norms = {
        ip_node: math.sqrt(sum(idf[cred] ** 2 for cred in graph.neighbors(ip_node)))
        for ip_node in ip_nodes
    }

    dot_products: Counter = Counter()
    shared_counts: Counter = Counter()
    for cred_node in cred_nodes:
        weight_sq = idf[cred_node] ** 2
        neighbors = sorted(graph.neighbors(cred_node))
        for ip_a, ip_b in combinations(neighbors, 2):
            dot_products[(ip_a, ip_b)] += weight_sq
            shared_counts[(ip_a, ip_b)] += 1

    similarity = nx.Graph()
    similarity.add_nodes_from(ip_nodes)
    for (ip_a, ip_b), dot in dot_products.items():
        shared = shared_counts[(ip_a, ip_b)]
        if shared < min_shared_pairs:
            continue
        denom = norms[ip_a] * norms[ip_b]
        score = dot / denom if denom else 0.0
        if score > 0 and score >= min_similarity:
            similarity.add_edge(ip_a, ip_b, weight=score, shared_pairs=shared)

    return similarity


def assign_clusters(graph: nx.Graph) -> pl.DataFrame:
    """Runs nx.connected_components over `graph` and returns one row per IP
    node. Normally called on build_ip_similarity_graph's IP-only output;
    also accepts the raw bipartite graph, where credential-pair nodes are
    skipped as graph-internal plumbing rather than reported as rows.
    cluster_id is assigned by component size descending, tie-broken by the
    component's minimum src_ip ascending, so results are reproducible
    regardless of set/dict iteration order."""
    components = []
    for component in nx.connected_components(graph):
        ip_addresses = sorted(node[1] for node in component if node[0] == "ip")
        if ip_addresses:
            components.append(ip_addresses)

    components.sort(key=lambda ips: (-len(ips), ips[0]))

    rows = [
        {"src_ip": src_ip, "cluster_id": cluster_id, "cluster_size": len(ip_addresses)}
        for cluster_id, ip_addresses in enumerate(components)
        for src_ip in ip_addresses
    ]

    return pl.DataFrame(
        rows,
        schema={"src_ip": pl.Utf8, "cluster_id": pl.Int64, "cluster_size": pl.Int64},
    )
