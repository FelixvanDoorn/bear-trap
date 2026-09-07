# honeypot/analysis/network/graph.py
#
# Builds a bipartite graph of src_ip <-> (username, password) credential
# pairs from raw session events, then clusters IPs via connected components
# -- IPs that ever attempted the same credential pair end up in the same
# cluster, a first-pass signal for shared campaign/botnet infrastructure.
from dataclasses import dataclass

import networkx as nx
import polars as pl


@dataclass
class ClusterAssignment:
    src_ip: str
    cluster_id: int
    cluster_size: int


# Credential pairs attempted by >=20 distinct src_ips in the 2026-08-29
# offline snapshot (honeypot/analysis/network/notebooks/eda.ipynb) -- the
# frequency distribution has a real cliff right here: 460 pairs at >=15
# distinct IPs drops to 60 at >=20. Without excluding these, each one acts
# as a hub node that connected_components merges every attempting IP
# through, regardless of whether those IPs share anything else -- this is
# exactly what produced a single 690-IP cluster out of 883 total IPs before
# this list existed. Includes both genuinely generic default credentials
# (admin/admin, root/123456, ...) and a handful of SIP/HTTP protocol-probe
# artifacts that land in the username/password columns from non-SSH/Telnet
# traffic (Cowrie has no dedicated field for those) -- neither carries real
# credential-sharing signal, so both get excluded. A manual, data-derived
# list rather than a frequency computed at graph-build time, so cluster
# membership stays stable/reproducible across runs regardless of what a
# given snapshot's live distribution looks like.
_GENERIC_CREDENTIAL_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("admin", "admin"),
        ("root", "root"),
        ("root", "123456"),
        ("root", "12345"),
        ("root", "password"),
        ("root", "1234"),
        ("root", "admin"),
        ("support", "support"),
        ("guest", "guest"),
        ("root", "vizxv"),
        ("admin", "1234"),
        ("", ""),
        ("user", "user"),
        ("admin", "123456"),
        ("root", "root123"),
        ("root", "123"),
        ("admin", "password"),
        ("root", "123456789"),
        ("admin", "admin123"),
        ("admin", "admin1234"),
        ("root", "111111"),
        ("root", "12345678"),
        ("root", "admin123"),
        ("root", "1234567890"),
        ("Call-ID: 50000", "CSeq: 42 OPTIONS"),
        ("root", "pass"),
        ("Max-Forwards: 70", "Content-Length: 0"),
        ("GET / HTTP/1.0", ""),
        ("Contact: <sip:nm@nm>", "Accept: application/sdp"),
        ("From: <sip:nm@nm>;tag=root", "To: <sip:nm2@nm2>"),
        ("root", "123123"),
        ("OPTIONS sip:nm SIP/2.0", "Via: SIP/2.0/TCP nm;branch=foo"),
        ("root", "ubuntu"),
        ("root", "1q2w3e4r"),
        ("root", "123321"),
        ("root", ""),
        ("root", "abc123"),
        ("root", "P@ssw0rd"),
        ("root", "000000"),
        ("root", "passw0rd"),
        ("root", "xc3511"),
        ("root", "qwerty"),
        ("root", "qwerty123"),
        ("nobody", "nobody"),
        ("admin", "123456789"),
        ("root", "1qaz2wsx"),
        ("root", "P@ssw0rd123"),
        ("default", "default"),
        ("root", "Passw0rd"),
        ("ubuntu", "ubuntu"),
        ("root", "1qaz@WSX"),
        ("test", "test"),
        ("user", "1234"),
        ("root", "7ujMko0admin"),
        ("admin", "abc123"),
        ("root", "Zte521"),
        ("root", "zlxx."),
        ("admin", "12345"),
        ("admin", ""),
        ("postgres", "123"),
    }
)


def build_credential_graph(
    events: pl.DataFrame,
    exclude_pairs: frozenset[tuple[str, str]] = _GENERIC_CREDENTIAL_PAIRS,
) -> nx.Graph:
    """Bipartite graph: one node per distinct src_ip, one node per distinct
    (username, password) pair, an edge between them for every src_ip that
    attempted that pair. Node ids are typed tuples -- ("ip", src_ip) /
    ("cred", username, password) -- rather than string-prefixed labels, so
    an attacker-controlled username/password value can never collide with a
    real IP string. Rows missing either username or password (a partial
    attempt) contribute no node or edge, and so does any pair in
    `exclude_pairs` (see _GENERIC_CREDENTIAL_PAIRS) -- an IP whose every
    attempt is excluded or partial ends up absent from the graph entirely,
    not an isolated node. Pass exclude_pairs=frozenset() to disable
    filtering."""
    graph = nx.Graph()
    attempts = (
        events.filter(
            pl.col("username").is_not_null() & pl.col("password").is_not_null()
        )
        .select("src_ip", "username", "password")
        .unique()
    )

    for src_ip, username, password in attempts.iter_rows():
        if (username, password) in exclude_pairs:
            continue
        ip_node = ("ip", src_ip)
        cred_node = ("cred", username, password)
        graph.add_edge(ip_node, cred_node)

    return graph


def generic_credential_attempts(
    events: pl.DataFrame,
    exclude_pairs: frozenset[tuple[str, str]] = _GENERIC_CREDENTIAL_PAIRS,
) -> pl.DataFrame:
    """The mirror image of build_credential_graph's filtering: one row per
    denylisted (username, password) pair that was actually attempted, with
    distinct_ips = how many distinct src_ips tried it -- the same signal
    used to derive _GENERIC_CREDENTIAL_PAIRS in the first place. For
    surfacing what build_credential_graph excludes, not for graph
    construction; sorted by distinct_ips descending."""
    attempts = (
        events.filter(
            pl.col("username").is_not_null() & pl.col("password").is_not_null()
        )
        .select("src_ip", "username", "password")
        .unique()
    )

    rows = [
        {"src_ip": src_ip, "username": username, "password": password}
        for src_ip, username, password in attempts.iter_rows()
        if (username, password) in exclude_pairs
    ]
    filtered = pl.DataFrame(
        rows, schema={"src_ip": pl.Utf8, "username": pl.Utf8, "password": pl.Utf8}
    )

    return (
        filtered.group_by("username", "password")
        .agg(pl.col("src_ip").n_unique().alias("distinct_ips"))
        .sort("distinct_ips", descending=True)
    )


def assign_clusters(graph: nx.Graph) -> pl.DataFrame:
    """Runs nx.connected_components over the bipartite graph and returns one
    row per IP node (credential-pair nodes are graph-internal plumbing, not
    part of the deliverable). cluster_id is assigned by component size
    descending, tied-broken by the component's minimum src_ip ascending, so
    results are reproducible regardless of set/dict iteration order."""
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
