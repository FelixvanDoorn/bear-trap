# honeypot/analysis/network/tests/test_graph.py
import math

import networkx as nx
import polars as pl
import pytest

from honeypot.analysis.network.graph import (
    DEFAULT_MIN_SHARED_PAIRS,
    assign_clusters,
    build_credential_graph,
    build_ip_similarity_graph,
    generic_credential_attempts,
    username_targeting_breakdown,
)


def _attempt(src_ip: str, username: str | None, password: str | None) -> dict:
    return {"src_ip": src_ip, "username": username, "password": password}


class TestBuildCredentialGraph:
    def test_two_ips_sharing_a_pair_are_linked_via_a_shared_cred_node(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "root", "toor"),
            ]
        )
        graph = build_credential_graph(events)

        cred_node = ("cred", "root", "toor")
        assert graph.has_edge(("ip", "1.1.1.1"), cred_node)
        assert graph.has_edge(("ip", "2.2.2.2"), cred_node)

    def test_partial_attempt_missing_username_creates_no_node(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", None, "toor")])
        graph = build_credential_graph(events)

        assert ("ip", "1.1.1.1") not in graph

    def test_partial_attempt_missing_password_creates_no_node(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "root", None)])
        graph = build_credential_graph(events)

        assert ("ip", "1.1.1.1") not in graph

    def test_duplicate_attempts_do_not_create_duplicate_edges(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("1.1.1.1", "root", "toor"),
            ]
        )
        graph = build_credential_graph(events)

        assert graph.number_of_edges() == 1

    def test_ip_never_present_with_both_fields_is_absent_from_graph(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", None),
                _attempt("1.1.1.1", None, "toor"),
            ]
        )
        graph = build_credential_graph(events)

        assert ("ip", "1.1.1.1") not in graph
        assert graph.number_of_nodes() == 0


class TestFrequencyBasedFiltering:
    def test_pair_at_threshold_is_excluded(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "root", "toor"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=2)

        assert ("cred", "root", "toor") not in graph
        assert graph.number_of_edges() == 0

    def test_pair_below_threshold_is_unaffected(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "root", "toor"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=3)

        assert graph.has_edge(("ip", "1.1.1.1"), ("cred", "root", "toor"))
        assert graph.has_edge(("ip", "2.2.2.2"), ("cred", "root", "toor"))

    def test_ip_whose_only_attempt_is_excluded_is_absent_from_graph(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "admin", "admin"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=2)

        assert ("ip", "1.1.1.1") not in graph
        assert graph.number_of_nodes() == 0

    def test_filtering_can_be_disabled(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "admin", "admin"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)

        assert graph.has_edge(("ip", "1.1.1.1"), ("cred", "admin", "admin"))
        assert graph.has_edge(("ip", "2.2.2.2"), ("cred", "admin", "admin"))

    def test_default_threshold_is_twenty(self) -> None:
        below = pl.DataFrame(
            [_attempt(f"10.0.0.{i}", "admin", "admin") for i in range(19)]
        )
        assert ("cred", "admin", "admin") in build_credential_graph(below)

        at_threshold = pl.DataFrame(
            [_attempt(f"10.0.0.{i}", "admin", "admin") for i in range(20)]
        )
        assert ("cred", "admin", "admin") not in build_credential_graph(at_threshold)

    def test_excluded_pair_appears_in_generic_credential_attempts(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "admin", "admin"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=2)
        generic = generic_credential_attempts(events, min_ips_to_exclude=2)

        assert ("cred", "admin", "admin") not in graph
        assert generic["username"].to_list() == ["admin"]


class TestGenericCredentialAttempts:
    def test_counts_distinct_ips_per_pair_at_or_above_threshold(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "admin", "admin"),
                _attempt("3.3.3.3", "admin", "admin"),
            ]
        )
        result = generic_credential_attempts(events, min_ips_to_exclude=3)

        row = result.row(0, named=True)
        assert row["username"] == "admin"
        assert row["password"] == "admin"
        assert row["distinct_ips"] == 3

    def test_excludes_pairs_below_threshold(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "admin", "admin"),
                _attempt("3.3.3.3", "root", "toor"),
            ]
        )
        result = generic_credential_attempts(events, min_ips_to_exclude=2)

        assert result["username"].to_list() == ["admin"]

    def test_excludes_partial_attempts(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "admin", None)])
        result = generic_credential_attempts(events, min_ips_to_exclude=1)

        assert result.height == 0

    def test_no_pairs_at_or_above_threshold_returns_empty_dataframe_with_correct_schema(
        self,
    ) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "root", "toor")])
        result = generic_credential_attempts(events, min_ips_to_exclude=2)

        assert result.height == 0
        assert result.schema["username"] == pl.Utf8
        assert result.schema["password"] == pl.Utf8
        assert result.schema["distinct_ips"] == pl.UInt32

    def test_sorted_by_distinct_ips_descending(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "root"),
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "admin", "admin"),
            ]
        )
        result = generic_credential_attempts(events, min_ips_to_exclude=1)

        assert result["username"].to_list() == ["admin", "root"]
        assert result["distinct_ips"].to_list() == [2, 1]

    def test_default_threshold_is_twenty(self) -> None:
        events = pl.DataFrame(
            [_attempt(f"10.0.0.{i}", "admin", "admin") for i in range(20)]
        )
        result = generic_credential_attempts(events)

        assert result["username"].to_list() == ["admin"]


class TestUsernameTargetingBreakdown:
    def test_counts_distinct_ips_per_username(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "postgres", "123"),
                _attempt("2.2.2.2", "postgres", "postgres123"),
                _attempt("3.3.3.3", "root", "toor"),
            ]
        )
        result = username_targeting_breakdown(events)

        row = result.row(0, named=True)
        assert row["username"] == "postgres"
        assert row["distinct_ips"] == 2

    def test_same_ip_multiple_passwords_for_one_username_counts_once(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "postgres", "123"),
                _attempt("1.1.1.1", "postgres", "postgres123"),
            ]
        )
        result = username_targeting_breakdown(events)

        assert result.row(0, named=True)["distinct_ips"] == 1

    def test_excludes_partial_attempts(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "postgres", None)])
        result = username_targeting_breakdown(events)

        assert result.height == 0

    def test_sorted_by_distinct_ips_descending(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("1.1.1.1", "postgres", "123"),
                _attempt("2.2.2.2", "postgres", "postgres123"),
            ]
        )
        result = username_targeting_breakdown(events)

        assert result["username"].to_list() == ["postgres", "root"]
        assert result["distinct_ips"].to_list() == [2, 1]

    def test_not_affected_by_pair_level_exclusion(self) -> None:
        # Same username, a different password every time -- no single pair
        # crosses any pair-level threshold, but the username itself is
        # still clearly a popular target and should show up as such.
        events = pl.DataFrame(
            [_attempt(f"10.0.0.{i}", "postgres", f"pw{i}") for i in range(25)]
        )
        result = username_targeting_breakdown(events)

        assert result.row(0, named=True)["distinct_ips"] == 25


class TestBuildIpSimilarityGraph:
    # Most tests here pass min_shared_pairs=1 to isolate the cosine-similarity
    # behavior; the min_shared_pairs tests at the end of the class cover
    # that threshold and its interaction with min_similarity.
    def test_ubiquitous_pair_contributes_no_similarity(self) -> None:
        # The only pair here is shared by every IP in the dataset (df ==
        # ip_count), so its idf -- and therefore the resulting cosine
        # similarity -- is exactly 0. No edge, even with min_similarity=0.
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "root", "toor"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)
        similarity = build_ip_similarity_graph(
            graph, min_similarity=0.0, min_shared_pairs=1
        )

        assert not similarity.has_edge(("ip", "1.1.1.1"), ("ip", "2.2.2.2"))

    def test_rare_shared_pair_produces_similarity_ubiquitous_pair_does_not(
        self,
    ) -> None:
        # root/toor is shared by all four IPs -- maximally ubiquitous,
        # contributes nothing. custom/xyz123 is shared only by the first
        # two -- rare, and the only thing that should link them.
        events = pl.DataFrame(
            [
                _attempt(ip, "root", "toor")
                for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4")
            ]
            + [
                _attempt("1.1.1.1", "custom", "xyz123"),
                _attempt("2.2.2.2", "custom", "xyz123"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)
        similarity = build_ip_similarity_graph(
            graph, min_similarity=0.0, min_shared_pairs=1
        )

        assert similarity.has_edge(("ip", "1.1.1.1"), ("ip", "2.2.2.2"))
        assert not similarity.has_edge(("ip", "1.1.1.1"), ("ip", "3.3.3.3"))

    def test_identical_credential_sets_have_similarity_one(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("1.1.1.1", "admin", "letmein"),
                _attempt("2.2.2.2", "root", "toor"),
                _attempt("2.2.2.2", "admin", "letmein"),
                # Third IP so root/toor and admin/letmein aren't ubiquitous
                # (idf 0), which would make this an empty-vector edge case
                # instead of a real "identical vectors" one.
                _attempt("3.3.3.3", "unrelated", "unrelated"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)
        similarity = build_ip_similarity_graph(
            graph, min_similarity=0.0, min_shared_pairs=1
        )

        edge = similarity.get_edge_data(("ip", "1.1.1.1"), ("ip", "2.2.2.2"))
        assert edge["weight"] == pytest.approx(1.0)

    def test_cosine_similarity_matches_manual_tfidf_calculation(self) -> None:
        # 4 IPs. root/toor: all four (idf 0). x/y: A and B (idf log(4/2)).
        # z/w: B only (idf log(4/1), inflates B's norm but isn't shared).
        events = pl.DataFrame(
            [
                _attempt("A", "root", "toor"),
                _attempt("B", "root", "toor"),
                _attempt("C", "root", "toor"),
                _attempt("D", "root", "toor"),
                _attempt("A", "x", "y"),
                _attempt("B", "x", "y"),
                _attempt("B", "z", "w"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)
        similarity = build_ip_similarity_graph(
            graph, min_similarity=0.0, min_shared_pairs=1
        )

        idf_xy = math.log(4 / 2)
        idf_zw = math.log(4 / 1)
        norm_a = idf_xy
        norm_b = math.sqrt(idf_xy**2 + idf_zw**2)
        expected = (idf_xy * idf_xy) / (norm_a * norm_b)

        edge = similarity.get_edge_data(("ip", "A"), ("ip", "B"))
        assert edge["weight"] == pytest.approx(expected)

    def test_min_similarity_threshold_excludes_below_and_includes_above(
        self,
    ) -> None:
        # Same scenario as the manual-calculation test above -- expected
        # similarity is ~0.447, so 0.9 should exclude it and 0.1 include it.
        events = pl.DataFrame(
            [
                _attempt("A", "root", "toor"),
                _attempt("B", "root", "toor"),
                _attempt("C", "root", "toor"),
                _attempt("D", "root", "toor"),
                _attempt("A", "x", "y"),
                _attempt("B", "x", "y"),
                _attempt("B", "z", "w"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)

        strict = build_ip_similarity_graph(
            graph, min_similarity=0.9, min_shared_pairs=1
        )
        lenient = build_ip_similarity_graph(
            graph, min_similarity=0.1, min_shared_pairs=1
        )

        assert not strict.has_edge(("ip", "A"), ("ip", "B"))
        assert lenient.has_edge(("ip", "A"), ("ip", "B"))

    def test_ips_with_no_shared_pairs_are_unconnected_but_present(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "svc-deploy", "Xk9mP2vQ7z"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)
        similarity = build_ip_similarity_graph(
            graph, min_similarity=0.0, min_shared_pairs=1
        )

        assert ("ip", "1.1.1.1") in similarity
        assert ("ip", "2.2.2.2") in similarity
        assert not similarity.has_edge(("ip", "1.1.1.1"), ("ip", "2.2.2.2"))

    def test_empty_graph_returns_empty_similarity_graph(self) -> None:
        similarity = build_ip_similarity_graph(nx.Graph())

        assert similarity.number_of_nodes() == 0

    def test_ubiquitous_pair_does_not_bridge_otherwise_unrelated_groups(
        self,
    ) -> None:
        # Regression for the actual problem this function fixes: two
        # otherwise-unrelated groups used to merge into one giant component
        # via connected_components as soon as any member of each group
        # shared even one pair. Group A (1.1.1.1/2.2.2.2) and group B
        # (3.3.3.3/4.4.4.4) each share a pair unique to that group -- rare,
        # real signal. All four also share root/toor, which -- being
        # maximally ubiquitous -- must not be enough to merge the groups.
        events = pl.DataFrame(
            [
                _attempt(ip, "root", "toor")
                for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4")
            ]
            + [
                _attempt("1.1.1.1", "a", "a"),
                _attempt("2.2.2.2", "a", "a"),
                _attempt("3.3.3.3", "b", "b"),
                _attempt("4.4.4.4", "b", "b"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)
        similarity = build_ip_similarity_graph(graph)
        clusters = assign_clusters(similarity)

        by_ip = dict(
            zip(clusters["src_ip"].to_list(), clusters["cluster_id"].to_list())
        )
        assert by_ip["1.1.1.1"] == by_ip["2.2.2.2"]
        assert by_ip["3.3.3.3"] == by_ip["4.4.4.4"]
        assert by_ip["1.1.1.1"] != by_ip["3.3.3.3"]

    def test_single_shared_rare_pair_is_not_enough_by_default(self) -> None:
        # Two small-dictionary IPs sharing exactly one rare pair: cosine is
        # high, but one coincidental pair is too little evidence -- the
        # failure mode that chained a 1,185-IP component in real data.
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "custom", "xyz123"),
                _attempt("2.2.2.2", "custom", "xyz123"),
                _attempt("1.1.1.1", "only-a", "a"),
                _attempt("2.2.2.2", "only-b", "b"),
                _attempt("3.3.3.3", "unrelated", "unrelated"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)

        default = build_ip_similarity_graph(graph, min_similarity=0.0)
        lenient = build_ip_similarity_graph(
            graph, min_similarity=0.0, min_shared_pairs=1
        )

        assert not default.has_edge(("ip", "1.1.1.1"), ("ip", "2.2.2.2"))
        assert lenient.has_edge(("ip", "1.1.1.1"), ("ip", "2.2.2.2"))

    def test_default_min_shared_pairs_is_two(self) -> None:
        assert DEFAULT_MIN_SHARED_PAIRS == 2

    def test_edge_carries_shared_pair_count(self) -> None:
        events = pl.DataFrame(
            [
                _attempt(ip, username, password)
                for ip in ("1.1.1.1", "2.2.2.2")
                for username, password in (("a", "1"), ("b", "2"), ("c", "3"))
            ]
            + [_attempt("3.3.3.3", "unrelated", "unrelated")]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)
        similarity = build_ip_similarity_graph(graph)

        edge = similarity.get_edge_data(("ip", "1.1.1.1"), ("ip", "2.2.2.2"))
        assert edge["shared_pairs"] == 3

    def test_enough_shared_pairs_but_low_similarity_is_not_linked(self) -> None:
        # Both thresholds must hold: two big-dictionary IPs that share 2
        # pairs out of ~20 each meet min_shared_pairs but have marginal
        # cosine similarity -- the weak links that chained the old
        # count-only approach's 273-IP component.
        events = pl.DataFrame(
            [_attempt("1.1.1.1", "shared", str(i)) for i in range(2)]
            + [_attempt("2.2.2.2", "shared", str(i)) for i in range(2)]
            + [_attempt("1.1.1.1", "a", str(i)) for i in range(20)]
            + [_attempt("2.2.2.2", "b", str(i)) for i in range(20)]
            + [_attempt("3.3.3.3", "unrelated", "unrelated")]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)
        similarity = build_ip_similarity_graph(graph, min_similarity=0.2)

        assert not similarity.has_edge(("ip", "1.1.1.1"), ("ip", "2.2.2.2"))

    def test_single_pair_links_do_not_chain_into_one_component(self) -> None:
        # Regression for the giant-component effect: A-B and B-C each share
        # exactly one (different) rare pair. With min_shared_pairs=1 that
        # chains A, B, C into one cluster; by default all three stay apart.
        events = pl.DataFrame(
            [
                _attempt("A", "p1", "p1"),
                _attempt("B", "p1", "p1"),
                _attempt("B", "p2", "p2"),
                _attempt("C", "p2", "p2"),
                _attempt("D", "unrelated", "unrelated"),
            ]
        )
        graph = build_credential_graph(events, min_ips_to_exclude=None)

        chained = assign_clusters(
            build_ip_similarity_graph(graph, min_similarity=0.0, min_shared_pairs=1)
        )
        default = assign_clusters(build_ip_similarity_graph(graph, min_similarity=0.0))

        assert chained["cluster_size"].max() == 3
        assert default["cluster_size"].max() == 1


class TestAssignClusters:
    def test_two_ips_sharing_a_pair_land_in_the_same_cluster(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "root", "toor"),
            ]
        )
        clusters = assign_clusters(build_credential_graph(events))

        cluster_ids = clusters.sort("src_ip")["cluster_id"].to_list()
        assert cluster_ids[0] == cluster_ids[1]
        assert clusters["cluster_size"].to_list() == [2, 2]

    def test_disjoint_pairs_land_in_different_clusters(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "svc-deploy", "Xk9mP2vQ7z"),
            ]
        )
        clusters = assign_clusters(build_credential_graph(events))

        by_ip = dict(
            zip(clusters["src_ip"].to_list(), clusters["cluster_id"].to_list())
        )
        assert by_ip["1.1.1.1"] != by_ip["2.2.2.2"]

    def test_transitive_chain_collapses_into_one_component(self) -> None:
        # IP1 and IP3 never share a pair directly, but both link through IP2.
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "root", "toor"),
                _attempt("2.2.2.2", "svc-deploy", "Xk9mP2vQ7z"),
                _attempt("3.3.3.3", "svc-deploy", "Xk9mP2vQ7z"),
            ]
        )
        clusters = assign_clusters(build_credential_graph(events))

        cluster_ids = set(clusters["cluster_id"].to_list())
        assert len(cluster_ids) == 1
        assert clusters["cluster_size"].to_list() == [3, 3, 3]

    def test_credential_pair_nodes_never_appear_as_output_rows(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "root", "toor")])
        clusters = assign_clusters(build_credential_graph(events))

        assert clusters["src_ip"].to_list() == ["1.1.1.1"]

    def test_unique_pair_is_a_singleton_cluster(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "root", "toor")])
        clusters = assign_clusters(build_credential_graph(events))

        assert clusters["cluster_size"].to_list() == [1]

    def test_cluster_ordering_is_deterministic_by_size_then_min_ip(self) -> None:
        events = pl.DataFrame(
            [
                # 3-IP cluster (higher IPs, but larger size -> cluster_id 0)
                _attempt("9.9.9.1", "a", "a"),
                _attempt("9.9.9.2", "a", "a"),
                _attempt("9.9.9.3", "a", "a"),
                # 2-IP cluster, lower IPs but smaller size -> cluster_id 1
                _attempt("1.1.1.1", "b", "b"),
                _attempt("1.1.1.2", "b", "b"),
            ]
        )
        clusters = assign_clusters(build_credential_graph(events))

        by_ip = dict(
            zip(clusters["src_ip"].to_list(), clusters["cluster_id"].to_list())
        )
        assert by_ip["9.9.9.1"] == 0
        assert by_ip["1.1.1.1"] == 1

    def test_empty_input_returns_empty_dataframe_with_correct_schema(self) -> None:
        events = pl.DataFrame(
            schema={"src_ip": pl.Utf8, "username": pl.Utf8, "password": pl.Utf8}
        )
        clusters = assign_clusters(build_credential_graph(events))

        assert clusters.height == 0
        assert clusters.schema["src_ip"] == pl.Utf8
        assert clusters.schema["cluster_id"] == pl.Int64
        assert clusters.schema["cluster_size"] == pl.Int64
