# honeypot/analysis/network/tests/test_graph.py
import polars as pl

from honeypot.analysis.network.graph import (
    _GENERIC_CREDENTIAL_PAIRS,
    assign_clusters,
    build_credential_graph,
    generic_credential_attempts,
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


class TestGenericCredentialFiltering:
    def test_default_denylist_excludes_known_generic_pair(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "admin", "admin"),
            ]
        )
        graph = build_credential_graph(events)

        assert ("cred", "admin", "admin") not in graph
        assert graph.number_of_edges() == 0

    def test_ip_whose_only_attempt_is_generic_is_absent_from_graph(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "admin", "admin")])
        graph = build_credential_graph(events)

        assert ("ip", "1.1.1.1") not in graph
        assert graph.number_of_nodes() == 0

    def test_non_generic_pair_is_unaffected(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "root", "toor")])
        graph = build_credential_graph(events)

        assert graph.has_edge(("ip", "1.1.1.1"), ("cred", "root", "toor"))

    def test_exclude_pairs_can_be_disabled(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "admin", "admin")])
        graph = build_credential_graph(events, exclude_pairs=frozenset())

        assert graph.has_edge(("ip", "1.1.1.1"), ("cred", "admin", "admin"))

    def test_custom_exclude_pairs_overrides_default(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "custom", "value")])
        graph = build_credential_graph(
            events, exclude_pairs=frozenset({("custom", "value")})
        )

        assert ("ip", "1.1.1.1") not in graph

    def test_generic_denylist_is_nonempty(self) -> None:
        # Sanity check the module-level constant itself, independent of
        # build_credential_graph's behavior.
        assert len(_GENERIC_CREDENTIAL_PAIRS) > 0
        assert ("admin", "admin") in _GENERIC_CREDENTIAL_PAIRS


class TestGenericCredentialAttempts:
    def test_counts_distinct_ips_per_denylisted_pair(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "admin", "admin"),
                _attempt("3.3.3.3", "admin", "admin"),
            ]
        )
        result = generic_credential_attempts(events)

        row = result.row(0, named=True)
        assert row["username"] == "admin"
        assert row["password"] == "admin"
        assert row["distinct_ips"] == 3

    def test_excludes_non_generic_pairs(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "admin", "admin"),
                _attempt("2.2.2.2", "root", "toor"),
            ]
        )
        result = generic_credential_attempts(events)

        assert result["username"].to_list() == ["admin"]

    def test_excludes_partial_attempts(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "admin", None)])
        result = generic_credential_attempts(events)

        assert result.height == 0

    def test_no_generic_attempts_returns_empty_dataframe_with_correct_schema(
        self,
    ) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "root", "toor")])
        result = generic_credential_attempts(events)

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
        result = generic_credential_attempts(events)

        assert result["username"].to_list() == ["admin", "root"]
        assert result["distinct_ips"].to_list() == [2, 1]

    def test_custom_exclude_pairs_overrides_default(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "custom", "value")])
        result = generic_credential_attempts(
            events, exclude_pairs=frozenset({("custom", "value")})
        )

        assert result["username"].to_list() == ["custom"]


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
