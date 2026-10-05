# honeypot/analysis/network/tests/test_charts.py
from pathlib import Path

import networkx as nx
import polars as pl

from honeypot.analysis.network.charts import (
    plot_ai_username_targeting,
    plot_cluster_size_distribution,
    plot_generic_credential_breakdown,
    plot_top_clusters_subgraph,
    plot_username_targeting_breakdown,
)


def _clusters_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "src_ip": ["1.1.1.1", "2.2.2.2", "3.3.3.3"],
            "cluster_id": [0, 0, 1],
            "cluster_size": [2, 2, 1],
        }
    )


def _small_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge(("ip", "1.1.1.1"), ("cred", "root", "toor"))
    graph.add_edge(("ip", "2.2.2.2"), ("cred", "root", "toor"))
    return graph


class TestPlotClusterSizeDistribution:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        path = plot_cluster_size_distribution(_clusters_df(), output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotGenericCredentialBreakdown:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        attempts = pl.DataFrame(
            {
                "username": ["admin", "root"],
                "password": ["admin", "root"],
                "distinct_ips": [241, 95],
            }
        )

        path = plot_generic_credential_breakdown(attempts, output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_empty_input_without_raising(self, tmp_path: Path) -> None:
        attempts = pl.DataFrame(
            schema={"username": pl.Utf8, "password": pl.Utf8, "distinct_ips": pl.UInt32}
        )

        path = plot_generic_credential_breakdown(attempts, output_dir=tmp_path)

        assert path.exists()


class TestPlotUsernameTargetingBreakdown:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        breakdown = pl.DataFrame(
            {
                "username": ["postgres", "grok"],
                "distinct_ips": [100, 12],
            }
        )

        path = plot_username_targeting_breakdown(breakdown, output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_empty_input_without_raising(self, tmp_path: Path) -> None:
        breakdown = pl.DataFrame(
            schema={"username": pl.Utf8, "distinct_ips": pl.UInt32}
        )

        path = plot_username_targeting_breakdown(breakdown, output_dir=tmp_path)

        assert path.exists()


class TestPlotTopClustersSubgraph:
    def test_writes_chart_file_for_a_small_cluster_set(self, tmp_path: Path) -> None:
        path = plot_top_clusters_subgraph(_small_graph(), output_dir=tmp_path)

        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 0

    def test_returns_none_and_writes_nothing_when_max_nodes_exceeded(
        self, tmp_path: Path
    ) -> None:
        path = plot_top_clusters_subgraph(
            _small_graph(), output_dir=tmp_path, max_nodes=1
        )

        assert path is None
        assert list(tmp_path.iterdir()) == []

    def test_returns_none_for_an_empty_graph(self, tmp_path: Path) -> None:
        path = plot_top_clusters_subgraph(nx.Graph(), output_dir=tmp_path)

        assert path is None


class TestPlotAiUsernameTargeting:
    def test_writes_chart_file(self, tmp_path: Path) -> None:
        breakdown = pl.DataFrame(
            {
                "username": ["claude", "ollama"],
                "category": ["agent_tooling", "llm_runtime"],
                "distinct_ips": [46, 2],
                "distinct_passwords": [12, 1],
            }
        )

        path = plot_ai_username_targeting(breakdown, output_dir=tmp_path)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_empty_input_without_raising(self, tmp_path: Path) -> None:
        breakdown = pl.DataFrame(
            schema={
                "username": pl.Utf8,
                "category": pl.Utf8,
                "distinct_ips": pl.UInt32,
                "distinct_passwords": pl.UInt32,
            }
        )

        path = plot_ai_username_targeting(breakdown, output_dir=tmp_path)

        assert path.exists()
