# honeypot/analysis/network/tests/test_ai_targeting.py
import polars as pl
import pytest

from honeypot.analysis.network.ai_targeting import (
    ai_targeting_ips,
    ai_username_targeting,
    classify_ai_username,
)


def _attempt(src_ip: str, username: str | None, password: str | None) -> dict:
    return {"src_ip": src_ip, "username": username, "password": password}


class TestClassifyAiUsername:
    @pytest.mark.parametrize(
        ("username", "category"),
        [
            ("claude", "agent_tooling"),
            ("openclaw", "agent_tooling"),
            ("clawdbot", "agent_tooling"),
            ("grok", "model_vendor"),
            ("ollama", "llm_runtime"),
            ("gpt4all", "llm_runtime"),
            ("gpu01", "ml_infra"),
            ("nvidia-persistenced", "ml_infra"),
            ("aiuser", "generic_ai"),
            ("Claude", "agent_tooling"),
            ("svc_claude", "agent_tooling"),
            ("llama.cpp", "llm_runtime"),
            ("open-webui", "llm_runtime"),
        ],
    )
    def test_matches_known_keywords(self, username: str, category: str) -> None:
        assert classify_ai_username(username) == category

    @pytest.mark.parametrize(
        "username",
        [
            "cai_an",
            "raymond",
            "aishwarya",
            "root",
            "admin",
            "",
            "-_-",
            "User-Agent: Mozilla/5.0 (X11; Linux x86_64)",
        ],
    )
    def test_substring_lookalikes_do_not_match(self, username: str) -> None:
        assert classify_ai_username(username) is None


class TestAiUsernameTargeting:
    def test_counts_distinct_ips_and_passwords_per_ai_username(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "claude", "claude"),
                _attempt("1.1.1.1", "claude", "123456"),
                _attempt("2.2.2.2", "claude", "claude"),
                _attempt("3.3.3.3", "ollama", "ollama"),
            ]
        )
        result = ai_username_targeting(events)

        assert result.to_dicts() == [
            {
                "username": "claude",
                "category": "agent_tooling",
                "distinct_ips": 2,
                "distinct_passwords": 2,
            },
            {
                "username": "ollama",
                "category": "llm_runtime",
                "distinct_ips": 1,
                "distinct_passwords": 1,
            },
        ]

    def test_excludes_non_ai_usernames_and_partial_attempts(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "root", "toor"),
                _attempt("2.2.2.2", "claude", None),
            ]
        )
        result = ai_username_targeting(events)

        assert result.height == 0
        assert result.columns == [
            "username",
            "category",
            "distinct_ips",
            "distinct_passwords",
        ]


class TestAiTargetingIps:
    def test_ai_share_separates_dedicated_from_broad_dictionary_ips(self) -> None:
        events = pl.DataFrame(
            [
                # Dedicated: only AI usernames.
                _attempt("1.1.1.1", "ollama", "ollama"),
                _attempt("1.1.1.1", "vllm", "vllm"),
                # Broad: one AI username among three.
                _attempt("2.2.2.2", "claude", "claude"),
                _attempt("2.2.2.2", "root", "toor"),
                _attempt("2.2.2.2", "admin", "admin"),
                # Never touches AI usernames -- absent from output.
                _attempt("3.3.3.3", "root", "toor"),
            ]
        )
        result = ai_targeting_ips(events)

        counts = result.select("src_ip", "ai_usernames", "total_usernames")
        assert counts.to_dicts() == [
            {"src_ip": "1.1.1.1", "ai_usernames": 2, "total_usernames": 2},
            {"src_ip": "2.2.2.2", "ai_usernames": 1, "total_usernames": 3},
        ]
        assert result["ai_share"].to_list() == pytest.approx([1.0, 1 / 3])

    def test_joins_cluster_assignment_when_given(self) -> None:
        events = pl.DataFrame(
            [
                _attempt("1.1.1.1", "claude", "claude"),
                _attempt("2.2.2.2", "codex", "codex"),
            ]
        )
        clusters = pl.DataFrame(
            {"src_ip": ["1.1.1.1"], "cluster_id": [0], "cluster_size": [5]}
        )
        result = ai_targeting_ips(events, clusters=clusters).sort("src_ip")

        assert result["cluster_id"].to_list() == [0, None]
        assert result["cluster_size"].to_list() == [5, None]

    def test_no_cluster_columns_without_clusters(self) -> None:
        events = pl.DataFrame([_attempt("1.1.1.1", "claude", "claude")])

        assert "cluster_id" not in ai_targeting_ips(events).columns
