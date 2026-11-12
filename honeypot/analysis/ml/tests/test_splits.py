import datetime as dt

import numpy as np
import polars as pl
import pytest

from honeypot.analysis.ml.splits import (
    SPLITS,
    assign_splits,
    cold_start,
    group_folds,
    last_days_start,
    split_summary,
    time_split,
)


def _profiles(n_asns: int = 400, ips_per_asn: int = 3) -> pl.DataFrame:
    rows = [
        {"src_ip": f"10.{a // 256}.{a % 256}.{i}", "asn": 1000 + a}
        for a in range(n_asns)
        for i in range(ips_per_asn)
    ]
    rows += [{"src_ip": f"192.0.2.{i}", "asn": None} for i in range(3)]
    return pl.DataFrame(rows, schema={"src_ip": pl.Utf8, "asn": pl.Int64})


def test_same_seed_same_split_and_different_seed_differs() -> None:
    profiles = _profiles()
    first = assign_splits(profiles, seed=0)
    assert first.equals(assign_splits(profiles, seed=0))
    assert not first["split"].equals(assign_splits(profiles, seed=1)["split"])
    assert first["src_ip"].to_list() == profiles["src_ip"].to_list()


def test_groups_never_cross_splits() -> None:
    splits = assign_splits(_profiles())
    per_group = splits.group_by("group").agg(pl.col("split").n_unique())
    assert per_group["split"].max() == 1
    # Null-ASN IPs are grouped by /24.
    no_asn = splits.filter(pl.col("asn").is_null())
    assert no_asn["group"].unique().to_list() == ["/24 192.0.2"]


def test_fractions_hold_for_many_groups() -> None:
    summary = split_summary(assign_splits(_profiles(n_asns=3000)))
    assert summary["split"].to_list() == list(SPLITS)
    shares = dict(zip(summary["split"], summary["ip_share"]))
    assert shares["train"] == pytest.approx(0.6, abs=0.03)
    assert shares["validation"] == pytest.approx(0.2, abs=0.03)
    assert shares["test"] == pytest.approx(0.2, abs=0.03)


def test_adding_rows_keeps_existing_assignments() -> None:
    small = assign_splits(_profiles(n_asns=200))
    large = assign_splits(_profiles(n_asns=400))
    joined = small.join(large, on="src_ip", suffix="_large")
    assert joined.height == small.height
    assert (joined["split"] == joined["split_large"]).all()


def test_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        assign_splits(_profiles(), fractions={"train": 0.5, "test": 0.5})


def test_time_split_cuts_at_midnight_of_the_last_days() -> None:
    starts = pl.Series(
        [dt.datetime(2026, 9, day, 23, tzinfo=dt.UTC) for day in range(1, 17)]
    )
    test_start = last_days_start(starts, test_days=7)
    assert test_start == dt.datetime(2026, 9, 10, tzinfo=dt.UTC)
    split = time_split(starts, test_start)
    assert split.to_list() == ["train"] * 9 + ["test"] * 7


def test_group_folds_grouped_stratified_deterministic() -> None:
    rng = np.random.default_rng(0)
    groups = pl.Series([f"AS{g}" for g in rng.integers(0, 200, 4000)])
    labels = pl.Series(rng.random(4000) < 0.05)
    folds = group_folds(groups, labels, n_folds=5, seed=0)
    assert np.array_equal(folds, group_folds(groups, labels, n_folds=5, seed=0))
    frame = pl.DataFrame({"group": groups, "y": labels, "fold": folds})
    assert frame.group_by("group").agg(pl.col("fold").n_unique()).max()[0, 1] == 1
    rates = frame.group_by("fold").agg(pl.col("y").mean())["y"]
    assert set(frame["fold"].unique().to_list()) == set(range(5))
    assert rates.max() - rates.min() < 0.03


def test_cold_start_marks_unseen_groups() -> None:
    mask = cold_start(pl.Series(["AS1", "AS2", "AS3"]), pl.Series(["AS2", "AS9"]))
    assert mask.to_list() == [True, False, True]
