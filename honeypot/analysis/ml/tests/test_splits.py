import polars as pl
import pytest

from honeypot.analysis.ml.splits import SPLITS, assign_splits, split_summary


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
