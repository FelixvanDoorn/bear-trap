# honeypot/analysis/ml/splits.py
#
# Train / validation / test split of the IP profiles, grouped by network.
#
# Grouped, because IPs in one ASN are near-duplicates (same hoster, same
# botnet operator, often the same campaign): a random-IP split puts
# siblings on both sides and makes the held-out score look better than it
# is -- geo/'s random_ip vs random_asn split (rho 0.93 vs 0.47) showed how
# much. IPs without an ASN are grouped by /24 instead.
#
# Deterministic and stable: a group's split is a hash of (seed, group),
# not a shuffle, so the same seed always gives the same split, and an ASN
# keeps its split when later snapshots add IPs or ASNs. The price is that
# the fractions hold for groups, not IPs: with 1,568 ASNs and the largest
# holding 4.7% of the IPs (2026-09-16 snapshot), the realized IP shares
# vary by a few points around the target (split_summary reports them).
import hashlib

import polars as pl

SPLITS = ("train", "validation", "test")
DEFAULT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}
DEFAULT_SEED = 0


def _unit_interval(seed: int, group: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    return int(digest[:16], 16) / 2**64


def split_groups(profiles: pl.DataFrame) -> pl.Series:
    """`group` per row: "AS<asn>", or "/24 a.b.c" when the ASN is null."""
    return profiles.select(
        pl.when(pl.col("asn").is_not_null())
        .then(pl.format("AS{}", pl.col("asn")))
        .otherwise(pl.format("/24 {}", pl.col("src_ip").str.replace(r"\.\d+$", "")))
        .alias("group")
    ).to_series()


def assign_splits(
    profiles: pl.DataFrame,
    seed: int = DEFAULT_SEED,
    fractions: dict[str, float] = DEFAULT_FRACTIONS,
) -> pl.DataFrame:
    """`profiles` (needs src_ip and asn) with `group` and `split` added."""
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"fractions must cover {SPLITS} and sum to 1")
    edges = []
    cumulative = 0.0
    for split in SPLITS:
        cumulative += fractions[split]
        edges.append((cumulative, split))

    def split_of(group: str) -> str:
        u = _unit_interval(seed, group)
        return next((split for edge, split in edges if u < edge), SPLITS[-1])

    with_groups = profiles.with_columns(split_groups(profiles))
    lookup = pl.DataFrame({"group": with_groups["group"].unique().sort()}).with_columns(
        pl.col("group").map_elements(split_of, return_dtype=pl.Utf8).alias("split")
    )
    return with_groups.join(lookup, on="group", how="left", maintain_order="left")


def split_summary(split_profiles: pl.DataFrame) -> pl.DataFrame:
    """Per split: IPs, groups, and the realized share of IPs."""
    return (
        split_profiles.group_by("split")
        .agg(pl.len().alias("ips"), pl.col("group").n_unique().alias("groups"))
        .with_columns((pl.col("ips") / pl.col("ips").sum()).alias("ip_share"))
        .join(
            pl.DataFrame({"split": SPLITS, "order": range(len(SPLITS))}),
            on="split",
        )
        .sort("order")
        .drop("order")
    )
