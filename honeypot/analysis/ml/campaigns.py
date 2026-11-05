# honeypot/analysis/ml/campaigns.py
#
# Campaign clustering of attacking IPs (DDS ch. 9, unsupervised), with
# hyperparameters chosen on a validation split and the result reported
# once on a held-out test split (splits.py).
#
# Pipeline per configuration:
#
#   profiles --FeatureEncoder--> sparse TF-IDF + behaviour (features.py)
#            --TruncatedSVD + L2 normalize--> dense embedding (LSA: on unit
#              vectors, Euclidean distance ranks pairs like cosine)
#            --k-means | Ward | HDBSCAN--> cluster labels (-1 = noise)
#
# No labels exist, so "does it generalize" means "do the clusters
# reproduce on IPs (and networks) the model never saw":
#
# - train: internal quality (silhouette, Davies-Bouldin) and stability --
#   recluster random 80% subsamples, mean adjusted Rand index (ARI) vs the
#   full-train labels.
# - validation: prediction strength (Tibshirani & Walther 2005). Cluster
#   the validation IPs on their own, and separately assign each to its
#   nearest train centroid; for every validation cluster, the share of its
#   pairs of IPs that the train model also puts together. Tibshirani &
#   Walther take the minimum over clusters; it's reported, but selection
#   uses the size-weighted mean: on the real data the minimum is always
#   set by one heterogeneous leftover cluster (0.13-0.56 for every config
#   in a first scan, weighted 0.80-0.91), so it can't tell configs apart.
#   Which campaigns don't reproduce is shown per campaign instead
#   (test_reproducibility in the catalog).
# - selection (`select_config`): among configs with train stability and
#   weighted validation prediction strength both >= their thresholds, the
#   highest weighted prediction strength; configs within ps_tolerance of
#   it count as tied, and the tie goes to the best validation silhouette,
#   then fewer clusters. Prediction strength can't tell a campaign from a
#   consistent sub-split of one (e.g. a campaign's 1-session vs 2-session
#   IPs: both halves reproduce perfectly), so preferring more clusters on
#   ties over-splits; silhouette on validation prefers the better
#   separated grouping. Their "largest k that passes" rule was also too
#   noisy here (k = 20 failed, 30 passed). tune() never sees the test
#   split.
# - test, once (`evaluate_on_test`): the chosen config refit on train +
#   validation, then prediction strength on test and agreement with
#   structure found independently -- ch. 4's credential-graph clusters and
#   each IP's dominant command sequence: do IPs those link end up in one
#   campaign (plus ARI / AMI, see external_agreement). Agreement, not
#   accuracy: neither is ground truth, and credentials are also one of the
#   feature blocks.
import itertools
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from math import comb
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from honeypot.analysis.fingerprinting.features import _hash_commands
from honeypot.analysis.ml.features import (
    BLOCK_WEIGHTS,
    DEFAULT_MIN_DF,
    FeatureEncoder,
)
from honeypot.analysis.ml.splits import DEFAULT_FRACTIONS
from honeypot.analysis.network.graph import (
    assign_clusters,
    build_credential_graph,
    build_ip_similarity_graph,
)

ALGORITHMS = ("kmeans", "ward", "hdbscan")
NOISE = -1


@dataclass(frozen=True)
class CampaignConfig:
    """Everything that determines a run's result besides the data. Its
    to_dict() is what runs.compute_run_id hashes."""

    seed: int = 0
    fractions: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FRACTIONS)
    )
    block_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(BLOCK_WEIGHTS)
    )
    min_df: int = DEFAULT_MIN_DF
    n_components: tuple[int, ...] = (10, 25, 50)
    kmeans_k: tuple[int, ...] = (4, 6, 8, 10, 12, 15, 20, 25, 30)
    ward_k: tuple[int, ...] = (4, 6, 8, 10, 12, 15, 20, 25, 30)
    hdbscan_min_cluster_size: tuple[int, ...] = (5, 10, 20, 40)
    stability_resamples: int = 10
    stability_fraction: float = 0.8
    min_stability: float = 0.8
    min_prediction_strength: float = 0.8
    ps_tolerance: float = 0.02
    min_cluster_ips: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CampaignConfig":
        return cls(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in values.items()
            }
        )

    def grid(self) -> list[dict[str, Any]]:
        """Every (n_components, algorithm, param) combination tuned."""
        params = {
            "kmeans": self.kmeans_k,
            "ward": self.ward_k,
            "hdbscan": self.hdbscan_min_cluster_size,
        }
        return [
            {"n_components": n, "algorithm": algorithm, "param": param}
            for n, algorithm in itertools.product(self.n_components, ALGORITHMS)
            for param in params[algorithm]
        ]


@dataclass
class Embedding:
    """FeatureEncoder + TruncatedSVD + L2 normalization, fit together on
    one set of profiles and applied to any other."""

    n_components: int
    seed: int
    block_weights: Mapping[str, float]
    min_df: int
    encoder: FeatureEncoder | None = None
    svd: TruncatedSVD | None = None

    def fit(self, profiles: pl.DataFrame) -> "Embedding":
        self.encoder = FeatureEncoder(self.block_weights, self.min_df).fit(profiles)
        matrix = self.encoder.transform(profiles)
        n = min(self.n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
        self.svd = TruncatedSVD(n, random_state=self.seed).fit(matrix)
        return self

    def transform(self, profiles: pl.DataFrame) -> np.ndarray:
        assert self.encoder is not None and self.svd is not None
        return normalize(self.svd.transform(self.encoder.transform(profiles)))

    @property
    def explained_variance(self) -> float:
        assert self.svd is not None
        return float(self.svd.explained_variance_ratio_.sum())


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Cluster ids renumbered 0, 1, ... by size descending (ties: by
    original id); noise stays -1. Makes labels comparable across reruns
    and readable (campaign 0 is the largest)."""
    ids, counts = np.unique(labels[labels != NOISE], return_counts=True)
    order = sorted(zip(-counts, ids))
    mapping = {old: new for new, (_, old) in enumerate(order)}
    return np.array([mapping.get(label, NOISE) for label in labels], dtype=int)


def fit_clusters(
    embedded: np.ndarray, algorithm: str, param: int, seed: int
) -> np.ndarray:
    """Cluster labels (relabeled by size) for one algorithm. `param` is k
    for kmeans/ward and min_cluster_size for hdbscan."""
    n = embedded.shape[0]
    if algorithm == "kmeans":
        model = KMeans(min(param, n), n_init=10, random_state=seed)
    elif algorithm == "ward":
        model = AgglomerativeClustering(min(param, n), linkage="ward")
    elif algorithm == "hdbscan":
        model = HDBSCAN(min_cluster_size=min(param, n), copy=True)
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}")
    return relabel_by_size(model.fit_predict(embedded))


def centroids(
    embedded: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(cluster ids, centroid matrix) of the non-noise clusters."""
    ids = np.unique(labels[labels != NOISE])
    return ids, np.vstack([embedded[labels == i].mean(axis=0) for i in ids])


def assign(embedded: np.ndarray, ids: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-centroid cluster id for every row."""
    if len(ids) == 0:
        return np.full(embedded.shape[0], NOISE)
    distances = ((embedded[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return ids[distances.argmin(axis=1)]


def pair_agreement(
    groups: np.ndarray, other: np.ndarray, min_size: int
) -> dict[int, tuple[int, float]]:
    """For each non-noise cluster of `groups` with >= min_size members:
    (size, share of its member pairs that `other` also puts in one
    cluster)."""
    agreement = {}
    for cluster in np.unique(groups[groups != NOISE]):
        members = other[groups == cluster]
        n = len(members)
        if n < min_size:
            continue
        _, counts = np.unique(members, return_counts=True)
        agreement[int(cluster)] = (n, sum(comb(int(c), 2) for c in counts) / comb(n, 2))
    return agreement


def prediction_strength(
    independent: np.ndarray, assigned: np.ndarray, min_size: int
) -> tuple[float, float]:
    """(minimum, size-weighted mean) over the `independent` clusters with
    >= min_size members of the share of their member pairs that `assigned`
    also puts in one cluster. NaN when no cluster is big enough."""
    agreement = pair_agreement(independent, assigned, min_size)
    if not agreement:
        return float("nan"), float("nan")
    sizes, shares = zip(*agreement.values())
    return float(min(shares)), float(np.average(shares, weights=sizes))


def internal_scores(embedded: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """n_clusters, noise_share, and silhouette / Davies-Bouldin over the
    non-noise rows (NaN with fewer than two clusters)."""
    kept = labels != NOISE
    n_clusters = len(np.unique(labels[kept]))
    scores = {
        "n_clusters": n_clusters,
        "noise_share": float(1 - kept.mean()),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
    }
    if 2 <= n_clusters < kept.sum():
        scores["silhouette"] = float(silhouette_score(embedded[kept], labels[kept]))
        scores["davies_bouldin"] = float(
            davies_bouldin_score(embedded[kept], labels[kept])
        )
    return scores


def stability(
    embedded: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    param: int,
    seed: int,
    resamples: int,
    fraction: float,
) -> float:
    """Mean ARI between `labels` and a reclustering of a random
    `fraction` subsample, over `resamples` subsamples (without
    replacement: duplicated points would distort density-based HDBSCAN)."""
    rng = np.random.default_rng(seed)
    n = embedded.shape[0]
    scores = []
    for _ in range(resamples):
        rows = np.sort(rng.choice(n, int(fraction * n), replace=False))
        sub = fit_clusters(embedded[rows], algorithm, param, int(rng.integers(2**31)))
        scores.append(adjusted_rand_score(labels[rows], sub))
    return float(np.mean(scores))


def tune(
    train: pl.DataFrame, validation: pl.DataFrame, config: CampaignConfig
) -> pl.DataFrame:
    """One row per config.grid() point: train internal scores and
    stability, validation prediction strength (min and weighted) and
    silhouette, plus the SVD's explained variance. The test split is not
    an argument on purpose."""
    rows = []
    for n_components in config.n_components:
        embedding = Embedding(
            n_components, config.seed, config.block_weights, config.min_df
        ).fit(train)
        train_z, val_z = embedding.transform(train), embedding.transform(validation)
        for point in (p for p in config.grid() if p["n_components"] == n_components):
            algorithm, param = point["algorithm"], point["param"]
            labels = fit_clusters(train_z, algorithm, param, config.seed)
            train_scores = internal_scores(train_z, labels)
            val_labels = fit_clusters(val_z, algorithm, param, config.seed)
            val_scores = internal_scores(val_z, val_labels)
            ps_min, ps_mean = prediction_strength(
                val_labels,
                assign(val_z, *centroids(train_z, labels))
                if train_scores["n_clusters"]
                else np.full(len(val_labels), NOISE),
                config.min_cluster_ips,
            )
            rows.append(
                point
                | {
                    "explained_variance": embedding.explained_variance,
                    **{f"train_{k}": v for k, v in train_scores.items()},
                    "train_stability": stability(
                        train_z,
                        labels,
                        algorithm,
                        param,
                        config.seed,
                        config.stability_resamples,
                        config.stability_fraction,
                    ),
                    "val_n_clusters": val_scores["n_clusters"],
                    "val_silhouette": val_scores["silhouette"],
                    "val_prediction_strength": ps_min,
                    "val_prediction_strength_weighted": ps_mean,
                }
            )
    return pl.DataFrame(rows)


def select_config(tuning: pl.DataFrame, config: CampaignConfig) -> dict[str, Any]:
    """The chosen grid row, with `eligible` (passed both thresholds) and
    `rule` (which branch chose it). Falls back to all rows if none passes
    both thresholds."""
    ps = "val_prediction_strength_weighted"
    scored = tuning.with_columns(
        (
            (pl.col("train_stability") >= config.min_stability)
            & (pl.col(ps) >= config.min_prediction_strength)
            & (pl.col("train_n_clusters") >= 2)
        )
        .fill_null(False)
        .alias("eligible")
    )
    candidates = scored.filter("eligible")
    rule = "best reproducibility among stable configs, ties -> val silhouette"
    if not candidates.height:
        candidates = scored.filter(pl.col(ps).is_not_nan() & pl.col(ps).is_not_null())
        rule = "fallback (none eligible): " + rule
    best = candidates[ps].max()
    chosen = (
        candidates.filter(pl.col(ps) >= best - config.ps_tolerance)
        .sort(
            ["val_silhouette", "train_n_clusters", "n_components"],
            descending=[True, False, False],
            nulls_last=True,
        )
        .row(0, named=True)
    )
    return chosen | {"rule": rule}


def tuning_with_selection(
    tuning: pl.DataFrame, chosen: Mapping[str, Any]
) -> pl.DataFrame:
    """tuning plus a `chosen` flag, for the tuning-curve chart."""
    return tuning.with_columns(
        (
            (pl.col("n_components") == chosen["n_components"])
            & (pl.col("algorithm") == chosen["algorithm"])
            & (pl.col("param") == chosen["param"])
        ).alias("chosen")
    )


def command_sequence_labels(events: pl.DataFrame) -> pl.DataFrame:
    """src_ip -> the IP's most common session command-sequence hash
    (fingerprinting's normalized exact-match hash; ties: smallest hash).
    IPs that ran no commands are absent."""
    sessions = (
        events.filter(
            (pl.col("eventid") == "cowrie.command.input")
            & pl.col("command_input").is_not_null()
        )
        .sort("event_timestamp")
        .group_by("src_ip", "session", maintain_order=True)
        .agg("command_input")
    )
    hashes = pl.DataFrame(
        [
            {"src_ip": src_ip, "label": _hash_commands(commands)}
            for src_ip, commands in sessions.select(
                "src_ip", "command_input"
            ).iter_rows()
        ],
        schema={"src_ip": pl.Utf8, "label": pl.Utf8},
    )
    return (
        hashes.group_by("src_ip", "label")
        .len()
        .sort(["src_ip", "len", "label"], descending=[False, True, False])
        .group_by("src_ip", maintain_order=True)
        .first()
        .select("src_ip", "label")
    )


def credential_cluster_labels(events: pl.DataFrame) -> pl.DataFrame:
    """src_ip -> ch. 4 credential-graph cluster (network/graph.py, default
    thresholds), as a string label."""
    clusters = assign_clusters(
        build_ip_similarity_graph(build_credential_graph(events))
    )
    return clusters.select("src_ip", pl.col("cluster_id").cast(pl.Utf8).alias("label"))


def external_agreement(
    campaigns: pl.DataFrame, references: Mapping[str, pl.DataFrame]
) -> pl.DataFrame:
    """Per reference labelling (src_ip, label), on the IPs in both:

    - linked_ips / linked_kept_together: IPs sharing a reference label
      with at least one other IP, and the size-weighted share of those
      groups' pairs that also share a campaign ("do IPs the reference
      links end up in one campaign?"). The meaningful number when the
      reference is much finer than the campaigns -- ch. 4's credential
      graph leaves most IPs as singletons;
    - ari / ami: adjusted Rand index and adjusted mutual information
      between the two partitions (noise counts as one campaign). Near 0
      by construction when one partition is mostly singletons."""
    rows = []
    for name, reference in references.items():
        both = campaigns.join(reference, on="src_ip", how="inner")
        row: dict[str, Any] = {
            "reference": name,
            "ips": both.height,
            "linked_ips": 0,
            "linked_kept_together": None,
            "ari": None,
            "ami": None,
        }
        if both.height >= 2:
            _, codes = np.unique(both["label"].to_numpy(), return_inverse=True)
            found = both["campaign"].to_numpy()
            agreement = pair_agreement(codes, found, min_size=2)
            if agreement:
                sizes, shares = zip(*agreement.values())
                row["linked_ips"] = int(sum(sizes))
                row["linked_kept_together"] = float(np.average(shares, weights=sizes))
            row["ari"] = float(adjusted_rand_score(codes, found))
            row["ami"] = float(adjusted_mutual_info_score(codes, found))
        rows.append(row)
    return pl.DataFrame(
        rows,
        schema={
            "reference": pl.Utf8,
            "ips": pl.Int64,
            "linked_ips": pl.Int64,
            "linked_kept_together": pl.Float64,
            "ari": pl.Float64,
            "ami": pl.Float64,
        },
    )


@dataclass
class FinalModel:
    """The chosen config refit on train + validation."""

    embedding: Embedding
    labels: np.ndarray
    ids: np.ndarray
    centers: np.ndarray


def fit_final(
    fit_profiles: pl.DataFrame, chosen: Mapping[str, Any], config: CampaignConfig
) -> FinalModel:
    embedding = Embedding(
        int(chosen["n_components"]), config.seed, config.block_weights, config.min_df
    ).fit(fit_profiles)
    embedded = embedding.transform(fit_profiles)
    labels = fit_clusters(
        embedded, str(chosen["algorithm"]), int(chosen["param"]), config.seed
    )
    return FinalModel(embedding, labels, *centroids(embedded, labels))


def evaluate_on_test(
    model: FinalModel,
    test: pl.DataFrame,
    chosen: Mapping[str, Any],
    config: CampaignConfig,
) -> tuple[np.ndarray, dict[str, float], pl.DataFrame]:
    """(test labels by nearest final centroid, test metrics, per-campaign
    reproducibility). Metrics: prediction strength of the chosen algorithm
    clustering test on its own vs the final model's assignment, and the
    silhouette of the assigned labels. Reproducibility per campaign: of
    the pairs of test IPs assigned to it, the share the independent test
    clustering also puts together (null under min_cluster_ips test IPs)."""
    test_z = model.embedding.transform(test)
    assigned = assign(test_z, model.ids, model.centers)
    independent = fit_clusters(
        test_z, str(chosen["algorithm"]), int(chosen["param"]), config.seed
    )
    ps_min, ps_mean = prediction_strength(independent, assigned, config.min_cluster_ips)
    agreement = pair_agreement(assigned, independent, config.min_cluster_ips)
    reproducibility = pl.DataFrame(
        [
            {"campaign": int(c), "test_ips": int((assigned == c).sum())}
            | {"test_reproducibility": agreement.get(int(c), (0, None))[1]}
            for c in model.ids
        ],
        schema={
            "campaign": pl.Int64,
            "test_ips": pl.Int64,
            "test_reproducibility": pl.Float64,
        },
    )
    return (
        assigned,
        {
            "ips": test.height,
            "prediction_strength": ps_min,
            "prediction_strength_weighted": ps_mean,
            "independent_n_clusters": len(np.unique(independent[independent != NOISE])),
            "silhouette_assigned": internal_scores(test_z, assigned)["silhouette"],
        },
        reproducibility,
    )


def embedding_2d(embedded: np.ndarray, seed: int) -> np.ndarray:
    """t-SNE map of the embedding, for the scatter chart only (t-SNE
    distances between far-apart groups aren't meaningful)."""
    perplexity = min(30.0, max(2.0, (embedded.shape[0] - 1) / 3))
    return TSNE(2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(
        embedded
    )


def _top(values: Iterable[str], n: int, width: int = 80) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:n]
    return "; ".join(
        f"{value if len(value) <= width else value[: width - 1] + '…'} ({count})"
        for value, count in ranked
    )


def campaign_profiles(labelled: pl.DataFrame, top_n: int = 3) -> pl.DataFrame:
    """One row per campaign (noise as -1): IPs, sessions, share of each
    split, top usernames, credential pairs, commands, client fingerprints,
    ASNs and countries, telnet / download / command shares, and first and
    last seen. `labelled` is ip_profiles + asn/as_label/country + split +
    campaign. Optional boolean columns `tried_ai_username` and `listed`
    (common.threatintel) become shares when present."""
    rows = []
    for campaign in sorted(labelled["campaign"].unique().to_list()):
        group = labelled.filter(pl.col("campaign") == campaign)
        tokens = group["credentials"].explode(empty_as_null=True).drop_nulls().to_list()
        row = {
            "campaign": campaign,
            "ips": group.height,
            "sessions": int(group["sessions"].sum()),
            **{
                f"{split}_ips": group.filter(pl.col("split") == split).height
                for split in ("train", "validation", "test")
            },
            "top_usernames": _top(
                (t.removeprefix("user=") for t in tokens if t.startswith("user=")),
                top_n,
            ),
            "top_pairs": _top(
                (t.removeprefix("pair=") for t in tokens if t.startswith("pair=")),
                top_n,
            ),
            "top_commands": _top(
                group["commands"].explode(empty_as_null=True).drop_nulls().to_list(),
                top_n,
            ),
            "top_clients": _top(
                group["client"].explode(empty_as_null=True).drop_nulls().to_list(),
                top_n,
            ),
            "top_asns": _top(group["as_label"].drop_nulls().to_list(), top_n),
            "top_countries": _top(group["country"].drop_nulls().to_list(), top_n),
            "asns": group["asn"].n_unique(),
            "telnet_share": float(group["telnet_share"].mean()),
            "command_ip_share": float(
                (group["commands"].list.len() > 0).mean()  # type: ignore[arg-type]
            ),
            "download_ip_share": float(group["downloaded"].mean()),
            "first_seen": group["first_seen"].min(),
            "last_seen": group["last_seen"].max(),
        }
        for column, name in (
            ("tried_ai_username", "ai_username_share"),
            ("listed", "blocklisted_share"),
        ):
            if column in group.columns:
                row[name] = float(group[column].cast(pl.Float64).mean())
        rows.append(row)
    return pl.DataFrame(rows).sort("ips", descending=True)


def daily_activity(events: pl.DataFrame, campaigns: pl.DataFrame) -> pl.DataFrame:
    """Distinct active IPs per campaign per UTC day (`day` as YYYY-MM-DD),
    from the session.connect events of the IPs in `campaigns` (src_ip,
    campaign)."""
    return (
        events.filter(pl.col("eventid") == "cowrie.session.connect")
        .join(campaigns, on="src_ip", how="inner")
        .group_by(
            "campaign", pl.col("event_timestamp").dt.strftime("%Y-%m-%d").alias("day")
        )
        .agg(pl.col("src_ip").n_unique().alias("ips"))
        .sort("campaign", "day")
    )
