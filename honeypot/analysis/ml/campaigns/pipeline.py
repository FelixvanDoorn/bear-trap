# honeypot/analysis/ml/campaigns/pipeline.py
#
# One versioned run of the campaign clustering (see runs.py for run ids,
# layout and idempotency, campaigns.py for the method):
#
#   snapshot -> IP profiles (+ DB-IP, blocklists, AI usernames)
#            -> grouped train / validation / test split
#            -> tune on train, select on validation
#            -> refit on train + validation, evaluate once on test
#            -> campaign catalog, tables, charts, metrics
#
# The run id covers the config, the snapshot, the DB-IP and blocklist
# files and the ml/ + common/ code, so rerunning with nothing changed
# skips straight to the existing run directory.
import argparse
import json
from collections.abc import Callable
from pathlib import Path

import polars as pl

from honeypot.analysis.common.enrich import (
    ENRICHMENT_DIR,
    enrich_ips,
    fetch_dbip,
    read_ranges,
)
from honeypot.analysis.common.extract import (
    SNAPSHOT_DIR,
    _latest_snapshot,
    get_sessions,
)
from honeypot.analysis.common.threatintel import (
    BLOCKLISTS,
    THREATINTEL_DIR,
    fetch_blocklist,
    match_blocklists,
    read_blocklist,
)
from honeypot.analysis.ml.campaigns.charts import (
    plot_campaign_activity,
    plot_campaign_map,
    plot_campaign_profiles,
    plot_campaign_sizes,
    plot_external_agreement,
    plot_tuning_curves,
)
from honeypot.analysis.ml.campaigns.clustering import (
    CampaignConfig,
    campaign_profiles,
    command_sequence_labels,
    credential_cluster_labels,
    daily_activity,
    embedding_2d,
    evaluate_on_test,
    external_agreement,
    fit_final,
    select_config,
    tune,
    tuning_with_selection,
)
from honeypot.analysis.ml.campaigns.profiles import ip_profiles
from honeypot.analysis.ml.runs import Run, RunResult, execute, experiment_dir
from honeypot.analysis.ml.splits import assign_splits, split_summary
from honeypot.analysis.network.ai_targeting import ai_targeting_ips

EXPERIMENT = "campaigns"
ASSIGNMENT_COLUMNS = (
    "src_ip",
    "split",
    "group",
    "campaign",
    "asn",
    "as_label",
    "country",
    "sessions",
    "login_attempts",
    "distinct_pairs",
    "first_seen",
    "last_seen",
    "tried_ai_username",
    "listed",
)


def _labelled_profiles(
    events: pl.DataFrame,
    country_ranges: pl.DataFrame,
    asn_ranges: pl.DataFrame,
    blocklists: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """ip_profiles with country/ASN, blocklist `listed` and
    `tried_ai_username` joined on (descriptive only, not features)."""
    profiles = ip_profiles(events)
    ips = profiles.select("src_ip")
    ai_ips = ai_targeting_ips(events)["src_ip"].implode()
    return (
        profiles.join(
            enrich_ips(ips, country_ranges, asn_ranges),
            on="src_ip",
            how="left",
            maintain_order="left",
        )
        .join(
            match_blocklists(ips, blocklists).select("src_ip", "listed"),
            on="src_ip",
            how="left",
            maintain_order="left",
        )
        .with_columns(pl.col("src_ip").is_in(ai_ips).alias("tried_ai_username"))
    )


def _campaign_body(
    events: pl.DataFrame,
    config: CampaignConfig,
    country_ranges: pl.DataFrame,
    asn_ranges: pl.DataFrame,
    blocklists: dict[str, pl.DataFrame],
) -> Callable[[Run], None]:
    def body(run: Run) -> None:
        run.log(f"{events.height} events, {events['src_ip'].n_unique()} source IPs")
        profiles = assign_splits(
            _labelled_profiles(events, country_ranges, asn_ranges, blocklists),
            seed=config.seed,
            fractions=dict(config.fractions),
        )
        summary = split_summary(profiles)
        run.write_table("split_summary", summary)
        train = profiles.filter(pl.col("split") == "train")
        validation = profiles.filter(pl.col("split") == "validation")
        test = profiles.filter(pl.col("split") == "test")
        run.log(f"split {summary.select('split', 'ips').rows()}")

        run.log(f"tuning {len(config.grid())} configurations")
        tuning = tune(train, validation, config)
        chosen = select_config(tuning, config)
        tuning = tuning_with_selection(tuning, chosen)
        run.write_table("tuning", tuning)
        run.log(
            f"chose {chosen['algorithm']} param={chosen['param']} "
            f"dims={chosen['n_components']} ({chosen['rule']})"
        )

        fit_profiles = pl.concat([train, validation])
        model = fit_final(fit_profiles, chosen, config)
        test_labels, test_metrics, reproducibility = evaluate_on_test(
            model, test, chosen, config
        )
        run.log(f"test {test_metrics}")

        labelled = pl.concat(
            [
                fit_profiles.with_columns(pl.Series("campaign", model.labels)),
                test.with_columns(pl.Series("campaign", test_labels)),
            ]
        ).sort("src_ip")
        assignments = labelled.select(ASSIGNMENT_COLUMNS)
        run.write_table("assignments", assignments)

        test_campaigns = assignments.filter(pl.col("split") == "test").select(
            "src_ip", "campaign"
        )
        agreement = external_agreement(
            test_campaigns,
            {
                "credential_clusters": credential_cluster_labels(events),
                "command_sequence": command_sequence_labels(events),
            },
        )
        run.write_table("external_agreement", agreement)

        catalog = campaign_profiles(labelled).join(
            reproducibility.drop("test_ips"), on="campaign", how="left"
        )
        run.write_table("catalog", catalog)
        activity = daily_activity(events, assignments.select("src_ip", "campaign"))
        run.write_table("activity", activity)

        coordinates = embedding_2d(model.embedding.transform(labelled), config.seed)
        run.write_table(
            "map_coordinates",
            assignments.select("src_ip", "split", "campaign").with_columns(
                pl.Series("x", coordinates[:, 0]), pl.Series("y", coordinates[:, 1])
            ),
        )

        plot_tuning_curves(tuning, config.min_prediction_strength, run.charts_dir)
        plot_campaign_map(coordinates, labelled["campaign"].to_numpy(), run.charts_dir)
        plot_campaign_sizes(catalog, run.charts_dir)
        plot_campaign_activity(activity, run.charts_dir)
        plot_campaign_profiles(catalog, run.charts_dir)
        plot_external_agreement(agreement, run.charts_dir)

        reproducible = catalog.filter(pl.col("test_reproducibility") >= 0.8).height
        run.log_metrics(
            data={
                "events": events.height,
                "source_ips": events["src_ip"].n_unique(),
                "clustered_ips": profiles.height,
                "connect_only_ips": events["src_ip"].n_unique() - profiles.height,
            },
            split={
                split: {"ips": ips, "groups": groups, "ip_share": share}
                for split, ips, groups, share in summary.rows()
            },
            chosen={
                key: chosen[key]
                for key in (
                    "algorithm",
                    "param",
                    "n_components",
                    "explained_variance",
                    "train_n_clusters",
                    "train_noise_share",
                    "train_silhouette",
                    "train_stability",
                    "val_silhouette",
                    "val_prediction_strength",
                    "val_prediction_strength_weighted",
                    "eligible",
                    "rule",
                )
            },
            test=test_metrics,
            campaigns={
                "count": int(len(model.ids)),
                "reproducible": reproducible,
                "largest_ips": int(catalog["ips"].max()),
            },
            agreement={row.pop("reference"): row for row in agreement.rows(named=True)},
        )

    return body


def run(
    mode: str = "offline",
    snapshot_path: Path | None = None,
    config: CampaignConfig | None = None,
    runs_dir: Path = experiment_dir(EXPERIMENT),
    snapshot_dir: Path = SNAPSHOT_DIR,
    enrichment_dir: Path = ENRICHMENT_DIR,
    threatintel_dir: Path = THREATINTEL_DIR,
    force: bool = False,
) -> RunResult:
    """Run (or find the finished run of) the campaign clustering. Live mode
    pulls a fresh snapshot first, which is then the run's input."""
    config = config or CampaignConfig()
    if mode == "live":
        get_sessions("live", snapshot_dir=snapshot_dir)
    elif mode != "offline":
        raise ValueError(f"Unknown mode: {mode!r}")
    snapshot = snapshot_path or _latest_snapshot(snapshot_dir)
    inputs = {
        "snapshot": snapshot,
        "dbip_country": fetch_dbip("country", enrichment_dir),
        "dbip_asn": fetch_dbip("asn", enrichment_dir),
        **{name: fetch_blocklist(name, threatintel_dir) for name in BLOCKLISTS},
    }
    body = _campaign_body(
        pl.read_parquet(snapshot),
        config,
        read_ranges(inputs["dbip_country"], "country"),
        read_ranges(inputs["dbip_asn"], "asn"),
        {name: read_blocklist(inputs[name]) for name in BLOCKLISTS},
    )
    return execute(body, config.to_dict(), inputs, runs_dir=runs_dir, force=force)


def load_config(path: Path | None, seed: int | None) -> CampaignConfig:
    """Defaults, overridden by the keys in a JSON file and then --seed."""
    values = CampaignConfig().to_dict()
    if path is not None:
        values |= json.loads(path.read_text())
    if seed is not None:
        values["seed"] = seed
    return CampaignConfig.from_dict(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster attacking IPs into campaigns (versioned run)."
    )
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument(
        "--snapshot", type=Path, default=None, help="Explicit snapshot parquet."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file overriding CampaignConfig fields (e.g. the k grid).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the seed.")
    parser.add_argument(
        "--force", action="store_true", help="Rerun even if this run id exists."
    )
    args = parser.parse_args()
    result = run(
        args.mode,
        snapshot_path=args.snapshot,
        config=load_config(args.config, args.seed),
        force=args.force,
    )
    metrics = json.loads((result.path / "metrics.json").read_text())
    status = "existing run (skipped)" if result.skipped else "new run"
    print(f"{status} {result.run_id}: {result.path}")
    print(
        json.dumps({k: metrics[k] for k in ("chosen", "test", "campaigns")}, indent=2)
    )


if __name__ == "__main__":
    main()
