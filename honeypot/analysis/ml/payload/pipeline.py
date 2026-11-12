# honeypot/analysis/ml/payload/pipeline.py
#
# One versioned run of the payload-drop models (see runs.py for run ids,
# layout and idempotency, sessions.py / models.py / evaluation.py for the
# method):
#
#   snapshot -> sessions with a login (+ DB-IP country / AS name / group)
#            -> per horizon k: feature rows up to the k-th command
#               -> time split: training weeks | test week
#               -> grouped, stratified folds on the training weeks
#               -> stage A + B cross-validation, selection, threshold,
#                  block ablation
#               -> final model refit on all training weeks, saved
#               -> test week, once: all sessions and cold start
#            -> tables, charts, metrics
#
# Errors are broken down by campaign when a finished campaigns run on the
# same snapshot exists: its assignments table becomes an input (hashed
# into the run id, recorded in the manifest), so the breakdown always
# names the exact clustering it used. Campaign ids are never features
# (the clustering saw the commands and downloads, i.e. the label).
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
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
from honeypot.analysis.ml.payload.charts import (
    plot_ablation,
    plot_calibration,
    plot_errors_by_campaign,
    plot_feature_weights,
    plot_horizon_curve,
    plot_imbalance_strategies,
    plot_model_comparison,
    plot_pr_curves,
)
from honeypot.analysis.ml.payload.evaluation import (
    calibration,
    errors_by_campaign,
    evaluate,
    pr_curve,
)
from honeypot.analysis.ml.payload.models import (
    Folds,
    PayloadConfig,
    ablation,
    feature_weights,
    final_pipeline,
    positive_scores,
    select,
    thresholds,
    tune_horizon,
)
from honeypot.analysis.ml.payload.sessions import (
    NUMERIC_BLOCKS,
    TOKEN_BLOCKS,
    horizon_rows,
    session_table,
)
from honeypot.analysis.ml.runs import (
    Run,
    RunResult,
    execute,
    experiment_dir,
    file_sha256,
    is_complete,
)
from honeypot.analysis.ml.splits import (
    cold_start,
    group_folds,
    last_days_start,
    split_groups,
    time_split,
)

EXPERIMENT = "payload_drop"
CAMPAIGNS_EXPERIMENT = "campaigns"
_CHOSEN_COLUMNS = (
    "horizon",
    "candidate",
    "model",
    "params",
    "strategy",
    "stage",
    "pr_auc_mean",
    "pr_auc_sd",
    "roc_auc_mean",
    "brier_mean",
    "recall_at_precision_mean",
    "best_candidate",
    "tied_candidates",
    "rule",
    "threshold_f1",
    "threshold_precision",
)


def campaign_assignments(snapshot: Path, campaigns_dir: Path) -> Path | None:
    """assignments.parquet of the most recently finished campaigns run on
    this exact snapshot (same sha256), or None."""
    digest = file_sha256(snapshot)
    found: list[tuple[str, Path]] = []
    for path in campaigns_dir.glob("*"):
        if not is_complete(path):
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        table = path / "tables" / "assignments.parquet"
        if manifest["inputs"]["snapshot"]["sha256"] == digest and table.is_file():
            found.append((manifest["finished_at"], table))
    return max(found)[1] if found else None


def _sessions(
    events: pl.DataFrame, country_ranges: pl.DataFrame, asn_ranges: pl.DataFrame
) -> pl.DataFrame:
    sessions = session_table(events)
    enriched = sessions.join(
        enrich_ips(sessions.select("src_ip").unique(), country_ranges, asn_ranges),
        on="src_ip",
        how="left",
        maintain_order="left",
    )
    return enriched.with_columns(split_groups(enriched))


def _horizon(
    run: Run,
    events: pl.DataFrame,
    sessions: pl.DataFrame,
    horizon: int,
    test_start: Any,
    config: PayloadConfig,
    campaigns: pl.DataFrame | None,
) -> dict[str, Any]:
    """Tune, select, refit, save and test one horizon; returns its tables
    (as frames) and metrics."""
    rows = horizon_rows(events, sessions, horizon)
    split = time_split(rows["start"], test_start)
    train = rows.filter(split == "train")
    test = rows.filter(split == "test")
    y_train = train["dropped"].to_numpy()
    run.log(
        f"k={horizon}: train {train.height} ({int(y_train.sum())} drops), "
        f"test {test.height} ({test['dropped'].sum()} drops)"
    )

    fold = group_folds(train["group"], train["dropped"], config.n_folds, config.seed)
    folds = Folds.encode(train, fold, config)
    summary, per_fold, oof = tune_horizon(horizon, folds, config, log=run.log)
    chosen = select(summary, per_fold)
    cut = thresholds(y_train, oof[chosen["candidate"]], config.target_precision)
    chosen |= cut
    run.log(f"k={horizon}: chose {chosen['candidate']} ({chosen['rule']})")
    blocks = ablation(train, fold, chosen, config) if config.ablation else None

    model = final_pipeline(chosen, config).fit(train, y_train)
    run.save_model(
        f"horizon_{horizon}",
        model,
        {
            "horizon": horizon,
            "candidate": chosen["candidate"],
            "model": chosen["model"],
            "params": json.loads(chosen["params"]),
            "strategy": chosen["strategy"],
            "thresholds": cut,
            "target_precision": config.target_precision,
            "trained_before": str(test_start),
            "training_sessions": train.height,
            "training_drops": int(y_train.sum()),
            "token_blocks": list(TOKEN_BLOCKS),
            "numeric_blocks": {k: list(v) for k, v in NUMERIC_BLOCKS.items()},
            "feature_names": model.named_steps["encode"].encoder_.feature_names(),
            "input": "horizon_rows(events, sessions, horizon) frame",
        },
    )

    # The test week, once.
    scores = positive_scores(model, test)
    y_test = test["dropped"].to_numpy()
    groups = test["group"].to_numpy()
    cold = cold_start(test["group"], train["group"]).to_numpy()
    views = {"all": np.ones(len(y_test), dtype=bool), "cold_start": cold}
    test_rows = []
    for view, mask in views.items():
        result = evaluate(
            y_test[mask],
            scores[mask],
            groups[mask],
            cut,
            config.target_precision,
            seed=config.seed,
        )
        test_rows.append(
            {
                "horizon": horizon,
                "view": view,
                "cv_pr_auc": chosen["pr_auc_mean"],
                "cv_pr_auc_sd": chosen["pr_auc_sd"],
            }
            | result
        )
        run.log(f"k={horizon} test {view}: {result}")

    curves = [pr_curve(y_train, oof[chosen["candidate"]], "cv")]
    for view, mask in views.items():
        if 0 < y_test[mask].sum() < mask.sum():
            label = "test" if view == "all" else "test_cold_start"
            curves.append(pr_curve(y_test[mask], scores[mask], label))
    weights = feature_weights(model)
    return {
        "summary": summary,
        "per_fold": per_fold,
        "chosen": chosen,
        "ablation": blocks,
        "test": test_rows,
        "curves": pl.concat(curves).with_columns(pl.lit(horizon).alias("horizon")),
        "calibration": calibration(y_test, scores).with_columns(
            pl.lit(horizon).alias("horizon")
        ),
        "errors": errors_by_campaign(
            test, scores >= cut["threshold_f1"], campaigns
        ).with_columns(pl.lit(horizon).alias("horizon")),
        "weights": weights.with_columns(
            pl.lit(horizon).alias("horizon"), pl.lit(chosen["model"]).alias("model")
        ),
        "population": {
            "horizon": horizon,
            "train_sessions": train.height,
            "train_drops": int(y_train.sum()),
            "test_sessions": test.height,
            "test_drops": int(y_test.sum()),
            "test_cold_start_sessions": int(cold.sum()),
            "test_cold_start_drops": int(y_test[cold].sum()),
        },
    }


def _payload_body(
    events: pl.DataFrame,
    config: PayloadConfig,
    country_ranges: pl.DataFrame,
    asn_ranges: pl.DataFrame,
    campaigns: pl.DataFrame | None,
) -> Any:
    def body(run: Run) -> None:
        sessions = _sessions(events, country_ranges, asn_ranges)
        test_start = last_days_start(sessions["start"], config.test_days)
        run.log(
            f"{sessions.height} sessions with a login, {sessions['dropped'].sum()} "
            f"drops; test week from {test_start}"
        )
        run.write_table(
            "label_summary",
            sessions.group_by("protocol", "drop_kind")
            .agg(pl.len().alias("sessions"))
            .sort("protocol", "drop_kind", nulls_last=True),
        )

        results = [
            _horizon(run, events, sessions, k, test_start, config, campaigns)
            for k in config.horizons
        ]
        cv_summary = pl.concat([r["summary"] for r in results])
        chosen = pl.DataFrame(
            [{key: r["chosen"].get(key) for key in _CHOSEN_COLUMNS} for r in results]
        )
        test_metrics = pl.DataFrame(
            [row for r in results for row in r["test"]], infer_schema_length=None
        )
        tables = {
            "population": pl.DataFrame([r["population"] for r in results]),
            "cv_summary": cv_summary,
            "cv_folds": pl.concat([r["per_fold"] for r in results]),
            "chosen": chosen,
            "test_metrics": test_metrics,
            "pr_curves": pl.concat([r["curves"] for r in results]),
            "calibration": pl.concat([r["calibration"] for r in results]),
            "errors_by_campaign": pl.concat([r["errors"] for r in results]),
            "feature_weights": pl.concat([r["weights"] for r in results]),
        }
        if config.ablation:
            tables["ablation"] = pl.concat([r["ablation"] for r in results])
        for name, table in tables.items():
            run.write_table(name, table)

        base_rates = test_metrics.filter(pl.col("view") == "all").select(
            "horizon", (pl.col("positives") / pl.col("sessions")).alias("base_rate")
        )
        plot_model_comparison(cv_summary, chosen, run.charts_dir)
        plot_imbalance_strategies(cv_summary, run.charts_dir)
        plot_pr_curves(tables["pr_curves"], base_rates, run.charts_dir)
        plot_calibration(tables["calibration"], run.charts_dir)
        plot_horizon_curve(test_metrics, run.charts_dir)
        plot_errors_by_campaign(tables["errors_by_campaign"], run.charts_dir)
        plot_feature_weights(tables["feature_weights"], run.charts_dir)
        if config.ablation:
            plot_ablation(tables["ablation"], run.charts_dir)

        run.log_metrics(
            data={
                "sessions": sessions.height,
                "drops": int(sessions["dropped"].sum()),
                "test_start": str(test_start),
                "campaign_breakdown": campaigns is not None,
            },
            **{
                f"horizon_{r['chosen']['horizon']}": {
                    "chosen": {
                        key: r["chosen"][key]
                        for key in (
                            "model",
                            "strategy",
                            "params",
                            "pr_auc_mean",
                            "pr_auc_sd",
                            "tied_candidates",
                            "threshold_f1",
                        )
                    },
                    "population": r["population"],
                    "test": {
                        row["view"]: {
                            k: v for k, v in row.items() if k not in ("horizon", "view")
                        }
                        for row in r["test"]
                    },
                }
                for r in results
            },
        )

    return body


def run(
    mode: str = "offline",
    snapshot_path: Path | None = None,
    config: PayloadConfig | None = None,
    runs_dir: Path = experiment_dir(EXPERIMENT),
    snapshot_dir: Path = SNAPSHOT_DIR,
    enrichment_dir: Path = ENRICHMENT_DIR,
    campaigns_dir: Path = experiment_dir(CAMPAIGNS_EXPERIMENT),
    force: bool = False,
) -> RunResult:
    """Run (or find the finished run of) the payload-drop models. Live
    mode pulls a fresh snapshot first, which is then the run's input."""
    config = config or PayloadConfig()
    if mode == "live":
        get_sessions("live", snapshot_dir=snapshot_dir)
    elif mode != "offline":
        raise ValueError(f"Unknown mode: {mode!r}")
    snapshot = snapshot_path or _latest_snapshot(snapshot_dir)
    inputs = {
        "snapshot": snapshot,
        "dbip_country": fetch_dbip("country", enrichment_dir),
        "dbip_asn": fetch_dbip("asn", enrichment_dir),
    }
    assignments = campaign_assignments(snapshot, campaigns_dir)
    if assignments is not None:
        inputs["campaign_assignments"] = assignments
    body = _payload_body(
        pl.read_parquet(snapshot),
        config,
        read_ranges(inputs["dbip_country"], "country"),
        read_ranges(inputs["dbip_asn"], "asn"),
        pl.read_parquet(assignments) if assignments is not None else None,
    )
    return execute(body, config.to_dict(), inputs, runs_dir=runs_dir, force=force)


def load_config(path: Path | None, seed: int | None) -> PayloadConfig:
    """Defaults, overridden by the keys in a JSON file and then --seed."""
    values = PayloadConfig().to_dict()
    if path is not None:
        values |= json.loads(path.read_text())
    if seed is not None:
        values["seed"] = seed
    return PayloadConfig.from_dict(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict payload drops early in a session (versioned run)."
    )
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument(
        "--snapshot", type=Path, default=None, help="Explicit snapshot parquet."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file overriding PayloadConfig fields (e.g. the model grid).",
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
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
