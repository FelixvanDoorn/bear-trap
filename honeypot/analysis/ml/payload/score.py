# honeypot/analysis/ml/payload/score.py
#
# Batch scoring with frozen payload-drop models: a versioned run of its own
# (experiment "payload_scoring") that loads the models a finished
# payload_drop run saved, never refits anything, and scores the sessions of
# a snapshot that the models were not trained on.
#
#   snapshot -> sessions with a login (+ DB-IP, same as training)
#            -> per horizon k: horizon_rows, sessions from `since` on
#               -> saved pipeline (encoder fit on the training weeks +
#                  classifier) -> score, flags at the saved thresholds
#               -> if the snapshot has both classes: the training run's
#                  test metrics (evaluate(), ASN-group bootstrap)
#               -> vocabulary drift: share of tokens the frozen encoder
#                  has never seen, per token block
#            -> tables, metrics
#
# The model run is pinned by id (config), never "the latest": the same
# command must always score with the same model. The run id hashes the
# snapshot, the DB-IP files and every scored model file, so a different
# model or snapshot is a different scoring run.
#
# Offline, this is evaluation rather than early warning: the sessions have
# already happened. Its use is the prospective test -- frozen models on
# weeks after their training window. `since` defaults to the end of the
# training window (`trained_before` in the model metadata), so on the
# training snapshot it reproduces the training run's test-week metrics
# exactly (tested); on a later snapshot it scores weeks nobody has looked
# at. No cold-start view: which ASN groups training saw is not stored with
# the model.
import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from honeypot.analysis.common.enrich import ENRICHMENT_DIR, fetch_dbip, read_ranges
from honeypot.analysis.common.extract import SNAPSHOT_DIR, _latest_snapshot
from honeypot.analysis.ml.payload.evaluation import evaluate
from honeypot.analysis.ml.payload.models import positive_scores
from honeypot.analysis.ml.payload.pipeline import EXPERIMENT as MODEL_EXPERIMENT
from honeypot.analysis.ml.payload.pipeline import enriched_sessions
from honeypot.analysis.ml.payload.sessions import TOKEN_BLOCKS, horizon_rows
from honeypot.analysis.ml.runs import (
    Run,
    RunResult,
    execute,
    experiment_dir,
    is_complete,
    load_model,
)

EXPERIMENT = "payload_scoring"
_SCORE_COLUMNS = ("session", "src_ip", "group", "start", "dropped")


@dataclass(frozen=True)
class ScoringConfig:
    """`model_run`: the payload_drop run whose models score (required).
    `horizons`: which of its models (None = all it saved). `since`: ISO
    timestamp, score sessions starting from it (None = the end of the
    models' training window). The bootstrap mirrors the training run's."""

    model_run: str
    horizons: tuple[int, ...] | None = None
    since: str | None = None
    resamples: int = 1000
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if self.horizons is not None:
            values["horizons"] = list(self.horizons)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ScoringConfig":
        values = dict(values)
        if values.get("horizons") is not None:
            values["horizons"] = tuple(values["horizons"])
        return cls(**values)


def model_names(model_run_path: Path, horizons: tuple[int, ...] | None) -> list[str]:
    """The saved models to score with, `horizon_<k>`, in horizon order."""
    if not is_complete(model_run_path):
        raise FileNotFoundError(f"No finished payload_drop run at {model_run_path}")
    manifest = json.loads((model_run_path / "manifest.json").read_text())
    saved = sorted(manifest["models"], key=lambda name: int(name.split("_")[-1]))
    if horizons is None:
        return saved
    names = [f"horizon_{k}" for k in horizons]
    missing = sorted(set(names) - set(saved))
    if missing:
        raise ValueError(f"Run {model_run_path.name} saved no model {missing}")
    return names


def vocabulary_drift(model: Any, rows: pl.DataFrame) -> list[dict[str, Any]]:
    """Per token block: how many token occurrences in `rows` the frozen
    encoder's vocabulary lacks (it scores them as absent), and how many
    rows have no known token in the block at all."""
    encoder = model.named_steps["encode"].encoder_
    out = []
    for block in TOKEN_BLOCKS:
        vocabulary = (
            encoder.vectorizers[block].vocabulary_
            if block in encoder.vectorizers
            else {}
        )
        token_lists = rows[block].to_list()
        tokens = [t for tokens in token_lists for t in tokens]
        unknown = sum(t not in vocabulary for t in tokens)
        no_known = sum(
            not any(t in vocabulary for t in tokens) for tokens in token_lists
        )
        out.append(
            {
                "block": block,
                "tokens": len(tokens),
                "unknown_tokens": unknown,
                "unknown_share": unknown / len(tokens) if tokens else None,
                "rows_without_known_token": no_known,
            }
        )
    return out


def _scoring_body(
    events: pl.DataFrame,
    config: ScoringConfig,
    model_run_path: Path,
    names: list[str],
    country_ranges: pl.DataFrame,
    asn_ranges: pl.DataFrame,
) -> Any:
    def body(run: Run) -> None:
        sessions = enriched_sessions(events, country_ranges, asn_ranges)
        scores, metrics, drift, summary = [], [], [], {}
        for name in names:
            model, meta = load_model(model_run_path, name)
            horizon = meta["horizon"]
            since = dt.datetime.fromisoformat(config.since or meta["trained_before"])
            rows = horizon_rows(events, sessions, horizon).filter(
                pl.col("start") >= since
            )
            if rows.is_empty():
                raise ValueError(f"{name}: no sessions from {since} on to score")
            score = positive_scores(model, rows)
            cut = meta["thresholds"]
            flags = {
                f"flagged_{key.removeprefix('threshold_')}": (
                    score >= value
                    if value is not None
                    else np.zeros(len(score), dtype=bool)
                )
                for key, value in cut.items()
            }
            scores.append(
                rows.select(_SCORE_COLUMNS).with_columns(
                    pl.lit(horizon).alias("horizon"),
                    pl.Series("score", score),
                    *(pl.Series(k, v) for k, v in flags.items()),
                )
            )
            y = rows["dropped"].to_numpy()
            result = evaluate(
                y,
                score,
                rows["group"].to_numpy(),
                cut,
                meta["target_precision"],
                resamples=config.resamples,
                seed=config.seed,
            )
            row = {
                "horizon": horizon,
                "model": meta["model"],
                "candidate": meta["candidate"],
                "since": str(since),
                "trained_before": meta["trained_before"],
                "flagged_f1": int(flags["flagged_f1"].sum()),
                "mean_score": float(score.mean()),
            } | result
            metrics.append(row)
            drift += [{"horizon": horizon} | d for d in vocabulary_drift(model, rows)]
            summary[name] = {
                k: v for k, v in row.items() if k not in ("horizon", "candidate")
            }
            run.log(f"{name} ({meta['candidate']}): {result}")

        run.write_table("scores", pl.concat(scores))
        run.write_table("metrics", pl.DataFrame(metrics, infer_schema_length=None))
        run.write_table("vocabulary_drift", pl.DataFrame(drift))
        run.log_metrics(model_run=config.model_run, **summary)

    return body


def run(
    config: ScoringConfig,
    snapshot_path: Path | None = None,
    runs_dir: Path = experiment_dir(EXPERIMENT),
    model_runs_dir: Path = experiment_dir(MODEL_EXPERIMENT),
    snapshot_dir: Path = SNAPSHOT_DIR,
    enrichment_dir: Path = ENRICHMENT_DIR,
    force: bool = False,
) -> RunResult:
    """Score a snapshot with a finished payload_drop run's saved models
    (or find the finished scoring run)."""
    model_run_path = model_runs_dir / config.model_run
    names = model_names(model_run_path, config.horizons)
    snapshot = snapshot_path or _latest_snapshot(snapshot_dir)
    inputs = {
        "snapshot": snapshot,
        "dbip_country": fetch_dbip("country", enrichment_dir),
        "dbip_asn": fetch_dbip("asn", enrichment_dir),
        **{name: model_run_path / "models" / f"{name}.joblib" for name in names},
    }
    body = _scoring_body(
        pl.read_parquet(snapshot),
        config,
        model_run_path,
        names,
        read_ranges(inputs["dbip_country"], "country"),
        read_ranges(inputs["dbip_asn"], "asn"),
    )
    return execute(body, config.to_dict(), inputs, runs_dir=runs_dir, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a snapshot with frozen payload-drop models (versioned run)."
    )
    parser.add_argument(
        "--model-run", required=True, help="payload_drop run id to score with."
    )
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument(
        "--since",
        default=None,
        help="ISO timestamp; default: the end of the models' training window.",
    )
    parser.add_argument("--horizons", type=int, nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = ScoringConfig(
        model_run=args.model_run,
        horizons=tuple(args.horizons) if args.horizons else None,
        since=args.since,
    )
    result = run(config, snapshot_path=args.snapshot, force=args.force)
    metrics = json.loads((result.path / "metrics.json").read_text())
    status = "existing run (skipped)" if result.skipped else "new run"
    print(f"{status} {result.run_id}: {result.path}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
