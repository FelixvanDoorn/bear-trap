# honeypot/analysis/ml/payload/models.py
#
# DDS ch. 9's supervised toolbox on one question (sessions.py): will this
# session try to drop a payload? Every model sees the same features
# (FeatureEncoder over sessions.py's blocks, fit on training folds only)
# and is scored by the same grouped, stratified cross-validation on the
# training weeks (splits.group_folds: no ASN in two folds). The test week
# is never seen here.
#
# Models, simplest first (MODEL_ORDER, also the tie-break order):
#
#   majority            constant: the training base rate (the floor)
#   one_rule            a decision stump: the single best split (1R)
#   naive_bayes         Bernoulli naive Bayes: tokens present / absent
#   logistic            L2 or L1 (sparse) logistic regression
#   tree                shallow decision tree (readable rules)
#   svm                 linear SVM, Platt-calibrated for probabilities
#   knn                 k nearest neighbours (rows are unit length, so
#                       Euclidean ranks neighbours like cosine)
#   random_forest       bagged trees
#   gradient_boosting   histogram gradient boosting
#
# Imbalance (3% positives): accuracy would reward "never", so the primary
# score is PR-AUC (average precision), which only depends on how well the
# positives are ranked; ROC-AUC, Brier score (calibration) and recall at
# precision >= target_precision are reported alongside. Strategies:
#
#   none           train on the data as is
#   class_weight   weight each class inversely to its frequency
#   undersample    drop random negatives to 1:1 (imbalanced-learn)
#   smote          synthesize positives between neighbours to 1:1
#
# Resampling runs inside the training folds only (imbalanced-learn's
# Pipeline resamples in fit, never in predict), so held-out folds keep
# the real class balance. Resampled models over-predict: their
# probabilities say "50% positives" -- the Brier score shows it.
#
# Two stages per horizon, to keep the grid affordable:
#
#   A. every model x its parameter grid, with its natural default
#      strategy (class_weight where the model supports it, else none);
#   B. each model's best stage-A parameters x the other strategies.
#
# Selection per horizon (`select`): the best mean PR-AUC; every candidate
# whose fold-by-fold shortfall from it is within one standard error counts
# as tied (the folds can't tell them apart), and the tie goes to the
# simplest model, then the higher mean. The decision threshold comes from
# the chosen candidate's out-of-fold predictions (max F1, and the lowest
# threshold reaching the target precision).
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from honeypot.analysis.ml.encoding import FeatureEncoder
from honeypot.analysis.ml.payload.sessions import (
    BLOCKS,
    HORIZONS,
    NUMERIC_BLOCKS,
    TOKEN_BLOCKS,
)

MODEL_ORDER = (
    "majority",
    "one_rule",
    "naive_bayes",
    "logistic",
    "tree",
    "svm",
    "knn",
    "random_forest",
    "gradient_boosting",
)
STRATEGIES = ("none", "class_weight", "undersample", "smote")
BASELINES = ("majority", "one_rule")
_WEIGHTED = ("logistic", "tree", "svm", "random_forest", "gradient_boosting")
_DENSE = ("knn", "gradient_boosting")

DEFAULT_GRID: dict[str, list[dict[str, Any]]] = {
    "majority": [{}],
    "one_rule": [{}],
    "naive_bayes": [{"alpha": 0.1}, {"alpha": 1.0}],
    "logistic": [
        {"C": 0.01},
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
        {"C": 0.01, "l1": True},
        {"C": 0.1, "l1": True},
    ],
    "tree": [{"max_depth": 3}, {"max_depth": 5}, {"max_depth": 8}],
    "svm": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
    "knn": [{"k": 10}, {"k": 50}],
    "random_forest": [{"min_samples_leaf": 1}, {"min_samples_leaf": 5}],
    "gradient_boosting": [
        {"learning_rate": 0.05, "max_leaf_nodes": 15},
        {"learning_rate": 0.1, "max_leaf_nodes": 31},
    ],
}


@dataclass(frozen=True)
class PayloadConfig:
    """Everything that determines a payload run (hashed into its id)."""

    seed: int = 0
    test_days: int = 7
    n_folds: int = 5
    horizons: tuple[int, ...] = HORIZONS
    min_df: int = 5
    block_weights: Mapping[str, float] = field(
        default_factory=lambda: {block: 1.0 for block in BLOCKS}
    )
    grid: Mapping[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {
            k: [dict(p) for p in v] for k, v in DEFAULT_GRID.items()
        }
    )
    strategies: tuple[str, ...] = STRATEGIES
    target_precision: float = 0.8
    ablation: bool = True

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["horizons"] = list(self.horizons)
        values["strategies"] = list(self.strategies)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PayloadConfig":
        values = dict(values)
        for key in ("horizons", "strategies"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def encoder(self, weights: Mapping[str, float] | None = None) -> FeatureEncoder:
        return FeatureEncoder(
            token_blocks=TOKEN_BLOCKS,
            numeric_blocks=NUMERIC_BLOCKS,
            weights=dict(self.block_weights if weights is None else weights),
            min_df=self.min_df,
        )


def default_strategy(model: str) -> str:
    return "class_weight" if model in _WEIGHTED else "none"


def applicable_strategies(model: str, strategies: tuple[str, ...]) -> list[str]:
    """Baselines only run as-is; class_weight needs a model that takes
    sample weights."""
    if model in BASELINES:
        return ["none"]
    return [s for s in strategies if s != "class_weight" or model in _WEIGHTED]


def _to_dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _classifier(
    model: str, params: Mapping[str, Any], weighted: bool, seed: int
) -> BaseEstimator:
    class_weight = "balanced" if weighted else None
    if model == "majority":
        return DummyClassifier(strategy="prior")
    if model == "one_rule":
        return DecisionTreeClassifier(max_depth=1, random_state=seed)
    if model == "naive_bayes":
        return BernoulliNB(alpha=params["alpha"])
    if model == "logistic":
        if params.get("l1"):
            return LogisticRegression(
                C=params["C"],
                l1_ratio=1.0,
                solver="liblinear",
                class_weight=class_weight,
                random_state=seed,
            )
        return LogisticRegression(
            C=params["C"], class_weight=class_weight, max_iter=2000
        )
    if model == "tree":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=seed,
        )
    if model == "svm":
        return CalibratedClassifierCV(
            LinearSVC(C=params["C"], class_weight=class_weight, random_state=seed),
            method="sigmoid",
            cv=3,
        )
    if model == "knn":
        return KNeighborsClassifier(params["k"], algorithm="brute")
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced_subsample" if weighted else None,
            n_jobs=-1,
            random_state=seed,
        )
    if model == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            class_weight=class_weight,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {model!r}")


def make_pipeline(
    model: str,
    params: Mapping[str, Any],
    strategy: str,
    seed: int,
    encoder: BaseEstimator | None = None,
) -> Pipeline:
    """[encoder] -> [resampler] -> [densify] -> classifier. Without an
    encoder it takes an already encoded matrix (cross-validation reuses
    each fold's encoding across candidates)."""
    steps: list[tuple[str, Any]] = []
    if encoder is not None:
        steps.append(("encode", encoder))
    if strategy == "undersample":
        steps.append(("resample", RandomUnderSampler(random_state=seed)))
    elif strategy == "smote":
        steps.append(("resample", SMOTE(random_state=seed)))
    if model in _DENSE:
        steps.append(("densify", FunctionTransformer(_to_dense, accept_sparse=True)))
    steps.append(
        ("model", _classifier(model, params, strategy == "class_weight", seed))
    )
    return Pipeline(steps)


class FrameEncoder(BaseEstimator, TransformerMixin):
    """FeatureEncoder as a pipeline step: polars frame -> sparse matrix,
    so a saved model takes horizon_rows() output directly."""

    def __init__(self, config: PayloadConfig) -> None:
        self.config = config

    def fit(self, frame: pl.DataFrame, y: Any = None) -> "FrameEncoder":
        self.encoder_ = self.config.encoder().fit(frame)
        return self

    def transform(self, frame: pl.DataFrame) -> sp.csr_matrix:
        return self.encoder_.transform(frame)


def final_pipeline(candidate: Mapping[str, Any], config: PayloadConfig) -> Pipeline:
    return make_pipeline(
        candidate["model"],
        json.loads(candidate["params"]),
        candidate["strategy"],
        config.seed,
        encoder=FrameEncoder(config),
    )


def positive_scores(pipeline: Pipeline, x: Any) -> np.ndarray:
    return pipeline.predict_proba(x)[:, 1]


def recall_at_precision(y: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Highest recall with precision >= target (0 if never reached)."""
    precision, recall, _ = precision_recall_curve(y, scores)
    reached = recall[precision >= target]
    return float(reached.max()) if reached.size else 0.0


def score(
    y: np.ndarray, scores: np.ndarray, target_precision: float
) -> dict[str, float]:
    both = len(np.unique(y)) == 2
    return {
        "pr_auc": float(average_precision_score(y, scores)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if both else float("nan"),
        "brier": float(brier_score_loss(y, scores)),
        "recall_at_precision": (
            recall_at_precision(y, scores, target_precision) if both else float("nan")
        ),
    }


def thresholds(
    y: np.ndarray, scores: np.ndarray, target_precision: float
) -> dict[str, float | None]:
    """max-F1 threshold, and the lowest threshold with precision >= target
    (None if no threshold reaches it)."""
    precision, recall, cut = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    reaching = np.flatnonzero(precision >= target_precision)
    return {
        "threshold_f1": float(cut[best]),
        "threshold_precision": float(cut[reaching[0]]) if reaching.size else None,
    }


@dataclass
class Folds:
    """One horizon's training rows, encoded once per fold."""

    y: np.ndarray
    fold: np.ndarray
    matrices: list[tuple[sp.csr_matrix, sp.csr_matrix]]
    encoders: list[FeatureEncoder]

    @classmethod
    def encode(
        cls,
        frame: pl.DataFrame,
        fold: np.ndarray,
        config: PayloadConfig,
        weights: Mapping[str, float] | None = None,
    ) -> "Folds":
        matrices, encoders = [], []
        for f in range(config.n_folds):
            train, held_out = frame.filter(fold != f), frame.filter(fold == f)
            encoder = config.encoder(weights).fit(train)
            matrices.append((encoder.transform(train), encoder.transform(held_out)))
            encoders.append(encoder)
        return cls(frame["dropped"].to_numpy(), fold, matrices, encoders)


def cross_validate(
    folds: Folds,
    model: str,
    params: Mapping[str, Any],
    strategy: str,
    config: PayloadConfig,
) -> tuple[list[dict[str, float]], np.ndarray, float]:
    """Per-fold scores, out-of-fold positive scores (one per training row)
    and the total fit seconds."""
    oof = np.zeros(len(folds.y))
    per_fold = []
    started = time.perf_counter()
    for f, (x_train, x_held_out) in enumerate(folds.matrices):
        train, held_out = folds.fold != f, folds.fold == f
        pipeline = make_pipeline(model, params, strategy, config.seed)
        pipeline.fit(x_train, folds.y[train])
        oof[held_out] = positive_scores(pipeline, x_held_out)
        per_fold.append(
            {"fold": f}
            | score(folds.y[held_out], oof[held_out], config.target_precision)
        )
    return per_fold, oof, time.perf_counter() - started


def candidate_id(model: str, params: Mapping[str, Any], strategy: str) -> str:
    return f"{model} {json.dumps(dict(params), sort_keys=True)} {strategy}"


def _summary(
    horizon: int,
    stage: str,
    model: str,
    params: Mapping[str, Any],
    strategy: str,
    per_fold: list[dict[str, float]],
) -> dict[str, Any]:
    frame = pl.DataFrame(per_fold)
    row: dict[str, Any] = {
        "horizon": horizon,
        "candidate": candidate_id(model, params, strategy),
        "stage": stage,
        "model": model,
        "params": json.dumps(dict(params), sort_keys=True),
        "strategy": strategy,
    }
    for metric in ("pr_auc", "roc_auc", "brier", "recall_at_precision"):
        row[f"{metric}_mean"] = float(frame[metric].mean())
        row[f"{metric}_sd"] = float(frame[metric].std())
    return row


def tune_horizon(
    horizon: int,
    folds: Folds,
    config: PayloadConfig,
    log: Callable[[str], None] = lambda _: None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, np.ndarray]]:
    """Stages A and B for one horizon: (summary per candidate, per-fold
    scores, out-of-fold scores per candidate)."""
    summaries, fold_rows, oof = [], [], {}

    def run(stage: str, model: str, params: Mapping[str, Any], strategy: str) -> None:
        per_fold, scores, seconds = cross_validate(
            folds, model, params, strategy, config
        )
        summary = _summary(horizon, stage, model, params, strategy, per_fold)
        summaries.append(summary)
        fold_rows.extend(
            {"horizon": horizon, "candidate": summary["candidate"]} | row
            for row in per_fold
        )
        oof[summary["candidate"]] = scores
        log(
            f"k={horizon} {summary['candidate']}: PR-AUC "
            f"{summary['pr_auc_mean']:.3f} +- {summary['pr_auc_sd']:.3f} "
            f"({seconds:.1f}s)"
        )

    models = [m for m in MODEL_ORDER if m in config.grid]
    for model in models:
        for params in config.grid[model]:
            run("A", model, params, default_strategy(model))
    stage_a = pl.DataFrame(summaries)
    for model in models:
        if model in BASELINES:
            continue
        best = (
            stage_a.filter(pl.col("model") == model)
            .sort("pr_auc_mean", descending=True)
            .row(0, named=True)
        )
        for strategy in applicable_strategies(model, config.strategies):
            if strategy != best["strategy"]:
                run("B", model, json.loads(best["params"]), strategy)
    return pl.DataFrame(summaries), pl.DataFrame(fold_rows), oof


def select(summary: pl.DataFrame, per_fold: pl.DataFrame) -> dict[str, Any]:
    """One horizon's chosen candidate (see header), with `rule`.

    Ties are judged on paired fold differences: every candidate is scored
    on the same folds, and how many positives a fold holds moves all of
    them together (with positives concentrated in a few ASNs, PR-AUC
    swings a lot between folds). So a candidate is tied with the best when
    its mean shortfall over the folds is within one standard error of that
    shortfall -- the one-standard-error rule on the differences, not on
    each candidate's own spread."""
    best = summary.sort("pr_auc_mean", descending=True).row(0, named=True)
    folds = per_fold.pivot(on="candidate", index="fold", values="pr_auc").sort("fold")
    best_scores = folds[best["candidate"]].to_numpy()
    gaps = []
    for candidate in summary["candidate"]:
        diff = best_scores - folds[candidate].to_numpy()
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        gaps.append({"candidate": candidate, "gap": float(diff.mean()), "gap_se": se})
    tied = (
        summary.join(pl.DataFrame(gaps), on="candidate")
        .filter(pl.col("gap") <= pl.col("gap_se"))
        .with_columns(
            pl.col("model")
            .replace_strict({m: i for i, m in enumerate(MODEL_ORDER)})
            .alias("order")
        )
    )
    chosen = tied.sort(["order", "pr_auc_mean"], descending=[False, True]).row(
        0, named=True
    )
    for key in ("order", "gap", "gap_se"):
        chosen.pop(key)
    chosen["best_candidate"] = best["candidate"]
    chosen["tied_candidates"] = tied.height
    chosen["rule"] = (
        f"simplest of {tied.height} candidates whose mean fold-by-fold "
        f"shortfall from the best ({best['pr_auc_mean']:.3f}) is within one "
        "standard error"
    )
    return chosen


def ablation(
    frame: pl.DataFrame,
    fold: np.ndarray,
    chosen: Mapping[str, Any],
    config: PayloadConfig,
) -> pl.DataFrame:
    """CV PR-AUC of the chosen candidate with each block left out in
    turn (and with all blocks, for reference)."""
    rows = []
    params = json.loads(chosen["params"])
    for dropped in (None, *BLOCKS):
        weights = dict(config.block_weights)
        if dropped is not None:
            if weights.get(dropped, 1.0) <= 0:
                continue
            weights[dropped] = 0.0
        folds = Folds.encode(frame, fold, config, weights)
        if not folds.matrices[0][0].shape[1]:
            continue
        per_fold, _, _ = cross_validate(
            folds, chosen["model"], params, chosen["strategy"], config
        )
        pr = [row["pr_auc"] for row in per_fold]
        rows.append(
            {
                "horizon": chosen["horizon"],
                "without": dropped or "(none)",
                "pr_auc_mean": float(np.mean(pr)),
                "pr_auc_sd": float(np.std(pr, ddof=1)),
            }
        )
    table = pl.DataFrame(rows)
    full = table.filter(pl.col("without") == "(none)")["pr_auc_mean"][0]
    return table.with_columns((pl.col("pr_auc_mean") - full).alias("change"))


def feature_weights(pipeline: Pipeline, top: int = 15) -> pl.DataFrame:
    """The fitted model's own view of its features, for the readable
    ones: coefficients (logistic, svm -- averaged over the calibration
    folds), log-probability ratios (naive Bayes) or impurity importances
    (trees, forest), the `top` largest by size. `kind` says which; empty
    for models without one (majority, knn, gradient boosting: see the
    block ablation instead)."""
    names = np.array(pipeline.named_steps["encode"].encoder_.feature_names())
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        kind, weights = "coefficient", model.coef_[0]
    elif isinstance(model, CalibratedClassifierCV):
        kind = "coefficient"
        weights = np.mean(
            [c.estimator.coef_[0] for c in model.calibrated_classifiers_], axis=0
        )
    elif isinstance(model, BernoulliNB):
        kind = "log_probability_ratio"
        weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    elif isinstance(model, DecisionTreeClassifier | RandomForestClassifier):
        kind, weights = "importance", model.feature_importances_
    else:
        return pl.DataFrame(
            schema={"feature": pl.Utf8, "weight": pl.Float64, "kind": pl.Utf8}
        )
    order = np.argsort(-np.abs(weights), kind="stable")[:top]
    return pl.DataFrame(
        {"feature": names[order], "weight": weights[order].astype(float)}
    ).with_columns(pl.lit(kind).alias("kind"))
