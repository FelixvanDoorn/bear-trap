# honeypot/analysis/ml/runs.py
#
# Versioned, idempotent pipeline runs. Every run writes into its own
# directory under RUNS_DIR, named by a run id that is a hash of everything
# that determines its result:
#
#   - the config (canonical JSON: sorted keys, no whitespace),
#   - the content of every input file (sha256),
#   - the source code that produces it (sha256 over CODE_DIRS' .py files,
#     tests excluded -- editing a test doesn't change any output).
#
# So the same inputs and code always give the same run id, and a rerun
# finds the finished directory and skips the work. A run is built in
# `<run_id>.tmp/` and renamed into place only once complete, so a crashed
# or interrupted run never looks finished; the next attempt clears the
# leftover tmp dir and starts over.
#
# Layout of one run:
#
#   <run_id>/config.json     the config exactly as hashed
#   <run_id>/manifest.json   inputs + hashes, git sha, package versions,
#                            start/end time (the only non-reproducible file)
#   <run_id>/metrics.json    headline numbers, flattened into the index
#   <run_id>/run.log         progress messages
#   <run_id>/tables/*.parquet, <run_id>/charts/*.png
#
# index.parquet / index.csv in RUNS_DIR, one row per finished run, are
# rebuilt from the run directories after every run rather than appended
# to, so they can't drift from what's on disk.
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl

ANALYSIS_DIR = Path(__file__).parent.parent
# Overridable like common/'s data dirs, so the notebook's execution test
# writes its run into a temp dir instead of the real run history.
RUNS_DIR = Path(
    os.environ.get("BEARTRAP_ML_RUNS_DIR") or Path(__file__).parent / "outputs" / "runs"
)
CODE_DIRS = (ANALYSIS_DIR / "ml", ANALYSIS_DIR / "common")
RUN_ID_LENGTH = 12
PACKAGES = ("polars", "numpy", "scikit-learn", "scipy", "matplotlib", "seaborn")

_TMP_SUFFIX = ".tmp"
_INDEX_NAME = "index"


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no whitespace: equal values always give
    the same string, so it's safe to hash."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha256(code_dirs: Iterable[Path] = CODE_DIRS) -> str:
    """sha256 over the relative path and content of every .py file under
    `code_dirs`, skipping tests/ and __pycache__/, in sorted order."""
    digest = hashlib.sha256()
    for directory in code_dirs:
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(directory)
            if {"tests", "__pycache__"} & set(relative.parts):
                continue
            digest.update(f"{directory.name}/{relative.as_posix()}\0".encode())
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def compute_run_id(
    config: Mapping[str, Any],
    inputs: Mapping[str, Path],
    code_dirs: Iterable[Path] = CODE_DIRS,
) -> str:
    """Short hash of config, input file contents and source code."""
    key = {
        "config": config,
        "inputs": {name: file_sha256(path) for name, path in inputs.items()},
        "code": code_sha256(code_dirs),
    }
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()[:RUN_ID_LENGTH]


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ANALYSIS_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n")


@dataclass
class Run:
    """Handle passed to a run's body: where to write, and what to log.
    `path` is the tmp directory while the body runs."""

    run_id: str
    path: Path
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def tables_dir(self) -> Path:
        return self.path / "tables"

    @property
    def charts_dir(self) -> Path:
        return self.path / "charts"

    def write_table(self, name: str, table: pl.DataFrame) -> Path:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self.tables_dir / f"{name}.parquet"
        table.write_parquet(path)
        return path

    def log_metrics(self, **values: Any) -> None:
        """Adds headline numbers to metrics.json (written when the run
        finishes). Nested dicts are fine; the index flattens them."""
        self.metrics.update(values)

    def log(self, message: str) -> None:
        with (self.path / "run.log").open("a") as handle:
            handle.write(f"{_now()} {message}\n")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    path: Path
    skipped: bool


def is_complete(path: Path) -> bool:
    return (path / "manifest.json").is_file()


def execute(
    body: Callable[[Run], None],
    config: Mapping[str, Any],
    inputs: Mapping[str, Path],
    runs_dir: Path = RUNS_DIR,
    force: bool = False,
    code_dirs: Iterable[Path] = CODE_DIRS,
) -> RunResult:
    """Runs `body` in a fresh run directory unless a finished run with the
    same id already exists (then returns it with skipped=True). `force`
    reruns and replaces it. Rebuilds the index either way."""
    code_dirs = tuple(code_dirs)
    run_id = compute_run_id(config, inputs, code_dirs)
    final = runs_dir / run_id
    if is_complete(final) and not force:
        rebuild_index(runs_dir)
        return RunResult(run_id, final, skipped=True)

    tmp = runs_dir / f"{run_id}{_TMP_SUFFIX}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    started_at = _now()
    run = Run(run_id, tmp)
    _write_json(tmp / "config.json", config)
    run.log(f"run {run_id} started")
    body(run)
    run.log(f"run {run_id} finished")
    _write_json(tmp / "metrics.json", run.metrics)
    git_status = _git("status", "--porcelain", "--", *map(str, code_dirs))
    # Written last: its presence is what marks the run as complete.
    _write_json(
        tmp / "manifest.json",
        {
            "run_id": run_id,
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in inputs.items()
            },
            "code_sha256": code_sha256(code_dirs),
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": None if git_status is None else bool(git_status),
            "packages": _package_versions(),
            "started_at": started_at,
            "finished_at": _now(),
        },
    )
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    rebuild_index(runs_dir)
    return RunResult(run_id, final, skipped=False)


def flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dict -> {"a.b": scalar}; lists and None are dropped (the
    index holds headline scalars only)."""
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, Mapping):
            flat |= flatten(item, f"{name}.")
        elif isinstance(item, bool | int | float | str):
            flat[name] = item
    return flat


def rebuild_index(runs_dir: Path = RUNS_DIR) -> pl.DataFrame:
    """One row per finished run, oldest first: run id, finish time, git
    sha, input file names, and every scalar metric (as `metrics.<name>`)
    and config value (as `config.<name>`). Written to index.parquet and
    index.csv in `runs_dir`."""
    rows = []
    for path in sorted(runs_dir.glob("*")):
        if path.name.endswith(_TMP_SUFFIX) or not is_complete(path):
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        rows.append(
            {
                "run_id": manifest["run_id"],
                "finished_at": manifest["finished_at"],
                "git_sha": manifest["git_sha"],
                "git_dirty": manifest["git_dirty"],
                **{
                    f"input.{name}": Path(item["path"]).name
                    for name, item in manifest["inputs"].items()
                },
                **flatten(json.loads((path / "metrics.json").read_text()), "metrics."),
                **flatten(json.loads((path / "config.json").read_text()), "config."),
            }
        )
    index = (
        pl.from_dicts(rows, infer_schema_length=None).sort("finished_at", "run_id")
        if rows
        else pl.DataFrame(schema={"run_id": pl.Utf8, "finished_at": pl.Utf8})
    )
    runs_dir.mkdir(parents=True, exist_ok=True)
    index.write_parquet(runs_dir / f"{_INDEX_NAME}.parquet")
    index.write_csv(runs_dir / f"{_INDEX_NAME}.csv")
    return index


def load_run(path: Path) -> dict[str, Any]:
    """config, metrics and manifest of a finished run, for the notebook."""
    if not is_complete(path):
        raise FileNotFoundError(f"No finished run at {path}")
    return {
        name: json.loads((path / f"{name}.json").read_text())
        for name in ("config", "metrics", "manifest")
    }
