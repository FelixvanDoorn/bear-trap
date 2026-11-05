import json
from pathlib import Path

import polars as pl
import pytest

from honeypot.analysis.ml.runs import (
    Run,
    canonical_json,
    code_sha256,
    compute_run_id,
    execute,
    file_sha256,
    flatten,
    load_run,
    rebuild_index,
)


@pytest.fixture
def setup(tmp_path: Path) -> dict:
    code = tmp_path / "code"
    (code / "tests").mkdir(parents=True)
    (code / "model.py").write_text("X = 1\n")
    (code / "tests" / "test_model.py").write_text("def test(): pass\n")
    data = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(data)
    return {
        "code_dirs": (code,),
        "inputs": {"snapshot": data},
        "runs_dir": tmp_path / "runs",
        "config": {"seed": 0, "grid": {"k": [2, 3]}},
    }


def _body(run: Run) -> None:
    run.write_table("values", pl.DataFrame({"v": [0.5, 1.5]}))
    run.charts_dir.mkdir()
    (run.charts_dir / "chart.png").write_bytes(b"png")
    run.log_metrics(score=0.75, detail={"k": 3, "ok": True}, ignored=[1, 2])


def _files(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): file_sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", "run.log"}
    }


def test_canonical_json_ignores_key_order() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json(
        {"a": [1, 2], "b": 1}
    )


def test_run_id_depends_on_config_inputs_and_code_not_tests(setup: dict) -> None:
    base = compute_run_id(setup["config"], setup["inputs"], setup["code_dirs"])
    assert len(base) == 12
    assert base == compute_run_id(
        dict(reversed(list(setup["config"].items()))),
        setup["inputs"],
        setup["code_dirs"],
    )
    assert base != compute_run_id(
        {**setup["config"], "seed": 1}, setup["inputs"], setup["code_dirs"]
    )

    code = setup["code_dirs"][0]
    (code / "tests" / "test_model.py").write_text("def test(): assert 1\n")
    assert base == compute_run_id(setup["config"], setup["inputs"], setup["code_dirs"])
    (code / "model.py").write_text("X = 2\n")
    assert base != compute_run_id(setup["config"], setup["inputs"], setup["code_dirs"])

    changed_code = compute_run_id(setup["config"], setup["inputs"], setup["code_dirs"])
    pl.DataFrame({"x": [1, 2, 4]}).write_parquet(setup["inputs"]["snapshot"])
    assert changed_code != compute_run_id(
        setup["config"], setup["inputs"], setup["code_dirs"]
    )


def test_code_hash_skips_pycache(setup: dict) -> None:
    code = setup["code_dirs"][0]
    before = code_sha256(setup["code_dirs"])
    (code / "__pycache__").mkdir()
    (code / "__pycache__" / "stale.py").write_text("Y = 1\n")
    assert code_sha256(setup["code_dirs"]) == before


def test_execute_writes_a_complete_run_and_index(setup: dict) -> None:
    result = execute(_body, **setup)

    assert not result.skipped
    assert result.path == setup["runs_dir"] / result.run_id
    assert not (setup["runs_dir"] / f"{result.run_id}.tmp").exists()
    loaded = load_run(result.path)
    assert loaded["config"] == setup["config"]
    assert loaded["metrics"]["score"] == 0.75
    assert loaded["manifest"]["inputs"]["snapshot"]["sha256"] == file_sha256(
        setup["inputs"]["snapshot"]
    )
    assert loaded["manifest"]["packages"]["polars"]
    assert (result.path / "tables" / "values.parquet").is_file()
    assert "finished" in (result.path / "run.log").read_text()

    index = pl.read_parquet(setup["runs_dir"] / "index.parquet")
    assert index["run_id"].to_list() == [result.run_id]
    assert index["metrics.detail.k"].to_list() == [3]
    assert index["config.seed"].to_list() == [0]
    assert "metrics.ignored" not in index.columns
    assert (setup["runs_dir"] / "index.csv").is_file()


def test_rerun_is_skipped_and_force_reproduces_identical_files(setup: dict) -> None:
    calls = []

    def body(run: Run) -> None:
        calls.append(run.run_id)
        _body(run)

    first = execute(body, **setup)
    before = _files(first.path)
    second = execute(body, **setup)
    assert second.skipped and second.run_id == first.run_id
    assert len(calls) == 1

    forced = execute(body, force=True, **setup)
    assert not forced.skipped and len(calls) == 2
    assert _files(forced.path) == before
    assert len(pl.read_parquet(setup["runs_dir"] / "index.parquet")) == 1


def test_changed_config_gives_a_second_run(setup: dict) -> None:
    first = execute(_body, **setup)
    second = execute(_body, **{**setup, "config": {"seed": 1}})

    assert first.run_id != second.run_id
    index = rebuild_index(setup["runs_dir"])
    assert sorted(index["run_id"].to_list()) == sorted([first.run_id, second.run_id])


def test_interrupted_run_is_never_complete_and_reruns_cleanly(setup: dict) -> None:
    def crash(run: Run) -> None:
        run.write_table("partial", pl.DataFrame({"v": [1]}))
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        execute(crash, **setup)
    run_id = compute_run_id(setup["config"], setup["inputs"], setup["code_dirs"])
    assert not (setup["runs_dir"] / run_id).exists()
    assert (setup["runs_dir"] / f"{run_id}.tmp").exists()
    assert rebuild_index(setup["runs_dir"]).height == 0
    with pytest.raises(FileNotFoundError):
        load_run(setup["runs_dir"] / run_id)

    result = execute(_body, **setup)
    assert not result.skipped
    assert not (setup["runs_dir"] / f"{run_id}.tmp").exists()
    assert not (result.path / "tables" / "partial.parquet").exists()
    assert json.loads((result.path / "metrics.json").read_text())["score"] == 0.75


def test_flatten_keeps_nested_scalars_only() -> None:
    assert flatten({"a": {"b": 1, "c": [1]}, "d": "x", "e": None}) == {
        "a.b": 1,
        "d": "x",
    }
