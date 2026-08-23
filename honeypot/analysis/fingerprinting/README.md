# Session behavioral fingerprinting

Classifies honeypot sessions as human / scripted-bot / AI-agent-driven using
behavioral signals (SSH client fingerprint, inter-command timing, command
sequences) — independent of the explicit prompt-injection self-report in
`agent_report.*`, which only fires when a trapped command is actually run.

## One-time setup

```sh
# From the repo root. Installs pytest + polars/pyarrow/matplotlib/
# google-cloud-bigquery/ipykernel -- everything needed for both the test
# suite and interactive use.
uv sync --group test --group analysis

# Registers a Jupyter kernel pinned to this repo's .venv, so the notebook
# always runs against exactly the versions pinned in uv.lock -- never a
# separate/global Jupyter environment. --env bakes this directory onto
# PYTHONPATH so the notebook can `import extract`/`features`/`charts`
# directly, with no sys.path hack and no dependency on Jupyter's cwd
# matching the notebook's location. $(pwd) is resolved by your shell at
# registration time, so this is correct wherever you've cloned the repo --
# nothing machine-specific is committed to git, same as the interpreter
# path below it.
uv run python -m ipykernel install --user --name bear-trap-analysis \
  --display-name "Python (bear-trap analysis)" \
  --env PYTHONPATH "$(pwd)/honeypot/analysis/fingerprinting"

# Registers nbdime as this repo's git diff/merge driver for .ipynb files
# (writes to .gitattributes, which is tracked, plus local .git/config,
# which isn't -- so this is a one-time step per clone, same reasoning as
# the kernel registration above). Gives cell-level notebook diffs via
# `git diff`/`git log -p` instead of raw JSON noise.
uv run nbdime config-git --enable
```

Then open `notebooks/eda.ipynb` and select the "Python (bear-trap analysis)"
kernel — imports assume that kernel's PYTHONPATH is set; there's no
`sys.path` fallback if you pick a different interpreter.

**Note**: a pre-commit hook (`nbstripout`) strips the notebook's outputs
(charts, tables) before every commit, so git history only ever tracks code
changes. That means the committed notebook renders with no output on
GitHub — re-run it locally (or `uv run python pipeline.py`) to see results.

## Data access

- **Offline (default, no credentials needed)**: reads the most recent
  parquet snapshot from `outputs/snapshots/`. No snapshot ships in the repo
  (that directory is gitignored), so you need at least one live pull first.
- **Live**: `get_sessions("live")` queries BigQuery directly via
  `gcloud`'s application-default credentials
  (`gcloud auth application-default login`, project
  `mineral-droplet-160709`) and writes a fresh snapshot as a side effect —
  every live pull doubles as the next offline run's input.

Regenerate the full dataset + chart set from the command line:

```sh
cd honeypot/analysis/fingerprinting
uv run python pipeline.py --mode live      # or --mode offline
```

## Known quirk: charts don't render inline

`charts.py` forces matplotlib's `Agg` backend (needed so tests and
`pipeline.py` runs stay headless-safe). That means calling e.g.
`plot_client_fingerprint_breakdown(features)` in the notebook only writes a
PNG and hands back its `Path` — it won't display in-cell. To preview a chart
inline, load the saved file explicitly:

```python
from IPython.display import Image

Image(plot_client_fingerprint_breakdown(features))
```
