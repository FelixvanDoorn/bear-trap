# Credential-sharing graph

Clusters honeypot `src_ip`s that reused the same `(username, password)`
credential pair, using a bipartite `networkx` graph (`src_ip` nodes <->
credential-pair nodes) and `connected_components` — a first-pass signal for
shared campaign/botnet infrastructure, independent of per-session
behavioral fingerprinting (see `../fingerprinting/`).

## One-time setup

```sh
# From the repo root.
uv sync --group test --group analysis

# Registers a Jupyter kernel pinned to this repo's .venv, with the repo
# root on PYTHONPATH so `honeypot.analysis.*` absolute imports resolve.
# This is the same "bear-trap-analysis" kernel `fingerprinting/notebooks
# /eda.ipynb` uses -- one shared kernel for all of honeypot/analysis/,
# since absolute imports mean there's no per-submodule path to scope.
uv run python -m ipykernel install --user --name bear-trap-analysis \
  --display-name "Python (bear-trap analysis)" \
  --env PYTHONPATH "$(pwd)"
```

Then open `notebooks/eda.ipynb` and select the "Python (bear-trap
analysis)" kernel — imports assume that kernel's PYTHONPATH is set;
there's no `sys.path` fallback if you pick a different interpreter.

**Note**: a pre-commit hook (`nbstripout`) strips the notebook's outputs
before every commit, so the committed notebook renders with no output on
GitHub — re-run it locally (or `uv run python -m
honeypot.analysis.network.pipeline`) to see results.

## Data access

Shared with every `honeypot/analysis/*` submodule via
`honeypot/analysis/common/extract.py`:

- **Offline (default, no credentials needed)**: reads the most recent
  parquet snapshot from `../common/outputs/snapshots/`. No snapshot ships
  in the repo (that directory is gitignored), so you need at least one
  live pull first (from this module or `fingerprinting/` — either
  refreshes the same shared snapshot store).
- **Live**: `get_sessions("live")` queries BigQuery directly via
  `gcloud`'s application-default credentials, project
  `mineral-droplet-160709`, and writes a fresh snapshot as a side effect.

Regenerate the full dataset + chart set from the command line, as a module
from the repo root (`honeypot.analysis.*` is a real Python package now, so
`python -m` resolves imports with no manual `PYTHONPATH`):

```sh
uv run python -m honeypot.analysis.network.pipeline --mode live      # or --mode offline
```

## Known quirk: charts don't render inline

Same as `fingerprinting/`: `charts.py` forces matplotlib's `Agg` backend,
so chart functions save a PNG and hand back its `Path` rather than
displaying in-cell. Preview via `Image(...)`:

```python
from IPython.display import Image

Image(plot_cluster_size_distribution(clusters))
```

## Generic credential-pair filtering

`build_credential_graph` excludes a manual denylist
(`graph._GENERIC_CREDENTIAL_PAIRS`) by default — without it, a few
high-frequency pairs (`admin`/`admin`, `root`/`123456`, ...) act as hub
nodes that merge unrelated campaigns into one oversized cluster purely
because they share a common weak password, not any real relationship.
Confirmed against real data before this was added: 690 of 883 IPs
collapsed into a single cluster without filtering.

The list was derived from the real credential-pair frequency distribution
in the 2026-08-29 offline snapshot (pairs attempted by ≥20 distinct
IPs — the point where the distribution has a real cliff), not guessed
up front. It also includes a handful of SIP/HTTP protocol-probe artifacts
that land in the `username`/`password` columns from non-SSH/Telnet
traffic, which carry no credential-sharing signal either. Pass
`exclude_pairs=frozenset()` to `build_credential_graph` to see the
unfiltered graph, or a custom `frozenset` to use a different list.
