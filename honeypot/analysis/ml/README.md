# Machine learning (DDS ch. 9)

*Data-Driven Security* ch. 9 ("Breaking into Machine Learning") splits ML
into unsupervised learning (find structure nobody labelled) and
supervised learning (predict a known label). This module does both on the
honeypot's data, as two experiments that share their building blocks and
their run tracking:

| experiment | question | details |
|---|---|---|
| `campaigns/` (unsupervised) | Which tools / campaigns attack the honeypot, how big is each, and which networks run them? | [campaigns/README.md](campaigns/README.md), `notebooks/campaigns.ipynb` |
| `payload/` (supervised) | From what a session shows at login and in its first commands, will it try to drop a payload? | [payload/README.md](payload/README.md), `notebooks/payload_drop.ipynb` |

Both are judged by how well they generalize: tuned on one part of the
data, reported once on a held-out part that's grouped by ASN (networks the
model never saw), and for the supervised half also later in time.

## What's in it

Shared, experiment-independent:

- `runs.py`: versioned, idempotent runs and saved models (below).
- `splits.py`: ASN-grouped splits (hashed, stable), the time cut, grouped
  and stratified cross-validation folds, the cold-start mask.
- `tokens.py`: per-session credential, command and client tokens.
- `encoding.py`: `FeatureEncoder`, token lists and numbers -> one sparse
  matrix of weighted blocks, fit on training rows only.
- `plotting.py`: chart palette, per-chart style, byte-stable PNGs.

Each experiment's subpackage holds its own features, method, charts and
pipeline. The two never import each other: the payload experiment reads a
finished campaigns run's `assignments` table (recorded as an input of its
run) to break its errors down by campaign.

## Runs: versioned and idempotent

Each pipeline's `run()` (and CLI) computes a run id: a 12-character
sha256 of the config (canonical JSON), the content of every input file
(snapshot, DB-IP, blocklists, a campaigns run's assignments, ...) and the
`ml/` + `common/` source code (tests excluded). Same inputs and code, same
id.

```
ml/outputs/runs/
  campaigns/ | payload_drop/   one directory per experiment
  payload_scoring/             scoring runs: frozen models on a snapshot
    index.parquet, index.csv   one row per finished run: metrics + config
    <run_id>/
      config.json              the config exactly as hashed
      manifest.json            inputs + sha256, code hash, git sha + dirty
                               flag, package versions, model hashes,
                               start / end time
      metrics.json             headline numbers (flattened into the index)
      run.log                  progress messages
      tables/*.parquet         the run's results (the table view of every
                               chart)
      charts/*.png
      models/<name>.joblib     trained models (payload: one per horizon)
      models/<name>.json       their metadata: features, threshold,
                               training window, sha256, package versions
```

- **Idempotent**: a finished run is found and skipped; `--force`
  recomputes it. A run is built in `<run_id>.tmp/` and renamed into place
  only when complete (manifest written last), so an interrupted run never
  looks finished and the next attempt starts clean.
- **Reproducible**: seeds are fixed and PNGs carry no timestamp, so a
  forced rerun reproduces every table and chart byte for byte (tested for
  both experiments). Only `manifest.json` and `run.log` carry times.
- **Versioned**: changing the config, data or code gives a new run id; old
  runs stay. The index is rebuilt from the run directories after every run
  (never appended), so it can't drift from what's on disk.
- **Models**: `Run.save_model` pickles a fitted pipeline with joblib and
  records its sha256 in the manifest. Unpickling runs code, so
  `runs.load_model` refuses a file whose hash doesn't match its metadata
  and the manifest of a finished run.
- `outputs/` is gitignored like every other module's; runs are versioned
  on disk, not in git. No MLflow: no server or extra dependencies, and the
  layout matches the repo's file-based outputs.

## Running it

```sh
# From the repo root.
uv sync --group test --group analysis
PYTHONPATH=. uv run python -m honeypot.analysis.ml.campaigns.pipeline   # ~1 min
PYTHONPATH=. uv run python -m honeypot.analysis.ml.payload.pipeline     # ~10 min
PYTHONPATH=. uv run python -m honeypot.analysis.ml.payload.score --model-run <run_id>
# both: --config overrides.json, --seed, --force, --mode live
```

Run the campaigns pipeline first if you want the payload errors broken
down by campaign (it picks up the latest finished campaigns run on the
same snapshot). Both reuse the cached DB-IP files in `../common/outputs/`.
The notebooks call the same `run()` and read the run directory, so opening
one after a pipeline run costs nothing. They use the shared "Python
(bear-trap analysis)" kernel (setup in `../network/README.md`).

## Tests

`tests/fixtures.py` plants four campaigns (telnet bot, SSH dropper, SSH
sprayer, SSH recon) with their own dictionaries, playbooks, clients and
session rates, spread over 24 ASNs; the SSH dropper doubles as the
positive class for the supervised tests. Per module:

- runs: same inputs -> same id, skip / force / interrupted runs, model
  round trip, a tampered model refused, separate experiment indexes;
- splits: grouped, deterministic, stable when data is added; time cut,
  grouped + stratified folds, cold start;
- tokens / encoding: per-session tokens, block weights and slices,
  vocabularies from training rows only;
- campaigns: planted campaigns recovered (ARI > 0.9), tuning selects four,
  byte-identical forced reruns;
- payload: label kinds and positions, populations per horizon, nothing
  after the cutoff reaches the features, fold encoders see training folds
  only, resampling only in fit, the paired selection rule, thresholds,
  evaluation and bootstrap, the pipeline end to end (saved models score
  `horizon_rows`, never trained on the test week, byte-identical forced
  reruns, the campaign breakdown); scoring with frozen models reproduces
  the training run's test week exactly and pins the model files it used.

`test_notebook.py` and `test_payload_notebook.py` run both notebooks end
to end on the fixture, writing their runs to a temp directory.

IP geolocation and ASN data by [DB-IP](https://db-ip.com), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
