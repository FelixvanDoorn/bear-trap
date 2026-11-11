# Attacker campaigns (DDS ch. 9, unsupervised)

*Data-Driven Security* ch. 9 ("Breaking into Machine Learning") splits ML
into supervised learning (predict a known label) and unsupervised
learning (find structure nobody labelled). This module is the
unsupervised half:

> Which tools / campaigns attack the honeypot, how big is each, and which
> networks run them?

Attacking IPs are clustered on how they behave: the credentials they
try, the commands they run, their SSH/telnet client and their timing.
Hyperparameters are tuned on a validation split, and the result is
reported once on a held-out test split. Every run is versioned and
reproducible.

## What's in it

Shared by the ml experiments (this one, and the supervised half to come):

- `runs.py`: versioned, idempotent runs (below).
- `splits.py`: train / validation / test, grouped by ASN.
- `tokens.py`: per-session credential, command and client tokens.
- `encoding.py`: `FeatureEncoder`, token lists and numbers -> one sparse
  matrix of weighted blocks, fit on training rows only.
- `plotting.py`: chart palette and byte-stable PNGs.

The campaign clustering, in `campaigns/`:

- `profiles.py`: one row per IP with a login attempt (`ip_profiles`), and
  `campaign_encoder()`: four equally weighted blocks (credentials,
  commands, client, behaviour).
- `clustering.py`: embedding (TF-IDF + truncated SVD), k-means / Ward /
  HDBSCAN, tuning, selection, test evaluation, external agreement and the
  campaign catalog.
- `charts.py`, `pipeline.py`, and `notebooks/campaigns.ipynb`.

## Method

**Unit: the IP, not the session.** Command sessions are near-duplicates:
55k command sessions in the 2026-09-16 snapshot contain 247 distinct
command sequences, the top 20 cover 95%. Clustering sessions would just
rediscover exact sequence hashes. A campaign is a set of IPs running the
same tooling. The 4,844 IPs that tried to log in are clustered; the 5,793
that only connected (scanners) carry almost no behaviour and are only
counted.

**Features** (`campaigns/profiles.py`), per IP:

| block | content | encoding |
|---|---|---|
| credentials | `user:password` pairs and usernames tried | TF-IDF |
| commands | commands run, C2 IPs and busybox tags stripped (fingerprinting's normalization) | TF-IDF |
| client | SSH version, hassh, telnet option-sequence hash | TF-IDF |
| behaviour | log sessions / attempts / pairs, success rate, active span, sessions per active hour, command / telnet / download shares | standardized |

Each block's rows are L2-normalized and weighted equally, so 8,500
credential columns can't drown out 12 behaviour numbers. A truncated SVD
reduces the matrix to 10, 25 or 50 dimensions (tuned), and each row is
normalized again so Euclidean distance ranks pairs like cosine.

**No leakage.** TF-IDF vocabularies, scaling and the SVD are fit on the
training split only and applied to validation and test.

**Splits** (`splits.py`): 60 / 20 / 20, grouped by ASN (IPs of one ASN are
near-duplicates; geo/'s random-IP vs random-ASN split showed how much a
random split flatters). An ASN's split is a hash of (seed, ASN), so the
same seed always gives the same split and an ASN keeps its split when new
snapshots add data. The fractions hold for the 1,568 ASNs, not for IPs:
realized IP shares are 63 / 18 / 19.

**Tuning** (`campaigns.tune`): 66 configurations, 3 SVD sizes x (k-means
and Ward with k in 4..30, HDBSCAN with min cluster size 5..40). Per
configuration:

- train: silhouette, Davies-Bouldin, and **stability**: mean adjusted
  Rand index (ARI) between the full-train clustering and reclusterings of
  ten random 80% subsamples;
- validation: **prediction strength** (Tibshirani & Walther 2005).
  Cluster the validation IPs on their own; separately assign each to its
  nearest train centroid. Per validation cluster, the share of its IP
  pairs the train model also keeps together.

**Selection** (`select_config`): among configurations with train
stability >= 0.8 and weighted validation prediction strength >= 0.8, the
highest prediction strength; configurations within 0.02 of it are tied,
and the tie goes to the best validation silhouette, then fewer clusters.
Two departures from Tibshirani & Walther, both made after a first scan of
the validation results and before the test split was touched:

- their minimum over clusters is reported but not used: on this data
  it's always set by one heterogeneous leftover cluster (0.13-0.56 for
  every configuration, size-weighted 0.80-0.91), so it can't tell
  configurations apart. Campaigns that don't reproduce are flagged
  individually instead (below).
- their "largest k that passes" rule was noisy (k-means k = 20 fails, 30
  passes), and preferring more clusters on ties over-split a planted
  campaign in the test fixture: prediction strength can't tell a campaign
  from a consistent sub-split of it. Validation silhouette can.

**Test, once** (`evaluate_on_test`): the chosen configuration is refit on
train + validation; test IPs are assigned to the nearest campaign centroid
and also clustered on their own. Reported: prediction strength,
silhouette, per-campaign **test reproducibility** (the share of pairs of
test IPs assigned to a campaign that the independent test clustering also
keeps together), and agreement with two labellings found independently:
ch. 4's credential-graph clusters and each IP's dominant command
sequence. For each, `linked_kept_together` is the share of IPs the
reference links (same label) that end up in one campaign; ARI / AMI are
reported too, but are near zero by construction against ch. 4's graph,
where 606 of 639 test IPs are singletons.

## Runs: versioned and idempotent

`pipeline.run()` (and the CLI) computes a run id: a 12-character sha256
of the config (canonical JSON), the content of every input file
(snapshot, DB-IP country and ASN, both blocklists) and the `ml/` +
`common/` source code (tests excluded). Same inputs and code, same id.

```
ml/outputs/runs/
  index.parquet, index.csv     one row per finished run: metrics + config
  <run_id>/
    config.json                the config exactly as hashed
    manifest.json              inputs + sha256, code hash, git sha + dirty
                               flag, package versions, start / end time
    metrics.json               headline numbers (flattened into the index)
    run.log                    progress messages
    tables/*.parquet           split_summary, tuning, assignments, catalog,
                               external_agreement, activity, map_coordinates
    charts/*.png               tuning_curves, campaign_sizes, campaign_map,
                               campaign_profiles, campaign_activity,
                               external_agreement
```

- **Idempotent**: a finished run is found and skipped; `--force`
  recomputes it. A run is built in `<run_id>.tmp/` and renamed into place
  only when complete (manifest written last), so an interrupted run never
  looks finished and the next attempt starts clean.
- **Reproducible**: seeds are fixed and PNGs carry no timestamp, so a
  forced rerun reproduces every table and chart byte for byte (tested).
  Only `manifest.json` and `run.log` carry times.
- **Versioned**: changing the config, data or code gives a new run id; old
  runs stay. The index is rebuilt from the run directories after every run
  (never appended), so it can't drift from what's on disk.
- `outputs/` is gitignored like every other module's; runs are versioned
  on disk, not in git. No MLflow: no server or extra dependencies, and the
  layout matches the repo's file-based outputs.

## Findings (2026-09-16 snapshot, run `03e4f5dea46f`)

**Chosen**: k-means, k = 12, on 10 SVD dimensions (65% of variance).
Train stability 0.995.

| | prediction strength (weighted) | (minimum) | silhouette |
|---|---|---|---|
| validation | 0.911 | 0.39 | 0.555 |
| test | 0.912 | 0.54 | 0.556 |

Validation and test agree, so the selection didn't overfit the
validation split: the campaigns reproduce on networks the model never
saw. 9 of 12 campaigns have test reproducibility >= 0.8. Of the IPs that
ch. 4's credential graph links, 77% land in one campaign; of IPs sharing
a dominant command sequence, 89% (ARI 0.67).

**The campaigns** (names are interpretations of the catalog;
`test` = test reproducibility):

| # | IPs | sessions | test | what it looks like |
|---|---|---|---|---|
| 0 | 749 | 917 | 1.00 | Telnet credential guessing with the Mirai default dictionary, no commands; residential ISPs (UA, RU, BR); steady all month |
| 1 | 758 | 3,003 | 0.98 | SSH brute force behind an `OpenSSH_7.4` client fingerprint, root/admin dictionary; CN, IN, US; bursts late Aug and early Sep |
| 2 | 664 | 17,953 | 1.00 | Go SSH client that logs in and runs `uname -a 2>/dev/null` (recon); VPS hosting (OVH, DigitalOcean, Hetzner); ~600 IPs a day on Sep 9-12 |
| 3 | 514 | 1,037 | 1.00 | `azureuser` password spray from a `RawPasswordConnectOnly` client; 37% from Microsoft's AS8075; a single burst, Aug 30-31 |
| 4 | 517 | 642 | 0.71 | Telnet `admin:admin` with a Gafgyt-style `echo -e` tag; Argentina (SION S.A.) -- doesn't reproduce |
| 5 | 325 | 29,388 | 0.95 | Telnet Mirai-family loader (`enable; system; shell; sh`), 16% of IPs download a payload; CN, KR |
| 6 | 291 | 885 | 0.88 | Go SSH client trying `crypto` / `wallet` / `ubuntu` usernames: hunting crypto nodes; VPS hosting; Sep 9-13 |
| 7 | 257 | 491 | 0.59 | Mixed SSH `root` attempts with `uname` variants, 21% blocklisted -- doesn't reproduce (catch-all) |
| 8 | 263 | 296 | 1.00 | `OpenSSH_7.4` fingerprint, device-default usernames (`support`, `operator`, `supervisor`), no commands; KR, IN |
| 9 | 176 | 79,463 | 0.82 | The heavy hitters: 52% of all sessions from 176 IPs, full recon script; AS197170 TechTies (also geo/'s outlier), NL; 58% blocklisted, 30% try AI-tooling usernames |
| 10 | 224 | 293 | 0.50 | Few `admin:admin` SSH attempts -- doesn't reproduce (catch-all) |
| 11 | 106 | 1,201 | 0.86 | Telnet scanners sending binary junk as usernames; cloud hosting (Google, Microsoft) |

What stands out:

- **Tooling separates campaigns better than geography.** Ten of the 12
  campaigns span 100-430 ASNs; the client fingerprint (hassh / telnet
  options) and dictionary define them, not the network. The exceptions
  run from a few hosters: 9 (35 ASNs, TechTies first) and 11 (28 ASNs,
  Google and Microsoft); campaign 3 spans 184 ASNs but 37% of its IPs
  are on Azure.
- **A few IPs make most of the noise.** Campaign 9 is 3.6% of the
  clustered IPs and half of all sessions, and it's where the
  AI-tooling-username probes and the blocklisted IPs concentrate (ch. 4
  found AI probes heavily listed; this says they're largely one
  operation).
- **Bursts vs background.** Campaigns 0, 4, 5 and 9 run all month;
  2, 3 and 6 are short bursts that a single daily count would misread as
  a change in the threat level.

## Caveats

- **An IP isn't an actor.** Residential IPs churn and CGNAT puts many
  devices behind one address. A campaign is IPs that behaved alike, not
  one operator; one operator running two tools shows up twice.
- **k-means assigns every IP.** Campaigns 4, 7 and 10 don't reproduce on
  test and should be read as "other", not as tools.
- **One deployment, one month.** Two sensors, 2026-08-17 to 2026-09-16.
  The grouped split shows the campaigns reproduce on unseen networks
  within that window, not that they'd look the same next month.
- **Unvalidated labels.** Agreement with ch. 4 and the command sequences
  supports the campaigns; the names above are interpretations.

## Running it

```sh
# From the repo root.
uv sync --group test --group analysis
PYTHONPATH=. uv run python -m honeypot.analysis.ml.campaigns.pipeline   # --config grid.json, --seed, --force, --mode live
```

`--config` takes a JSON file overriding any `CampaignConfig` field, e.g.
`{"kmeans_k": [8, 10, 12], "n_components": [10]}`. A full run takes about
a minute (the 66-configuration grid is most of it) and reuses the cached
DB-IP and blocklist files in `../common/outputs/`. The notebook calls the
same `run()` and reads the run directory, so opening it after a pipeline
run costs nothing. It uses the shared "Python (bear-trap analysis)" kernel
(setup in `../network/README.md`).

## Tests

`tests/fixtures.py` plants four campaigns (telnet bot, SSH dropper, SSH
sprayer, SSH recon) with their own dictionaries, playbooks, clients and
session rates, spread over 24 ASNs. The tests check that clustering
recovers them (ARI > 0.9), that tuning selects four, that the final model
reproduces on the test split, that the run tracker skips finished runs,
recovers from interrupted ones and gives byte-identical forced reruns,
and that the split is grouped, deterministic and stable when data is
added. `test_notebook.py` runs the notebook end to end on the fixture,
writing its run to a temp directory.

IP geolocation and ASN data by [DB-IP](https://db-ip.com), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
