# Predicting payload drops (DDS ch. 9, supervised)

The supervised half of `ml/` (overview, runs and tests in
`../README.md`):

> From what a session shows at login, and in its first command or two,
> will it try to drop a payload?

A session about to fetch malware is worth cutting off, recording in full
or routing to a richer sandbox *before* the payload arrives. Eight
classifiers from DDS ch. 9's toolbox, four ways of handling the rare
positives and three horizons are compared by grouped cross-validation on
the training weeks; the chosen model per horizon is tested once on the
last week. Every run is versioned and saves its trained models.

- `sessions.py`: the label, the horizon cut-off and the feature blocks.
- `models.py`: the model zoo, imbalance strategies, cross-validation,
  selection, thresholds, block ablation, feature weights.
- `evaluation.py`: the test week -- metrics with group-bootstrap
  intervals, PR curves, calibration, errors by campaign.
- `charts.py`, `pipeline.py`, and `../notebooks/payload_drop.ipynb`.

## Method

**Unit and label.** One row per session with a successful login. Positive
(`dropped`) when the session tries to get a file onto the box: a fetch
tool pointed at a network location (`wget` / `curl` / `tftp` / `ftpget` /
`lwp-download` plus a URL or IPv4), `scp` from a remote host, an echo
dropper (>= 32 hex-escaped bytes in one `echo`), or a Cowrie transfer
event (download, failed download, upload). Not positives: `busybox wget;
busybox STC` (checks wget exists) and short `echo -e "\x6F\x6B"` liveness
probes, both common in Mirai-family playbooks.

**Horizons.** At horizon *k* the model sees the session up to its login
(*k* = 0) or its *k*-th command. The population at *k*: sessions with at
least *k* commands and no drop among them. Already-dropped sessions are
decided, and sessions that ended before *k* commands are excluded so that
"fewer than *k* commands" can't leak "ended early, so negative". Nothing
after the cutoff reaches the features (tested by changing everything
after it).

**Features** (`sessions.py`), encoded by the shared `FeatureEncoder`
(TF-IDF token blocks, standardized numeric blocks, equal weights, fit on
training folds only):

| block | content |
|---|---|
| credentials | `pair=` / `user=` tokens of the attempts so far |
| client | SSH version, hassh, telnet option hash |
| commands | the first *k* normalized commands |
| context | protocol, sensor role |
| network | country, words of the AS name (`digitalocean`, `telecom`) |
| login | failed attempts before the success, usernames tried, first-try success |
| timing | log seconds connect -> login and login -> cutoff, hour of day on the circle |

Deliberately left out: **IP history** (the label is near-constant per IP:
only 28 of 620 dropping IPs ever had a non-dropping session, so history
turns it into a lookup), **campaign ids** (the clustering saw commands
and downloads, i.e. the label), **blocklists** (fetched after the fact,
partly built from these attacks) and the ASN number (useless on unseen
networks; its name's words can generalize).

**Splits** (`../splits.py`). Test = the last 7 days (from 2026-09-10),
touched once. Inside the training weeks, 5 folds grouped by ASN (no
network in two folds) and stratified (similar share of drops per fold).
The test week is reported for all sessions and for **cold start**:
sessions from ASN groups never seen in training.

**Models**, simplest first: majority (constant base rate), one-rule
(decision stump), Bernoulli naive Bayes, logistic regression (L2, C in
0.01..10; L1, C 0.01 / 0.1), decision tree (depth 3 / 5 / 8), linear SVM
(Platt-calibrated), *k*-nearest neighbours (k 10 / 50), random forest
(200 trees), histogram gradient boosting.

**Imbalance** (3.2% positives). Scored by **PR-AUC** (average precision),
which only depends on ranking the positives; ROC-AUC, Brier score and
recall at 80% precision alongside. Strategies: none, class weights,
random undersampling to 1:1, SMOTE to 1:1 (imbalanced-learn), resampling
inside training folds only. Stage A runs every model x parameters with its
natural strategy (class weights where supported); stage B runs each
model's best parameters with the other strategies.

**Selection**: the best mean PR-AUC; candidates whose mean fold-by-fold
shortfall from it is within one standard error count as tied, and the
simplest tied model wins. Paired, because positives cluster in a few
ASNs and the folds swing all candidates together (the majority baseline
alone ranges 0.05-0.35 across folds at *k* = 2); judged on each
candidate's own spread, 23 of 41 candidates tied at *k* = 2. The decision
threshold is the max-F1 point of the chosen candidate's out-of-fold
predictions.

**Test, once**: the chosen candidate refit on all training weeks; PR-AUC,
ROC-AUC, Brier, recall at 80% precision and precision / recall at the
threshold, each with a 95% bootstrap interval that resamples whole ASN
groups (a botnet's identical sessions are one draw).

## Findings (2026-09-16 snapshot, run `27ca55659c0a`)

**Data.** 58,668 sessions with a login, 1,880 drops (3.2%): 8.8% of telnet
sessions, 0.3% of SSH. Fetch commands 1,435, transfer events without a
matching command 426, echo droppers 19.

| *k* | train sessions (drops) | test sessions (drops) | cold start (drops) |
|---|---|---|---|
| 0 | 43,448 (1,633) | 15,220 (247) | 492 (37) |
| 1 | 40,863 (1,506) | 14,503 (217) | 456 (36) |
| 2 | 17,829 (1,427) | 1,275 (214) | 192 (36) |

Many bots run exactly one command and leave, so the *k* = 2 population is
a third of *k* = 1's, with 8.6% drops.

**Chosen models and the test week:**

| *k* | chosen | CV PR-AUC | test PR-AUC (95% CI) | cold start | precision / recall at threshold |
|---|---|---|---|---|---|
| 0 | logistic, C = 1, class weights | 0.745 ± 0.081 | 0.893 (0.771-0.963) | 0.858 (0.699-0.966) | 0.81 / 0.75 |
| 1 | logistic, C = 1, class weights | 0.905 ± 0.085 | 0.934 (0.805-0.986) | 0.747 (0.580-0.950) | 0.98 / 0.80 |
| 2 | naive Bayes, alpha 0.1, undersampled | 0.929 ± 0.050 | 0.939 (0.816-0.983) | 0.694 (0.505-0.951) | 0.91 / 0.86 |

Against a base rate of 1.6% (test week, *k* = 0) and baselines at 0.05
(majority, one-rule) in CV.

What stands out:

- **A drop is largely predictable at login.** Before any command,
  PR-AUC 0.89 on the test week, catching 3 in 4 drops at 81% precision.
  One or two commands lift that to 0.93-0.94 and 80-86% recall at
  91-98% precision.
- **Simple models are enough.** At every horizon the most complex
  candidates (gradient boosting, kNN) score best by a hair, but within
  one standard error of a logistic regression or naive Bayes: the folds
  can't tell them apart. Shallow decision trees are the clear losers
  (0.38-0.75): one split per token can't add up many weak signals.
- **Network and tooling carry the signal at login; commands after.**
  Leaving out the network block (country, AS-name words) costs 0.24
  PR-AUC at *k* = 0, client fingerprint and protocol 0.06 each; at *k* =
  1-2, the commands block matters most (0.12, 0.06). The heaviest weights
  at login: one SSH client fingerprint (hassh) and `protocol=telnet`
  toward a drop; GoDaddy-hosted, US and Go-client SSH sessions against.
- **Memorizing a network doesn't generalize, and grouped CV shows it.**
  On the full training weeks the best single split is the AS-name word
  `limited` (286 of 298 sessions behind it dropped), but it belongs to a
  few networks: on held-out networks the one-rule stump scores the base
  rate.
- **Unseen networks are harder later in the session.** Cold-start PR-AUC
  falls from 0.86 (*k* = 0) to 0.75 and 0.69 (*k* = 1-2) while the full
  test week rises. The intervals are wide (36-37 cold-start drops), but
  the drop suggests the command-stage models lean on network words more
  than the login-stage one does.
- **Imbalance handling matters less than the model.** No strategy wins
  everywhere (at *k* = 0: class weights for logistic regression and the
  SVM, no reweighting for the forest and boosting, SMOTE for naive Bayes
  and kNN, undersampling for the tree), and most gaps are within fold
  noise. What resampling reliably does is inflate probabilities: Brier
  scores mostly 2-10x worse than training on the data as is (the
  Platt-calibrated SVM shrugs off SMOTE, not undersampling).
- **Errors concentrate in one campaign.** On the test week, campaign 4
  (Gafgyt-style telnet) is half the drops and almost all are caught. Most
  misses and false alarms are campaign 5, the Mirai-family loader: the
  same playbook, but only some of its sessions fetch the payload, so its
  sessions look alike whether they drop or not.
- **Test beats CV at *k* = 0** (0.89 vs 0.75): the test week's drops are
  dominated by campaign 4, which is easy to spot, while the CV folds
  include rarer campaigns. The CIs overlap; one good week is not a
  guarantee.

## Caveats

- **The label is an attempt, not an infection**, and telnet dominates it.
- **One deployment, one month.** The test week shows the models carry
  over one week and to unseen networks; a new botnet with new credentials
  and client is exactly what they can't have seen.
- **AS-name words can stand in for a network** when one organization
  runs several ASNs (GoDaddy, DigitalOcean): grouped folds don't separate
  those. The cold-start view is the check, and it is weaker after the
  first commands.
- **Probabilities run high.** The chosen models were trained with class
  weights or undersampling, and sit below the diagonal in the calibration
  plot: act on the threshold, not on the raw score.
- **One accidental look at the test week.** While sizing the model grid,
  a timing script fit default-parameter models on the training weeks and
  scored them on the test week, before any selection code existed. No
  decision used those numbers: the grid and the selection rule were fixed
  in the spec beforehand, and selection only sees cross-validation on the
  training weeks.

## Running it

```sh
# From the repo root (runs the campaigns pipeline first for the
# per-campaign error breakdown).
PYTHONPATH=. uv run python -m honeypot.analysis.ml.payload.pipeline   # --config overrides.json, --seed, --force, --mode live
```

About 10 minutes. `--config` takes a JSON file overriding any
`PayloadConfig` field, e.g. `{"horizons": [0], "grid": {"logistic":
[{"C": 1.0}]}}`. The notebook reads the same file via
`BEARTRAP_ML_PAYLOAD_CONFIG`. Runs land in
`ml/outputs/runs/payload_drop/<run_id>/` with a saved model per horizon:

```python
from honeypot.analysis.ml.runs import load_model
from honeypot.analysis.ml.payload.models import positive_scores
model, meta = load_model(run_path, "horizon_0")   # sha256-checked
scores = positive_scores(model, horizon_rows(events, sessions, 0))
flagged = scores >= meta["thresholds"]["threshold_f1"]
```

IP geolocation and ASN data by [DB-IP](https://db-ip.com), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
