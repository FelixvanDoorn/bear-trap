# Honeypot overview

The "what did we collect" baseline — the descriptive-statistics ground of
*Data-Driven Security* ch. 3 — answered for a Cowrie deployment rather than
as generic security-event counting. Read this module's output before the
deeper ones (`../network/`, `../fingerprinting/`): it says what the dataset
does and doesn't contain.

## Why it's shaped this way: three Cowrie facts

Several "obvious" overview metrics are misleading on this setup, so they're
deliberately left out or reframed:

1. **Compare by `node_role`, not by cloud or sensor.** Each cloud runs a
   different role with a different config — AWS is `control_proxy` (SSH
   only), Azure is `agent_detector` (SSH + telnet + agent prompt
   handlers) — so "AWS vs Azure" is really "config A vs config B". And
   `sensor` is Cowrie's container id: it changes on every redeploy (the
   Azure VM shows up as two sensors because its container was replaced on
   2026-08-17).
2. **Authentication is configuration, not attacker skill.** `userdb.txt`
   accepts `root`/`admin` with any password (plus two fixed credentials),
   so a login success rate measures which usernames attackers try. The
   attacker-driven signal is what happens *after* login: commands, file
   transfers, tunnel requests.
3. **`dst_port` is the honeypot's own listening port** (2222 SSH, 2223
   telnet) on every event except `cowrie.direct-tcpip.request`, where it's
   the target of an attacker asking the honeypot to relay traffic. So
   there's no "top ports" table — destination ports only appear in
   `tunnel_targets`.

Covered elsewhere, not repeated here: SSH client versions, command
sequences, telnet options and timing (`../fingerprinting/`); usernames,
passwords and credential reuse (`../network/`).

## What's in it

All in `summary.py`, charted by `charts.py` (seaborn):

| Function | Answers | Chart |
|---|---|---|
| `coverage` | Which roles/protocols reported, over what window, through how many sensor containers | — |
| `session_funnel` | Share of sessions that tried a login, authenticated, ran commands, transferred files, requested a tunnel; median duration | `session_funnel.png` |
| `daily_activity` | Distinct source IPs per UTC day per role, new (first seen anywhere in the dataset) vs returning; sessions per day | `daily_activity.png` |
| `tunnel_targets` | Which hosts/ports attackers tried to relay through the honeypot, by requests, sessions and distinct source IPs | `tunnel_targets.png` |

`daily_activity` counts IPs and sessions rather than events — daily event
volume swings 16x while distinct IPs swing 5x, so event counts mostly
measure how chatty a few sessions were. It trims the dataset's first and
last dates (collection starts and stops mid-day), and every IP is "new"
for the first few days while the series warms up.

## First findings (2026-09-16 snapshot)

- **Telnet bots behave differently from SSH bots.** Telnet authenticates
  a larger share of sessions (41% vs 34% on the agent sensor's SSH) and
  runs far longer (median 37s vs ~2s), and about half of telnet sessions
  never attempt a login at all.
- **Two new-IP waves on the agent sensor** (08-30 and 09-09), the second
  followed by a returning-IP wave on 09-10/11 — a candidate for spike
  attribution once ASN enrichment lands. The control proxy sees a steady
  ~60 IPs/day, almost all new.
- **Spam-relay probing:** 732 distinct IPs each made one tunnel request to
  the same mail server on port 25 — the single biggest tunnel target.
  Most other targets are connectivity checks (`ip-who.com`, `google.com`).

## Running it

Setup (one-time `uv sync` + the shared "Python (bear-trap analysis)"
Jupyter kernel) is the same as `../network/README.md`. Then:

```sh
# From the repo root.
uv run python -m honeypot.analysis.overview.pipeline --mode offline   # or --mode live
```

Tables land in `outputs/tables/*.parquet`, charts in `outputs/charts/`
(both gitignored). For interactive exploration, open
`notebooks/eda.ipynb`; charts save a PNG and return its `Path`, so preview
them with `Image(...)` (matplotlib runs headless on the `Agg` backend).

## Tests

Same layout as the other analysis modules: `test_summary.py` (table
logic), `test_charts.py` (each chart writes a file, including empty
input), `test_pipeline.py` (offline end-to-end run) and `test_notebook.py`
(executes the real notebook against synthetic snapshots via nbconvert).
Shared synthetic-event builders live in `tests/fixtures.py`.
