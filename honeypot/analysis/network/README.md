# Credential-sharing graph

Clusters honeypot `src_ip`s that reused the same `(username, password)`
credential pairs into likely shared campaign/botnet infrastructure, and
separately surfaces which usernames/apps are most targeted — independent
of per-session behavioral fingerprinting (see `../fingerprinting/`).

Two questions, two different tools:
- **"Which IPs are likely related?"** — `build_credential_graph` builds a
  bipartite `networkx` graph (`src_ip` nodes <-> credential-pair nodes),
  `build_ip_similarity_graph` projects it onto IPs alone (an edge only when
  two IPs' credential sets are similar by TF-IDF cosine *and* they share
  at least two pairs), and `assign_clusters` runs `connected_components`
  over that. See "Filtering" below for why each filtering step exists.
- **"Which usernames/apps are attackers looking for?"** — `username_
  targeting_breakdown` counts distinct IPs per username directly, with no
  clustering involved at all. Deliberately separate: a widely-known
  service (`postgres`, `mysql`, ...) gets probed with the same handful of
  obvious passwords by many unrelated actors, so at the pair level it's
  exactly as noisy as `root`/`admin` — clustering can't answer "who's
  targeting X" for the same reason it needs filtering in the first place.

## Zooming in: AI-tooling usernames

`ai_targeting.py` narrows the username view to accounts tied to AI tooling
— coding agents (`claude`, `codex`, `cursor`, `openclaw`/`clawdbot`),
model vendors (`grok`, `gemini`, `deepseek`), self-hosted LLM runtimes
(`ollama`, `vllm`, `llama.cpp`) and GPU/ML hosts (`gpu`, `nvidia`,
`tensorflow`). Classification uses a curated keyword → category map
(`AI_USERNAME_KEYWORDS`) matched on whole username *tokens*, never raw
substrings — a substring match on `ai` pulls in ~100 unrelated `cai_*`
names. Usernames containing whitespace (HTTP/SIP probe lines like
`User-Agent: ...`) never match. Extend the map as new tools appear.

- `ai_username_targeting` — per AI username: category, distinct IPs,
  distinct passwords tried.
- `ai_targeting_ips` — per IP that tried any AI username: how many AI vs.
  total usernames it tried (`ai_share`), plus its credential cluster when
  `clusters` is passed. A high `ai_share` marks an IP actually hunting AI
  tooling; a low one, a broad dictionary that happens to include AI
  names (the dominant pattern so far: ~240-username lists where AI names
  are ~4%). A null `cluster_id` means every pair that IP tried was
  excluded from the graph as generic (e.g. `claude`/`claude` attempted by
  >= `min_ips_to_exclude` IPs).

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

## Filtering: two different problems, two different fixes

Naive connected-components clustering on raw credential sharing collapses
almost the whole dataset into one meaningless cluster, but not for one
reason — there are two independent failure modes, confirmed against real
data one at a time.

**1. A handful of pairs get attempted broadly and independently.**
`build_credential_graph` excludes any `(username, password)` pair attempted
by `>= min_ips_to_exclude` distinct IPs (default 20, see `graph.
DEFAULT_MIN_IPS_TO_EXCLUDE`) — without this, pairs like `admin`/`admin` or
`root`/`123456` act as hub nodes that merge every IP that ever tried them,
regardless of any other relationship. This threshold is computed fresh
from whatever `events` you pass in, not a fixed list — an earlier version
hardcoded a denylist derived from one snapshot, and it was already stale
by the next live pull (pairs like `root`/`admin` and `support`/`support`
were never added to it by hand). Pass `min_ips_to_exclude=None` to disable
this filter. `generic_credential_attempts` surfaces exactly what gets
excluded at a given threshold, for review — not silently dropped.

**2. Weak links chain transitively into a giant component.**
Even after (1), raw `connected_components` on the bipartite graph still
produced a 2,507-of-3,401-IP mega-cluster. The cause wasn't a few popular
pairs — the 39,786 credential-pair nodes inside that cluster had a
*median* distinct-IP count of 1. With tens of thousands of attempts,
individually-rare shared pairs chain transitively into one giant
component regardless of any per-pair frequency threshold (the classic
random-graph "giant component" effect).

`build_ip_similarity_graph` fixes this by projecting the bipartite graph
onto IPs and linking two IPs only when **both** of these hold:
- **TF-IDF cosine similarity >= `min_similarity`** (default 0.1, `graph.
  DEFAULT_MIN_SIMILARITY`) — how *distinctive* the overlap is. A pair
  shared by 19 IPs counts for far less than one shared by 2, and cosine
  normalizes away IPs that just try huge dictionaries.
- **>= `min_shared_pairs` pairs in common** (default 2, `graph.
  DEFAULT_MIN_SHARED_PAIRS`) — how *much* evidence there is.

Each check alone fails in its own way. Measured on the 2026-09-16
snapshot (3,401 IPs; reproduce with `--min-similarity` /
`--min-shared-pairs`):

| Edge rule | Largest clusters | Singletons |
|---|---|---|
| shared >= 2 only | 273, 23, 21 | 86% |
| cosine >= 0.1 only | 1,185, 321, 22 | 34% |
| **cosine >= 0.1 and shared >= 2** (default) | **23, 21, 20** | **94%** |
| cosine >= 0.5 and shared >= 2 | 19, 14, 13 | 96% |

- **Cosine only:** 88% of its edges at 0.1 rest on a *single* shared pair.
  Two IPs with tiny dictionaries that share one rare pair score close to
  1.0, so raising the threshold doesn't remove them (it only moves the
  largest cluster from 1,185 to 321 IPs at 0.5).
- **Shared-count only:** the 273-IP cluster looked convincing from its
  totals (~26,000 shared `root`/password pairs, ~95 pairs per IP), but
  internally it's sparse: density 0.022, median edge cosine 0.048. A
  real core (top-decile edges share 75+ pairs at cosine > 0.7) was
  chained to big-dictionary IPs that overlapped only slightly.

Each similarity-graph edge stores both values (`weight` = cosine,
`shared_pairs` = count) for inspection. Known limitation:
`connected_components` is still single-linkage, so one surviving weak
edge can merge two groups. The default may also over-split real
campaigns. Community detection (e.g. Louvain) on the weighted graph is the
natural next step if that becomes a problem.
