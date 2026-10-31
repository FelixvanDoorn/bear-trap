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

## Do AI probes differ from other attackers? (DDS ch. 4)

`ai_contrast.py` builds one profile per login-attempting IP (`ip_profiles`)
and screens it for differences between IPs that tried an AI-tooling
username and IPs that didn't (`compare_ai_probes`), using the generic
`../common/stats.py` (`compare_groups`):

- **Properties**: country, ASN, blocklist status, SSH client version,
  hassh, ssh/telnet mix, node role, session outcomes (authenticated / ran
  commands / file transfer / tunnel request), and distinct usernames /
  passwords, credential attempts, sessions, active days, median session
  length.
- **Tests**: chi-square for categoricals (rare levels pooled into `other`
  until every expected cell count is >= 5), Fisher's exact test for 2x2
  tables, Mann-Whitney U for numerics. Effect sizes: Cramér's V and
  rank-biserial r. **Read effect size with the q-value.** With a few
  hundred IPs, a negligible difference can still come out significant.
- **Multiple testing**: Benjamini-Hochberg FDR (`q_bh`, `significant_bh`)
  is the decision criterion. This is an exploratory screen of ~17
  correlated properties, so controlling the false-discovery rate fits
  better than family-wise error. BH stays valid under the positive
  dependence expected here (country vs ASN, client vs hassh). Holm and
  Bonferroni are reported alongside for a stricter reading. Untestable
  properties (one level, no variation) don't count toward m.
- **Size-matched**: most AI probes are broad ~240-name dictionaries, so
  against *all* IPs every test would mostly measure "big dictionary vs
  one-shot". Both groups are restricted to IPs with >= `min_usernames`
  (default 10, `--min-usernames`) distinct usernames. `distinct_usernames`
  stays in the screen as a balance check. It still differs (median 38 vs
  18 in the 2026-09-16 snapshot), so volume-type properties
  (`distinct_passwords`, `credential_attempts`, `sessions`) partly echo
  dictionary size rather than anything AI-specific.
- **Unit is the IP**, but IPs in one campaign aren't independent. In the
  2026-09-16 snapshot, 44 of the 104 AI-probe IPs (plus 13 non-AI ones)
  sit in one ASN, AS197170 TechTies (NL), all sharing a single hassh. That
  one campaign drives the country, ASN and hassh results: excluding the
  ASN, country is no longer significant and hassh drops from V = 0.55 to
  0.12. Check `category_breakdown` before reading a categorical hit as
  "AI probes in general".
- **One IP per credential cluster** (`one_ip_per_cluster`): reruns the
  screen keeping only the busiest IP per (cluster, group), so a campaign
  sharing one wordlist counts once per group. Written as
  `ai_probe_comparison_by_cluster.parquet`. **It understates dependence
  for large campaigns.** `build_credential_graph` drops pairs tried by >=
  `min_ips_to_exclude` (20) IPs as generic, and a big campaign's own
  wordlist crosses that line. In the 2026-09-16 snapshot, 22 of the 64
  IPs on the campaign's hassh aren't in the graph at all and the rest
  split over three clusters plus singletons. The matched set only shrinks
  from 589 to 546 IPs and hassh stays at V = 0.44. So a property that
  survives this check isn't shown to be independent of that campaign; use
  `without_top_ai_level` for that.

Outputs (`outputs/ai_contrast/`): `ip_enrichment.parquet`,
`ai_probe_profiles.parquet` (drill-down input for
`stats.category_breakdown`), `ai_probe_comparison.parquet`,
`ai_probe_comparison_by_cluster.parquet`.

**Visual walkthrough: `notebooks/ai_contrast.ipynb`.** It shows the screen
table, then an effect-size dot plot comparing all matched IPs with the
same screen after `without_top_ai_level` removes the dominant ASN, and
with `one_ip_per_cluster`. A long bar between a property's dots means
that effect came from one campaign. After that come share-of-IPs bars for country, ASN and hassh,
and box plots for the numeric properties. The chart functions
(`plot_ai_probe_effect_sizes`, `plot_ai_probe_breakdown`,
`plot_ai_probe_distributions`) are in `charts.py`. Its execution test
points `BEARTRAP_ENRICHMENT_DIR` at fixture DB-IP files, the enrichment
equivalent of `BEARTRAP_SNAPSHOT_DIR`.

### IP enrichment

`../common/enrich.py` adds `country`, `asn`, `as_org` and `as_label` from
DB-IP's free Lite databases (IPv4 only). The first run downloads the
current month's country and ASN CSVs (~11 MB) into
`../common/outputs/enrichment/` (gitignored), and later runs reuse them.
Delete them to refresh. These lookups show who owns or where DB-IP places
an address *now*, not at attack time.

IP geolocation and ASN data by [DB-IP](https://db-ip.com), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Already on a blocklist? (DDS ch. 4)

`../common/threatintel.py` checks every source IP against two free
public lists and adds `on_blocklist_de_ssh`, `on_firehol_level1` and
`listed` (on either):

- **blocklist.de `ssh.txt`**: IPs reported for SSH attacks. The closest
  match to our traffic, but partly fed by other honeypots, so hits are
  partly circular.
- **FireHOL `level1`**: a low-false-positive aggregate of DShield, Feodo,
  Spamhaus DROP and bogon ranges, mostly whole networks. It already
  includes Spamhaus DROP, so that list isn't fetched separately.

The first run downloads both (~150 KB) into
`../common/outputs/threatintel/` (gitignored) with the date in the
filename. Later runs reuse them; delete a file to refresh it. Like the
DB-IP data, **a list shows what was listed on the download date, not at
attack time**. blocklist.de drops IPs once reports stop, so for an older
snapshot it partly measures "still active".

`listed` is also a property in the AI-probe screen, a column in
`reportable_ips.csv`, and the basis of `outputs/threatintel/`
(`ip_listings.parquet`, `coverage_by_report_tier.parquet`,
`coverage_by_ai_probe.parquet`).

2026-09-16 snapshot against lists downloaded 2026-10-27:

- **7.5% of all 10,637 source IPs are listed** (2.2% blocklist.de, 6.1%
  FireHOL). 2,464 of the 2,651 reportable IPs (93%) are on neither list,
  so most of what this honeypot sees isn't in these public feeds.
- **AI probes look heavily listed (50% vs 7%, V = 0.47), but that's the
  campaign again.** 59 of the 64 IPs on hassh `0a07365c...` are on FireHOL
  level1, because their whole networks are: all 74 AS197170 TechTies IPs
  in the data, plus the 2 BANATSYNC ones. The 5 campaign IPs elsewhere
  aren't listed. Outside the campaign, 2.4% of matched IPs are on FireHOL.

## Reporting abusive IPs

`report.py` picks out the source IPs worth reporting and writes them to
`outputs/report/` in two formats. **Nothing is submitted anywhere.**
Uploading stays a manual step, because abuse reports are public and hard
to retract.

- `reportable_ips.csv`: one row per IP with the evidence: first/last
  seen, sessions, protocols, login attempts, distinct usernames,
  authenticated / ran commands / tunnel request / payload download,
  country and ASN, blocklist flags (see above), `tier` and
  `abuseipdb_categories`.
- `abuseipdb_bulk.csv`: AbuseIPDB's
  [bulk-report](https://www.abuseipdb.com/bulk-report) format (`IP`,
  `Categories`, `ReportDate` = the IP's last event, `Comment`).

An IP is reportable once it reaches any tier; `tier` is the highest:

| Tier | Evidence | AbuseIPDB categories |
|---|---|---|
| 1 brute force | >= `--min-login-attempts` (default 10) login attempts | 18 Brute-Force |
| 2 post-login activity | ran commands or requested a TCP tunnel | + 15 Hacking |
| 3 payload delivery | a `wget`/`curl`/`tftp`/`ftpget` command, or a Cowrie file download/upload event (failed downloads count) | + 20 Exploited Host |

Every row also gets 22 (SSH) and/or 23 (IoT Targeted, for Telnet;
AbuseIPDB has no Telnet category). Category 18 needs the attempt
threshold even when a higher tier applies. Comments state counts and
behaviour only, never the honeypot's address, credentials or command
text. Private, loopback and reserved addresses are dropped, and
`--exclude-ip` (repeatable) keeps your own scanners or monitoring out.

In the 2026-09-16 snapshot, 2,651 of 10,637 IPs are reportable: 75 tier
1, 1,956 tier 2, 620 tier 3. Tier 2 dominates because `userdb.txt` lets
root/admin in with any password, so most brute-forcers authenticate within
a few tries and go straight to running commands.

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
