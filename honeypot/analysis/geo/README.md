# Attackers vs address space (DDS ch. 5)

Raw "attacks by country" rankings mostly show where the address space is:
the US and China top them because they hold the most IPv4 addresses.
*Data-Driven Security* ch. 5 ("From Maps to Regression") makes the point
that counts need modelling against exposure before a country "stands out".
This module asks:

> Which countries send more, or fewer, attacking IPs than their IPv4
> address space predicts, and does that hold up?

Per country, a count regression on address space:

    log E[attacking IPs] = b0 + beta * log(IPv4 addresses)

`beta` is estimated rather than fixed at 1 (an offset). `beta = 1` would
mean attackers scale in proportion to address space.

## What's in it

- `exposure.py`: IPv4 addresses per country from the cached DB-IP Lite
  ranges (`common/enrich.py`, no extra download), one row per attacking
  IP (`attacker_ips`), and per-country counts with zeros
  (`country_counts`). Three responses: `ips`, distinct `/24` networks
  (`slash24s`) and distinct ASNs (`asns`).
- `regression.py`: Poisson and negative binomial (NB2) fits, the
  model-choice checks, country residuals with BH-adjusted tests, and the
  predictive band.
- `validation.py`: split stability across five split methods, response
  robustness, influence, bootstrap and cross-validation.
- `charts.py`, `pipeline.py`, `notebooks/regression.ipynb`.

## Method

### Which count model?

Poisson assumes variance = mean. Attack counts are bursty (one campaign or
one infected ISP adds dozens of IPs at once), so the negative binomial
(variance = mu + alpha * mu^2) is fitted alongside. It nests Poisson at
alpha = 0. The notebook opens with the checks, from model-free to formal:

1. **Mean vs variance in address-space bins** (`mean_variance_table`). No
   model involved: Poisson needs variance / mean ~ 1 in every bin.
2. **Fit statistics** (`model_checks`): log-likelihood, AIC, Pearson
   dispersion (~1 when the variance model is right), observed vs expected
   share of zero countries.
3. **Tests of alpha = 0** (`overdispersion_tests`): likelihood ratio (p
   halved, since alpha = 0 is on the boundary) and Cameron-Trivedi. The
   latter is least squares on squared residuals, dominated by the largest
   countries and low-powered at ~250 rows, so it's reported but doesn't
   decide.
4. **Count bands / rootogram** (`frequency_table`): observed vs expected
   number of countries with 0, 1-4, 5-19, 20-99 and 100+ attacking IPs.
5. **Variance across the range** (`quantile_residuals`,
   `variance_by_fitted`): steps 1-4 check the variance on average; this
   checks it at every fitted size. The plot uses randomized quantile
   residuals (Dunn & Smyth 1996), which are exactly N(0, 1) at every
   fitted value when the model is right, zeros included. Pearson
   residuals are the natural first choice but mislead as a picture here:
   at y = 0 they are fixed by mu (the 89 zero countries form a curve, not
   a scatter) and they're strongly skewed at small counts, so +-2 means
   little. Pearson residuals stay where they belong: the dispersion
   statistic, the per-bin mean of squared residuals (~1 per bin under the
   right model) and naming outliers (`pearson_residuals`).

`assumption_scorecard` puts them in one table.

### Validation: splitting the data

The model is explanatory and fitted once, so validation means checking
whether its conclusions survive, not producing a test score. For split
stability, the attacking IPs are divided into parts, the model is refitted
per part, and the countries' log ratios (`log((obs + 0.5) / (exp + 0.5))`)
are rank-correlated between parts. Countries need >= 10 IPs to enter the
correlation.

| Method | Parts | What it tests | Pitfall |
|---|---|---|---|
| `random_ip` | each IP a coin flip | sampling noise | leaks: a campaign's IPs land on both sides, so agreement is inflated (naive baseline) |
| `random_asn` | each ASN a coin flip | do countries stand out across different networks? | strict: a country carried by one ISP fails it |
| `time_blocks` | IPs by first sighting, equal-size consecutive blocks | would next month look the same? | campaigns can persist across blocks |
| `node_role` | IPs seen by each honeypot | a second vantage point | the nodes have different configs |
| `protocol` | ssh-only vs telnet-only IPs | two attacker populations | not a replication |

For clustered data like this, a grouped split (`random_asn`, the
`GroupKFold` idea) is the best-practice guard against leakage. Comparing
it with `random_ip` shows how much the naive split flatters the result.
`time_blocks` is the natural out-of-sample test for this question.

The other checks: refit with `/24`s and ASNs as the response, Cook's
distance with refits after dropping the most influential countries, a
bootstrap over countries (500 resamples), and repeated 5-fold
cross-validation (held-out log-likelihood, compared per fold) between
Poisson, NB and NB with a quadratic term.

## Findings (2026-09-16 snapshot, DB-IP 2026-10)

10,637 attacking IPs in 245 countries; 89 countries sent none.

**Negative binomial, not Poisson.** Every check except Cameron-Trivedi
agrees:
- Variance / mean rises from 1.0 in the smallest-country bins to 497 in
  the largest. The model-free alpha (2.0) is close to the fitted 1.55.
- Likelihood ratio p < 1e-16; AIC 11,509 (Poisson) vs 1,441 (NB).
- Zero countries: 36% observed, Poisson expects 11%, NB 35%.
- Pearson dispersion 80.5 (Poisson) vs 2.6 (NB).
- Out of sample, Poisson's held-out log-likelihood is ~22 per country worse
  in every fold.
- Cameron-Trivedi alone is borderline (one-sided p = 0.06).

NB's dispersion of 2.6 isn't a wrong variance curve:
- Quantile residuals: NB has 4% outside +-2 (5% expected under N(0, 1))
  and no funnel; Poisson 56%, fanning out with the fitted count.
- Mean squared Pearson residual by fitted-value bin: Poisson climbs from
  0.8 to 310; NB is 1.2-1.5 in the two largest bins. The middle bin's
  8.3 is UZ and MM (its median is 0.23). In the smallest bin (fitted
  ~0.2) NB sits at 0.26, slightly overstating the variance of countries
  expected to send almost nothing.
- So three countries carry the excess (Pearson residuals: UZ 15, MM 13,
  AD 7.5).

**Slope: beta = 0.87, but "below 1" is fragile.**
- The Wald CI is [0.78, 0.95] (p = 0.003 for beta = 1), but the bootstrap
  CI is [0.77, 1.00].
- Dropping the three most influential countries (UZ, MM, AD) gives 0.94
  [0.86, 1.03].
- A quadratic term beats the straight line in 74% of CV folds (small gain:
  +0.02 log-likelihood per country): the line bends. The slope is about 1
  for mid-sized countries and flatter for the largest address holders
  (JP, US, much of it legacy space).
- So attackers scale roughly in proportion to address space, except that
  the biggest holders send fewer per address.

**Which countries stand out.**
- After BH correction only **Uzbekistan (103 IPs vs 4.9 expected, 21x)**
  and **Myanmar (66 vs 3.6, 18x)** are individually significant.
- Next come Andorra 12x, Ukraine 8x, Argentina 7x, Pakistan 7x and Iraq
  6x (q 0.10-0.51).
- Lowest: Japan 0.05x, Australia 0.09x, Ireland, Finland and Egypt ~0.1x.
  None is individually significant. The NB predictive distribution is
  wide, so a ratio of 5 can still be ordinary.
- **The raw top two are below the line.** The US (1,663 IPs, 0.28x) and
  China (881, 0.52x) lead the raw ranking only because they hold the most
  address space. Per address, both send fewer attackers than expected.

**Does it hold up?**
- **Time: yes.** First vs second half, Spearman rho = 0.90, beta 0.84 /
  0.85. Four time blocks: median rho 0.78.
- **Naive vs grouped split: big gap.** Random IP halves rho = 0.93;
  random ASN halves rho = 0.47 (median of 5 seeds), and no country stays
  significant in both. The extremes (UZ, MM, UA, AR, PK) stay above
  expectation in both ASN halves for all 5 seeds, so those are
  country-wide patterns. The middle of the
  ranking is carried by individual networks. Read a mid-table country's
  ratio as "one or two networks", not "the country".
- **Vantage point and protocol:** rho = 0.77 for each. Slopes 0.79-0.86
  by node, and 0.81 (ssh) vs 0.91 (telnet).
- **Response:** counting `/24`s changes almost nothing (rho = 0.99, beta
  0.87), so IP churn within subnets isn't driving the ranking. Counting
  ASNs gives a flatter beta (0.72, rho = 0.84): bigger countries have
  attackers spread over more networks, but not proportionally more.

**Follow-ups: what carries the outliers** (notebook Part 4,
`country_networks`, `compare_country_residuals`, `compare_slopes`):

| Country | IPs vs expected | /24 count vs expected | Networks | Read as |
|---|---|---|---|---|
| Myanmar | 18x (q 0.002) | 15x (q 0.005) | 23 ASNs, largest 15% | the strongest country-wide finding |
| Uzbekistan | 21x (q 0.0005) | 10x (q 0.07) | 75% one ISP (AS8193 Uzbektelekom), 2.6 IPs per /24 | real, but concentrated in one national ISP |
| Ukraine | 8x | 8x | 233 ASNs, largest 7% | country-wide, mostly Telnet |
| Argentina | 7x | 6x | 157 ASNs, largest 18% | country-wide |
| Pakistan | 7x | 4.6x | 50% one ISP (AS9541 Cyber Internet Services) | Telnet, half one ISP |
| Andorra | 12x | 4x | 15 of 16 IPs one hosting provider (AS48090), SSH only | a hosting artefact, not the country |

- **SSH vs Telnet slopes don't differ significantly.** SSH-only IPs give
  beta 0.81 [0.73, 0.90]; Telnet-only 0.91 [0.79, 1.03], consistent with
  proportional scaling. The difference is -0.10 [-0.24, 0.05], p = 0.19.
- **Which countries stand out does differ by protocol.** Pakistan is 21x
  for Telnet (q 0.002) but 1.5x for SSH, and Ukraine 13x for Telnet (q
  0.05) vs 5.6x for SSH. Their excess is the Telnet IoT botnet traffic
  described in `../network/README.md`. Uzbekistan and Myanmar stand out
  under both protocols; Andorra only under SSH.

## Caveats

- **Exposure is allocation, not use.** DB-IP counts every allocated
  address, so countries with large dark legacy blocks (US, JP) look
  under-represented partly by construction. That's part of the curvature
  above.
- **Placement is current, not historical.** DB-IP shows who holds, and
  where it places, an address in its 2026-10 release, not at attack time.
- **Country is where the device is, not the operator.** Infected routers
  in a country say nothing about who runs the botnet.
- **One honeypot pair, one month.** Two sensors see a small slice of the
  internet. The time split shows stability within this month, not
  beyond it.
- Missing covariates (internet users, broadband penetration, patching
  practice) likely explain some residuals: rich, well-administered
  countries sit below the line. Adding them is a separate step and should
  be judged with the cross-validation already here.

## Running it

```sh
# From the repo root.
uv sync --group test --group analysis
PYTHONPATH=. uv run python -m honeypot.analysis.geo.pipeline   # --n-boot, --repeats, --min-ips
```

Outputs go to `outputs/` (gitignored): `model/` and `validation/` parquet
tables, `charts/` PNGs. The notebook uses the shared "Python (bear-trap
analysis)" kernel (setup in `../network/README.md`).

## Tests

`tests/fixtures.py` builds synthetic data with a known answer: countries
whose counts are drawn from a negative binomial with a chosen slope and
dispersion, so the tests check that the fit recovers them, that Poisson
data passes the overdispersion checks, and that an injected outlier is
flagged and ranked most influential. `test_notebook.py` runs the notebook
end to end on a synthetic world with fixture DB-IP files.

IP geolocation and ASN data by [DB-IP](https://db-ip.com), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
