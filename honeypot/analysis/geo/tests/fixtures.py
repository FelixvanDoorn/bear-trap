# honeypot/analysis/geo/tests/fixtures.py
#
# Synthetic data with a known answer: countries of varying address space
# whose attacker counts are drawn from a negative binomial with a chosen
# slope (beta) and dispersion (alpha), so tests can check the model
# recovers them. Shared by the geo unit, pipeline and notebook tests.
import datetime as dt
import gzip
from pathlib import Path

import numpy as np
import polars as pl

from honeypot.analysis.overview.tests.fixtures import make_event, make_events

WORLD_COUNTRIES = 40
WORLD_BETA = 0.8
WORLD_ALPHA = 0.5
_START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def nb_draw(rng: np.random.Generator, mu: np.ndarray, alpha: float) -> np.ndarray:
    """NB2 counts with mean mu and variance mu + alpha * mu^2, as a
    gamma-Poisson mixture."""
    return rng.poisson(rng.gamma(1 / alpha, alpha * mu))


def simulated_counts(
    n: int = 300,
    beta: float = WORLD_BETA,
    alpha: float = WORLD_ALPHA,
    seed: int = 0,
    poisson: bool = False,
) -> pl.DataFrame:
    """country_counts-shaped frame (country, addresses, ips, slash24s,
    asns) with log E[ips] = -6 + beta * log(addresses). slash24s and asns
    are thinned copies of ips, enough to exercise the other responses."""
    rng = np.random.default_rng(seed)
    addresses = np.round(np.exp(rng.uniform(np.log(1e3), np.log(1e9), n)))
    mu = np.exp(-6 + beta * np.log(addresses))
    ips = rng.poisson(mu) if poisson else nb_draw(rng, mu, alpha)
    return pl.DataFrame(
        {
            "country": [f"C{i:03d}" for i in range(n)],
            "addresses": addresses.astype(np.int64),
            "ips": ips.astype(np.int64),
            "slash24s": np.ceil(ips * 0.8).astype(np.int64),
            "asns": np.ceil(ips * 0.3).astype(np.int64),
        }
    )


def _country_code(i: int) -> str:
    return chr(ord("A") + i // 26) + chr(ord("A") + i % 26)


def _world_blocks() -> list[tuple[str, int, int]]:
    """(country, first octet, prefix size in addresses): country i owns
    the start of i.0.0.0 with a block of 2^(8 + i % 16) addresses."""
    return [
        (_country_code(i), i + 1, 2 ** (8 + i % 16)) for i in range(WORLD_COUNTRIES)
    ]


def write_world_dbip(enrichment_dir: Path, month: str = "2026-01") -> None:
    """DB-IP Lite country and ASN CSVs for the synthetic world, in the
    published filename format so fetch_dbip never downloads. Each
    country's block is split into two ASNs."""
    enrichment_dir.mkdir(parents=True, exist_ok=True)
    country_rows, asn_rows = [], []
    for i, (country, octet, size) in enumerate(_world_blocks()):
        start = octet * 2**24
        half = size // 2
        country_rows.append(f"{_ip(start)},{_ip(start + size - 1)},{country}")
        for j, (lo, hi) in enumerate(
            ((start, start + half - 1), (start + half, start + size - 1))
        ):
            asn = 64500 + 2 * i + j
            asn_rows.append(f'{_ip(lo)},{_ip(hi)},{asn},"Net {country}{j}"')
    for kind, rows in (("country", country_rows), ("asn", asn_rows)):
        path = enrichment_dir / f"dbip-{kind}-lite-{month}.csv.gz"
        path.write_bytes(gzip.compress(("\n".join(rows) + "\n").encode()))


def _ip(value: int) -> str:
    return ".".join(str((value >> shift) & 255) for shift in (24, 16, 8, 0))


def world_events(seed: int = 0) -> pl.DataFrame:
    """Snapshot-shaped events for the synthetic world: per country an NB
    number of attacking IPs (log E = -5 + WORLD_BETA * log(size)), each
    with one connect event at a random time over four weeks, on a random
    node role and protocol."""
    rng = np.random.default_rng(seed)
    rows = []
    for country, octet, size in _world_blocks():
        mu = np.exp(-5 + WORLD_BETA * np.log(size))
        n_ips = min(int(nb_draw(rng, np.array([mu]), WORLD_ALPHA)[0]), size)
        offsets = rng.choice(size, n_ips, replace=False)
        for offset in offsets:
            ip = _ip(octet * 2**24 + int(offset))
            role = rng.choice(["agent_detector", "control_proxy"])
            protocol = (
                "ssh" if role == "control_proxy" else rng.choice(["ssh", "telnet"])
            )
            event = make_event(
                f"s-{ip}",
                "cowrie.session.connect",
                src_ip=ip,
                node_role=str(role),
                protocol=str(protocol),
            )
            event["event_timestamp"] = _START + dt.timedelta(
                minutes=int(rng.integers(0, 28 * 24 * 60))
            )
            rows.append(event)
    return make_events(rows)
