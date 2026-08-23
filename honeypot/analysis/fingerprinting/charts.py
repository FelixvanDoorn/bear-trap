# honeypot/analysis/fingerprinting/charts.py
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

# Must be set before pyplot is imported, so a headless run never tries to
# initialize a GUI backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "outputs" / "charts"

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MAX_LABEL_LEN = 60


def _sanitize_label(value: str, max_len: int = _MAX_LABEL_LEN) -> str:
    """Session-derived text (client_version, command sequences, ...) is
    attacker-controlled and frequently isn't clean text at all -- port
    scanners and other-protocol probes (TLS ClientHellos, RDP cookies, ...)
    land here verbatim as raw bytes, which has already crashed matplotlib's
    mathtext parser once on a stray '$' in the noise. Strip control bytes
    and truncate rather than trust it as plain text."""
    cleaned = _CONTROL_CHARS.sub("�", value)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "…"
    return cleaned


_NO_BANNER_LABEL = "(no banner captured)"
_NON_SSH_LABEL = "(non-SSH protocol noise)"


def _bucket_client_versions(features: pl.DataFrame) -> pl.DataFrame:
    """Real SSH banners (per is_ssh_client, RFC 4253's "SSH-" prefix) are
    kept distinct; everything else -- TLS/HTTP/FTP probes, scanner
    payloads, and sessions with no captured banner at all -- is collapsed
    into single buckets rather than one-off values each, since roughly 40%
    of what lands in this field on a real deployment isn't an SSH client at
    all, and showing each garbage value individually buries the real
    signal in a long tail. Returns value_counts-shaped output."""
    bucketed = features.select(
        pl.when(pl.col("client_version").is_null())
        .then(pl.lit(_NO_BANNER_LABEL))
        .when(pl.col("is_ssh_client"))
        .then(pl.col("client_version"))
        .otherwise(pl.lit(_NON_SSH_LABEL))
        .alias("client_version")
    )
    return bucketed["client_version"].value_counts(sort=True)


def plot_client_fingerprint_breakdown(
    features: pl.DataFrame, output_dir: Path = OUTPUT_DIR
) -> Path:
    """Bar chart of session counts per SSH client fingerprint bucket."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = _bucket_client_versions(features)
    labels = [_sanitize_label(v) for v in counts["client_version"].to_list()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, counts["count"].to_list())
    # Sanitizing strips control bytes, but printable-but-adversarial
    # characters ($, \, _, ^, {}) can still reach mathtext -- disable math
    # parsing per-label so they're always rendered literally.
    for label in ax.get_yticklabels():
        label.set_parse_math(False)
    ax.invert_yaxis()
    ax.set_xlabel("Session count")
    ax.set_title("SSH client fingerprint breakdown")
    fig.tight_layout()

    path = output_dir / "client_fingerprint_breakdown.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _cluster_command_sequences(features: pl.DataFrame) -> pl.DataFrame:
    """Exact-match clusters of session command sequences (see
    features._hash_commands), sorted by size descending. Sessions that ran
    an identical sequence -- same commands, same order -- share one row,
    carrying a representative `commands` list plus the cluster's session
    count. Singleton clusters (count == 1) are sessions with no match to a
    known playbook -- the residual worth a closer look, not generic
    automated tooling replaying a fixed script."""
    with_commands = features.filter(pl.col("command_sequence_hash").is_not_null())
    return (
        with_commands.group_by("command_sequence_hash")
        .agg(pl.len().alias("count"), pl.col("commands").first())
        .sort("count", descending=True)
    )


def plot_command_sequence_clusters(
    features: pl.DataFrame, output_dir: Path = OUTPUT_DIR, top_n: int = 20
) -> Path:
    """Bar chart of the largest exact-match command-sequence clusters.
    Singleton (unique, unmatched) clusters aren't plotted individually --
    there could be thousands -- but are summarized in the title, since
    that count is the actual signal of interest: how much traffic doesn't
    match a known replayed playbook."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters = _cluster_command_sequences(features)

    n_sequences = clusters.height
    n_singletons = clusters.filter(pl.col("count") == 1).height
    n_sessions = int(clusters["count"].sum()) if n_sequences else 0

    top = clusters.head(top_n)
    labels = [
        _sanitize_label(" ; ".join(cmds), max_len=80)
        for cmds in top["commands"].to_list()
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, top["count"].to_list())
    for label in ax.get_yticklabels():
        label.set_parse_math(False)
    ax.invert_yaxis()
    ax.set_xlabel("Session count")
    ax.set_title(
        f"Top {top.height} command-sequence clusters of {n_sequences} "
        f"distinct sequences across {n_sessions} sessions "
        f"({n_singletons} unique/unmatched)"
    )
    fig.tight_layout()

    path = output_dir / "command_sequence_clusters.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_inter_command_timing(
    features: pl.DataFrame, output_dir: Path = OUTPUT_DIR
) -> Path:
    """Histogram of each session's mean inter-command interval."""
    output_dir.mkdir(parents=True, exist_ok=True)
    values = features["inter_command_mean_s"].drop_nulls().to_list()

    fig, ax = plt.subplots(figsize=(10, 6))
    if values:
        ax.hist(values, bins=30)
    ax.set_xlabel("Mean inter-command interval (s)")
    ax.set_ylabel("Session count")
    ax.set_title("Inter-command timing distribution")
    fig.tight_layout()

    path = output_dir / "inter_command_timing.png"
    fig.savefig(path)
    plt.close(fig)
    return path
