# honeypot/analysis/network/ai_targeting.py
#
# Zooms in on login attempts against usernames tied to AI tooling -- coding
# agents (claude, codex, openclaw, ...), local LLM runtimes (ollama, vllm,
# ...), model vendors (grok, gemini, ...) and the GPU/ML hosts they run on.
# Same per-username lens as graph.username_targeting_breakdown, narrowed to
# a curated keyword taxonomy, plus a per-IP view separating dedicated
# AI-hunting IPs from broad dictionaries that just happen to include a few
# AI names (the dominant pattern in the 2026-09-16 snapshot: ~240-username
# dictionaries where AI names are ~4% of the list).
import re

import polars as pl

from honeypot.analysis.network.graph import _pair_attempts

# Matched against whole username tokens (see _username_tokens), never as
# raw substrings: a substring match on "ai" pulls in ~100 unrelated
# cai_*/aishwarya-style names, and "ray" pulls in raymond/murray. Ambiguous
# entries (agent, gpu, nvidia) are kept deliberately -- in real data they
# get probed by the same IPs, in the same runs, as the unambiguous names --
# and ai_targeting_ips' ai_share is what tells a dedicated actor apart from
# a generic dictionary either way.
AI_USERNAME_KEYWORDS: dict[str, str] = {
    # Coding agents / AI assistants run on dev hosts.
    "claude": "agent_tooling",
    "codex": "agent_tooling",
    "cursor": "agent_tooling",
    "copilot": "agent_tooling",
    "aider": "agent_tooling",
    "cline": "agent_tooling",
    "windsurf": "agent_tooling",
    "opencode": "agent_tooling",
    "openhands": "agent_tooling",
    # OpenClaw and its earlier names (Clawdbot -> Moltbot -> OpenClaw).
    "openclaw": "agent_tooling",
    "clawdbot": "agent_tooling",
    "clawd": "agent_tooling",
    "moltbot": "agent_tooling",
    # Model vendors / hosted model names.
    "grok": "model_vendor",
    "gemini": "model_vendor",
    "chatgpt": "model_vendor",
    "gpt": "model_vendor",
    "openai": "model_vendor",
    "anthropic": "model_vendor",
    "deepseek": "model_vendor",
    "qwen": "model_vendor",
    "mistral": "model_vendor",
    "llama": "model_vendor",
    # Self-hosted LLM / generative runtimes and their UIs.
    "ollama": "llm_runtime",
    "vllm": "llm_runtime",
    "gpt4all": "llm_runtime",
    "lmstudio": "llm_runtime",
    "llamacpp": "llm_runtime",
    "localai": "llm_runtime",
    "sglang": "llm_runtime",
    "openwebui": "llm_runtime",
    "comfyui": "llm_runtime",
    "invokeai": "llm_runtime",
    # GPU / ML training infrastructure.
    "gpu": "ml_infra",
    "gpuadmin": "ml_infra",
    "nvidia": "ml_infra",
    "cuda": "ml_infra",
    "tensorflow": "ml_infra",
    "pytorch": "ml_infra",
    "torch": "ml_infra",
    "huggingface": "ml_infra",
    "mlflow": "ml_infra",
    "kubeflow": "ml_infra",
    # Generic AI-flavored service accounts.
    "ai": "generic_ai",
    "aiuser": "generic_ai",
    "llm": "generic_ai",
    "agent": "generic_ai",
}

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_TRAILING_DIGITS = re.compile(r"\d+$")


def _username_tokens(username: str) -> list[str]:
    """Lowercased tokens split on any non-alphanumeric run, each also tried
    with trailing digits stripped (gpu01 -> gpu, claude2 -> claude) --
    the unstripped form comes first so a keyword that itself ends in a
    digit (gpt4all) still matches as-is. The whole username with
    separators removed is tried before any single token, so a dotted or
    hyphenated product name (llama.cpp -> llamacpp, open-webui ->
    openwebui) wins over a component token (llama)."""
    parts = [part for part in _TOKEN_SPLIT.split(username.lower()) if part]
    tokens = ["".join(parts)] if len(parts) > 1 else []
    for token in parts:
        tokens.append(token)
        stripped = _TRAILING_DIGITS.sub("", token)
        if stripped and stripped != token:
            tokens.append(stripped)
    return tokens


def classify_ai_username(username: str) -> str | None:
    """AI_USERNAME_KEYWORDS category of the first token that matches, or
    None if no token does. A username containing whitespace is never a
    match: those are HTTP/SIP protocol-probe lines that land in the
    username column (e.g. "User-Agent: Mozilla/5.0 ..." would otherwise
    match "agent"), not real account names."""
    if any(ch.isspace() for ch in username):
        return None
    for token in _username_tokens(username):
        category = AI_USERNAME_KEYWORDS.get(token)
        if category is not None:
            return category
    return None


def _ai_attempts(attempts: pl.DataFrame) -> pl.DataFrame:
    """`attempts` (see graph._pair_attempts) restricted to AI usernames,
    with a `category` column. Classifies each distinct username once
    rather than per row."""
    usernames = attempts.select("username").unique()
    categories = usernames.with_columns(
        pl.col("username")
        .map_elements(classify_ai_username, return_dtype=pl.Utf8)
        .alias("category")
    ).filter(pl.col("category").is_not_null())
    return attempts.join(categories, on="username", how="inner")


def ai_username_targeting(events: pl.DataFrame) -> pl.DataFrame:
    """One row per AI-tooling username attempted alongside a non-null
    password: its category, distinct_ips (how many distinct src_ips tried
    it) and distinct_passwords (how many different passwords were tried
    against it). Sorted by distinct_ips descending, then username."""
    ai = _ai_attempts(_pair_attempts(events))
    return (
        ai.group_by("username", "category")
        .agg(
            pl.col("src_ip").n_unique().alias("distinct_ips"),
            pl.col("password").n_unique().alias("distinct_passwords"),
        )
        .sort(["distinct_ips", "username"], descending=[True, False])
    )


def ai_targeting_ips(
    events: pl.DataFrame, clusters: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per src_ip that attempted at least one AI-tooling username:
    ai_usernames (distinct AI usernames it tried), total_usernames
    (distinct usernames of any kind) and ai_share = their ratio. A high
    ai_share marks an IP actually hunting AI tooling; a low one, an IP
    running a broad dictionary that happens to include some AI names. If
    `clusters` (graph.assign_clusters output) is given, its cluster_id and
    cluster_size are left-joined on, null for IPs absent from the
    credential graph. Sorted by ai_usernames descending, then ai_share
    descending, then src_ip."""
    attempts = _pair_attempts(events)
    ai = _ai_attempts(attempts)

    per_ip = (
        ai.group_by("src_ip")
        .agg(pl.col("username").n_unique().alias("ai_usernames"))
        .join(
            attempts.group_by("src_ip").agg(
                pl.col("username").n_unique().alias("total_usernames")
            ),
            on="src_ip",
            how="left",
        )
        .with_columns(
            (pl.col("ai_usernames") / pl.col("total_usernames")).alias("ai_share")
        )
    )

    if clusters is not None:
        per_ip = per_ip.join(
            clusters.select("src_ip", "cluster_id", "cluster_size"),
            on="src_ip",
            how="left",
        )

    return per_ip.sort(
        ["ai_usernames", "ai_share", "src_ip"], descending=[True, True, False]
    )
