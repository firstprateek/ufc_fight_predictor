"""Download the source CSVs at one upstream commit into the raw cache."""

from __future__ import annotations

from pathlib import Path

import httpx

REPO = "Greco1899/scrape_ufc_stats"
BRANCH = "main"
FILES = [
    "ufc_event_details",
    "ufc_fight_details",
    "ufc_fight_results",
    "ufc_fight_stats",
    "ufc_fighter_details",
    "ufc_fighter_tott",
]
USER_AGENT = "ufc-pipeline/0.1 (research)"
LATEST_FILE = "latest.txt"


def latest_sha(client: httpx.Client) -> str:
    r = client.get(
        f"https://api.github.com/repos/{REPO}/commits/{BRANCH}",
        headers={"Accept": "application/vnd.github.sha"},
    )
    r.raise_for_status()
    return r.text.strip()


def fetch(raw_dir: Path, sha: str | None = None, client: httpx.Client | None = None) -> Path:
    """Fetch the six CSVs for `sha` (default: current head of the branch). Files already in the
    cache are kept. Returns the snapshot directory and records it as the latest."""
    own = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
    )
    try:
        sha = sha or latest_sha(client)
        snap = raw_dir / sha
        snap.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            dest = snap / f"{name}.csv"
            if dest.exists():
                continue
            url = f"https://raw.githubusercontent.com/{REPO}/{sha}/{name}.csv"
            r = client.get(url)
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)
        (raw_dir / LATEST_FILE).write_text(sha + "\n")
        return snap
    finally:
        if own:
            client.close()


def latest_snapshot(raw_dir: Path) -> Path:
    f = raw_dir / LATEST_FILE
    if not f.exists():
        raise FileNotFoundError(f"no snapshot in {raw_dir}; run `ufc fetch` first")
    return raw_dir / f.read_text().strip()
