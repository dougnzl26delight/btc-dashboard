"""Publish the latest dashboard caches to GitHub so Streamlit Cloud stays fresh.

Local precompute (the Crypto_precompute_dashboard task) regenerates .panel_cache
every 15 min but never publishes it — so the cloud only updated when the hourly
GitHub Action happened to fire, which GitHub's free tier skips for hours at a
time. This task commits the latest caches + brief and pushes to origin/main
hourly, so the cloud refreshes straight from this (always-on) machine. The
GitHub Action stays as the laptop-off fallback.

Push-only by design: it does NOT recompute anything and makes no API calls, so
it's cheap and never races the precompute job for data.

HOW (rewritten 2026-10-04): the commit is built with git PLUMBING in a throwaway
index (same pattern as publish_live_cache.py) - origin/main's tree with the local
cache files laid on top, committed as a child of origin/main and pushed as a
fast-forward. It never touches the working tree, the real index, or HEAD.

Why: the old version committed on top of the LOCAL branch, which is always
behind origin because the Worker-driven Action pushes caches at :25 every hour.
So nearly every push was rejected, and the "self-heal" ran `git reset --hard
origin/main` - wiping the local precompute output (the live blob then published
the Action's copies, e.g. ETF flows built on GitHub's runners where Farside
fails = "$+0M"), and it would also have wiped any uncommitted CODE in this repo.
It also ran `git add` + `git commit` on the real index, so anything already
staged by hand was swept into a "data: refresh dashboard caches" commit.
Result: 31 of 34 cache commits in the 30h before the rewrite were the Action's.

Run via the standard dispatcher:
    pythonw.exe _scheduler/run.py push_dashboard push_dashboard
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Exactly what the GitHub Action commits — the files Streamlit Cloud reads.
# (*.pkl, not the whole folder: _store's atomic-write .pkl.tmp files must not ride along.)
PATHSPECS = [".panel_cache/*.pkl", ".simpleton_daily_brief.json", ".simpleton_brief_state.json"]
MESSAGE = "data: refresh dashboard caches [skip ci]"
ATTEMPTS = 2   # one rebuild-on-new-base retry if the Action pushed mid-run


def git(*args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, env=env)
    out = (p.stdout + p.stderr).strip()
    print(f"$ git {' '.join(args)}" + (f"\n{out}" if out and len(out) < 400 else ""))
    if check and p.returncode != 0:
        raise SystemExit(f"git {args[0]} failed ({p.returncode})")
    return p


def publish_once() -> str:
    """Build + push one cache commit on top of origin/main.
    Returns 'published' | 'unchanged' | 'rejected'."""
    git("fetch", "origin", "main")
    base = git("rev-parse", "origin/main").stdout.strip()
    base_tree = git("rev-parse", f"{base}^{{tree}}").stdout.strip()

    env = dict(os.environ)
    idx = tempfile.NamedTemporaryFile(prefix="push_idx_", suffix=".idx", delete=False)
    idx.close()
    env["GIT_INDEX_FILE"] = idx.name
    try:
        git("read-tree", base, env=env)
        # --ignore-removal: a panel missing locally must not be deleted from main.
        git("add", "-f", "--ignore-removal", "--", *PATHSPECS, env=env, check=False)
        tree = git("write-tree", env=env).stdout.strip()
        if tree == base_tree:
            return "unchanged"
        commit = git("commit-tree", tree, "-p", base, "-m", MESSAGE, env=env).stdout.strip()
    finally:
        try:
            os.unlink(idx.name)
        except OSError:
            pass
    if git("push", "origin", f"{commit}:refs/heads/main", check=False).returncode != 0:
        return "rejected"
    print(f"published {commit[:9]} on {base[:9]}")
    return "published"


def main() -> None:
    print(f"--- push_dashboard {datetime.now(timezone.utc).isoformat()} ---")
    for attempt in range(1, ATTEMPTS + 1):
        result = publish_once()
        if result == "unchanged":
            print("no cache changes vs origin/main — nothing to publish")
            return
        if result == "published":
            print("published OK")
            return
        print(f"push rejected (origin moved) — attempt {attempt}/{ATTEMPTS}")
    print("lost the push race twice — next cycle re-publishes on the new base")


if __name__ == "__main__":
    main()
