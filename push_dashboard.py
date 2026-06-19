"""Publish the latest dashboard caches to GitHub so Streamlit Cloud stays fresh.

Local precompute (the Crypto_precompute_dashboard task) regenerates .panel_cache
every 15 min but never publishes it — so the cloud only updated when the hourly
GitHub Action happened to fire, which GitHub's free tier skips for hours at a
time. This task commits the latest caches + brief and pushes to origin/main
hourly, so the cloud refreshes straight from this (always-on) machine. The
GitHub Action stays as the laptop-off fallback.

Push-only by design: it does NOT recompute anything and makes no API calls, so
it's cheap and never races the precompute job for data. Safe to run alongside
the Action — the rebase-on-reject step self-heals if both push.

Run via the standard dispatcher:
    pythonw.exe _scheduler/run.py push_dashboard push_dashboard
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Exactly what the GitHub Action commits — the files Streamlit Cloud reads.
FILES = [".panel_cache", ".simpleton_daily_brief.json", ".simpleton_brief_state.json"]


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    print(f"$ git {' '.join(args)}" + (f"\n{out}" if out else ""))
    if check and p.returncode != 0:
        raise SystemExit(f"git {args[0]} failed ({p.returncode})")
    return p


def main() -> None:
    print(f"--- push_dashboard {datetime.now(timezone.utc).isoformat()} ---")

    # Self-heal: clear any rebase a prior run left stuck, so we never wedge.
    if (ROOT / ".git" / "rebase-merge").exists() or (ROOT / ".git" / "rebase-apply").exists():
        print("clearing stuck rebase from a prior run")
        git("rebase", "--abort", check=False)

    # Stage only the published artefacts (force: most are gitignored).
    git("add", "-f", *FILES, check=False)

    # Nothing new? Done. (git diff --cached --quiet exits 0 when no staged change.)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("no cache changes to publish")
        return

    git("commit", "-m", "data: refresh dashboard caches [skip ci]")

    # Push; if we're behind (the Action pushed too), rebase onto origin and retry.
    if git("push", "origin", "main", check=False).returncode != 0:
        print("push rejected (behind origin) — rebasing and retrying")
        git("pull", "--rebase", "--autostash", "origin", "main", check=False)
        git("push", "origin", "main")

    print("published OK")


if __name__ == "__main__":
    main()
