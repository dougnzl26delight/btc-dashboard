"""Publish the live dashboard blob to the `live-data` branch (Tier-1 real-time).

Why this exists
---------------
The Streamlit Cloud app reaches production only via commits to `main`, and
Streamlit REBOOTS the app on every push to the served branch. So we cannot push
data to `main` frequently without the app perpetually restarting.

This task DECOUPLES data from code: it bundles the current panel caches into one
small gzipped blob and force-pushes it to a dedicated `live-data` branch that
Streamlit does NOT watch. The app fetches that blob out-of-band (see
core/dashboard_cache.py `_load` overlay) every ~60s and prefers it over the
committed pickles when fresher — so the dashboard goes near-live with ZERO
redeploys and ZERO reboot thrash.

Design notes
------------
* Uses git PLUMBING with a throwaway index (GIT_INDEX_FILE) so it never touches
  the working tree, the real index, or HEAD. Safe to run concurrently with the
  main-branch precompute/push tasks.
* Each publish is an ORPHAN commit (no parent), force-pushed. `live-data`
  therefore always holds exactly ONE commit — no history growth, ever.
* Excludes `swift_charts` (1.9 MB of chart imagery, slow-moving) so the blob
  stays ~130 KB -> gzips to a few tens of KB, cheap to fetch every 60s.
* Fail-safe: any error is logged and the task exits non-zero, but it never
  corrupts anything (plumbing-only). The app's disk-pickle fallback means a
  missed publish just makes the app fall back to `main`'s cadence.

Invoked by Task Scheduler as:
  pythonw.exe _scheduler/run.py publish_live_cache publish_live_cache
"""
from __future__ import annotations

import gzip
import os
import pickle
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".panel_cache"
BRANCH = "live-data"
BLOB_NAME = "live_cache.pkl.gz"

# Panels excluded from the live blob (large + slow-moving -> not worth shipping
# every 3 min; they stay on the committed-pickle cadence and the app falls back
# to disk for them).
EXCLUDE = {"swift_charts"}


def _log(msg: str) -> None:
    print(f"[publish_live_cache {datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def _git(*args: str, env=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(ROOT), env=env,
        capture_output=True, text=True, check=check,
    )


def build_blob() -> bytes:
    """Bundle {key: (ts, value)} for every non-excluded panel cache into a
    gzipped pickle. Returns the compressed bytes."""
    panels: dict[str, tuple] = {}
    for path in sorted(CACHE_DIR.glob("*.pkl")):
        key = path.stem
        if key in EXCLUDE:
            continue
        try:
            with path.open("rb") as f:
                loaded = pickle.load(f)
            # Panel caches are stored as (timestamp, value) tuples.
            if isinstance(loaded, tuple) and len(loaded) == 2:
                panels[key] = loaded
        except Exception as e:  # noqa: BLE001
            _log(f"skip {key}: {e}")
            continue
    payload = {"generated": time.time(), "n_panels": len(panels), "panels": panels}
    raw = gzip.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=6)
    _log(f"blob: {len(panels)} panels, {len(raw)/1024:.1f} KB gzipped")
    return raw


def publish(raw: bytes) -> None:
    """Force-push `raw` as {BLOB_NAME} to an orphan single-commit `live-data`
    branch, using a throwaway index so the working tree is never touched."""
    tmp_blob = ROOT / ".live_cache.tmp.gz"
    tmp_blob.write_bytes(raw)

    env = dict(os.environ)
    # Deterministic identity so commit-tree never prompts / fails.
    env.setdefault("GIT_AUTHOR_NAME", "doug")
    env.setdefault("GIT_AUTHOR_EMAIL", "dougnzl26@gmail.com")
    env.setdefault("GIT_COMMITTER_NAME", "doug")
    env.setdefault("GIT_COMMITTER_EMAIL", "dougnzl26@gmail.com")

    idx = tempfile.NamedTemporaryFile(prefix="live_idx_", suffix=".idx", delete=False)
    idx.close()
    env["GIT_INDEX_FILE"] = idx.name
    try:
        _git("read-tree", "--empty", env=env)
        sha = _git("hash-object", "-w", str(tmp_blob), env=env).stdout.strip()
        _git("update-index", "--add", "--cacheinfo", f"100644,{sha},{BLOB_NAME}", env=env)
        tree = _git("write-tree", env=env).stdout.strip()
        commit = _git(
            "commit-tree", tree, "-m", f"live blob {int(time.time())}", env=env
        ).stdout.strip()
        _git("update-ref", f"refs/heads/{BRANCH}", commit, env=env)
        push = _git("push", "-f", "origin", f"{BRANCH}:{BRANCH}", env=env, check=False)
        if push.returncode != 0:
            _log(f"push FAILED: {push.stderr.strip()}")
            raise SystemExit(1)
        _log(f"pushed {BRANCH} @ {commit[:10]} ({sha[:10]})")
    finally:
        try:
            os.unlink(idx.name)
        except OSError:
            pass
        tmp_blob.unlink(missing_ok=True)


def main() -> None:
    if not CACHE_DIR.exists():
        _log("no .panel_cache dir; nothing to publish")
        raise SystemExit(0)
    raw = build_blob()
    publish(raw)


if __name__ == "__main__":
    main()
