"""Bulletproof disk-backed cache for dashboard panels.

Why this exists
---------------
Streamlit's `@st.cache_data` is process-local. When the streamlit server
restarts (crash, redeploy, machine reboot), every panel re-computes from
scratch on the next page hit. With slow upstreams (FRED, on-chain APIs),
that means a 30-60 second black screen for the user.

This module adds a DISK layer below `@st.cache_data`. Even when the
streamlit cache is cold, the disk cache returns the last-good value
INSTANTLY, then a background refresh recomputes for next time.

Pattern:
    @disk_cached("top_scorecard", ttl=1800)
    def compute_top_scorecard():
        ...  # slow, may hit FRED/etc

    @st.cache_data(ttl=1800)
    def cached_top_scorecard():
        return compute_top_scorecard()  # already protected by disk cache
"""
from __future__ import annotations

import functools
import gzip
import json
import os
import pickle
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".panel_cache"


# ============================================================
# Live overlay (Tier-1 real-time) — see publish_live_cache.py
# ============================================================
# The always-on home PC force-pushes a small gzipped blob of the panel caches
# to the `live-data` branch every few minutes. Streamlit does NOT watch that
# branch, so data refreshes WITHOUT redeploying the app. Here we fetch that blob
# out-of-band and let `_load` prefer it over the committed pickle when it is
# fresher — turning ~1h staleness into ~minutes with zero reboot thrash.
#
# Safety: the fetch is TTL-guarded (at most once per _LIVE_TTL), only runs under
# a live Streamlit session (never during local precompute/publish), and is fully
# wrapped in try/except — any failure silently falls back to the on-disk pickle,
# so the app behaves EXACTLY as before whenever the blob is unreachable.
_LIVE_URL = os.environ.get(
    "DASHBOARD_LIVE_URL",
    "https://raw.githubusercontent.com/dougnzl26delight/btc-dashboard/live-data/live_cache.pkl.gz",
)
_LIVE_ENABLED = os.environ.get("DASHBOARD_LIVE_OVERLAY", "1") != "0"
_LIVE_TTL = 60  # seconds between remote re-fetches
_live_state: dict[str, Any] = {"ts": 0.0, "panels": {}}


def _streamlit_active() -> bool:
    """True only inside a running Streamlit script (Cloud or local cockpit).
    False during precompute/publish so those never hit the network."""
    try:
        from streamlit.runtime import exists  # type: ignore
        return bool(exists())
    except Exception:
        return False


def _refresh_live_blob() -> None:
    """Refresh the in-memory live blob at most once per _LIVE_TTL. Never raises."""
    now = time.time()
    if now - _live_state["ts"] < _LIVE_TTL:
        return
    _live_state["ts"] = now  # stamp first so failures don't hammer the network
    try:
        req = urllib.request.Request(_LIVE_URL, headers={"User-Agent": "btcdelight-app"})
        raw = urllib.request.urlopen(req, timeout=4).read()
        data = pickle.loads(gzip.decompress(raw))
        panels = data.get("panels")
        if isinstance(panels, dict):
            _live_state["panels"] = panels
    except Exception:
        pass  # keep last-good; disk fallback still applies


def _live_get(key: str) -> Optional[tuple[float, Any]]:
    """Return the live blob's (ts, value) for key, or None. Never raises."""
    if not (_LIVE_ENABLED and _streamlit_active()):
        return None
    try:
        _refresh_live_blob()
        entry = _live_state["panels"].get(key)
        if isinstance(entry, tuple) and len(entry) == 2:
            return entry
    except Exception:
        pass
    return None


def _cache_path(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)
    return CACHE_DIR / f"{safe}.pkl"


def _load_disk(key: str) -> Optional[tuple[float, Any]]:
    """Read (timestamp, value) from the on-disk pickle, or None."""
    path = _cache_path(key)
    if not path.exists(): return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load(key: str) -> Optional[tuple[float, Any]]:
    """Return (timestamp, value) — the FRESHER of the live blob and the on-disk
    pickle. Falls back cleanly to either alone (or None)."""
    live = _live_get(key)
    disk = _load_disk(key)
    if live is not None and disk is not None:
        return live if live[0] >= disk[0] else disk
    return live if live is not None else disk


def _store(key: str, value: Any) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        path = _cache_path(key)
        # Atomic write: tmp file then rename
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump((time.time(), value), f)
        tmp.replace(path)
    except Exception:
        pass


def disk_cached(key: str, ttl: int = 1800):
    """Decorator: cache function result on disk under `key` for `ttl` seconds.

    Returns CACHED value if fresh.
    Returns STALE cached value (logged warning) if recompute fails.
    """
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            cached = _load(key)
            now = time.time()
            # Fresh hit
            if cached is not None:
                ts, value = cached
                if now - ts < ttl:
                    return value
            # Try recompute
            try:
                value = fn(*args, **kwargs)
                _store(key, value)
                return value
            except Exception:
                # Recompute failed — return stale cache (any age) if any
                if cached is not None:
                    return cached[1]
                raise
        return wrapper
    return deco


def get_cached(key: str, max_age: Optional[int] = None) -> Optional[Any]:
    """Read cached value, optionally with max age. Returns None if missing/stale."""
    cached = _load(key)
    if cached is None: return None
    ts, value = cached
    if max_age is not None and (time.time() - ts) > max_age:
        return None
    return value


def cache_age_seconds(key: str) -> Optional[float]:
    """Return age in seconds of the cached entry, or None if missing."""
    cached = _load(key)
    if cached is None: return None
    return time.time() - cached[0]


# ============================================================
# FRED Circuit Breaker
# ============================================================
# When FRED fails, mark it down for FRED_DOWN_TTL seconds and reject
# all subsequent calls instantly. Without this, every dashboard load
# pays the full timeout for every FRED series even when FRED is down.

_FRED_STATE_FILE = REPO_ROOT / ".fred_circuit_state.json"
FRED_DOWN_TTL = 600  # 10 minutes


def fred_is_down() -> bool:
    """True if FRED was marked down within the last FRED_DOWN_TTL seconds."""
    if not _FRED_STATE_FILE.exists(): return False
    try:
        st = json.loads(_FRED_STATE_FILE.read_text())
        last_fail = st.get("last_fail_ts", 0)
        return (time.time() - last_fail) < FRED_DOWN_TTL
    except Exception:
        return False


def mark_fred_down() -> None:
    """Mark FRED as down (called on any FRED request failure)."""
    try:
        _FRED_STATE_FILE.write_text(json.dumps({"last_fail_ts": time.time()}))
    except Exception:
        pass


def mark_fred_up() -> None:
    """Clear the FRED-down flag (called on first successful FRED request)."""
    try:
        if _FRED_STATE_FILE.exists():
            _FRED_STATE_FILE.unlink()
    except Exception:
        pass
