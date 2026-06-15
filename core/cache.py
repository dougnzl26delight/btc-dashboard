"""TTL file-based cache for market data and signal computations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _safe_key(key: str) -> str:
    return key.replace("/", "_").replace(":", "_").replace(" ", "_")


def get_or_fetch(key: str, fetch: Callable[[], Any], ttl_s: int) -> Any:
    """Return cached value if newer than ttl_s; otherwise call fetch() and cache it."""
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"{_safe_key(key)}.json"
    if f.exists() and (time.time() - f.stat().st_mtime) < ttl_s:
        return json.loads(f.read_text())
    val = fetch()
    f.write_text(json.dumps(val, default=str))
    return val


def invalidate(key: str | None = None) -> int:
    """Delete cached entries. Pass None to clear all. Returns number deleted."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json")) if key is None else [
        CACHE_DIR / f"{_safe_key(key)}.json"
    ]
    deleted = 0
    for f in files:
        if f.exists():
            f.unlink()
            deleted += 1
    return deleted
