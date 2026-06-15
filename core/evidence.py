"""Evidence ledger — append-only JSONL log of strategy claims + their backing data."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LEDGER = Path(__file__).resolve().parent.parent / ".evidence_ledger.jsonl"


def record(strategy: str, claim: str, evidence: dict[str, Any]) -> None:
    entry = {
        "ts": time.time(),
        "strategy": strategy,
        "claim": claim,
        "evidence": evidence,
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_all(strategy: str | None = None) -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if strategy is None or e.get("strategy") == strategy:
            out.append(e)
    return out
