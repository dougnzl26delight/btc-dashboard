"""Portfolio-level drawdown kill switch.

If total equity drops more than KILL_DD_PCT from peak, liquidate all
positions and write a kill-marker file. The orchestrator checks for that
marker before each cycle and skips trading if active.

The kill-marker is sticky — manual review and removal is required to
resume trading. That's the design: a kill switch exists to force a human
to look before continuing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.broker import Broker
from core.data import ticker
from ops.alerts import alert


REPO_ROOT = Path(__file__).resolve().parent.parent
PEAK_FILE = REPO_ROOT / ".peak_equity.json"
KILL_FILE = REPO_ROOT / ".kill_switch.json"
STATE_FILE = REPO_ROOT / ".paper_state.json"

KILL_DD_PCT = 0.08      # 2026-05-28: liquidate at 8% portfolio DD (trader 10% max)
WARN_DD_PCT = 0.05      # alert at 5% portfolio DD
START_EQUITY = 100_000.0


def _load_peak() -> float:
    if PEAK_FILE.exists():
        try:
            return float(json.loads(PEAK_FILE.read_text()).get("peak", START_EQUITY))
        except Exception:
            pass
    return START_EQUITY


def _save_peak(peak: float) -> None:
    PEAK_FILE.write_text(json.dumps({"peak": peak}))


def is_killed() -> bool:
    return KILL_FILE.exists()


def reset_kill_switch() -> None:
    if KILL_FILE.exists():
        KILL_FILE.unlink()


def _compute_equity() -> float:
    if not STATE_FILE.exists():
        return START_EQUITY
    state = json.loads(STATE_FILE.read_text())
    cash = float(state.get("cash_quote", 0))
    positions = state.get("positions", {})
    quote = state.get("quote_currency", "USDT")
    equity = cash
    for asset, qty in positions.items():
        if abs(qty) < 1e-9:
            continue
        try:
            price = float(ticker(f"{asset}/{quote}")["last"])
            equity += qty * price
        except Exception:
            pass
    return equity


def run_circuit_breaker(mode: str = "paper", long_only: bool = False) -> dict:
    if is_killed():
        return {"status": "already_killed"}

    equity = _compute_equity()
    peak = max(_load_peak(), equity)
    _save_peak(peak)
    drawdown = max(0.0, 1 - equity / peak) if peak > 0 else 0.0

    if drawdown > KILL_DD_PCT:
        if not STATE_FILE.exists():
            return {"status": "no_state_to_kill"}
        state = json.loads(STATE_FILE.read_text())
        positions = state.get("positions", {})
        quote = state.get("quote_currency", "USDT")
        broker = Broker(mode=mode, long_only=long_only)
        liquidated = []
        for asset, qty in list(positions.items()):
            if abs(qty) < 1e-9:
                continue
            pair = f"{asset}/{quote}"
            try:
                price = float(ticker(pair)["last"])
                close_side = "sell" if qty > 0 else "buy"
                broker.place_market_order(pair, close_side, abs(qty) * price)
                liquidated.append(asset)
            except Exception:
                pass

        KILL_FILE.write_text(json.dumps({
            "killed_at_equity": equity,
            "peak": peak,
            "drawdown_pct": drawdown,
            "liquidated": liquidated,
        }, indent=2))
        alert(
            f"CIRCUIT BREAKER FIRED: dd={drawdown:.1%} > kill {KILL_DD_PCT:.0%}. "
            f"Liquidated {len(liquidated)} positions. Trading disabled until reset.",
            level="critical",
        )
        return {"status": "killed", "drawdown": drawdown, "equity": equity, "liquidated": liquidated}

    if drawdown > WARN_DD_PCT:
        alert(
            f"Drawdown warning: equity ${equity:,.0f} vs peak ${peak:,.0f} "
            f"= dd {drawdown:.1%}. Kill at {KILL_DD_PCT:.0%}.",
            level="warning",
        )

    return {"status": "ok", "drawdown": drawdown, "equity": equity, "peak": peak}


if __name__ == "__main__":
    print(run_circuit_breaker())
