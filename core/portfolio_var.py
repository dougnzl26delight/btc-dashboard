"""Daily VaR enforcement at the orchestrator level.

Computes 1-day 95% VaR across all open positions (spot + perp + sleeves).
Hard-blocks new trades when daily VaR > MAX_DAILY_VAR_PCT of equity.

Methodology: parametric VaR
    1. For each open position, compute its 30-day realized daily vol.
    2. Position dollar VaR = abs(position_value) * vol * z(0.95)  ≈ 1.645 * sigma
    3. Portfolio VaR = sqrt(sum of (pos_var)² + 2*sum(pos_i_var * pos_j_var * rho))
       Simplification: assume rho=0.7 for all crypto majors (close to historical avg).

Per top-1% pre-trade risk control: VaR limit is a HARD STOP on new exposure,
but does not force-close existing positions (those are handled by sleeve CBs).

Limit: 1% of portfolio equity per day at 95% confidence.
       Means: a 1-in-20 day should not exceed 1% loss.
       Annual implied vol ≈ 1% × sqrt(252) = ~16% — moderate risk.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import data
from core.perp_broker import PerpBroker


MAX_DAILY_VAR_PCT = 0.01           # 1% of equity per day at 95% confidence
Z_95 = 1.645                        # 95th percentile of standard normal
AVG_CRYPTO_CORRELATION = 0.7        # heuristic; major crypto pairs cluster high
LOOKBACK_DAYS_FOR_VOL = 30


def _realized_vol(pair: str) -> float:
    """30-day realized daily vol (decimal, e.g. 0.05 = 5% daily)."""
    try:
        df = data.ohlcv_extended(pair, days_back=LOOKBACK_DAYS_FOR_VOL + 10)
        if df.empty or len(df) < LOOKBACK_DAYS_FOR_VOL:
            return 0.05  # default conservative 5% daily vol
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        return float(rets.iloc[-LOOKBACK_DAYS_FOR_VOL:].std())
    except Exception:
        return 0.05


def _gather_positions() -> list[dict]:
    """Aggregate ALL open positions across spot + perp."""
    positions = []
    spot_state = REPO_ROOT / ".paper_state.json"
    if spot_state.exists():
        s = json.loads(spot_state.read_text())
        for asset, qty in s.get("positions", {}).items():
            if abs(qty) < 1e-12:
                continue
            pair = f"{asset}/USDT"
            try:
                df = data.ohlcv_extended(pair, days_back=2)
                px = float(df["close"].iloc[-1])
            except Exception:
                continue
            positions.append({
                "venue": "spot",
                "pair": pair,
                "qty": qty,
                "price": px,
                "notional": abs(qty * px),
                "vol_daily": _realized_vol(pair),
            })
    perp = PerpBroker(mode="paper")
    for asset, qty in perp._state.positions.items():
        if abs(qty) < 1e-12:
            continue
        pair = f"{asset}/USDT"
        try:
            df = data.ohlcv_extended(pair, days_back=2)
            px = float(df["close"].iloc[-1])
        except Exception:
            continue
        positions.append({
            "venue": "perp",
            "pair": pair,
            "qty": qty,
            "price": px,
            "notional": abs(qty * px),
            "vol_daily": _realized_vol(pair),
        })
    return positions


def _portfolio_var_dollars(positions: list[dict], correlation: float = AVG_CRYPTO_CORRELATION) -> float:
    """Parametric VaR with single-correlation simplification."""
    if not positions:
        return 0.0
    pos_var = np.array([p["notional"] * p["vol_daily"] * Z_95 for p in positions])
    # Covariance-aware aggregation: variance = sum(var_i²) + 2 * rho * sum(var_i * var_j) for i<j
    var_sq = (pos_var ** 2).sum()
    cross_sum = 0.0
    for i in range(len(pos_var)):
        for j in range(i + 1, len(pos_var)):
            cross_sum += pos_var[i] * pos_var[j]
    total_var_sq = var_sq + 2 * correlation * cross_sum
    return float(np.sqrt(max(total_var_sq, 0.0)))


def current_var_status() -> dict:
    """Compute current portfolio VaR vs limit. Used by run.py to gate new trades."""
    positions = _gather_positions()
    var_dollars = _portfolio_var_dollars(positions)

    # Equity = combined paper accounts
    spot_state = REPO_ROOT / ".paper_state.json"
    cash = 0.0
    pos_value = 0.0
    if spot_state.exists():
        s = json.loads(spot_state.read_text())
        cash = float(s.get("cash_quote", 0))
        for asset, qty in s.get("positions", {}).items():
            if abs(qty) < 1e-12:
                continue
            try:
                px = float(data.ohlcv_extended(f"{asset}/USDT", days_back=2)["close"].iloc[-1])
                pos_value += qty * px
            except Exception:
                pass
    perp = PerpBroker(mode="paper")
    perp_cash = float(perp._state.cash_quote)
    perp_mtm = 0.0
    for asset, qty in perp._state.positions.items():
        if abs(qty) < 1e-12:
            continue
        try:
            px = float(data.ohlcv_extended(f"{asset}/USDT", days_back=2)["close"].iloc[-1])
            entry = perp._state.entry_prices.get(asset, px)
            perp_mtm += qty * (px - entry)
        except Exception:
            pass
    equity = cash + pos_value + perp_cash + perp_mtm

    var_pct = var_dollars / equity if equity > 0 else 0
    limit_dollars = equity * MAX_DAILY_VAR_PCT
    headroom = limit_dollars - var_dollars
    return {
        "var_dollars": var_dollars,
        "var_pct": var_pct,
        "equity": equity,
        "limit_pct": MAX_DAILY_VAR_PCT,
        "limit_dollars": limit_dollars,
        "headroom_dollars": headroom,
        "headroom_pct": (headroom / equity) if equity > 0 else 0,
        "exceeded": var_pct > MAX_DAILY_VAR_PCT,
        "n_positions": len(positions),
    }


def gate_new_trade(additional_notional: float, pair: str) -> dict:
    """Check whether adding `additional_notional` of `pair` would exceed VaR limit.

    Returns: {allowed: bool, reason: str, projected_var_pct: float}
    """
    positions = _gather_positions()
    additional_vol = _realized_vol(pair)
    # Hypothetical new position
    new_positions = positions + [{
        "notional": abs(additional_notional),
        "vol_daily": additional_vol,
    }]
    new_var = _portfolio_var_dollars(new_positions)

    status = current_var_status()
    equity = status["equity"]
    new_pct = new_var / equity if equity > 0 else 0
    allowed = new_pct <= MAX_DAILY_VAR_PCT
    return {
        "allowed": allowed,
        "reason": "" if allowed else (
            f"VaR limit exceeded: adding ${additional_notional:,.0f} of {pair} "
            f"would push portfolio VaR to {new_pct*100:.2f}% (cap {MAX_DAILY_VAR_PCT*100:.1f}%)"
        ),
        "current_var_pct": status["var_pct"],
        "projected_var_pct": new_pct,
    }


def main():
    """CLI: print VaR status snapshot."""
    s = current_var_status()
    print("=" * 80)
    print("PORTFOLIO VaR STATUS")
    print("=" * 80)
    print(f"  Equity:              ${s['equity']:,.2f}")
    print(f"  Daily VaR (95%):     ${s['var_dollars']:,.2f}  ({s['var_pct']*100:.2f}% of equity)")
    print(f"  Limit:               ${s['limit_dollars']:,.2f}  ({s['limit_pct']*100:.1f}% of equity)")
    print(f"  Headroom:            ${s['headroom_dollars']:,.2f}  ({s['headroom_pct']*100:.2f}%)")
    print(f"  Open positions:      {s['n_positions']}")
    print(f"  Status:              {'EXCEEDED' if s['exceeded'] else 'OK'}")
    print()
    if s["exceeded"]:
        print("  WARNING: Current portfolio exceeds daily VaR limit.")
        print("  New trades will be BLOCKED at the orchestrator until exposure reduces.")


if __name__ == "__main__":
    main()
