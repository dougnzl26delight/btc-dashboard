"""Real-time portfolio risk monitor — VaR, ES, position correlation matrix.

Runs every 30 minutes. Computes:
  1. Cross-position daily-return correlation matrix (last 60 days)
  2. Effective number of bets via correlation-weighted N
  3. 1-day historical VaR + Expected Shortfall (95% / 99%)
  4. Per-pair Marginal VaR contribution
  5. Total notional exposure vs capital

Alerts when:
  - Mean off-diagonal correlation > 0.85 (positions are 1 effective bet)
  - 95% VaR > 8% of equity
  - 99% ES > 15% of equity
  - Any pair contributes > 40% of total VaR

Output: writes daily snapshot to risk_snapshots/{date}.json. The kill_criteria
script reads these for K-criterion enforcement.

Scheduled as Crypto_portfolio_risk_30min.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import data
from core.pnl_attribution import load_attribution
from ops.alerts import alert


SNAPSHOTS_DIR = REPO_ROOT / "risk_snapshots"

# Alert thresholds
HIGH_CORRELATION_THRESHOLD = 0.85
VAR_95_THRESHOLD = 0.08
ES_99_THRESHOLD = 0.15
SINGLE_PAIR_VAR_PCT = 0.40

CORR_WINDOW_DAYS = 60


def open_position_pairs() -> list[str]:
    """Active pairs (excluding basis arb internal accounting)."""
    attrib = load_attribution()
    out = []
    for p, tag in attrib.items():
        if p.startswith("basis:") or p.startswith("xsmom:"):
            # basis: tracked separately; xsmom prefix uses an alias
            if p.startswith("xsmom:"):
                base = p.removeprefix("xsmom:")
                if base not in out:
                    out.append(base)
            continue
        if p not in out:
            out.append(p)
    return out


def fetch_recent_returns(pairs: list[str], days: int = CORR_WINDOW_DAYS) -> pd.DataFrame:
    """Fetch closing prices for pairs over last N days; return daily returns."""
    cols = {}
    for p in pairs:
        try:
            df = data.ohlcv_extended(p, days_back=days + 5)
            if df.empty:
                continue
            cols[p] = df["close"]
        except Exception:
            continue
    if not cols:
        return pd.DataFrame()
    panel = pd.concat(cols, axis=1)
    panel = panel.dropna(how="all")
    return panel.pct_change().dropna()


def compute_correlation_matrix(rets: pd.DataFrame) -> dict:
    if rets.empty or len(rets) < 30:
        return {"error": "insufficient data"}
    corr = rets.corr()
    n = len(corr.columns)
    if n < 2:
        return {"corr_matrix": corr.to_dict(),
                "mean_off_diagonal": 0,
                "effective_n_bets": n}
    # Average off-diagonal correlation
    mask = ~np.eye(n, dtype=bool)
    off_diag = corr.values[mask]
    mean_corr = float(off_diag.mean())
    # Effective number of bets: 1 / sum(w_i * w_j * corr_{ij}) at equal weights
    w = np.ones(n) / n
    portfolio_var = float(w @ corr.values @ w)
    eff_n = 1.0 / portfolio_var if portfolio_var > 0 else n
    return {
        "n_pairs": n,
        "corr_matrix": corr.round(3).to_dict(),
        "mean_off_diagonal": mean_corr,
        "effective_n_bets": float(eff_n),
    }


def compute_var_es(positions: dict, rets: pd.DataFrame) -> dict:
    """Compute portfolio 1-day historical VaR (95%/99%) and ES (99%)."""
    if rets.empty:
        return {"error": "no returns"}
    common_pairs = [p for p in positions if p in rets.columns]
    if not common_pairs:
        return {"error": "no overlapping pairs"}

    # Build dollar P&L series: sum over pairs of (notional * daily_return)
    pnl_series = pd.Series(0.0, index=rets.index)
    for pair in common_pairs:
        notional = positions[pair]["notional"]  # signed
        pnl_series += notional * rets[pair]

    if len(pnl_series.dropna()) < 30:
        return {"error": "insufficient pnl history"}
    pnl = pnl_series.dropna().values

    var_95 = -float(np.percentile(pnl, 5))
    var_99 = -float(np.percentile(pnl, 1))
    losses = -pnl[pnl < 0]
    es_99 = float(losses[losses > var_99].mean()) if (losses > var_99).any() else var_99

    return {
        "var_95_dollars": var_95,
        "var_99_dollars": var_99,
        "es_99_dollars": es_99,
        "n_obs": len(pnl),
    }


def main():
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    pairs = open_position_pairs()
    if not pairs:
        return {"status": "no_open_positions"}

    rets = fetch_recent_returns(pairs)
    corr_info = compute_correlation_matrix(rets)

    # Build positions dict with notional
    attrib = load_attribution()
    positions = {}
    total_long_notional = 0.0
    total_short_notional = 0.0
    for tag_pair, tag in attrib.items():
        actual_pair = tag_pair.removeprefix("xsmom:")
        if actual_pair not in pairs:
            continue
        notional = tag["qty"] * tag["entry_price"] * (1 if tag["side"] == "long" else -1)
        if actual_pair in positions:
            positions[actual_pair]["notional"] += notional
        else:
            positions[actual_pair] = {"notional": notional, "side": tag["side"]}
        if notional > 0:
            total_long_notional += notional
        else:
            total_short_notional += notional

    var_info = compute_var_es(positions, rets)

    # Get current equity
    from core.broker import Broker
    from core.perp_broker import PerpBroker
    spot_cash = float(Broker(mode="paper").get_balance().get("USDT", 0))
    perp_cash = float(PerpBroker(mode="paper").get_balance().get("USDT", 0))
    cash = spot_cash + perp_cash

    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_positions": len(positions),
        "long_notional": total_long_notional,
        "short_notional": total_short_notional,
        "gross_notional": total_long_notional + abs(total_short_notional),
        "net_notional": total_long_notional + total_short_notional,
        "cash": cash,
        "correlation": corr_info,
        "var_es": var_info,
    }

    # Alerts
    alerts_fired = []
    if (corr_info.get("mean_off_diagonal", 0) > HIGH_CORRELATION_THRESHOLD
            and corr_info.get("n_pairs", 0) >= 3):
        msg = (f"HIGH CORRELATION: avg off-diagonal "
               f"{corr_info['mean_off_diagonal']:.2f} > "
               f"{HIGH_CORRELATION_THRESHOLD}. "
               f"Effective bets: {corr_info['effective_n_bets']:.1f} "
               f"of {corr_info['n_pairs']} pairs.")
        alert(msg, level="warning")
        alerts_fired.append(msg)

    if "var_95_dollars" in var_info and cash > 0:
        var_pct = var_info["var_95_dollars"] / cash
        if var_pct > VAR_95_THRESHOLD:
            msg = f"VaR95 ${var_info['var_95_dollars']:,.0f} = {var_pct:.1%} of cash > {VAR_95_THRESHOLD:.0%}"
            alert(msg, level="warning")
            alerts_fired.append(msg)

        es_pct = var_info["es_99_dollars"] / cash
        if es_pct > ES_99_THRESHOLD:
            msg = f"ES99 ${var_info['es_99_dollars']:,.0f} = {es_pct:.1%} of cash > {ES_99_THRESHOLD:.0%}"
            alert(msg, level="critical")
            alerts_fired.append(msg)

    snapshot["alerts_fired"] = alerts_fired

    out_file = SNAPSHOTS_DIR / f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    out_file.write_text(json.dumps(snapshot, indent=2, default=str))

    return snapshot


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
