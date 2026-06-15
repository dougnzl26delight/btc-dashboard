"""Tail risk hedging engine — BTC put protection suggestions.

Taleb's antifragility principle: small, cheap insurance against tail events
turns drawdowns from disasters into opportunities. Specifically: OTM BTC puts
on Deribit, priced cheaply during euphoria, pay off massively during crashes.

This module computes:
    1. Whether the rig should currently hold tail protection (regime trigger)
    2. Suggested strike + expiry + notional
    3. Maximum acceptable premium (% of bankroll/year)

Triggers tail hedge BUY when:
    - cycle_score > 60 (late bull / euphoria) AND
    - VIX rising > 25% from 30d low (regime change brewing) AND
    - BTC up > 30% in 30 days (parabolic move)

Triggers tail hedge SELL when:
    - cycle_score < 30 (bear in progress) AND
    - puts have realized 5x+ payoff

References:
    Nassim Taleb (2012) *Antifragile*
    Universa Investments — Spitznagel's tail-hedge framework
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_hedge_recommendation(bankroll_usd: float = 200_000) -> dict:
    """Compute current tail-hedge recommendation.

    Returns:
        {
            should_hedge: bool,
            urgency: 'critical' | 'recommended' | 'optional' | 'unnecessary',
            reasoning: list[str],
            suggested_structure: dict | None,
            max_premium_pct: float,
        }
    """
    reasoning = []
    risk_factors = 0  # 0-4 score

    # 1. Cycle position
    try:
        from core.onchain import cycle_position
        cp = cycle_position()
        score = cp.get("score", 50)
        if score > 80:
            risk_factors += 2
            reasoning.append(f"Cycle EUPHORIA: score {score:.0f} — historical top zone")
        elif score > 60:
            risk_factors += 1
            reasoning.append(f"Cycle LATE BULL: score {score:.0f}")
    except Exception:
        pass

    # 2. VIX / macro regime
    try:
        from core.macro_correlation import latest_metrics
        m = latest_metrics()
        vix = m.get("VIX", {}).get("value")
        vix_5d_chg = m.get("VIX", {}).get("ret_5d", 0)
        if vix is not None:
            if vix > 30:
                risk_factors += 1
                reasoning.append(f"VIX {vix:.1f} > 30 — elevated risk")
            if vix_5d_chg > 0.20:
                risk_factors += 1
                reasoning.append(f"VIX +{vix_5d_chg*100:.0f}% in 5d — regime change brewing")
    except Exception:
        pass

    # 3. BTC parabolic move
    try:
        from core import data
        df = data.ohlcv_extended("BTC/USDT", days_back=35)
        if not df.empty and len(df) > 30:
            ret_30d = float(df["close"].iloc[-1] / df["close"].iloc[-31] - 1)
            if ret_30d > 0.40:
                risk_factors += 2
                reasoning.append(f"BTC +{ret_30d*100:.0f}% in 30d — parabolic; mean reversion likely")
            elif ret_30d > 0.25:
                risk_factors += 1
                reasoning.append(f"BTC +{ret_30d*100:.0f}% in 30d — extended move")
    except Exception:
        pass

    # 4. F&G extreme greed
    try:
        from core.fear_greed import latest
        fg = latest()
        fg_value = fg.get("value")
        if fg_value is not None and fg_value > 80:
            risk_factors += 1
            reasoning.append(f"F&G EXTREME GREED at {fg_value} — contrarian sell signal")
    except Exception:
        pass

    # Determine urgency
    if risk_factors >= 4:
        urgency = "critical"
        max_premium_pct = 0.02   # 2% of bankroll on insurance
        should_hedge = True
    elif risk_factors >= 2:
        urgency = "recommended"
        max_premium_pct = 0.01
        should_hedge = True
    elif risk_factors == 1:
        urgency = "optional"
        max_premium_pct = 0.005
        should_hedge = False
    else:
        urgency = "unnecessary"
        max_premium_pct = 0.0
        should_hedge = False

    # Suggested structure
    suggested_structure = None
    if should_hedge:
        try:
            from core.options_iv import get_atm_iv
            iv = get_atm_iv("BTC")
            spot = iv.get("spot")
            if spot:
                # OTM put 20% below spot, 30-60 day expiry
                strike = spot * 0.80
                max_premium_usd = bankroll_usd * max_premium_pct
                # Rough put estimate: ATM IV ~ 50%, 20% OTM put 30d ≈ 1-2% of spot
                est_premium_pct_of_spot = 0.015 if iv.get("atm_iv_pct", 50) > 0.5 else 0.008
                est_premium_usd_per_btc = spot * est_premium_pct_of_spot
                n_puts = max_premium_usd / est_premium_usd_per_btc if est_premium_usd_per_btc > 0 else 0
                suggested_structure = {
                    "instrument": "BTC put",
                    "strike": round(strike, -3),  # round to nearest 1000
                    "expiry_days": 45,
                    "n_contracts_est": round(n_puts, 2),
                    "premium_usd_est": max_premium_usd,
                    "venue": "Deribit",
                    "notes": "Buy via Deribit web UI (no live execution from rig). "
                             "Roll quarterly. Sell if regime turns bear OR cycle_score < 30.",
                }
        except Exception:
            pass

    return {
        "should_hedge": should_hedge,
        "urgency": urgency,
        "risk_factor_count": risk_factors,
        "max_premium_pct_of_bankroll": max_premium_pct,
        "reasoning": reasoning,
        "suggested_structure": suggested_structure,
    }


def main():
    print("=" * 80)
    print("TAIL RISK HEDGE RECOMMENDATION")
    print("=" * 80)
    r = compute_hedge_recommendation(bankroll_usd=200_000)
    print(f"\nUrgency: {r['urgency'].upper()}")
    print(f"Risk factors triggered: {r['risk_factor_count']}/6")
    print(f"Max premium budget: {r['max_premium_pct_of_bankroll']*100:.2f}% of bankroll")
    print()
    print("Reasoning:")
    if not r["reasoning"]:
        print("  No tail-risk indicators triggered.")
    for line in r["reasoning"]:
        print(f"  - {line}")
    print()
    if r["suggested_structure"]:
        print("Suggested hedge structure:")
        for k, v in r["suggested_structure"].items():
            print(f"  {k:<20s}  {v}")
    else:
        print("No hedge needed at current regime.")


if __name__ == "__main__":
    main()
