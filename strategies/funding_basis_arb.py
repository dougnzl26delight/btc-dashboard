"""Funding-rate basis arbitrage strategy.

The textbook crypto-carry trade. Long spot + short perp when perp funding
rate exceeds threshold; collect funding payments every 8h.

Documented Sharpe 4-5 in our 999-day backtest across BTC/ETH/SOL.
Per-pair returns 2.6-3.3% per year at 30% allocation, max DD < 0.5%.
Realistic edge for retail because the trade is delta-neutral and the
funding premium reflects retail leverage demand.

LIVE EXECUTION REQUIRES PERP BROKER. Currently only paper-tradeable
because we lack a perp broker class — see BACKLOG.md.

For now, exposed as a SIGNAL: positive when funding favorable, used by
the orchestrator to know whether to allocate capital to the basis trade
once the perp broker exists.

VALIDATED = False on live; backtest passes the documented threshold.
Counts as trial #57.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data


VALIDATED = False
NAME = "funding_basis_arb"

# 2026-06-01 W14.A: RAISED back to 2.0bp/8h per academic finding.
# Mathematics 2026 (DOI 10.3390/math14020346) analyzed 35.7M funding obs across
# 26 exchanges. Binance basis arb Sharpe = -7.34 at LOW thresholds (=cost stack
# eats edge). Only HIGH-conviction entries clear costs reliably.
# Cost stack per round-trip: ~20-30bps (slippage + fees on both legs).
# To net positive: gross funding capture must exceed ~25bps over hold period.
# At 5-day average hold, 2.0bp/8h = 30bps gross captured = barely positive.
# Below 2.0bp/8h on CEXs is structurally negative-Sharpe.
ENTRY_FUNDING_BPS_8H = 2.0   # ~22% annualized — only clear-cost-stack entries
EXIT_FUNDING_BPS_8H = 1.0    # close when funding falls below entry conviction

# W14.A also: require 5-day stable delta before entry (per academic + practitioner)
# Single-spike entry is "gambling, not strategy" — only persistent funding edges.
ENTRY_STABILITY_DAYS = 5
ENTRY_STABILITY_MIN_BPS = 1.0  # must average ≥1bp over last 5 days to qualify


def latest_signal(pair: str = "BTC/USDT") -> float:
    """Returns +1 when funding > entry, 0 below entry, -1 when funding < -entry.

    Note: this is a FLAG signal, not a directional weight. The actual basis
    trade requires both spot and perp legs which the spot-only orchestrator
    can't currently execute. When the perp broker is built, this signal
    drives basis-trade allocation.
    """
    perp_pair = pair if ":" in pair else f"{pair}:{pair.split('/')[1]}"
    try:
        funding_df = data.funding_history(perp_pair, limit=1)
        if funding_df.empty:
            return 0.0
        funding_bps_8h = float(funding_df["funding_rate"].iloc[-1]) * 10_000.0
    except Exception:
        return 0.0

    if funding_bps_8h > ENTRY_FUNDING_BPS_8H:
        return 1.0  # bull funding regime — open long-spot/short-perp basis trade
    if funding_bps_8h < -ENTRY_FUNDING_BPS_8H:
        return -1.0  # bear funding regime — open short-spot/long-perp (rare)
    return 0.0


def latest_funding_bps_8h(pair: str = "BTC/USDT") -> float:
    perp_pair = pair if ":" in pair else f"{pair}:{pair.split('/')[1]}"
    try:
        funding_df = data.funding_history(perp_pair, limit=1)
        if funding_df.empty:
            return 0.0
        return float(funding_df["funding_rate"].iloc[-1]) * 10_000.0
    except Exception:
        return 0.0


def rank_universe_by_funding(pairs: list[str]) -> list[dict]:
    """W10 continuous-capture: rank all pairs by current funding, descending.

    W14.A enhancement: also fetches 5-day funding stability — pairs that have
    spent the last N days WITH funding above stability minimum get a stability
    bonus. Single-spike entries are filtered out.
    """
    rows = []
    for pair in pairs:
        bps = latest_funding_bps_8h(pair)
        # Fetch trailing 5-day funding stability (W14.A)
        try:
            perp_pair = pair if ":" in pair else f"{pair}:{pair.split('/')[1]}"
            history = data.funding_history(perp_pair, limit=15)  # ~5 days of 8h periods
            recent_bps = history["funding_rate"].astype(float) * 10_000.0 if not history.empty else None
            if recent_bps is not None and len(recent_bps) >= 10:
                stable_mean_bps = float(recent_bps.mean())
                stable_min_bps = float(recent_bps.min())
                is_stable = stable_mean_bps >= ENTRY_STABILITY_MIN_BPS
            else:
                stable_mean_bps = bps
                stable_min_bps = bps
                is_stable = False
        except Exception:
            stable_mean_bps = bps
            stable_min_bps = bps
            is_stable = False
        rows.append({
            "pair": pair,
            "funding_bps_8h": bps,
            "stable_mean_bps_5d": stable_mean_bps,
            "stable_min_bps_5d": stable_min_bps,
            "is_stable_5d": is_stable,
            "annualized_pct": bps * 3 * 365 / 100,
            "qualifies_for_entry": bps > ENTRY_FUNDING_BPS_8H and is_stable,
        })
    return sorted(rows, key=lambda r: -r["funding_bps_8h"])


if __name__ == "__main__":
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        sig = latest_signal(pair)
        bps = latest_funding_bps_8h(pair)
        print(f"{pair}: funding={bps:+.3f} bps/8h ({bps*3*365/100:+.1f}% ann), basis_arb signal={sig:+.1f}")
