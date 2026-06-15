"""Diverse signal ensemble — momentum + ETF flows + ETH/BTC reversion.

Hypothesis: combining signals that capture FUNDAMENTALLY DIFFERENT axes
(price trend, institutional flow, relative valuation) gives more
diversification benefit than ensembling within a single family.

Tests:
  1. Pairwise correlation of the three signals (should be low)
  2. Equal-weighted ensemble — expected to raise OOS Sharpe and lower min-fold
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from research import signals as sig
from research.etf_flows import etf_flow_signal, fetch_etf_flows
from research.sweep import evaluate_candidate


def build_signals(days_back: int = 2000):
    """Construct the three signal components, all aligned to BTC daily index."""
    btc = data.ohlcv_extended("BTC/USDT", days_back=days_back)["close"]
    eth = data.ohlcv_extended("ETH/USDT", days_back=days_back)["close"]
    eth_btc = (eth / btc).dropna()
    flows = fetch_etf_flows()

    sig_momentum = sig.tsmom_multi(btc, horizons=(30, 90))
    sig_revert = sig.zscore_revert(eth_btc.reindex(btc.index).ffill().bfill(), window=20)
    sig_etf = (
        etf_flow_signal(flows, ema_window=21).reindex(btc.index).ffill().bfill()
        if not flows.empty
        else pd.Series(0.0, index=btc.index)
    )

    return btc, sig_momentum, sig_revert, sig_etf


def correlations(s1: pd.Series, s2: pd.Series, s3: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"momentum": s1, "ethbtc_revert": s2, "etf_flows": s3}).dropna()
    return df.corr()


def ensemble_signal(s1: pd.Series, s2: pd.Series, s3: pd.Series) -> pd.Series:
    aligned = pd.DataFrame({"m": s1, "r": s2, "e": s3}).fillna(0)
    return aligned.mean(axis=1)


def main():
    btc, s_mom, s_revert, s_etf = build_signals()
    bench = btc.pct_change().fillna(0)

    print("=== Pairwise correlations of signal components ===")
    print(correlations(s_mom, s_revert, s_etf).round(3).to_string())

    candidates = [
        {
            "name": "diverse3_mom+etf+ethbtc",
            "fn": (lambda mom=s_mom, rev=s_revert, etf=s_etf: lambda p: ensemble_signal(
                mom.reindex(p.index).ffill().fillna(0),
                rev.reindex(p.index).ffill().fillna(0),
                etf.reindex(p.index).ffill().fillna(0),
            ))(),
        },
        {
            "name": "diverse2_mom+etf",
            "fn": (lambda mom=s_mom, etf=s_etf: lambda p: (
                mom.reindex(p.index).ffill().fillna(0)
                + etf.reindex(p.index).ffill().fillna(0)
            ) / 2)(),
        },
        {
            "name": "diverse2_mom+ethbtc",
            "fn": (lambda mom=s_mom, rev=s_revert: lambda p: (
                mom.reindex(p.index).ffill().fillna(0)
                + rev.reindex(p.index).ffill().fillna(0)
            ) / 2)(),
        },
    ]

    # num_trials inflated: 35 (sweep) + 4 (XS) + 3 (ensembles) + 3 (diverse) ~= 45
    num_trials = 45

    print("\n=== Diverse ensembles (num_trials=45) ===")
    for c in candidates:
        r = evaluate_candidate(c, btc, bench, num_trials=num_trials)
        print(
            f"{c['name']:30s} OOS={r.get('mean_sharpe_oos',0):+.2f} "
            f"(min={r.get('min_sharpe_oos',0):+.2f}) alpha_t={r.get('alpha_t',0):+.2f} "
            f"beta={r.get('beta',0):+.3f} dsr={r.get('dsr',0):.2f} val={r.get('validated', False)}"
        )


if __name__ == "__main__":
    main()
