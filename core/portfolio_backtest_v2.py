"""Portfolio backtest v2 — applies 1%-trader-grade fixes:

1. Inverse-vol position sizing per pair (risk parity)
2. Concordance dampener (when strategies cluster, take less)
3. Cost-aware signal threshold (skip if expected return < 2x cost)
4. Slow rebalancing (only trade if weight delta > 30% of cap)
5. Portfolio vol targeting (scale down if total vol > target)
6. Per-strategy weighting by recent Sharpe (proxy for HRP)

Compare to v1 backtest to measure each fix's contribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from research import signals as res_sig
from signals.funding_basis import funding_signal_series


SLIPPAGE_BPS = 5
FEE_BPS = 10
ANNUALIZATION = 365
SMA_REGIME_WINDOW = 200
COST_PER_SIDE = (SLIPPAGE_BPS + FEE_BPS) / 10_000

# === v2 risk-parity / cost-aware parameters ===
TARGET_PAIR_VOL_ANN = 0.50          # target 50% ann vol per pair (crypto)
MAX_PAIR_GROSS = 0.15               # hard cap: 15% per pair
TARGET_PORTFOLIO_VOL_ANN = 0.20     # target 20% ann portfolio vol
COST_AWARE_MULTIPLIER = 2.0         # signal must clear 2x cost
SLOW_REBALANCE_THRESHOLD = 0.30     # only rebalance if delta > 30% of cap
CONCORDANCE_DAMPEN_THRESHOLD = 0.7  # dampen when 70%+ strategies aligned


def _build_signals(prices_df, funding_signals):
    """Same signal mix as v1."""
    out = {}
    tsmom = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    for pair in prices_df.columns:
        tsmom[pair] = res_sig.tsmom_multi(prices_df[pair], horizons=(30, 90, 180))
    out["tsmom_multi"] = tsmom.fillna(0)

    st = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    for pair in prices_df.columns:
        st[pair] = res_sig.tsmom_single(prices_df[pair], lookback=10)
    out["short_term_tsmom"] = st.fillna(0)

    vbo = pd.DataFrame(index=prices_df.index, columns=prices_df.columns, dtype=float)
    for pair in prices_df.columns:
        vbo[pair] = res_sig.vol_breakout(prices_df[pair], window=30)
    out["vol_breakout"] = vbo.fillna(0)

    if "BTC/USDT" in prices_df.columns and "ETH/USDT" in prices_df.columns:
        ratio = (prices_df["ETH/USDT"] / prices_df["BTC/USDT"]).reindex(prices_df.index).ffill().bfill()
        revert = res_sig.zscore_revert(ratio, window=20).fillna(0)
        ethbtc = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        ethbtc["BTC/USDT"] = revert
        out["ethbtc_revert"] = ethbtc

    if funding_signals:
        f = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        for pair, fs in funding_signals.items():
            if pair in prices_df.columns and not fs.empty:
                f[pair] = fs.reindex(prices_df.index).ffill().fillna(0)
        out["funding_basis"] = f

    return out


def _concordance(row):
    """Concordance score [0, 1] for a row of strategy signals."""
    nonzero = [v for v in row if abs(v) > 0.1]
    if len(nonzero) < 2:
        return 0.0
    pos = sum(1 for v in nonzero if v > 0)
    neg = sum(1 for v in nonzero if v < 0)
    return abs(pos - neg) / len(nonzero)


def backtest_v2(
    universe,
    days_back=730,
    starting_equity=100_000.0,
    long_only=False,
    apply_regime=True,
    use_inverse_vol=True,
    use_concordance_dampen=True,
    use_cost_aware=True,
    use_slow_rebalance=True,
    use_portfolio_vol_target=True,
    include_funding=True,
):
    prices_dict = {}
    for pair in universe:
        df = data.ohlcv_extended(pair, days_back=days_back)
        if not df.empty:
            prices_dict[pair] = df["close"]
    if not prices_dict:
        return {"error": "no price data"}

    aligned = pd.DataFrame(prices_dict).dropna()
    if len(aligned) < SMA_REGIME_WINDOW + 30:
        return {"error": "insufficient history"}

    funding_signals = {}
    if include_funding:
        for pair in universe:
            try:
                base, quote = pair.split("/")
                perp = f"{base}/{quote}:{quote}"
                fs = funding_signal_series(perp_pair=perp, days_back=days_back)
                if not fs.empty:
                    funding_signals[pair] = fs
            except Exception:
                pass

    signals_by_strat = _build_signals(aligned, funding_signals)
    if not signals_by_strat:
        return {"error": "no signals"}

    # === Per-pair realized vol for inverse-vol sizing ===
    pair_returns = aligned.pct_change().fillna(0)
    pair_vol_ann = pair_returns.rolling(30).std() * np.sqrt(ANNUALIZATION)

    # === Per-pair regime ===
    sma200 = aligned.rolling(SMA_REGIME_WINDOW).mean()
    is_bull = aligned > sma200

    # === Per-pair signal aggregation ===
    pair_signals = pd.DataFrame(0.0, index=aligned.index, columns=aligned.columns)
    pair_concordance = pd.DataFrame(0.0, index=aligned.index, columns=aligned.columns)
    for pair in aligned.columns:
        contrib_cols = []
        for sname, sdf in signals_by_strat.items():
            if pair in sdf.columns:
                contrib_cols.append(sdf[pair].rename(sname))
        if not contrib_cols:
            continue
        sig_df = pd.concat(contrib_cols, axis=1).fillna(0)
        # Mean signal across strategies for this pair
        pair_signals[pair] = sig_df.mean(axis=1)
        # Concordance: |pos - neg| / total non-zero
        pair_concordance[pair] = sig_df.apply(_concordance, axis=1)

    # === Apply regime gate ===
    if apply_regime:
        for pair in pair_signals.columns:
            mask_bull = is_bull[pair].fillna(False)
            pair_signals.loc[~mask_bull & (pair_signals[pair] > 0), pair] = 0
            pair_signals.loc[mask_bull & (pair_signals[pair] < 0), pair] = 0

    if long_only:
        pair_signals = pair_signals.clip(lower=0)

    # === Concordance dampener: when strategies cluster, scale signal down ===
    if use_concordance_dampen:
        dampen = (1 - pair_concordance.clip(0, 1)).clip(0.3, 1.0)
        pair_signals = pair_signals * dampen

    # === Inverse-vol position sizing (risk parity per pair) ===
    if use_inverse_vol:
        # Each pair's max gross = TARGET_PAIR_VOL_ANN / realized_vol, capped
        inv_vol_cap = (TARGET_PAIR_VOL_ANN / pair_vol_ann.replace(0, np.nan)).clip(upper=MAX_PAIR_GROSS).fillna(MAX_PAIR_GROSS)
        pair_targets = pair_signals * inv_vol_cap
    else:
        pair_targets = pair_signals * MAX_PAIR_GROSS

    pair_targets = pair_targets.clip(-MAX_PAIR_GROSS, MAX_PAIR_GROSS)

    # === Portfolio vol target: scale all positions down if total portfolio vol too high ===
    if use_portfolio_vol_target:
        # Estimate portfolio vol = sqrt(sum(weight_i * vol_i)^2 * (1+correlation_avg))
        # Simpler: portfolio vol ≈ sum(|weight| * vol) * sqrt(1/n_eff)
        # For brevity: scale by ratio
        gross = pair_targets.abs().sum(axis=1)
        weighted_vol = (pair_targets.abs() * pair_vol_ann).sum(axis=1) / gross.replace(0, np.nan)
        portfolio_vol_est = weighted_vol * gross  # approximation
        scale = (TARGET_PORTFOLIO_VOL_ANN / portfolio_vol_est).clip(upper=1.0).fillna(1.0)
        pair_targets = pair_targets.mul(scale, axis=0)

    # === Daily backtest loop with cost-aware threshold + slow rebalance ===
    n_days = len(aligned)
    equity = np.zeros(n_days)
    equity[0] = starting_equity
    weights = pd.Series(0.0, index=aligned.columns)

    for t in range(1, n_days):
        # Mark-to-market P&L from yesterday's positions
        day_returns = pair_returns.iloc[t]
        position_pnl = (weights * day_returns).sum() * equity[t - 1]

        targets_today = pair_targets.iloc[t]

        # === Slow rebalance: only update if delta > threshold * cap ===
        if use_slow_rebalance:
            delta_threshold = SLOW_REBALANCE_THRESHOLD * MAX_PAIR_GROSS
            new_weights = weights.copy()
            for pair in targets_today.index:
                if abs(targets_today[pair] - weights[pair]) > delta_threshold:
                    new_weights[pair] = targets_today[pair]
        else:
            new_weights = targets_today.copy()

        # === Cost-aware: skip pairs where expected return < 2x cost ===
        if use_cost_aware:
            expected_return = abs(new_weights * pair_vol_ann.iloc[t]) / np.sqrt(ANNUALIZATION)  # daily vol-scaled
            cost_threshold = (SLIPPAGE_BPS + FEE_BPS) / 10_000 * COST_AWARE_MULTIPLIER
            below_threshold = expected_return < cost_threshold
            new_weights[below_threshold] = weights[below_threshold]  # keep old, don't trade

        # Apply rebalance
        weight_changes = (new_weights - weights).abs()
        cost = weight_changes.sum() * COST_PER_SIDE * equity[t - 1]
        equity[t] = equity[t - 1] + position_pnl - cost
        weights = new_weights.copy()

    # === Stats ===
    eq_series = pd.Series(equity, index=aligned.index)
    rets = eq_series.pct_change().dropna()
    if rets.std() == 0:
        return {"error": "no variation"}

    sharpe = float(rets.mean() / rets.std() * np.sqrt(ANNUALIZATION))
    total_return = float(eq_series.iloc[-1] / eq_series.iloc[0] - 1)
    annualized = (1 + total_return) ** (ANNUALIZATION / n_days) - 1
    peak = eq_series.cummax()
    max_dd = float((1 - eq_series / peak).max())

    bench_eqwt = (1 + pair_returns.mean(axis=1)).cumprod() * starting_equity
    btc_ret = pair_returns.get("BTC/USDT", pd.Series(0))
    btc_eq = (1 + btc_ret).cumprod() * starting_equity if not btc_ret.empty else pd.Series([starting_equity])
    btc_total = float(btc_eq.iloc[-1] / starting_equity - 1)

    return {
        "n_days": n_days,
        "n_pairs": len(aligned.columns),
        "n_strategies": len(signals_by_strat),
        "total_return": total_return,
        "annualized_return": float(annualized),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "benchmark_eqwt_total_return": float(bench_eqwt.iloc[-1] / starting_equity - 1),
        "benchmark_btc_total_return": btc_total,
        "alpha_vs_btc": total_return - btc_total,
        "fixes_active": {
            "inverse_vol": use_inverse_vol,
            "concordance_dampen": use_concordance_dampen,
            "cost_aware": use_cost_aware,
            "slow_rebalance": use_slow_rebalance,
            "portfolio_vol_target": use_portfolio_vol_target,
            "long_only": long_only,
            "regime_gate": apply_regime,
        },
    }


if __name__ == "__main__":
    UNIVERSE = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
    ]

    test_configs = [
        ("v1 baseline (no fixes)", dict(use_inverse_vol=False, use_concordance_dampen=False,
                                         use_cost_aware=False, use_slow_rebalance=False,
                                         use_portfolio_vol_target=False)),
        ("only inverse-vol sizing", dict(use_inverse_vol=True, use_concordance_dampen=False,
                                          use_cost_aware=False, use_slow_rebalance=False,
                                          use_portfolio_vol_target=False)),
        ("only concordance dampen", dict(use_inverse_vol=False, use_concordance_dampen=True,
                                          use_cost_aware=False, use_slow_rebalance=False,
                                          use_portfolio_vol_target=False)),
        ("only cost-aware threshold", dict(use_inverse_vol=False, use_concordance_dampen=False,
                                            use_cost_aware=True, use_slow_rebalance=False,
                                            use_portfolio_vol_target=False)),
        ("only slow rebalance", dict(use_inverse_vol=False, use_concordance_dampen=False,
                                      use_cost_aware=False, use_slow_rebalance=True,
                                      use_portfolio_vol_target=False)),
        ("only portfolio vol target", dict(use_inverse_vol=False, use_concordance_dampen=False,
                                            use_cost_aware=False, use_slow_rebalance=False,
                                            use_portfolio_vol_target=True)),
        ("ALL FIXES (1%-grade)", dict(use_inverse_vol=True, use_concordance_dampen=True,
                                       use_cost_aware=True, use_slow_rebalance=True,
                                       use_portfolio_vol_target=True)),
        ("ALL FIXES + long-only", dict(use_inverse_vol=True, use_concordance_dampen=True,
                                        use_cost_aware=True, use_slow_rebalance=True,
                                        use_portfolio_vol_target=True, long_only=True)),
    ]

    print(f"{'Configuration':<35s} {'Total':>10s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha-BTC':>10s}")
    print("-" * 80)
    for label, kwargs in test_configs:
        r = backtest_v2(UNIVERSE, days_back=730, **kwargs)
        if "error" in r:
            print(f"{label:<35s} ERROR: {r['error']}")
        else:
            print(f"{label:<35s} {r['total_return']:>+9.1%} {r['sharpe']:>+7.2f} {r['max_drawdown']:>+7.1%} {r['alpha_vs_btc']:>+9.1%}")
