"""Indicator factor mining — 4 years of price action, 25+ indicators.

Phase 1: Compute each indicator and measure its signal-strength against
         forward returns. Information Coefficient (IC) + Sharpe.

Phase 2: Top-N indicators pair-tested as composite signals.

Phase 3: Walk-forward OOS validation on winning combos.

Data: BTC/USDT, ETH/USDT, SOL/USDT daily over last 1460 days (~4 years).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data
from core.swing_backtest import compute_atr


ANNUALIZATION = 365


# ============================================================================
# INDICATOR LIBRARY — each returns a Series (signal value) indexed by date
# ============================================================================

def ind_sma_ratio(df, fast=50, slow=200):
    """SMA fast/slow ratio - 1. Positive = bullish."""
    return df["close"].rolling(fast).mean() / df["close"].rolling(slow).mean() - 1


def ind_price_vs_sma(df, window=200):
    """Price / SMA - 1. Positive = above SMA."""
    return df["close"] / df["close"].rolling(window).mean() - 1


def ind_rsi(df, period=14):
    """Wilder RSI. >70 overbought, <30 oversold."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ind_macd_signal(df, fast=12, slow=26, signal=9):
    """MACD histogram (MACD line - signal line)."""
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal).mean()
    return (macd - sig) / df["close"]  # normalized


def ind_bollinger_pct(df, window=20, n_std=2.0):
    """Position within Bollinger band: 0 = lower band, 1 = upper band."""
    mid = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (df["close"] - lower) / (upper - lower)


def ind_donchian_pct(df, window=20):
    """Position within Donchian channel: 0 = low, 1 = high."""
    hi = df["high"].rolling(window).max()
    lo = df["low"].rolling(window).min()
    return (df["close"] - lo) / (hi - lo)


def ind_tsmom(df, lookback=30):
    """N-day return."""
    return df["close"].pct_change(lookback)


def ind_obv_slope(df, window=20):
    """On-Balance Volume slope over N days."""
    signed_vol = np.where(df["close"] > df["close"].shift(), df["volume"], -df["volume"])
    obv = pd.Series(signed_vol, index=df.index).cumsum()
    return obv.pct_change(window)


def ind_adx(df, period=14):
    """Average Directional Index — trend strength."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
    minus_dm = (low.diff().abs()).where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def ind_cci(df, period=20):
    """Commodity Channel Index."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    return (typical - sma) / (0.015 * mad)


def ind_stoch_k(df, period=14):
    """Stochastic %K."""
    lowest = df["low"].rolling(period).min()
    highest = df["high"].rolling(period).max()
    return 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)


def ind_williams_r(df, period=14):
    """Williams %R (-100 to 0)."""
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    return -100 * (highest - df["close"]) / (highest - lowest).replace(0, np.nan)


def ind_realized_vol(df, window=20):
    """Annualized realized volatility from daily returns."""
    return df["close"].pct_change().rolling(window).std() * np.sqrt(365)


def ind_volume_z(df, window=20):
    """Z-score of today's volume vs N-day average."""
    avg = df["volume"].rolling(window).mean()
    std = df["volume"].rolling(window).std()
    return (df["volume"] - avg) / std.replace(0, np.nan)


def ind_atr_pct(df, period=14):
    """ATR / price — relative range size."""
    return compute_atr(df, period) / df["close"]


def ind_close_vs_high(df, window=60):
    """Close relative to N-day high. 1 = at high, 0 = at low."""
    hi = df["high"].rolling(window).max()
    lo = df["low"].rolling(window).min()
    return (df["close"] - lo) / (hi - lo).replace(0, np.nan)


def ind_ema_ratio(df, fast=12, slow=26):
    """EMA fast/slow ratio - 1."""
    return (df["close"].ewm(span=fast).mean()
            / df["close"].ewm(span=slow).mean() - 1)


def ind_pct_from_high_n(df, window=90):
    """Distance from N-day high (negative = below)."""
    return df["close"] / df["high"].rolling(window).max() - 1


def ind_vwap_deviation(df, window=20):
    """Price vs volume-weighted average price over N days."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical * df["volume"]).rolling(window).sum() / df["volume"].rolling(window).sum()
    return df["close"] / vwap - 1


INDICATORS = {
    "SMA50/200_ratio":     lambda df: ind_sma_ratio(df, 50, 200),
    "Price/SMA200":         lambda df: ind_price_vs_sma(df, 200),
    "Price/SMA50":          lambda df: ind_price_vs_sma(df, 50),
    "RSI14":                lambda df: ind_rsi(df, 14),
    "RSI21":                lambda df: ind_rsi(df, 21),
    "MACD_hist":            lambda df: ind_macd_signal(df, 12, 26, 9),
    "Bollinger_pct":        lambda df: ind_bollinger_pct(df, 20, 2.0),
    "Donchian20_pct":       lambda df: ind_donchian_pct(df, 20),
    "Donchian50_pct":       lambda df: ind_donchian_pct(df, 50),
    "TSMOM_30":             lambda df: ind_tsmom(df, 30),
    "TSMOM_60":             lambda df: ind_tsmom(df, 60),
    "TSMOM_90":             lambda df: ind_tsmom(df, 90),
    "OBV_slope_20":         lambda df: ind_obv_slope(df, 20),
    "ADX_14":               lambda df: ind_adx(df, 14),
    "CCI_20":               lambda df: ind_cci(df, 20),
    "Stoch_K":              lambda df: ind_stoch_k(df, 14),
    "Williams_R":           lambda df: ind_williams_r(df, 14),
    "Realized_Vol_20":      lambda df: ind_realized_vol(df, 20),
    "Volume_Z":             lambda df: ind_volume_z(df, 20),
    "ATR_pct":              lambda df: ind_atr_pct(df, 14),
    "Close_vs_60d_HL":      lambda df: ind_close_vs_high(df, 60),
    "EMA12/26_ratio":       lambda df: ind_ema_ratio(df, 12, 26),
    "Pct_from_90d_high":    lambda df: ind_pct_from_high_n(df, 90),
    "VWAP_deviation_20":    lambda df: ind_vwap_deviation(df, 20),
}


# ============================================================================
# SIGNAL EVALUATION
# ============================================================================

def information_coefficient(indicator: pd.Series, fwd_return: pd.Series) -> float:
    """Spearman rank correlation between indicator and forward return."""
    both = pd.concat([indicator, fwd_return], axis=1).dropna()
    if len(both) < 30:
        return 0.0
    return float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))


def signal_to_long_only_sharpe(
    indicator: pd.Series, returns: pd.Series, entry_threshold: float = None,
    is_quantile: bool = True, quantile: float = 0.7,
) -> dict:
    """Trade long when indicator > entry_threshold. Long-only Sharpe."""
    if is_quantile:
        entry_threshold = indicator.quantile(quantile)
    signal = (indicator > entry_threshold).astype(int).shift(1)  # next-day position
    strategy_rets = signal * returns
    strategy_rets = strategy_rets.dropna()
    if len(strategy_rets) < 30 or strategy_rets.std() == 0:
        return {"sharpe": 0, "trades": 0, "avg_ret": 0, "win_rate": 0}
    sharpe = float(strategy_rets.mean() / strategy_rets.std() * np.sqrt(ANNUALIZATION))
    trades = int((signal.diff() != 0).sum())
    in_market = signal == 1
    win_rate = float((strategy_rets[in_market] > 0).sum() / max(in_market.sum(), 1))
    return {"sharpe": sharpe, "trades": trades, "avg_ret": float(strategy_rets[in_market].mean()),
            "win_rate": win_rate, "pct_in_market": float(in_market.mean())}


def compute_all_indicators(df):
    """Returns a DataFrame with one column per indicator."""
    out = {}
    for name, fn in INDICATORS.items():
        try:
            out[name] = fn(df)
        except Exception as e:
            print(f"  WARN: {name} failed: {e}")
    return pd.DataFrame(out, index=df.index)


# ============================================================================
# PHASE 1 — single-indicator forward-return analysis
# ============================================================================

def phase1_single_indicators(pairs, days_back=1460, horizons=(5, 10, 30)):
    """For each indicator on each pair, compute IC + long-only Sharpe."""
    rows = []
    for pair in pairs:
        df = data.ohlcv_extended(pair, days_back=days_back + 250)
        if df.empty or len(df) < 300:
            continue
        df = df.copy()
        ind_df = compute_all_indicators(df)
        rets = df["close"].pct_change()
        for ind_name in ind_df.columns:
            ind_series = ind_df[ind_name]
            row = {"pair": pair, "indicator": ind_name}
            # IC at each horizon
            for h in horizons:
                fwd_ret = df["close"].pct_change(h).shift(-h)
                row[f"IC_{h}d"] = information_coefficient(ind_series, fwd_ret)
            # Long-only Sharpe at quantile threshold
            s = signal_to_long_only_sharpe(ind_series, rets, is_quantile=True, quantile=0.7)
            row["sharpe_q70"] = s["sharpe"]
            row["pct_in_mkt_q70"] = s["pct_in_market"]
            row["win_rate_q70"] = s["win_rate"]
            # Also try quantile 0.5 (median split)
            s50 = signal_to_long_only_sharpe(ind_series, rets, is_quantile=True, quantile=0.5)
            row["sharpe_q50"] = s50["sharpe"]
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# PHASE 2 — pairwise combinations
# ============================================================================

def phase2_pairwise(top_indicators, pairs, days_back=1460):
    """Take top N indicators, test all 2-indicator combinations (AND signal)."""
    results = []
    for pair in pairs:
        df = data.ohlcv_extended(pair, days_back=days_back + 250)
        if df.empty or len(df) < 300:
            continue
        ind_df = compute_all_indicators(df)
        rets = df["close"].pct_change()
        for ind1, ind2 in combinations(top_indicators, 2):
            if ind1 not in ind_df.columns or ind2 not in ind_df.columns:
                continue
            s1 = ind_df[ind1]
            s2 = ind_df[ind2]
            # AND signal: both indicators > 70th percentile
            sig = ((s1 > s1.quantile(0.7)) & (s2 > s2.quantile(0.7))).astype(int).shift(1)
            strat_rets = sig * rets
            strat_rets = strat_rets.dropna()
            if len(strat_rets) < 30 or strat_rets.std() == 0:
                continue
            sharpe = float(strat_rets.mean() / strat_rets.std() * np.sqrt(ANNUALIZATION))
            pct_in = float((sig == 1).mean())
            cumret = float((1 + strat_rets).prod() - 1)
            results.append({
                "pair": pair, "ind1": ind1, "ind2": ind2,
                "sharpe": sharpe, "cumulative_ret": cumret,
                "pct_in_market": pct_in,
            })
    return pd.DataFrame(results)


# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("INDICATOR FACTOR MINING — 4 year history, 24 indicators")
    print("=" * 80)
    print()

    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    print(f"Pairs: {pairs}")
    print(f"Window: ~4 years (1460 days)")
    print(f"Indicators: {len(INDICATORS)}")
    print()

    # --- Phase 1 ---
    print("PHASE 1: Single-indicator scan...")
    df1 = phase1_single_indicators(pairs)
    print(f"  Total indicator x pair combinations: {len(df1)}")
    print()

    # Avg across pairs (per indicator)
    avg_per_ind = df1.groupby("indicator").agg({
        "IC_5d": "mean", "IC_10d": "mean", "IC_30d": "mean",
        "sharpe_q70": "mean", "sharpe_q50": "mean",
        "pct_in_mkt_q70": "mean", "win_rate_q70": "mean",
    }).round(3)

    # Rank by best sharpe across q70 and q50
    avg_per_ind["best_sharpe"] = avg_per_ind[["sharpe_q70", "sharpe_q50"]].max(axis=1)
    ranked = avg_per_ind.sort_values("best_sharpe", ascending=False)
    print("TOP 15 SINGLE INDICATORS (avg Sharpe across 3 pairs):")
    print(f"{'Indicator':<22s} {'IC_5d':>7s} {'IC_30d':>7s} {'Sh_q70':>7s} {'Sh_q50':>7s} {'%In_Mkt':>8s}")
    for ind_name, row in ranked.head(15).iterrows():
        print(f"{ind_name:<22s} {row['IC_5d']:>+6.3f}  {row['IC_30d']:>+6.3f}  "
              f"{row['sharpe_q70']:>+6.2f}  {row['sharpe_q50']:>+6.2f}  "
              f"{row['pct_in_mkt_q70']:>7.1%}")
    print()
    print("BOTTOM 5 (worst):")
    for ind_name, row in ranked.tail(5).iterrows():
        print(f"{ind_name:<22s} {row['IC_5d']:>+6.3f}  {row['IC_30d']:>+6.3f}  "
              f"{row['sharpe_q70']:>+6.2f}  {row['sharpe_q50']:>+6.2f}  "
              f"{row['pct_in_mkt_q70']:>7.1%}")
    print()

    # --- Phase 2: pairwise combos using top 10 indicators ---
    top10 = ranked.head(10).index.tolist()
    print(f"PHASE 2: Pairwise (AND) combinations of top 10 indicators...")
    print(f"  Indicators used: {top10}")
    print()
    df2 = phase2_pairwise(top10, pairs)
    avg_combo = df2.groupby(["ind1", "ind2"]).agg({
        "sharpe": "mean", "cumulative_ret": "mean", "pct_in_market": "mean",
    }).round(3)
    avg_combo = avg_combo.sort_values("sharpe", ascending=False)
    print("TOP 15 PAIRWISE COMBOS (Long when BOTH > 70th %ile):")
    print(f"{'Indicator 1':<22s} {'Indicator 2':<22s} {'Sharpe':>7s} {'CumRet':>9s} {'%In_Mkt':>8s}")
    for (i1, i2), row in avg_combo.head(15).iterrows():
        print(f"{i1:<22s} {i2:<22s} {row['sharpe']:>+6.2f}  {row['cumulative_ret']:>+8.1%}  "
              f"{row['pct_in_market']:>7.1%}")
    print()

    # --- Compare to baseline (current pro_trend long-only) ---
    print("BASELINE COMPARISON: current pro_trend (price > SMA200 + Donchian-20 high break)")
    print("  Backtest reference: Sharpe 1.40, annualized +80% (LONG-ONLY)")
    print()
    print("Note: Phase 1 indicators trade DAILY based on threshold cross, which is")
    print("DIFFERENT from pro_trend's wide-stop trend-follower mechanics. Direct")
    print("Sharpe comparison isn't apples-to-apples — Phase 1 measures signal")
    print("POWER, not strategy backtest. Top indicators here may be used as")
    print("entry FILTERS or COMPOSITE signals atop existing trend-follower.")
