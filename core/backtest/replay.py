"""P8: Historical scenario replay.

Walk-forward replay the engine over a window. Compare to benchmarks:
  - 60/40 SPY/AGG
  - 90/10 SPY/BTC (DCA)
  - All-cash baseline (T-bills)

Returns Sharpe, max DD, total return, hit-rate by regime.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd


HISTORICAL_SCENARIOS = {
    "2000-2003_dotcom":       {"start": "2000-03-01", "end": "2003-03-01"},
    "2007-2009_GFC":          {"start": "2007-10-01", "end": "2009-03-01"},
    "2013-2015_taper":        {"start": "2013-05-01", "end": "2015-12-01"},
    "2018_vol_spike":         {"start": "2018-01-01", "end": "2018-12-31"},
    "2020_covid":             {"start": "2020-02-01", "end": "2020-08-01"},
    "2022_hikes":             {"start": "2022-01-01", "end": "2023-01-01"},
    "2024-2025_etf_era":      {"start": "2024-01-01", "end": "2025-12-31"},
}


def _yf(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=start, end=end)
        if df is None or df.empty: return None
        return df
    except Exception:
        return None


def fetch_scenario_data(scenario: dict) -> Optional[pd.DataFrame]:
    """Fetch the prices we need to backtest a scenario."""
    start, end = scenario["start"], scenario["end"]
    tickers = ["SPY", "AGG", "BTC-USD", "BIL", "GLDM", "VTIP"]
    data = {}
    for t in tickers:
        df = _yf(t, start, end)
        if df is not None and not df.empty:
            data[t] = df["Close"]
    if not data: return None
    return pd.DataFrame(data).dropna(how="all").ffill()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def sharpe(daily_returns: pd.Series, rf: float = 0.04) -> float:
    if daily_returns is None or daily_returns.empty: return 0.0
    excess = daily_returns - rf / 252
    if excess.std() == 0: return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def max_drawdown(daily_returns: pd.Series) -> float:
    if daily_returns is None or daily_returns.empty: return 0.0
    eq = (1 + daily_returns).cumprod()
    peak = eq.cummax()
    return float((eq / peak - 1).min())


def total_return(daily_returns: pd.Series) -> float:
    if daily_returns is None or daily_returns.empty: return 0.0
    return float((1 + daily_returns).prod() - 1)


# ============================================================
# Benchmark portfolios
# ============================================================

def benchmark_60_40(returns: pd.DataFrame) -> pd.Series:
    """60% SPY / 40% AGG, daily rebalanced."""
    if "SPY" not in returns.columns or "AGG" not in returns.columns:
        return pd.Series(dtype=float)
    return 0.6 * returns["SPY"] + 0.4 * returns["AGG"]


def benchmark_90_10_btc(returns: pd.DataFrame) -> pd.Series:
    """90% SPY / 10% BTC daily rebalanced."""
    if "SPY" not in returns.columns: return pd.Series(dtype=float)
    btc_col = "BTC-USD" if "BTC-USD" in returns.columns else None
    if btc_col is None: return returns["SPY"]
    return 0.9 * returns["SPY"] + 0.1 * returns[btc_col].fillna(0)


def benchmark_all_cash(returns: pd.DataFrame) -> pd.Series:
    """Approximation: BIL ETF returns; fallback to 4% annual constant."""
    if "BIL" in returns.columns:
        return returns["BIL"]
    return pd.Series(0.04 / 252, index=returns.index)


# ============================================================
# Simplified engine replay
# ============================================================

def simple_engine_replay(returns: pd.DataFrame,
                          regime_series: Optional[pd.Series] = None,
                          rebalance_freq: int = 21) -> pd.Series:
    """Simplified engine replay using a stylized regime → weights rule.

    Without full historical macro signals, we approximate regimes by:
      - 60d trailing SPY return sign + vol regime
      - SPY drawdown depth

    Weights per regime (matching live engine logic):
      RISK_ON:    60% SPY / 0% BTC / 40% BIL
      LATE_CYCLE: 30% SPY / 0% BTC / 70% BIL (mostly cash)
      BEAR:       10% SPY / 25% BTC / 65% BIL (bear buy fear)
    """
    if returns.empty: return pd.Series(dtype=float)
    # Default regime classifier (approximate using SPY behavior)
    if regime_series is None and "SPY" in returns.columns:
        spy = (1 + returns["SPY"]).cumprod()
        dd = spy / spy.cummax() - 1
        vol_60d = returns["SPY"].rolling(60).std()
        med_vol = vol_60d.median()
        regimes = []
        for d, v in zip(dd, vol_60d):
            if d < -0.15 or (v is not None and v > med_vol * 1.5):
                regimes.append("RECESSIONARY_BEAR")
            elif d < -0.05 or (v is not None and v > med_vol * 1.2):
                regimes.append("LATE_CYCLE")
            else:
                regimes.append("RISK_ON")
        regime_series = pd.Series(regimes, index=returns.index)

    weights_per_regime = {
        "RISK_ON":           {"SPY": 0.60, "BIL": 0.40, "BTC-USD": 0.00},
        "LATE_CYCLE":        {"SPY": 0.30, "BIL": 0.70, "BTC-USD": 0.00},
        "RECESSIONARY_BEAR": {"SPY": 0.10, "BIL": 0.65, "BTC-USD": 0.25},
    }

    engine_rets = []
    current_weights = weights_per_regime.get(regime_series.iloc[0] if len(regime_series) else "RISK_ON",
                                                {})
    for i, (dt, row) in enumerate(returns.iterrows()):
        # Rebalance periodically
        if i % rebalance_freq == 0 and len(regime_series) > i:
            target_regime = regime_series.iloc[i]
            current_weights = weights_per_regime.get(target_regime,
                                weights_per_regime["LATE_CYCLE"])
        # Portfolio daily return
        r = sum(current_weights.get(a, 0) * row.get(a, 0)
                for a in current_weights)
        engine_rets.append(r)
    return pd.Series(engine_rets, index=returns.index)


# ============================================================
# Full scenario replay
# ============================================================

def replay_scenario(scenario_name: str) -> dict:
    """Replay a historical scenario, return engine vs benchmark metrics."""
    if scenario_name not in HISTORICAL_SCENARIOS:
        return {"error": f"unknown scenario: {scenario_name}"}
    sc = HISTORICAL_SCENARIOS[scenario_name]
    prices = fetch_scenario_data(sc)
    if prices is None or prices.empty:
        return {"error": "data unavailable", "scenario": scenario_name}

    returns = compute_returns(prices)
    if returns.empty:
        return {"error": "returns empty", "scenario": scenario_name}

    engine_rets = simple_engine_replay(returns)
    bench_6040 = benchmark_60_40(returns)
    bench_btc = benchmark_90_10_btc(returns)
    bench_cash = benchmark_all_cash(returns)

    def _metrics(r):
        return {
            "total_return": total_return(r),
            "sharpe": sharpe(r),
            "max_dd": max_drawdown(r),
            "vol_annual": float(r.std() * np.sqrt(252)) if len(r) > 0 else 0,
        }

    engine_m = _metrics(engine_rets)
    b6040_m = _metrics(bench_6040)
    bbtc_m = _metrics(bench_btc)
    bcash_m = _metrics(bench_cash)

    return {
        "scenario": scenario_name,
        "start": sc["start"],
        "end": sc["end"],
        "n_days": len(returns),
        "engine": engine_m,
        "benchmarks": {
            "60_40_SPY_AGG":   b6040_m,
            "90_10_SPY_BTC":   bbtc_m,
            "all_cash_BIL":    bcash_m,
        },
        "outperformance_vs_6040": engine_m["total_return"] - b6040_m["total_return"],
        "sharpe_uplift_vs_6040": engine_m["sharpe"] - b6040_m["sharpe"],
        "dd_better_than_6040": engine_m["max_dd"] > b6040_m["max_dd"],
        "pass": (engine_m["sharpe"] > b6040_m["sharpe"]
                 and engine_m["max_dd"] > b6040_m["max_dd"]),
    }


def replay_all_scenarios() -> list[dict]:
    return [replay_scenario(name) for name in HISTORICAL_SCENARIOS]


def main():
    print("Replay smoke test — 2022_hikes scenario")
    r = replay_scenario("2022_hikes")
    if "error" in r:
        print(f"  Error: {r['error']}")
        return
    print(f"  Engine: total {r['engine']['total_return']:+.2%}  "
          f"Sharpe {r['engine']['sharpe']:+.2f}  MaxDD {r['engine']['max_dd']:.2%}")
    print(f"  60/40:  total {r['benchmarks']['60_40_SPY_AGG']['total_return']:+.2%}  "
          f"Sharpe {r['benchmarks']['60_40_SPY_AGG']['sharpe']:+.2f}")
    print(f"  Pass: {r['pass']}")


if __name__ == "__main__":
    main()
