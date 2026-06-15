"""P1: Standardization layer.

Every raw signal gets converted into:
  - rolling z-score (10y window default)
  - rolling percentile rank (5y window default)
  - velocity (Δ over 30/90d)

Raw thresholds drift across regimes. The only durable representation
is a regime-agnostic z-score / percentile.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# Trading-day windows
TEN_YEARS = 2520
FIVE_YEARS = 1260
TWO_YEARS = 504


def rolling_zscore(series: pd.Series, window: int = TEN_YEARS,
                    min_periods: int = 252,
                    winsorize: float = 3.0) -> pd.Series:
    """Rolling z-score over `window` days, winsorized at ±`winsorize` std."""
    s = pd.Series(series).astype(float)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    z = (s - mean) / std
    return z.clip(-winsorize, winsorize)


def rolling_percentile(series: pd.Series, window: int = FIVE_YEARS,
                        min_periods: int = 252) -> pd.Series:
    """Rolling percentile rank in [0, 1]."""
    s = pd.Series(series).astype(float)
    return s.rolling(window, min_periods=min_periods).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) if len(x) > 0 else np.nan,
        raw=False,
    )


def velocity(series: pd.Series, lookback: int = 30) -> pd.Series:
    """Absolute change over `lookback` days."""
    s = pd.Series(series).astype(float)
    return s - s.shift(lookback)


def velocity_z(series: pd.Series, lookback: int = 30,
                  zscore_window: int = FIVE_YEARS) -> pd.Series:
    """Z-score of the velocity — captures acceleration/deceleration."""
    v = velocity(series, lookback)
    return rolling_zscore(v, window=zscore_window)


def standardize_signal(series: pd.Series, name: str = "",
                         z_window: int = TEN_YEARS,
                         pct_window: int = FIVE_YEARS) -> dict:
    """Return a single dict summarising a signal's current standardized state.

    Used by the live engine to fetch one feature row per signal.
    """
    if series is None or len(series) == 0:
        return {"name": name, "raw": None, "z": None, "percentile": None,
                "v30": None, "v90": None, "v30_z": None}
    z = rolling_zscore(series, window=z_window)
    p = rolling_percentile(series, window=pct_window)
    v30 = velocity(series, 30)
    v90 = velocity(series, 90)
    v30z = velocity_z(series, lookback=30, zscore_window=pct_window)

    def _last(x):
        try:
            v = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
            return float(v) if not pd.isna(v) else None
        except Exception:
            return None

    return {
        "name": name,
        "raw": _last(series),
        "z": _last(z),
        "percentile": _last(p),
        "v30": _last(v30),
        "v90": _last(v90),
        "v30_z": _last(v30z),
    }


def standardize_batch(signals: dict[str, pd.Series],
                        z_window: int = TEN_YEARS,
                        pct_window: int = FIVE_YEARS) -> dict[str, dict]:
    """Standardize a batch of named signals. Returns dict-of-dicts."""
    out = {}
    for name, series in signals.items():
        try:
            out[name] = standardize_signal(series, name=name,
                                             z_window=z_window,
                                             pct_window=pct_window)
        except Exception as e:
            out[name] = {"name": name, "error": str(e)[:80]}
    return out


def main():
    # Smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 3000
    series = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100,
                        index=pd.date_range("2015-01-01", periods=n))
    r = standardize_signal(series, name="synthetic")
    print("Standardize smoke test:")
    for k, v in r.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
