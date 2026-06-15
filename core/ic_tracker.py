"""Information coefficient tracker — detects signal decay over time.

IC = correlation between signal at t and forward return at t+1.
Track its rolling mean; if recent IC drops materially below historical, the
signal is decaying and the strategy should be paused or retired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ic(signal: pd.Series, forward_returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Spearman correlation between signal and forward returns."""
    df = pd.DataFrame({"sig": signal, "ret": forward_returns}).dropna()
    if len(df) < window:
        return pd.Series(dtype=float)

    out_vals: list[float] = []
    out_idx: list = []
    for end in range(window, len(df) + 1):
        chunk = df.iloc[end - window: end]
        if chunk["sig"].std() == 0 or chunk["ret"].std() == 0:
            corr = 0.0
        else:
            corr = chunk["sig"].corr(chunk["ret"], method="spearman")
            if np.isnan(corr):
                corr = 0.0
        out_vals.append(float(corr))
        out_idx.append(df.index[end - 1])
    return pd.Series(out_vals, index=out_idx)


def degradation_alert(
    ic: pd.Series, recent_window: int = 30, threshold: float = 0.5
) -> dict:
    """Flag if recent IC has materially dropped vs historical IC."""
    if len(ic) < recent_window * 2:
        return {"degraded": False, "reason": "insufficient observations"}
    historical = float(ic.iloc[:-recent_window].mean())
    recent = float(ic.iloc[-recent_window:].mean())
    if historical <= 0:
        return {"degraded": False, "historical_ic": historical, "recent_ic": recent}
    decay = 1.0 - recent / historical
    return {
        "degraded": decay > threshold,
        "historical_ic": historical,
        "recent_ic": recent,
        "decay": decay,
    }
