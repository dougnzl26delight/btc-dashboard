"""P2: Information Coefficient table.

For each signal, measure Spearman rank-correlation with forward returns
of SPY and BTC across multiple horizons.

Decision rules:
  - IC > 0.05 = decent
  - IC > 0.10 = rare/valuable
  - IC > 0.15 = gold
  - |IC| < 0.03 across BOTH 6m and 12m horizons = drop / demote to "watch"

Robustness check: optimal horizon's IC sign must agree with 2 nearest
horizons (otherwise the signal is noise that happens to spike at one h).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IC_CACHE = REPO_ROOT / ".ic_table.json"

# Default horizons (trading days): 1m, 3m, 6m, 12m, 24m
DEFAULT_HORIZONS = [21, 63, 126, 252, 504]


def information_coefficient(signal: pd.Series, returns: pd.Series,
                              horizon_days: int) -> dict:
    """Spearman rank correlation: signal[t] vs sum(returns[t:t+h])."""
    if signal is None or returns is None:
        return {"ic": None, "n": 0, "horizon_days": horizon_days}

    s = pd.Series(signal).dropna()
    r = pd.Series(returns).dropna()
    # Align indexes
    aligned = pd.concat([s, r], axis=1, join="inner").dropna()
    if len(aligned) < horizon_days + 100:
        return {"ic": None, "n": len(aligned), "horizon_days": horizon_days}
    aligned.columns = ["sig", "ret"]

    # Forward sum return
    fwd_ret = aligned["ret"].rolling(horizon_days).sum().shift(-horizon_days)
    paired = pd.concat([aligned["sig"], fwd_ret], axis=1).dropna()
    if len(paired) < 100:
        return {"ic": None, "n": len(paired), "horizon_days": horizon_days}

    rho, p = spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
    if np.isnan(rho):
        return {"ic": None, "n": len(paired), "horizon_days": horizon_days}
    ir = float(rho) * np.sqrt(len(paired) / 252)  # info ratio approx
    return {
        "ic": float(rho), "p_value": float(p),
        "n": len(paired), "horizon_days": horizon_days,
        "ir": float(ir),
    }


def find_optimal_horizon(signal: pd.Series, returns: pd.Series,
                          horizons: list = None) -> dict:
    """Test multiple horizons, return one with max |IC| + robustness."""
    horizons = horizons or DEFAULT_HORIZONS
    results = {h: information_coefficient(signal, returns, h) for h in horizons}
    valid = {h: r for h, r in results.items() if r.get("ic") is not None}
    if not valid:
        return {"ic": None, "robust": False, "horizon_days": None,
                "all_horizons": results}

    best_h, best_r = max(valid.items(), key=lambda x: abs(x[1]["ic"]))
    # Robustness: neighbors agree on sign
    horizons_sorted = sorted(valid.keys())
    idx = horizons_sorted.index(best_h)
    sign = np.sign(best_r["ic"])
    neighbors_ok = True
    if idx > 0:
        neighbors_ok &= np.sign(valid[horizons_sorted[idx-1]]["ic"]) == sign
    if idx < len(horizons_sorted) - 1:
        neighbors_ok &= np.sign(valid[horizons_sorted[idx+1]]["ic"]) == sign

    return {
        **best_r,
        "robust": bool(neighbors_ok),
        "all_horizons": {h: r.get("ic") for h, r in valid.items()},
    }


def signal_passes_oos_gate(signal: pd.Series, returns: pd.Series,
                              min_ic: float = 0.03,
                              horizons: list = None) -> dict:
    """Pass if BOTH 6m and 12m ICs exceed min_ic in absolute value."""
    horizons = horizons or [126, 252]
    ics = [information_coefficient(signal, returns, h)["ic"] for h in horizons]
    valid = [ic for ic in ics if ic is not None]
    if not valid:
        return {"pass": False, "reason": "no_valid_ic"}
    all_pass = all(abs(ic) > min_ic for ic in valid)
    return {
        "pass": bool(all_pass),
        "ic_6m": ics[0] if ics else None,
        "ic_12m": ics[1] if len(ics) > 1 else None,
        "reason": "ok" if all_pass else "ic_below_threshold",
    }


def build_ic_table(signals_dict: dict[str, pd.Series],
                    returns_spy: pd.Series,
                    returns_btc: Optional[pd.Series] = None,
                    horizons: list = None) -> pd.DataFrame:
    """Compute IC table for a dictionary of signals vs SPY (+ BTC if provided)."""
    horizons = horizons or DEFAULT_HORIZONS
    rows = []
    for name, sig in signals_dict.items():
        best_spy = find_optimal_horizon(sig, returns_spy, horizons)
        gate_spy = signal_passes_oos_gate(sig, returns_spy)
        row = {
            "signal": name,
            "best_ic_spy": best_spy.get("ic"),
            "best_h_spy": best_spy.get("horizon_days"),
            "robust_spy": best_spy.get("robust"),
            "oos_pass_spy": gate_spy.get("pass"),
            "ic_6m_spy": gate_spy.get("ic_6m"),
            "ic_12m_spy": gate_spy.get("ic_12m"),
        }
        if returns_btc is not None:
            best_btc = find_optimal_horizon(sig, returns_btc, horizons)
            gate_btc = signal_passes_oos_gate(sig, returns_btc)
            row.update({
                "best_ic_btc": best_btc.get("ic"),
                "best_h_btc": best_btc.get("horizon_days"),
                "robust_btc": best_btc.get("robust"),
                "oos_pass_btc": gate_btc.get("pass"),
                "ic_6m_btc": gate_btc.get("ic_6m"),
                "ic_12m_btc": gate_btc.get("ic_12m"),
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by best |IC vs SPY|
    if "best_ic_spy" in df.columns:
        df["abs_ic_spy"] = df["best_ic_spy"].abs()
        df = df.sort_values("abs_ic_spy", ascending=False).drop(columns=["abs_ic_spy"])
    return df


def save_ic_table(df: pd.DataFrame) -> None:
    """Persist as JSON for downstream use as IC weights."""
    try:
        out = {row["signal"]: row for row in df.to_dict(orient="records")}
        IC_CACHE.write_text(json.dumps(out, indent=2, default=str))
    except Exception:
        pass


def load_ic_table() -> dict:
    if not IC_CACHE.exists(): return {}
    try: return json.loads(IC_CACHE.read_text())
    except Exception: return {}


def load_ic_weights(theme: Optional[str] = None) -> dict:
    """Return {signal_name: ic_weight}.

    Weight = max(0, IC_OOS) so negative-IC signals get zero weight.
    Use IC vs SPY by default. theme arg currently unused — reserved for
    theme-specific IC weights in future.
    """
    tbl = load_ic_table()
    if not tbl: return {}
    weights = {}
    for name, row in tbl.items():
        ic = row.get("ic_6m_spy") or row.get("best_ic_spy") or 0
        try: ic_v = float(ic) if ic else 0.0
        except Exception: ic_v = 0.0
        weights[name] = max(0.0, ic_v)
    return weights


def main():
    # Synthetic smoke test
    rng = np.random.default_rng(7)
    n = 1500
    idx = pd.date_range("2020-01-01", periods=n)
    # Real signal: leads the return by 60 days
    base = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=idx)
    leading = base.shift(-60) + rng.normal(0, 5, n)  # noisy leading
    returns = base.diff().fillna(0) / 100  # synthetic daily returns
    # Lagging signal: follows returns
    lagging = pd.Series(returns.cumsum() + rng.normal(0, 3, n), index=idx)
    # Noise
    noise = pd.Series(rng.normal(0, 1, n), index=idx)

    signals = {"leading": leading, "lagging": lagging, "noise": noise}
    df = build_ic_table(signals, returns)
    print("Smoke test IC table:")
    print(df.to_string())


if __name__ == "__main__":
    main()
