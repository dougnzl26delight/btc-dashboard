"""Portfolio combiner — aggregates strategy signals into final target weights
under regime gates and risk caps.

Conventions:
- Each strategy emits signal in [-1, 1] = direction * confidence (NOT allocation)
- Each strategy gets at most max_strategy_alloc of book; equal-weighted otherwise
- Regime gate zeros directionally-blocked signals and scales by vol regime
"""

from __future__ import annotations

from typing import Dict

from core import regime
from core.risk import DEFAULT, RiskCaps


def combine(
    strategy_signals: Dict[str, float],
    pair: str = "BTC/USDT",
    caps: RiskCaps = DEFAULT,
    apply_regime: bool = True,
) -> Dict[str, dict]:
    reg = regime.overall(pair) if apply_regime else {
        "scale": 1.0, "long_ok": True, "short_ok": True, "vol": {}, "trend": {},
    }

    n = max(len(strategy_signals), 1)
    alloc_per = min(caps.max_strategy_alloc, 1.0 / n)

    out: Dict[str, dict] = {}
    for name, raw in strategy_signals.items():
        sig = raw
        if sig > 0 and not reg["long_ok"]:
            sig = 0.0
        elif sig < 0 and not reg["short_ok"]:
            sig = 0.0

        adjusted = sig * reg["scale"]
        weight = max(-caps.max_strategy_alloc, min(caps.max_strategy_alloc, adjusted * alloc_per))

        out[name] = {
            "raw_signal": raw,
            "regime_adjusted": adjusted,
            "alloc_cap": alloc_per,
            "final_weight": weight,
        }

    total = sum(c["final_weight"] for c in out.values())
    out["__total__"] = {"final_weight": total, "regime": reg}
    return out
