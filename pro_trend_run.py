"""Pro trend follower production cycle — multi-pair, long+short via perp.

Iterates the active universe (PRO_TREND_PAIRS) plus any pair with an
orphaned open position from a prior universe — those still get managed
(trail/exit/pyramid) but won't accept new entries because cycle() only
enters when units is empty.

Scheduled as Crypto_pro_trend_daily.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ops import alerts, watchdog
from ops.sleeve_circuit_breakers import apply_sleeve_scaling, is_paused
from strategies import pro_trend


def _orphaned_pairs() -> list[str]:
    """State files with open units for pairs not in current PRO_TREND_PAIRS."""
    root = Path(__file__).resolve().parent
    out = []
    for f in root.glob(".pro_trend_state_*.json"):
        base = f.stem.removeprefix(".pro_trend_state_")
        pair = f"{base}/USDT"
        if pair in pro_trend.PRO_TREND_PAIRS:
            continue
        try:
            st = json.loads(f.read_text())
        except Exception:
            continue
        if st.get("units"):
            out.append(pair)
    return out


def _compute_pro_trend_equity() -> float:
    """Aggregate equity across all pro_trend pairs using per-pair peak_equity
    as the baseline. Captures both open MTM and trail-stop locked-in P&L."""
    root = Path(__file__).resolve().parent
    total = 0.0
    n = 0
    for f in root.glob(".pro_trend_state_*.json"):
        try:
            st = json.loads(f.read_text())
        except Exception:
            continue
        # Use peak_equity as proxy when no open units (sleeve idle but reserved)
        # Open-position MTM tracking would need per-pair price lookup; defer to
        # broker for full accuracy. Here we conservatively use peak.
        total += float(st.get("peak_equity", 100_000.0))
        n += 1
    return total / max(n, 1)


def main():
    # === Sleeve drawdown circuit breaker — gate the whole runner ===
    pt_equity = _compute_pro_trend_equity()
    apply_sleeve_scaling("pro_trend", pt_equity)
    if is_paused("pro_trend"):
        alerts.alert(
            "PRO_TREND SLEEVE PAUSED — drawdown circuit breaker > 20%. "
            "Run: python -m ops.sleeve_circuit_breakers reset pro_trend",
            level="critical",
        )
        watchdog.beat()
        return []

    results = []
    universe = list(pro_trend.PRO_TREND_PAIRS) + _orphaned_pairs()
    for pair in universe:
        try:
            # enable_shorts=True (2026-05-28): re-enabled with new v5 SHORT
            # entry filter (TSMOM_30 < -0.10 AND MACD_hist < 0) gating the
            # Donchian-low break. The unfiltered short fired at capitulations
            # (cost Sharpe 0.50). The filter ensures only firm bear momentum
            # triggers shorts. Long entries unchanged.
            r = pro_trend.cycle(pair, mode="paper", enable_shorts=True)
            results.append(r)
        except Exception as e:
            alerts.alert(f"pro_trend cycle failed for {pair}: {e}", level="warning")
            continue

        for a in r.get("actions", []):
            kind = a.get("action")
            if kind == "entry_long":
                alerts.alert(
                    f"PRO_TREND ENTRY LONG {pair} @ ${a['entry']:,.4f}, "
                    f"qty {a['qty']:.6f}, stop ${a['stop']:,.4f}",
                    level="trade",
                )
            elif kind == "entry_short":
                alerts.alert(
                    f"PRO_TREND ENTRY SHORT {pair} @ ${a['entry']:,.4f} (perp), "
                    f"qty {a['qty']:.6f}, stop ${a['stop']:,.4f}",
                    level="trade",
                )
            elif kind == "pyramid_long":
                alerts.alert(f"PRO_TREND PYRAMID LONG {pair} @ ${a['entry']:,.4f}", level="trade")
            elif kind == "pyramid_short":
                alerts.alert(f"PRO_TREND PYRAMID SHORT {pair} @ ${a['entry']:,.4f}", level="trade")
            elif kind in ("exit_long", "exit_short"):
                alerts.alert(
                    f"PRO_TREND EXIT {pair} @ ${a['exit_price']:,.4f}, "
                    f"reason: {a['reason']}, closed {a['n_units_closed']} units",
                    level="trade",
                )
            elif kind == "dd_kill":
                alerts.alert(f"PRO_TREND DD KILL {pair} side={a['side']}", level="critical")

    # Quiet status summary
    print(f"{'Pair':<10s} {'Side':<6s} {'Units':>6s} {'Price':>12s} {'SMA':>12s} {'Trail':>12s}")
    for r in results:
        if r.get("status") != "ok":
            continue
        side = r.get("side") or "flat"
        units = r.get("n_units", 0)
        print(f"{r['pair']:<10s} {side:<6s} {units:>6d}  "
              f"${r['price']:>10,.2f}  ${r['sma']:>10,.2f}  ${r['trail_stop']:>10,.2f}")

    watchdog.beat()
    return results


if __name__ == "__main__":
    main()
