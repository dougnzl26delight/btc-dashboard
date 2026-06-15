"""Top-level orchestrator. One trading cycle.

Pipeline:
  1. Collect signal from each strategy
  2. Combine via portfolio (applies regime + risk caps)
  3. Compute target qty for the pair
  4. Place corrective order if delta > min trade size
  5. Heartbeat the watchdog
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from core import attribution, correlation_monitor, evidence, execution, multi_exchange, portfolio, portfolio_var, regime
from core.broker import Broker
from core.data import ohlcv_extended, ticker
from ops import alerts, circuit_breaker, position_monitor, watchdog
from strategies import (
    diverse_mom_ethbtc,
    funding_basis,
    short_term_momentum,
    tsmom,
    vol_breakout,
)


# Multi-strategy ensemble. All VALIDATED=False; live trading still gated by
# regime + validation flags. In paper mode, shorts are simulated via negative
# positions (live deployment would need a perp broker for short execution).
STRATEGIES = [
    # v2 (2026-05-10) — same signal set as the v2 portfolio backtest, applied
    # with risk parity, concordance dampening, and portfolio vol targeting.
    # 729-day backtest with these settings: -1.7% return, -6.4% max DD,
    # -33.7% alpha vs BTC. Capital-preserving overlay, not an alpha generator.
    diverse_mom_ethbtc,    # BTC-specific composite
    tsmom,                 # TSMOM(60) any pair
    short_term_momentum,   # TSMOM(10) any pair
    vol_breakout,          # vol regime breakout any pair
    funding_basis,         # perp funding contrarian any pair
]
UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
]
MIN_TRADE_USDT = 50.0
# v2: long-only confirmed best in 729-day backtest. Crypto's structural
# bull bias means shorts drag returns even when "right" on individual moves.
LONG_ONLY = True

# v2 risk-parity / portfolio-vol parameters
TARGET_PAIR_VOL_ANN = 0.50         # target 50% ann vol per pair
MAX_PAIR_GROSS = 0.15              # hard cap per pair
TARGET_PORTFOLIO_VOL_ANN = 0.20    # target 20% ann portfolio vol
CONCORDANCE_DAMPEN_FLOOR = 0.30    # don't dampen below 30% even at 100% concordance


def _pair_realized_vol(pair: str, window: int = 30) -> float:
    """Annualized realized vol over `window` days."""
    try:
        df = ohlcv_extended(pair, days_back=window * 3)
        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(log_ret) < window:
            return TARGET_PAIR_VOL_ANN
        return float(log_ret.iloc[-window:].std() * np.sqrt(365))
    except Exception:
        return TARGET_PAIR_VOL_ANN


def _pair_cap(pair: str) -> float:
    """Inverse-vol per-pair gross cap (risk parity)."""
    rv = _pair_realized_vol(pair)
    if rv <= 0:
        return MAX_PAIR_GROSS
    return min(MAX_PAIR_GROSS, TARGET_PAIR_VOL_ANN / rv)


def _concordance_dampener(score: float) -> float:
    """Scale factor in [floor, 1.0]. Higher concordance → lower scale."""
    return max(CONCORDANCE_DAMPEN_FLOOR, 1.0 - score)


def _portfolio_vol_scale(targets: dict[str, float], vols: dict[str, float]) -> float:
    """Scale factor to bring portfolio gross vol to target."""
    if not targets:
        return 1.0
    weighted_vol = sum(abs(targets[p]) * vols.get(p, TARGET_PAIR_VOL_ANN) for p in targets)
    if weighted_vol <= TARGET_PORTFOLIO_VOL_ANN:
        return 1.0
    return TARGET_PORTFOLIO_VOL_ANN / weighted_vol


def cycle(mode: str = "paper") -> dict:
    """Multi-pair cycle with risk gates run BEFORE signal evaluation:
      1. Circuit breaker — kill all positions if portfolio dd > 20%
      2. Position monitor — close any position that hit stop/TP/trailing-stop
      3. Signal cycle — re-evaluate signals per pair and rebalance
    """
    # === Pre-cycle cross-venue divergence check ===
    # Refresh divergence state; halt cycle if Binance vs Kraken disagree > 1%.
    try:
        cross = multi_exchange.run_cross_check(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        if cross["halted"]:
            alerts.alert("Cycle SKIPPED — cross-venue divergence active", level="critical")
            watchdog.beat()
            return {"status": "multi_exchange_halt", "cross_check": cross}
    except Exception as e:
        alerts.alert(f"multi_exchange check failed (non-blocking): {e}", level="warning")

    cb = circuit_breaker.run_circuit_breaker(mode=mode, long_only=LONG_ONLY)
    if cb.get("status") in ("killed", "already_killed"):
        alerts.alert(
            f"Trading skipped — circuit breaker active ({cb.get('status')}). "
            f"Inspect .kill_switch.json and call ops.circuit_breaker.reset_kill_switch() "
            f"to resume.",
            level="warning",
        )
        watchdog.beat()
        return {"status": cb.get("status"), "circuit_breaker": cb}

    mon = position_monitor.run_monitor(mode=mode, long_only=LONG_ONLY)

    # 2026-05-28 W9: orchestrator owns its OWN sub-account. Can no longer
    # eat positions belonging to BAH/oversold/etc — they're isolated.
    broker = Broker(mode=mode, long_only=LONG_ONLY, sleeve="orchestrator")
    bal = broker.get_balance()
    cash = float(bal.get("USDT", 0.0))

    # Snapshot prices for the universe (cached at the broker level)
    prices: dict[str, float] = {}
    for pair in UNIVERSE:
        try:
            prices[pair] = float(ticker(pair)["last"])
        except Exception as e:
            alerts.alert(f"ticker fetch failed for {pair}: {e}", level="warning")

    # Mark-to-market equity across all open positions
    equity = cash
    for pair in UNIVERSE:
        base = pair.split("/")[0]
        held_qty = float(bal.get(base, 0.0))
        if held_qty != 0 and pair in prices:
            equity += held_qty * prices[pair]

    cycle_summary: dict[str, dict] = {}
    signals_per_pair: dict[str, dict[str, float]] = {}
    n_traded = 0
    n_blocked_liquidity = 0

    # === Pass 1: compute per-pair signals + raw targets (no portfolio scaling yet) ===
    pair_data: dict[str, dict] = {}
    for pair in UNIVERSE:
        if pair not in prices:
            continue
        price = prices[pair]
        base = pair.split("/")[0]
        held = float(bal.get(base, 0.0))

        signals: dict[str, float] = {}
        for s in STRATEGIES:
            try:
                signals[s.NAME] = float(s.latest_signal(pair))
            except Exception:
                signals[s.NAME] = 0.0

        concordance = correlation_monitor.check_concordance(signals, pair=pair)
        signals_per_pair[pair] = signals

        combined = portfolio.combine(signals, pair=pair)
        raw_signal = combined["__total__"]["final_weight"]

        # === v2 fix #1: concordance dampener ===
        dampen = _concordance_dampener(concordance.get("score", 0.0))
        dampened_signal = raw_signal * dampen

        # === v2 fix #2: long-only clamp ===
        if LONG_ONLY:
            dampened_signal = max(0.0, dampened_signal)

        # === v2 fix #3: inverse-vol per-pair cap (risk parity) ===
        pair_cap = _pair_cap(pair)
        target_weight = max(-pair_cap, min(pair_cap, dampened_signal))

        pair_data[pair] = {
            "price": price,
            "held": held,
            "base": base,
            "signals": signals,
            "concordance": concordance,
            "raw_signal": raw_signal,
            "dampened_signal": dampened_signal,
            "pair_cap": pair_cap,
            "target_weight": target_weight,
            "regime": combined["__total__"]["regime"],
        }

    # === Pass 2: apply portfolio vol target ===
    pair_vols = {p: _pair_realized_vol(p) for p in pair_data}
    targets_only = {p: pd["target_weight"] for p, pd in pair_data.items()}
    portfolio_scale = _portfolio_vol_scale(targets_only, pair_vols)

    # === Pass 3: trade execution per pair with scaled targets ===
    for pair, pd in pair_data.items():
        price = pd["price"]
        held = pd["held"]
        target_weight = pd["target_weight"] * portfolio_scale

        target_qty = target_weight * equity / price
        delta_qty = target_qty - held
        delta_value = delta_qty * price

        traded = None
        liquidity_check = None
        var_check = None
        if abs(delta_value) > MIN_TRADE_USDT:
            side = "buy" if delta_value > 0 else "sell"
            # Determine if this trade is REDUCING risk (closing toward 0).
            # Exit orders always bypass liquidity AND VaR check — risk reduction
            # is always allowed regardless of current exposure.
            is_reducing = (held != 0 and abs(target_qty) < abs(held)) or (target_qty == 0 and held != 0)
            liquidity_check = execution.check_liquidity(pair, side, abs(delta_value))

            # Pre-trade VaR check — block new exposure that exceeds daily VaR limit
            if not is_reducing:
                var_check = portfolio_var.gate_new_trade(abs(delta_value), pair)
                if not var_check["allowed"]:
                    alerts.alert(
                        f"{pair} trade BLOCKED by VaR: {var_check['reason']}",
                        level="warning",
                    )
            allow = (liquidity_check["ok"] or is_reducing) and (
                is_reducing or (var_check is None) or var_check["allowed"]
            )
            if not allow:
                n_blocked_liquidity += 1
                alerts.alert(
                    f"{pair} trade BLOCKED by liquidity: {liquidity_check['reason']}",
                    level="warning",
                )
            else:
                try:
                    traded = broker.place_market_order(pair, side, abs(delta_value))
                    n_traded += 1
                    extra = " [EXIT-OVERRIDE]" if (is_reducing and not liquidity_check["ok"]) else ""
                    alerts.alert(
                        f"{pair} traded ${delta_value:+,.0f} (w={target_weight:+.3f}, "
                        f"spread={liquidity_check['spread_bps']:.1f}bps){extra}",
                        level="trade",
                    )
                    bal = broker.get_balance()
                except Exception as e:
                    alerts.alert(f"{pair} trade failed: {e}", level="warning")

        cycle_summary[pair] = {
            "price": price,
            "held": held,
            "target_weight": target_weight,
            "raw_signal": pd["raw_signal"],
            "dampened_signal": pd["dampened_signal"],
            "pair_cap": pd["pair_cap"],
            "portfolio_scale": portfolio_scale,
            "delta_value": delta_value,
            "traded": bool(traded),
            "blocked_liquidity": liquidity_check is not None and not liquidity_check["ok"],
            "regime": pd["regime"],
            "concordance": pd["concordance"],
            "signals": {k: round(v, 4) for k, v in pd["signals"].items()},
        }

    # Snapshot for per-strategy attribution analysis
    attribution.snapshot_signals(signals_per_pair, prices, equity)

    watchdog.beat()

    evidence.record(
        "orchestrator",
        f"multi-pair cycle n_pairs={len(UNIVERSE)} traded={n_traded}",
        {
            "equity": equity,
            "n_pairs": len(UNIVERSE),
            "n_traded": n_traded,
            "summary": cycle_summary,
        },
    )

    return {
        "equity": equity,
        "n_pairs": len(UNIVERSE),
        "n_traded": n_traded,
        "n_blocked_liquidity": n_blocked_liquidity,
        "summary": cycle_summary,
        "circuit_breaker": cb,
        "monitor": mon,
    }


if __name__ == "__main__":
    import json
    result = cycle()
    print(f"Equity: ${result['equity']:,.2f}")
    print(f"Pairs evaluated: {result['n_pairs']}, trades fired: {result['n_traded']}")
    print()
    for pair, info in result["summary"].items():
        marker = "**" if info["traded"] else "  "
        print(
            f"{marker} {pair:10s}  w={info['target_weight']:+.3f}  "
            f"price=${info['price']:>11,.2f}  delta=${info['delta_value']:+10,.0f}  "
            f"regime={info['regime']['trend']['regime']}/{info['regime']['vol']['regime']}"
        )
