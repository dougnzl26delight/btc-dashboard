"""Daily BTC + alt exit-signal monitor.

Tracks the backtested exit rule:
    Signal: MACD bear cross + close<EMA21 within 5 days
    Backtest score 1.51, capture 66.6%, avoided 126% drawdown (n=5).

Output state per pair:
    PEAK ZONE  - all conditions aligned, signal firing now (NEW FIRE)
    ARMED      - RSI >= 70 (overbought) — watch for confirm
    EARLY      - signal fired recently but possible false (regime watch)
    WAITING    - no exit conditions met yet
    STALE      - past fire, asset resumed uptrend (false signal)

Saves state to btc_exit_signal_state.json so we detect NEW fires across runs.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from core import data


STATE_FILE = Path(__file__).resolve().parent / "btc_exit_signal_state.json"

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "BNB/USDT",
    "DOT/USDT", "ATOM/USDT",
    # Lower-cap additions
    "TAO/USDT", "ONDO/USDT",
]

FEB_LOW_START = "2026-02-01"


def calc_indicators(df):
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    c = df["close"]
    df["sma_20"] = c.rolling(20).mean()
    df["sma_200"] = c.rolling(200).mean()
    df["ema_21"] = c.ewm(span=21).mean()
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9).mean()
    df["macd_hist"] = macd - macd_sig
    bb_std = c.rolling(20).std()
    df["bb_upper"] = df["sma_20"] + 2 * bb_std
    df["bb_lower"] = df["sma_20"] - 2 * bb_std
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["mayer"] = c / df["sma_200"]
    return df


def signal_state(df):
    """Classify the current state and most recent fire date."""
    macd_bear = (df["macd_hist"].shift(1) > 0) & (df["macd_hist"] < 0)
    ema_break = (df["close"].shift(1) > df["ema_21"].shift(1)) & (df["close"] < df["ema_21"])
    sig = macd_bear.rolling(5).max().astype(bool) & ema_break

    last = df.iloc[-1]
    today = df.index[-1]
    fires = sig[sig.fillna(False)]
    last_fire = fires.index[-1] if len(fires) > 0 else None
    days_since = (today - last_fire).days if last_fire is not None else None

    rsi = last["rsi"]
    mayer = last["mayer"]
    bb_pct = last["bb_pct"]
    macd_h = last["macd_hist"]
    above_ema = last["close"] > last["ema_21"]
    ema21_price = float(last["ema_21"])
    dist_from_ema21 = float(last["close"] / last["ema_21"] - 1)

    # Trailing-stop alert level
    if not above_ema:
        stop_alert = "BROKEN"
    elif dist_from_ema21 <= 0.02:
        stop_alert = "NEAR"
    elif dist_from_ema21 <= 0.04:
        stop_alert = "WATCH"
    else:
        stop_alert = "OK"

    # Was RSI > 70 in the last 10 days?
    rsi_hot_recent = (df["rsi"].iloc[-10:] > 70).any()

    # Priority order: current readings dominate stale signal history
    if rsi >= 80:
        state = "EXTREME OB"
    elif days_since is not None and days_since <= 3 and rsi_hot_recent:
        state = "PEAK ZONE"
    elif rsi >= 70 and bb_pct >= 0.95:
        state = "ARMED"
    elif days_since is not None and 3 < days_since <= 14:
        state = "FIRED" if rsi_hot_recent else "EARLY"
    elif days_since is not None and days_since > 30 and above_ema:
        state = "STALE"
    elif days_since is not None:
        state = f"{days_since}d ago"
    else:
        state = "WAITING"

    return {
        "state": state,
        "last_fire": str(last_fire.date()) if last_fire is not None else None,
        "days_since": days_since,
        "rsi": float(rsi),
        "mayer": float(mayer),
        "bb_pct": float(bb_pct),
        "macd_hist": float(macd_h),
        "above_ema": bool(above_ema),
        "rsi_hot_recent": bool(rsi_hot_recent),
        "price": float(last["close"]),
        "ema21_price": ema21_price,
        "dist_from_ema21": dist_from_ema21,
        "stop_alert": stop_alert,
    }


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def compute_status(pairs=None):
    """Compute exit-signal status for each pair. Returns dict[pair] -> state dict.

    Adds 'from_feb_low' to each state dict. Returns empty dict if no data.
    """
    if pairs is None:
        pairs = PAIRS
    out = {}
    for pair in pairs:
        try:
            df = data.ohlcv_extended(pair, days_back=400)
        except Exception:
            continue
        if df.empty:
            continue
        df = calc_indicators(df)
        try:
            feb_low = float(df.loc[df.index >= pd.Timestamp(FEB_LOW_START), "low"].min())
            from_low = df["close"].iloc[-1] / feb_low - 1
        except Exception:
            from_low = None
        s = signal_state(df)
        s["from_feb_low"] = from_low
        s["pair"] = pair
        out[pair] = s
    return out


def detect_alerts(current, prior):
    """Compare current vs prior state, return list of alert strings."""
    msgs = []
    for pair, s in current.items():
        p = prior.get(pair, {})
        prior_fire = p.get("last_fire")
        prior_state_label = p.get("state")
        prior_stop_alert = p.get("stop_alert")

        new_fire = (s.get("last_fire") != prior_fire
                    and s.get("days_since") is not None
                    and s["days_since"] <= 5)
        state_changed = (s["state"] != prior_state_label and prior_state_label is not None)

        if new_fire:
            msgs.append(f"NEW SIGNAL FIRE: {pair} @ ${s['price']:,.4f} on {s['last_fire']}")
        elif state_changed and s["state"] in ("EXTREME OB", "PEAK ZONE", "ARMED", "FIRED"):
            msgs.append(f"STATE CHANGE: {pair} {prior_state_label} -> {s['state']}")

        if s["stop_alert"] == "BROKEN" and prior_stop_alert != "BROKEN":
            msgs.append(f"EMA21 BROKEN: {pair} @ ${s['price']:,.4f} "
                        f"(EMA21=${s['ema21_price']:,.4f}) — EXIT TRIGGER")
        elif s["stop_alert"] == "NEAR" and prior_stop_alert not in ("NEAR", "BROKEN"):
            msgs.append(f"NEAR EMA21: {pair} only {s['dist_from_ema21']*100:.1f}% above stop "
                        f"(${s['ema21_price']:,.4f}) — set sell orders")
        elif s["stop_alert"] == "WATCH" and prior_stop_alert == "OK":
            msgs.append(f"WATCH: {pair} now {s['dist_from_ema21']*100:.1f}% above EMA21 "
                        f"(${s['ema21_price']:,.4f})")
    return msgs


def main():
    print("=" * 100)
    print(f"BTC + ALT EXIT-SIGNAL MONITOR  ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})")
    print("=" * 100)
    print("Signal: MACD bear cross + close<EMA21 w/in 5d  |  Backtest n=5, capture 67%, avoided 126%")
    print()

    prior_state = load_state()
    current_state = {}
    alerts = []

    print(f"{'Pair':<10s} {'Price':>11s} {'FromLow':>8s} {'RSI':>4s} {'Mayer':>6s} {'BB%':>5s} "
          f"{'EMA21($)':>10s} {'Dist':>6s} {'Stop':>7s} {'LastFire':<12s} {'STATE':<12s}")
    print("-" * 110)

    for pair in PAIRS:
        try:
            df = data.ohlcv_extended(pair, days_back=400)
        except Exception as e:
            print(f"   {pair}: data fetch failed ({e})")
            continue
        if df.empty:
            continue
        df = calc_indicators(df)

        feb_low = float(df.loc[df.index >= pd.Timestamp(FEB_LOW_START), "low"].min())
        from_low = df["close"].iloc[-1] / feb_low - 1

        s = signal_state(df)
        current_state[pair] = s

        prior = prior_state.get(pair, {})
        prior_fire = prior.get("last_fire")
        prior_state_label = prior.get("state")

        new_fire = (s["last_fire"] != prior_fire and s["days_since"] is not None and s["days_since"] <= 5)
        state_changed = s["state"] != prior_state_label and prior_state_label is not None

        prior_stop_alert = prior.get("stop_alert")
        stop_changed = s["stop_alert"] != prior_stop_alert and prior_stop_alert is not None

        if new_fire:
            alerts.append(f"*** NEW SIGNAL FIRE: {pair} @ ${s['price']:,.4f} on {s['last_fire']}")
        elif state_changed and s["state"] in ("EXTREME OB", "PEAK ZONE", "ARMED", "FIRED"):
            alerts.append(f"*** STATE CHANGE: {pair} {prior_state_label} -> {s['state']}")

        # Trailing-stop alerts
        if s["stop_alert"] == "BROKEN" and prior_stop_alert != "BROKEN":
            alerts.append(f"!!! EMA21 BROKEN: {pair} @ ${s['price']:,.4f} (EMA21 = ${s['ema21_price']:,.4f}) — EXIT TRIGGER")
        elif s["stop_alert"] == "NEAR" and prior_stop_alert not in ("NEAR", "BROKEN"):
            alerts.append(f"!!  NEAR EMA21: {pair} only {s['dist_from_ema21']*100:.1f}% above stop ({s['ema21_price']:,.4f}) — set sell orders")
        elif stop_changed and s["stop_alert"] == "WATCH":
            alerts.append(f"!   WATCH: {pair} now {s['dist_from_ema21']*100:.1f}% above EMA21 ({s['ema21_price']:,.4f})")

        price_fmt = f"${s['price']:>9,.4f}" if s['price'] < 100 else f"${s['price']:>9,.2f}"
        ema_fmt = f"${s['ema21_price']:>8,.4f}" if s['ema21_price'] < 100 else f"${s['ema21_price']:>8,.2f}"
        dist_str = f"{s['dist_from_ema21']*100:+.1f}%"
        last_fire_str = s["last_fire"] if s["last_fire"] else "never"
        flag = "***" if new_fire else ("!! " if s["state"] in ("EXTREME OB", "PEAK ZONE", "ARMED", "FIRED") or s["stop_alert"] in ("BROKEN", "NEAR") else "   ")

        print(f"{flag}{pair:<7s} {price_fmt:>11s} {from_low:>+7.1%} {s['rsi']:>4.0f} "
              f"{s['mayer']:>6.2f} {s['bb_pct']:>4.0%} {ema_fmt:>10s} {dist_str:>6s} "
              f"{s['stop_alert']:>7s} {last_fire_str:<12s} {s['state']:<12s}")

    print()
    print("=" * 110)
    print("TRAILING-STOP SUMMARY — sorted by distance to EMA21 (smallest = sell first)")
    print("=" * 110)
    print(f"{'Pair':<10s} {'Price':>12s} {'EMA21':>11s} {'Distance':>10s} {'Drop to stop':>13s} {'Status':<10s}")
    print("-" * 80)
    sorted_pairs = sorted(
        ((p, st) for p, st in current_state.items()),
        key=lambda x: x[1]["dist_from_ema21"] if x[1] else 999
    )
    for p, st in sorted_pairs:
        if st is None:
            continue
        price_fmt = f"${st['price']:,.4f}" if st['price'] < 100 else f"${st['price']:,.2f}"
        ema_fmt = f"${st['ema21_price']:,.4f}" if st['ema21_price'] < 100 else f"${st['ema21_price']:,.2f}"
        drop_to_stop = (st["ema21_price"] / st["price"] - 1) * 100 if st["above_ema"] else 0
        print(f"{p:<10s} {price_fmt:>12s} {ema_fmt:>11s} {st['dist_from_ema21']*100:>+9.1f}%  "
              f"{drop_to_stop:>+11.1f}%  {st['stop_alert']:<10s}")
    print()
    print("Status legend:")
    print("  OK     - more than 4% above EMA21, plenty of room")
    print("  WATCH  - 2-4% above EMA21, getting close")
    print("  NEAR   - within 2% of EMA21 — sell orders should be live")
    print("  BROKEN - already crossed below EMA21 — EXIT signal active")
    print()
    print("=" * 100)
    print("BTC-SPECIFIC PEAK TARGETS")
    print("=" * 100)
    btc_s = current_state.get("BTC/USDT")
    if btc_s:
        feb_low = 62910
        btc_price = btc_s["price"]
        cur_from_low = btc_price / feb_low - 1
        print(f"BTC current: ${btc_price:,.2f}  (+{cur_from_low*100:.1f}% from Feb low ${feb_low:,})")
        print()
        print("Historical relief-peak targets (mirroring prior bears):")
        print(f"  2018 BTC pattern (+54%):  peak ${feb_low * 1.54:,.0f}  signal at ${feb_low * 1.54 * 0.91:,.0f}")
        print(f"  2022 BTC pattern (+43%):  peak ${feb_low * 1.43:,.0f}  signal at ${feb_low * 1.43 * 0.91:,.0f}")
        print(f"  Average:                  peak ${feb_low * 1.485:,.0f}  signal at ${feb_low * 1.485 * 0.91:,.0f}")
        print()
        print(f"Distance to historical-average signal fire price:  "
              f"+{(feb_low * 1.485 * 0.91 / btc_price - 1) * 100:.1f}%")
        print(f"Distance to historical-average relief peak:        "
              f"+{(feb_low * 1.485 / btc_price - 1) * 100:.1f}%")
        print()
        print("Conditions needed for BTC clean signal fire:")
        conditions = [
            ("RSI(14) >= 70", btc_s["rsi"] >= 70, f"now {btc_s['rsi']:.0f}"),
            ("Mayer Mult >= 1.05", btc_s["mayer"] >= 1.05, f"now {btc_s['mayer']:.2f}"),
            ("BB% >= 0.95", btc_s["bb_pct"] >= 0.95, f"now {btc_s['bb_pct']*100:.0f}%"),
            ("MACD hist NEGATIVE", btc_s["macd_hist"] < 0, f"now {btc_s['macd_hist']:+.2f}"),
            ("Close BELOW EMA21", not btc_s["above_ema"], f"now {'above' if btc_s['above_ema'] else 'BELOW'}"),
        ]
        for label, met, current in conditions:
            mark = "[YES]" if met else "[no] "
            print(f"  {mark}  {label:<25s}  ({current})")
        n_met = sum(1 for _, met, _ in conditions if met)
        print()
        print(f"Conditions aligned: {n_met}/5.  Sell trigger requires 4+/5 (with MACD bear + EMA21 break required).")

    print()
    if alerts:
        print("=" * 100)
        print("!!! ALERTS !!!")
        print("=" * 100)
        for a in alerts:
            print(a)
    else:
        print("No new alerts since last run.")

    save_state(current_state)
    print()
    print(f"State saved to {STATE_FILE.name}")
    print("Run daily.  To schedule: add to your daily cron in core/ops_orchestrator.py")


if __name__ == "__main__":
    main()
