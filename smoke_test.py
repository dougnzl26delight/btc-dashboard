"""Phase 1-6 smoke test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from core import data, risk, deflated_sharpe, backtest, evidence, ic_tracker, regime, portfolio
from ops import alerts, daily_log, watchdog
from signals.tsmom import tsmom_signal
from strategies import tsmom, funding_basis


def main():
    print("=== Phase 1: backbone ===")
    df = data.ohlcv("BTC/USDT", timeframe="1d", limit=300)
    print(f"  ohlcv rows={len(df)} last_close={df['close'].iloc[-1]:.2f}")
    print(f"  position_size 10% of 100k @ {df['close'].iloc[-1]:.0f} = {risk.position_size(100000, 0.10, df['close'].iloc[-1]):.6f} BTC")

    sig = tsmom_signal(df["close"], lookback_days=60)
    bt = backtest.run(df["close"], sig)
    summary = backtest.summarize(bt)
    print(f"  TSMOM bt: sharpe={summary['sharpe']:.2f} total={summary['total_return']:.2%} dd={summary['max_drawdown']:.2%}")

    hurdle = deflated_sharpe.passes_quant_hurdle(bt["ret"].dropna(), num_trials=20)
    print(f"  hurdle: dsr={hurdle['dsr']:.3f} t={hurdle['t_stat']:.2f} passes={hurdle['passes_combined']}")

    fwd_ret = df["close"].pct_change().shift(-1)
    ic = ic_tracker.rolling_ic(sig, fwd_ret, window=60)
    if not ic.empty:
        print(f"  ic: rolling 60d mean={ic.mean():.3f} latest={ic.iloc[-1]:.3f}")

    evidence.record("smoke", "phase 1-3 wired", {"sharpe": summary['sharpe'], "dsr": hurdle['dsr']})

    print("\n=== Phase 5: regime ===")
    reg = regime.overall("BTC/USDT")
    print(f"  vol regime: {reg['vol']['regime']} (rv={reg['vol']['realized_vol']:.2%}, scale={reg['vol']['scale']})")
    print(f"  trend: {reg['trend']['regime']} (price/sma={reg['trend']['price_vs_sma']:.3f})")
    print(f"  long_ok={reg['long_ok']} short_ok={reg['short_ok']}")

    print("\n=== Phase 2-5 strategies + portfolio combiner ===")
    s_tsmom = tsmom.latest_signal("BTC/USDT")
    print(f"  TSMOM latest: {s_tsmom:+.4f}")
    try:
        s_funding = funding_basis.latest_signal("BTC/USDT", "BTC/USDT:USDT")
        print(f"  funding_basis latest: {s_funding:+.4f}")
    except Exception as e:
        s_funding = 0.0
        print(f"  funding_basis: skipped ({type(e).__name__}: {e})")

    combined = portfolio.combine({"tsmom": s_tsmom, "funding_basis": s_funding}, pair="BTC/USDT")
    print(f"  combined total weight: {combined['__total__']['final_weight']:+.4f}")
    for name, c in combined.items():
        if name == "__total__":
            continue
        print(f"    {name}: raw={c['raw_signal']:+.3f} adjusted={c['regime_adjusted']:+.3f} final={c['final_weight']:+.3f}")

    print("\n=== Phase 6: ops ===")
    print(f"  alerts: {alerts.alert_status()}")
    watchdog.beat()
    print(f"  watchdog: {watchdog.check()}")

    print("\nALL OK")


if __name__ == "__main__":
    main()
