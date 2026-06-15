"""Systematic parameter sweep across indicator families.

Discipline notes:
- num_trials in DSR equals the candidate count in THIS sweep
- All results reported (winners and losers)
- Validation requires passing all three gates (walk-forward + factor + DSR/t)
- Even if a strategy passes here, that does NOT make it tradeable — it only
  makes it a candidate for the next stage of the funnel (more data, more OOS,
  paper trading, then live with small allocation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import backtest, cv, data, deflated_sharpe, factor_decomp
from research import signals as sig
from research.etf_flows import etf_flow_signal, fetch_etf_flows
from signals.funding_basis import funding_signal_series


def make_candidates(eth_btc_ratio: pd.Series | None = None) -> list[dict]:
    cands: list[dict] = []

    for lb in (30, 60, 90, 120, 180, 365):
        cands.append(
            {"name": f"tsmom_single_{lb}", "fn": (lambda lb=lb: lambda p: sig.tsmom_single(p, lookback=lb))()}
        )

    for hs in [(30, 90), (30, 90, 180), (60, 180, 365), (30, 60, 90, 180)]:
        name = "tsmom_multi_" + "_".join(map(str, hs))
        cands.append({"name": name, "fn": (lambda hs=hs: lambda p: sig.tsmom_multi(p, horizons=hs))()})

    # ETH/BTC ratio signals — relative-strength play, uncorrelated with absolute price moves
    if eth_btc_ratio is not None and not eth_btc_ratio.empty:
        for lb in (30, 60, 90):
            def eb_tsmom_fn(p, lb=lb, ratio=eth_btc_ratio):
                ratio_aligned = ratio.reindex(p.index).ffill().bfill()
                return sig.tsmom_single(ratio_aligned, lookback=lb)
            cands.append({"name": f"ethbtc_ratio_tsmom_{lb}", "fn": eb_tsmom_fn})
        for w in (20, 60):
            def eb_zscore_fn(p, w=w, ratio=eth_btc_ratio):
                ratio_aligned = ratio.reindex(p.index).ffill().bfill()
                return sig.zscore_revert(ratio_aligned, window=w)
            cands.append({"name": f"ethbtc_ratio_zrevert_{w}", "fn": eb_zscore_fn})

    for f, s in [(10, 30), (20, 50), (50, 200), (10, 50)]:
        cands.append(
            {"name": f"ma_cross_{f}_{s}", "fn": (lambda f=f, s=s: lambda p: sig.ma_crossover(p, fast=f, slow=s))()}
        )

    for w, ns in [(20, 2.0), (20, 2.5), (50, 2.0)]:
        cands.append(
            {"name": f"boll_revert_{w}_{ns}", "fn": (lambda w=w, ns=ns: lambda p: sig.bollinger_revert(p, window=w, n_std=ns))()}
        )

    for w, lo, hi in [(14, 30, 70), (14, 25, 75), (7, 30, 70)]:
        cands.append(
            {"name": f"rsi_{w}_{lo}_{hi}", "fn": (lambda w=w, lo=lo, hi=hi: lambda p: sig.rsi_revert(p, window=w, low=lo, high=hi))()}
        )

    for w in (20, 50, 100):
        cands.append({"name": f"donchian_{w}", "fn": (lambda w=w: lambda p: sig.donchian_breakout(p, window=w))()})

    for w in (20, 60):
        cands.append({"name": f"zscore_revert_{w}", "fn": (lambda w=w: lambda p: sig.zscore_revert(p, window=w))()})

    cands.append({"name": "vol_breakout_30", "fn": lambda p: sig.vol_breakout(p, window=30)})

    # Funding-basis: signal independent of prices, aligned to price index
    fund = funding_signal_series(perp_pair="BTC/USDT:USDT", days_back=1000)
    if not fund.empty:
        def funding_fn(p):
            return fund.reindex(p.index, method="ffill").fillna(0)
        cands.append({"name": "funding_basis", "fn": funding_fn})

    # ETF flow signals (Farside scrape, 2024-01 onward)
    flows = fetch_etf_flows()
    if not flows.empty:
        for ema_w in (3, 7, 21):
            etf_sig = etf_flow_signal(flows, ema_window=ema_w)
            def etf_fn(p, s=etf_sig):
                return s.reindex(p.index, method="ffill").fillna(0)
            cands.append({"name": f"etf_flows_ema{ema_w}", "fn": etf_fn})

    return cands


def evaluate_candidate(
    cand: dict,
    prices: pd.Series,
    bench_returns: pd.Series,
    num_trials: int,
) -> dict:
    try:
        wf = cv.walk_forward(prices, signal_fn=cand["fn"], n_folds=5, min_train=365)
    except Exception as e:
        return {"name": cand["name"], "error": str(e), "validated": False}

    sig_full = cand["fn"](prices)
    bt = backtest.run(prices, sig_full)
    decomp = factor_decomp.decompose(bt["ret"], bench_returns)

    if wf.get("n_folds", 0) > 0 and len(wf.get("concatenated_returns", [])) >= 30:
        hurdle = deflated_sharpe.passes_quant_hurdle(
            wf["concatenated_returns"], num_trials=num_trials
        )
    else:
        hurdle = {"passes_combined": False, "dsr": 0.0, "t_stat": 0.0}

    return {
        "name": cand["name"],
        "mean_sharpe_oos": round(wf.get("mean_sharpe_oos", 0.0), 3),
        "min_sharpe_oos": round(wf.get("min_sharpe_oos", 0.0), 3),
        "std_sharpe_oos": round(wf.get("std_sharpe_oos", 0.0), 3),
        "wf_passes": wf.get("passes", False),
        "alpha_ann": round(decomp.get("alpha_annualized", 0.0), 4),
        "alpha_t": round(decomp.get("alpha_t", 0.0), 2),
        "beta": round(decomp.get("beta", 0.0), 3),
        "alpha_pass": decomp.get("passes_alpha_t", False),
        "dsr": round(hurdle.get("dsr", 0.0), 3),
        "hurdle_t": round(hurdle.get("t_stat", 0.0), 2),
        "hurdle_pass": hurdle.get("passes_combined", False),
        "validated": (
            wf.get("passes", False)
            and decomp.get("passes_alpha_t", False)
            and hurdle.get("passes_combined", False)
        ),
    }


def run_sweep(pair: str = "BTC/USDT", days_back: int = 2000) -> pd.DataFrame:
    df = data.ohlcv_extended(pair, days_back=days_back)
    bench = df["close"].pct_change().fillna(0)

    # ETH/BTC ratio for relative-strength signals
    try:
        eth = data.ohlcv_extended("ETH/USDT", days_back=days_back)["close"]
        btc = df["close"]
        eth_btc = (eth / btc).dropna()
    except Exception:
        eth_btc = None

    candidates = make_candidates(eth_btc_ratio=eth_btc)
    num_trials = len(candidates)
    print(f"Sweeping {num_trials} candidates on {pair} ({len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()})\n")

    rows: list[dict] = []
    for i, c in enumerate(candidates, 1):
        r = evaluate_candidate(c, df["close"], bench, num_trials=num_trials)
        rows.append(r)
        marker = "VALID" if r.get("validated") else ""
        print(
            f"  [{i:2d}/{num_trials}] {r['name']:30s} "
            f"OOS={r.get('mean_sharpe_oos',0):+.2f} (min={r.get('min_sharpe_oos',0):+.2f}) "
            f"alpha_t={r.get('alpha_t',0):+.2f} dsr={r.get('dsr',0):.2f} {marker}"
        )

    out = pd.DataFrame(rows).sort_values("mean_sharpe_oos", ascending=False)
    out_path = Path(__file__).resolve().parent.parent / "research_sweep_results.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    df = run_sweep()
    print("\n" + "=" * 60)
    print("RANKED BY MEAN OOS SHARPE")
    print("=" * 60)
    cols = ["name", "mean_sharpe_oos", "min_sharpe_oos", "alpha_t", "beta", "dsr", "hurdle_t", "validated"]
    print(df[cols].to_string(index=False))
    print(f"\nValidated: {df['validated'].sum()}/{len(df)}")
