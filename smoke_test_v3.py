"""Smoke test for the five new practitioner-technique modules:
  1. core/cpcv.py       — Combinatorial Purged CV (López de Prado AFML 7.4)
  2. core/hrp.py        — Hierarchical Risk Parity (López de Prado 2016)
  3. core/meta_labeling.py — Meta-labeling (López de Prado AFML 3.6)
  4. core/garch_vol.py  — GARCH(1,1) (Engle 1982, Bollerslev 1986)
  5. core/shrinkage.py  — Ledoit-Wolf shrinkage cov (Ledoit/Wolf 2003)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from core import cpcv, data, garch_vol, hrp, meta_labeling, shrinkage
from research import signals as res_sig


def main():
    print("=" * 60)
    print("FETCHING DATA")
    print("=" * 60)
    btc = data.ohlcv_extended("BTC/USDT", days_back=2000)["close"]
    eth = data.ohlcv_extended("ETH/USDT", days_back=2000)["close"]
    sol = data.ohlcv_extended("SOL/USDT", days_back=2000)["close"]
    print(f"  BTC: {len(btc)} bars, ETH: {len(eth)} bars, SOL: {len(sol)} bars")

    print("\n" + "=" * 60)
    print("1. CPCV (Combinatorial Purged CV)")
    print("=" * 60)
    splits = cpcv.cpcv_split(n_obs=2000, n_groups=6, k_test=2, embargo=5)
    print(f"  6 groups, k=2 -> {len(splits)} combinations (C(6,2) = 15)")
    train_lens = [len(t) for t, _ in splits]
    test_lens = [len(t) for _, t in splits]
    print(f"  train sizes: mean={np.mean(train_lens):.0f}, range=[{min(train_lens)}, {max(train_lens)}]")
    print(f"  test sizes:  mean={np.mean(test_lens):.0f}, range=[{min(test_lens)}, {max(test_lens)}]")

    sig_fn = lambda p: res_sig.tsmom_multi(p, horizons=(30, 90))
    cpcv_result = cpcv.cpcv_evaluate_signal(btc, sig_fn, n_groups=6, k_test=2, embargo=5)
    print(f"  TSMOM(30,90) under CPCV: {cpcv_result['n_evaluated']} evaluable folds")
    print(f"    mean OOS Sharpe: {cpcv_result['mean_sharpe_oos']:+.2f}")
    print(f"    std OOS Sharpe:  {cpcv_result['std_sharpe_oos']:+.2f}")
    print(f"    OOS observations: {cpcv_result['n_oos_obs']}")
    print(f"    (vs walk-forward earlier: 5 folds, 1630 OOS obs)")

    print("\n" + "=" * 60)
    print("2. HRP (Hierarchical Risk Parity)")
    print("=" * 60)
    aligned = pd.DataFrame({"BTC": btc, "ETH": eth, "SOL": sol}).dropna()
    rets = aligned.pct_change().dropna()
    weights = hrp.hrp_weights(rets)
    print(f"  HRP weights across BTC/ETH/SOL daily returns:")
    for asset, w in weights.items():
        print(f"    {asset}: {w:.3f}")
    print(f"  sum: {weights.sum():.3f}")
    eq_weight_var = float(np.dot(np.dot([1/3]*3, rets.cov().values), [1/3]*3))
    hrp_var = float(np.dot(np.dot(weights.values, rets.cov().values), weights.values))
    print(f"  portfolio variance — equal weight: {eq_weight_var:.6f}, HRP: {hrp_var:.6f}")

    print("\n" + "=" * 60)
    print("3. META-LABELING (RandomForest filter)")
    print("=" * 60)
    primary = res_sig.tsmom_multi(btc, horizons=(30, 90))
    log_ret = np.log(btc / btc.shift(1))
    features = pd.DataFrame({
        "rsi_14": _rsi(btc, 14),
        "rsi_28": _rsi(btc, 28),
        "vol_30d": log_ret.rolling(30).std() * np.sqrt(365),
        "vol_60d": log_ret.rolling(60).std() * np.sqrt(365),
        "trail_30d_ret": log_ret.rolling(30).sum(),
        "trail_90d_ret": log_ret.rolling(90).sum(),
        "price_vs_sma200": btc / btc.rolling(200).mean(),
    }).dropna()
    X, y = meta_labeling.build_meta_dataset(
        btc, primary, features, horizon=30, pt_sigma=2.0, sl_sigma=1.5
    )
    print(f"  events generated: {len(X)} | base rate (P(profit)): {y.mean():.3f}")
    if len(X) > 0:
        clf, diag = meta_labeling.train_meta_classifier(X, y, n_splits=5)
        print(f"  CV accuracies: {[f'{a:.3f}' for a in diag['cv_accuracies']]}")
        print(f"  mean CV accuracy: {diag['mean_cv_accuracy']:.3f} (vs base rate {diag['base_rate']:.3f})")
        lift = diag['mean_cv_accuracy'] - diag['base_rate']
        print(f"  lift over base rate: {lift:+.3f}")
        print(f"  top features:")
        sorted_imp = sorted(diag['feature_importance'].items(), key=lambda x: -x[1])
        for name, imp in sorted_imp[:3]:
            print(f"    {name}: {imp:.3f}")

    print("\n" + "=" * 60)
    print("4. GARCH(1,1) (Engle/Bollerslev)")
    print("=" * 60)
    btc_log_ret = np.log(btc / btc.shift(1)).dropna()
    cond_vol = garch_vol.garch_conditional_vol(btc_log_ret)
    realized_vol = btc_log_ret.rolling(30).std() * np.sqrt(365)
    fc_vol = garch_vol.garch_forecast_vol(btc_log_ret, horizon=1)
    params = garch_vol.garch_params(btc_log_ret)
    print(f"  fitted GARCH(1,1) parameters:")
    print(f"    omega:  {params['omega']:.6f}")
    print(f"    alpha:  {params['alpha']:.4f}  (impact of new shocks)")
    print(f"    beta:   {params['beta']:.4f}  (vol persistence)")
    print(f"    a+b:    {params['persistence']:.4f}  (stationary: {params['stationary']})")
    print(f"  current GARCH conditional vol (annualized): {cond_vol.iloc[-1]:.2%}")
    print(f"  current realized vol (30d, annualized):     {realized_vol.iloc[-1]:.2%}")
    print(f"  next-period GARCH forecast vol:              {fc_vol:.2%}")

    print("\n" + "=" * 60)
    print("5. LEDOIT-WOLF SHRINKAGE")
    print("=" * 60)
    sample_cov = rets.cov()
    shrunk = shrinkage.shrinkage_cov(rets, method="ledoit_wolf")
    intensity = shrinkage.shrinkage_intensity(rets)
    print(f"  shrinkage intensity (alpha): {intensity:.3f}")
    print(f"  sample covariance:")
    print(sample_cov.round(6).to_string())
    print(f"  shrunk covariance:")
    print(shrunk.round(6).to_string())

    print("\nALL FIVE MODULES OK")


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


if __name__ == "__main__":
    main()
