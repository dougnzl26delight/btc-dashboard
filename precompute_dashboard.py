"""Precompute all dashboard panels to disk cache.

Runs on a schedule (every 15 min) so the streamlit dashboard never blocks
on slow upstream APIs (FRED, Deribit, CoinMetrics). The dashboard reads
from the disk cache populated by this job.

Usage:
    python precompute_dashboard.py              # precompute all
    python precompute_dashboard.py --quick      # only fast panels
    python precompute_dashboard.py --panel KEY  # one panel by key
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.dashboard_cache import disk_cached, _store, cache_age_seconds  # noqa: E402


# Each panel: (cache_key, callable)
def _panels() -> dict:
    """Return dict of panel_key -> (callable_returning_value, importable).

    Importable is True if we can safely import the underlying module without
    side effects; False for ones that hit network on import (rare).
    """
    panels = {}

    def _safe(key, fn):
        panels[key] = fn

    # Top scorecard bundle (FRED-heavy, 30+s cold)
    def _top_scorecard():
        from core.btc_top_scorecard import (
            top_confirmation_scorecard, phased_exit_recommendation, historical_backtest
        )
        return {
            "scorecard": top_confirmation_scorecard(),
            "recommendation": phased_exit_recommendation(current_equity_pct=70),
            "backtest": historical_backtest(),
        }
    _safe("top_scorecard", _top_scorecard)

    def _rotation():
        from core.btc_macro_rotation import rotation_phase
        return rotation_phase()
    _safe("rotation", _rotation)

    def _early_rotation():
        from core.btc_early_rotation import early_rotation_signal
        return early_rotation_signal(current_equity_pct=70, current_btc_pct=30, total_stake_nzd=130_000)
    _safe("early_rotation", _early_rotation)

    # === Top-tier macro layer + regime + unified decision ===
    def _macro_layer():
        from core.btc_macro_layer import all_macro_signals
        return all_macro_signals()
    _safe("macro_layer", _macro_layer)

    def _regime():
        from core.btc_macro_layer import all_macro_signals
        from core.btc_regime import full_regime_analysis
        from core.btc_unified_decision import net_liquidity_z
        macro = all_macro_signals()
        liq = net_liquidity_z()
        return full_regime_analysis(macro=macro, liquidity_z=liq["z"])
    _safe("regime", _regime)

    def _unified_decision():
        from core.btc_unified_decision import unified_decision
        return unified_decision(current_equity_pct=70, current_btc_pct=30, total_stake_nzd=130_000)
    _safe("unified_decision", _unified_decision)

    def _predictor_engine():
        from core.predictor_engine import predictor_engine_state
        return predictor_engine_state(current_equity_pct=70, current_btc_pct=30, total_stake_nzd=130_000)
    _safe("predictor_engine", _predictor_engine)

    # I1: BTC-native top scorecard (the gap from cycle 5 backtest)
    def _btc_native_top():
        from core.btc_native_top_scorecard import btc_native_top_scorecard
        return btc_native_top_scorecard()
    _safe("btc_native_top_scorecard", _btc_native_top)

    # G1: BTC-native BOTTOM scorecard — guru-tier signals (Swift/Woo/Edwards/Mayer)
    def _btc_native_bottom():
        from core.btc_native_bottom_scorecard import btc_native_bottom_scorecard
        return btc_native_bottom_scorecard()
    _safe("btc_native_bottom_scorecard", _btc_native_bottom)

    # === Guru-grade upgrade panels (2026-06-13) — precompute so the page NEVER
    #     computes them on render (each hits yfinance / SPX / QQQ / DXY = slow) ===
    def _bottom_confirmation():
        from core.btc_bottom_scorecard import bottom_confirmation_scorecard
        return bottom_confirmation_scorecard()        # includes theme-breadth overlay
    _safe("bottom_confirmation", _bottom_confirmation)

    def _regime_tag():
        from core.regime_tag import regime_tag
        return regime_tag()
    _safe("regime_tag", _regime_tag)

    def _btc_equity_relval():
        from core.btc_equity_relval import btc_equity_relative_value
        return btc_equity_relative_value()
    _safe("btc_equity_relval", _btc_equity_relval)

    def _etf_flow_quality():
        from core.etf_flow_quality import etf_flow_quality
        return etf_flow_quality()
    _safe("etf_flow_quality", _etf_flow_quality)

    # Olson AI summary — background only. The LLM call is hash-guarded inside, so
    # this is a no-op (0 tokens) unless his feed actually changed since last time.
    def _olson_ai_summary():
        from core.olson_ai_summary import olson_ai_summary
        return olson_ai_summary()
    _safe("olson_ai_summary", _olson_ai_summary)

    # S1: Phillip Swift / LookIntoBitcoin indicator suite
    def _swift():
        from core.btc_swift_indicators import all_swift_indicators
        return all_swift_indicators()
    _safe("swift_indicators", _swift)

    # S2: Phillip Swift chart suite — Rainbow, Pi Cycle history, multiplier bands
    def _swift_charts():
        from core.btc_swift_charts import all_swift_charts
        return all_swift_charts()
    _safe("swift_charts", _swift_charts)

    # CD1: Cycle Dials — Swift indicators as at-a-glance gauges (Charts tab)
    def _cycle_dials():
        from core.btc_cycle_dials import all_cycle_dials
        return all_cycle_dials()
    _safe("cycle_dials", _cycle_dials)

    # F1: Free-tier proxies for paid metrics (HODL Waves, Reserve Risk, CVDD, etc.)
    def _proxies():
        from core.btc_advanced_proxies import all_proxies
        return all_proxies()
    _safe("free_proxies", _proxies)

    # SW1: Phillip Swift Watch (Risk Index, Thermocap, Profitable Days, 200wMA + content)
    def _swift_watch():
        from core.btc_swift_watch import all_swift_watch
        return all_swift_watch()
    _safe("swift_watch", _swift_watch)

    # SD1: Swift Dials (Halving Clock, BTC Dominance, S2F, Open Interest, cycle overlay)
    def _swift_dials():
        from core.btc_swift_dials import all_swift_dials
        return all_swift_dials()
    _safe("swift_dials", _swift_dials)

    # EQ1: Jesse Olson's equity-side technical layer for QQQ — 589 gap, 200wMA, MACD, RSI
    def _equity_olson():
        from core.equity_olson import qqq_olson_verdict
        return qqq_olson_verdict()
    _safe("equity_olson", _equity_olson)

    # EQ2: Semis leading tell (SOXX) — early warning ahead of QQQ
    def _equity_semis():
        from core.equity_semis import semis_tell
        return semis_tell()
    _safe("equity_semis", _equity_semis)

    # RT1: Equity→BTC rotation trigger — 3 paths, 2-of-2 each (single-shot rotation)
    def _rotation_trigger():
        from core.rotation_trigger import evaluate_rotation_trigger
        return evaluate_rotation_trigger()
    _safe("rotation_trigger", _rotation_trigger)

    # RV1: Rotation Validation — backtest + correlation + sensitivity + confidence + cycle6
    def _rotation_validation():
        from core.rotation_validation import all_validation
        return all_validation()
    _safe("rotation_validation", _rotation_validation)

    # GN1: Glassnode-grade proxies (LTH NPC + aSOPR proxy + cohort P/L)
    def _glassnode_proxies():
        from core.glassnode_proxies import all_glassnode_proxies
        return all_glassnode_proxies()
    _safe("glassnode_proxies", _glassnode_proxies)

    # GI1: Guru intelligence — track records + recent high-relevance calls
    def _guru_intelligence():
        from core.guru_intelligence import all_guru_intelligence
        return all_guru_intelligence()
    _safe("guru_intelligence", _guru_intelligence)

    # SO1: BTC top scale-out trigger — phased exit ladder for the next bull
    def _scale_out_trigger():
        from core.scale_out_trigger import evaluate_scale_out_trigger
        return evaluate_scale_out_trigger()
    _safe("scale_out_trigger", _scale_out_trigger)

    # I3: Pattern target zones — current price vs all supply/support zones
    def _pattern_zones():
        from core.btc_pattern_target_alert import all_zones_status
        return all_zones_status()
    _safe("pattern_zones", _pattern_zones)

    # I4: ETF flow regime classification
    def _etf_regime():
        from core.btc_etf_regime_detector import classify_regime
        return classify_regime()
    _safe("etf_regime", _etf_regime)

    def _state():
        from core.btc_prediction import state_of_btc
        return state_of_btc()
    _safe("state_of_btc", _state)

    def _bottom_signals():
        from core.btc_bottom_signals import all_bottom_signals
        return all_bottom_signals()
    _safe("bottom_signals", _bottom_signals)

    def _date_predictions():
        from core.btc_date_predictions import (
            indicator_extrapolation, cycle_4_analog,
            macro_calendar, bottom_date_convergence,
        )
        return {
            "extrapolation":   indicator_extrapolation(),
            "cycle_4_analog":  cycle_4_analog(),
            "macro_calendar":  macro_calendar(180),
            "convergence":     bottom_date_convergence(),
        }
    _safe("date_predictions", _date_predictions)

    def _realized_price():
        from core.btc_cost_basis import realized_price
        return realized_price()
    _safe("realized_price", _realized_price)

    def _sth():
        from core.btc_cost_basis import sth_cost_basis
        return sth_cost_basis()
    _safe("sth_cost_basis", _sth)

    def _rcap_dd():
        from core.btc_cost_basis import realized_cap_drawdown_depth
        return realized_cap_drawdown_depth()
    _safe("realized_cap_drawdown", _rcap_dd)

    def _olson():
        from core.btc_jesse_olson import olson_combined_verdict
        return olson_combined_verdict()
    _safe("olson", _olson)

    def _ohlcv():
        from core import data
        return data.ohlcv_extended("BTC/USDT", days_back=90)
    _safe("ohlcv_90d", _ohlcv)

    # 'Today's update' daily brief — regenerate EVERY cycle (added LAST so it reads
    # the freshly-computed caches above) instead of being frozen once at 6am. Keeps
    # it current all day AND immune to a missed 6am run (reboot/sleep). The same-day
    # snapshot is replaced in its own state file, so the 'vs yesterday' diff holds.
    def _simpleton_brief():
        from core.simpleton_daily_brief import build_daily_brief
        return build_daily_brief()
    _safe("simpleton_brief", _simpleton_brief)

    return panels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", help="Only precompute one panel by key")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slowest panels (FRED-heavy)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    panels = _panels()
    if args.panel:
        if args.panel not in panels:
            print(f"Unknown panel '{args.panel}'. Available: {', '.join(panels)}",
                  file=sys.stderr)
            return 1
        targets = {args.panel: panels[args.panel]}
    elif args.quick:
        skip = {"top_scorecard", "rotation", "date_predictions"}
        targets = {k: v for k, v in panels.items() if k not in skip}
    else:
        targets = panels

    total_t0 = time.time()
    succeeded, failed = 0, 0
    for key, fn in targets.items():
        t0 = time.time()
        try:
            if not args.quiet:
                age = cache_age_seconds(key)
                age_str = f"prev age={age/60:.1f}min" if age else "no prev cache"
                print(f"[{time.time()-total_t0:5.1f}s] computing {key:25s} ({age_str})...",
                      flush=True)
            value = fn()
            _store(key, value)
            if not args.quiet:
                print(f"[{time.time()-total_t0:5.1f}s]   OK ({time.time()-t0:.1f}s)",
                      flush=True)
            succeeded += 1
        except Exception as e:
            if not args.quiet:
                print(f"[{time.time()-total_t0:5.1f}s]   FAIL {key}: {e!r}",
                      flush=True)
                traceback.print_exc(limit=2)
            failed += 1

    elapsed = time.time() - total_t0
    if not args.quiet:
        print(f"\nDone in {elapsed:.1f}s: {succeeded} ok, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
