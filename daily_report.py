"""Daily report generator — emails a plain-text summary of exit-signal state.

Runs at 19:00 NZT (= 07:00 UTC). Snapshots:
  - Overnight alerts (state changes since last report)
  - Trailing-stop ranking with EMA21 distance
  - BTC peak-conditions countdown
  - Recommended actions

Writes the report to daily_reports/YYYY-MM-DD.txt for archival, then emails
via ops.alerts.send_email_report() if EMAIL_* env vars are configured.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ops import alerts, watchdog
import btc_exit_signal_alert as monitor


REPORT_DIR = Path(__file__).resolve().parent / "daily_reports_crypto"
REPORT_DIR.mkdir(exist_ok=True)

LAST_REPORT_STATE = Path(__file__).resolve().parent / "btc_exit_signal_report_state.json"

FEB_LOW = 62910  # BTC Feb 2026 cycle-relief low


def _nzt_now() -> str:
    """Return current NZ time as a string. NZ = UTC+12 (NZST) or +13 (NZDT).

    DST: NZDT runs last Sun of Sept through first Sun of April. May = NZST.
    """
    # Simple offset; for May we're in NZST = UTC+12. Good enough for report headers.
    nz = datetime.now(timezone.utc) + timedelta(hours=12)
    return nz.strftime("%Y-%m-%d %H:%M NZT")


def build_report(current: dict, alert_msgs: list[str]) -> tuple[str, str]:
    """Return (subject, body) for the daily email."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    nzt = _nzt_now()

    lines = []
    lines.append("CRYPTO RIG DAILY REPORT")
    lines.append(f"Generated: {nzt}")
    lines.append("=" * 70)
    lines.append("")

    # === Section 1: overnight alerts ===
    lines.append("OVERNIGHT ALERTS")
    lines.append("-" * 70)
    if alert_msgs:
        for m in alert_msgs:
            lines.append(f"  ! {m}")
    else:
        lines.append("  No state changes since last report.")
    lines.append("")

    # === Section 2: trailing-stop ranking ===
    lines.append("TRAILING-STOP RANKING (sorted by urgency)")
    lines.append("-" * 70)
    lines.append(f"  {'Pair':<10s} {'Price':>11s} {'EMA21':>11s} {'Dist':>7s} "
                 f"{'Drop':>7s} {'RSI':>4s} {'Stop':<8s} {'State':<12s}")
    sort_key = lambda kv: ({"BROKEN": 0, "NEAR": 1, "WATCH": 2, "OK": 3}.get(
        kv[1].get("stop_alert", "OK"), 4), kv[1].get("dist_from_ema21", 0))
    for pair, s in sorted(current.items(), key=sort_key):
        price = s.get("price", 0)
        ema = s.get("ema21_price", 0)
        dist = s.get("dist_from_ema21", 0)
        drop = (ema / price - 1) * 100 if price > 0 and s.get("above_ema") else 0
        rsi = s.get("rsi", 0)
        stop = s.get("stop_alert", "?")
        state = s.get("state", "?")
        price_str = f"${price:,.4f}" if price < 100 else f"${price:,.2f}"
        ema_str = f"${ema:,.4f}" if ema < 100 else f"${ema:,.2f}"
        lines.append(f"  {pair:<10s} {price_str:>11s} {ema_str:>11s} "
                     f"{dist*100:>+6.1f}% {drop:>+6.1f}% {rsi:>4.0f} "
                     f"{stop:<8s} {state:<12s}")
    lines.append("")

    # === Section 3: BTC peak countdown ===
    btc = current.get("BTC/USDT")
    if btc:
        lines.append("BTC PEAK COUNTDOWN")
        lines.append("-" * 70)
        btc_price = btc["price"]
        from_low = btc_price / FEB_LOW - 1
        avg_peak = FEB_LOW * 1.485
        avg_signal = avg_peak * 0.91
        lines.append(f"  Current: ${btc_price:,.2f}  (+{from_low*100:.1f}% from Feb low)")
        lines.append(f"  Avg historical signal-fire target: ${avg_signal:,.0f} "
                     f"(+{(avg_signal/btc_price - 1)*100:.1f}% from here)")
        lines.append(f"  Avg historical relief peak:        ${avg_peak:,.0f} "
                     f"(+{(avg_peak/btc_price - 1)*100:.1f}% from here)")
        lines.append("")
        conditions = [
            ("RSI(14) >= 70", btc["rsi"] >= 70, f"now {btc['rsi']:.0f}"),
            ("Mayer Mult >= 1.05", btc["mayer"] >= 1.05, f"now {btc['mayer']:.2f}"),
            ("BB% >= 0.95", btc["bb_pct"] >= 0.95, f"now {btc['bb_pct']*100:.0f}%"),
            ("MACD hist NEGATIVE", btc["macd_hist"] < 0, f"now {btc['macd_hist']:+.2f}"),
            ("Close BELOW EMA21", not btc["above_ema"], "above" if btc["above_ema"] else "BELOW"),
        ]
        n_met = sum(1 for _, met, _ in conditions if met)
        lines.append(f"  Exit conditions aligned: {n_met}/5")
        for label, met, current_val in conditions:
            mark = "YES" if met else "no "
            lines.append(f"    [{mark}] {label:<25s} ({current_val})")
        lines.append("")

    # === Section 4: recommended actions ===
    lines.append("RECOMMENDED ACTIONS")
    lines.append("-" * 70)
    actions = []
    for pair, s in current.items():
        if s.get("stop_alert") == "BROKEN":
            actions.append(f"  SELL NOW:    {pair} — EMA21 broken (price ${s['price']:,.4f} vs EMA21 ${s['ema21_price']:,.4f})")
        elif s.get("stop_alert") == "NEAR":
            actions.append(f"  SET LIMIT:   {pair} — stop-loss at ${s['ema21_price']:,.4f} ({s['dist_from_ema21']*100:.1f}% below current)")
        elif s.get("state") == "EXTREME OB":
            actions.append(f"  SCALE OUT:   {pair} — RSI {s['rsi']:.0f}, sell tranche on next +3% bounce")
        elif s.get("state") == "PEAK ZONE":
            actions.append(f"  EXIT SIGNAL: {pair} — backtested rule fired, sell immediately")
    if not actions:
        actions.append("  No urgent actions. Hold and monitor.")
    lines.extend(actions)
    lines.append("")

    # === Section 4.5: process compliance (Mark Douglas) ===
    try:
        from ops.process_compliance import compute_daily_score
        comp = compute_daily_score()
        lines.append("PROCESS COMPLIANCE (Douglas: 'track process, not P&L')")
        lines.append("-" * 70)
        oc = comp.get("overall_compliance")
        if oc is not None:
            lines.append(f"  Overall: {oc*100:.0f}%   Verdict: {comp.get('verdict', '?')}")
        else:
            lines.append("  Overall: NO DATA")
        lines.append(f"  Manual overrides today: {comp.get('n_manual_overrides', 0)}")
        for sleeve_name, d in comp.get("per_sleeve", {}).items():
            compl_str = f"{d['compliance']*100:.0f}%" if d.get('compliance') is not None else "n/a"
            lines.append(f"  {sleeve_name:<22s} sig={d['signals']:>3d} trd={d['trades']:>3d} compliance={compl_str}")
        lines.append("")
    except Exception as _e:
        lines.append(f"PROCESS COMPLIANCE — failed to load: {_e}")
        lines.append("")

    # === Section 4.6: meta-confidence (Lopez de Prado) ===
    try:
        from core.meta_confidence import CONFIDENCE_FUNCS, get_meta_confidence
        lines.append("META-CONFIDENCE per sleeve (Lopez de Prado AFML 3.7)")
        lines.append("-" * 70)
        for sleeve_name in CONFIDENCE_FUNCS:
            mc = get_meta_confidence(sleeve_name)
            if mc >= 1.3:
                tag = "STRONG (upsize)"
            elif mc >= 1.0:
                tag = "normal"
            elif mc >= 0.7:
                tag = "weak"
            else:
                tag = "VERY WEAK"
            lines.append(f"  {sleeve_name:<22s} {mc:.2f}x   {tag}")
        lines.append("")
    except Exception as _e:
        lines.append(f"META-CONFIDENCE — failed: {_e}")
        lines.append("")

    # === Section 4.7: loss acceptance locks (Douglas + Livermore) ===
    try:
        from ops.loss_acceptance_lock import status as lock_status
        locks = lock_status()
        active_locks = {k: v for k, v in locks.items() if v.get("locked")}
        if active_locks:
            lines.append("LOSS ACCEPTANCE LOCKS — sleeve cooldowns active")
            lines.append("-" * 70)
            for s, d in active_locks.items():
                lines.append(f"  {s:<22s} expires {d['expires_at'][:19]} (Douglas: take the loss; wait before adjusting)")
            lines.append("")
    except Exception:
        pass

    # === Section 4.8: BTC key levels + macro regime ===
    try:
        from core.btc_key_levels import get_status as btc_status
        bs = btc_status()
        lines.append("BTC KEY LEVELS (Glassnode + Hayes)")
        lines.append("-" * 70)
        lines.append(f"  Price ${bs.get('price', 0):,.0f}  Regime {bs.get('regime', '?')}")
        if bs.get("sth_mvrv_proxy") is not None:
            lines.append(f"  STH-MVRV proxy {bs['sth_mvrv_proxy']:.2f}  ({bs.get('sth_mvrv_regime', '?')})")
        for action in bs.get("actions", []):
            lines.append(f"  -> {action}")
        lines.append("")
    except Exception:
        pass

    try:
        from core.macro_correlation import regime_status as macro_status
        ms = macro_status()
        lines.append("MACRO REGIME (Hayes thesis)")
        lines.append("-" * 70)
        lines.append(f"  Regime: {ms.get('regime', '?').upper()}  de_risk_level={ms.get('de_risk_level', 0)}")
        for flag in ms.get("flags", []):
            lines.append(f"  ! {flag}")
        lines.append("")
    except Exception:
        pass

    # === Section 4.9: W16.B Fear & Greed + cycle composite ===
    try:
        from core.fear_greed import latest as fg_latest, cycle_composite_score
        fg = fg_latest()
        comp = cycle_composite_score()
        lines.append("FEAR & GREED + CYCLE COMPOSITE (W16.B)")
        lines.append("-" * 70)
        if fg.get("value") is not None:
            chg = fg.get("7d_change", 0)
            arrow = "rising +" if chg > 5 else ("falling -" if chg < -5 else "stable =")
            lines.append(f"  F&G Index: {fg['value']:>3d} ({fg.get('classification', '?')})  7d {arrow}{abs(chg)}")
            lines.append(f"  -> {fg.get('action', '?')}")
        if comp.get("composite_score") is not None:
            lines.append(f"  Composite (F&G+MVRV): {comp['composite_score']:.0f}/100  "
                         f"(F&G {comp.get('fg_value', '?')} + cycle {comp.get('cycle_score', 0):.0f})")
            if comp.get("composite_action"):
                lines.append(f"  -> {comp['composite_action']}")
        lines.append("")
    except Exception as _e:
        lines.append(f"FEAR & GREED — failed: {_e}")
        lines.append("")

    # === Section 4.10: W16.H BTC dominance regime ===
    try:
        from core.btc_dominance import status as dom_status
        ds = dom_status()
        if not ds.get("error"):
            lines.append("BTC DOMINANCE REGIME (W16.H)")
            lines.append("-" * 70)
            lines.append(f"  BTC.D: {ds['btc_dominance_pct']:.2f}%  Regime: {ds.get('regime', '?')}")
            if ds.get("eth_dominance_pct"):
                lines.append(f"  ETH.D: {ds['eth_dominance_pct']:.2f}%")
            lines.append(f"  Alt-sleeve scale: {ds.get('alt_scale', 1.0):.2f}x   "
                         f"BTC-sleeve scale: {ds.get('btc_scale', 1.0):.2f}x")
            lines.append(f"  -> {ds.get('action', '?')}")
            lines.append("")
    except Exception as _e:
        lines.append(f"BTC DOMINANCE — failed: {_e}")
        lines.append("")

    # === Section 4.11: W16.A liquidation pressure ===
    try:
        from core.liquidation_pressure import liquidation_pressure as lp_fn
        readings = []
        for _p in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            try:
                readings.append(lp_fn(_p))
            except Exception:
                continue
        actionable = [r for r in readings if r.get("edge_direction", "no_edge") != "no_edge"]
        if actionable:
            lines.append("LIQUIDATION PRESSURE (W16.A)")
            lines.append("-" * 70)
            for r in actionable:
                lines.append(f"  {r.get('pair', '?'):<10s}  long-cascade {r.get('cascade_long_probability', 0)*100:.0f}%  "
                             f"short-cascade {r.get('cascade_short_probability', 0)*100:.0f}%  "
                             f"edge: {r.get('edge_direction', '?')}")
                for note in r.get("reasoning", [])[:2]:
                    lines.append(f"    {note}")
            lines.append("")
    except Exception:
        pass

    # === Section 4.12: W16.E tail hedge recommendation ===
    try:
        from core.tail_hedge import compute_hedge_recommendation
        h = compute_hedge_recommendation(bankroll_usd=200_000)
        urgency = h.get("urgency", "?")
        if urgency in ("critical", "recommended"):
            lines.append("TAIL HEDGE RECOMMENDATION (W16.E)")
            lines.append("-" * 70)
            lines.append(f"  Urgency: {urgency.upper()}  ({h.get('risk_factor_count', 0)}/6 risk factors)")
            for r in h.get("reasoning", []):
                lines.append(f"  - {r}")
            ss = h.get("suggested_structure")
            if ss:
                lines.append(f"  Suggested: {ss.get('instrument', '?')} strike ${ss.get('strike', 0):,.0f} "
                             f"{ss.get('expiry_days', '?')}d  ~${ss.get('premium_usd_est', 0):,.0f} premium")
                lines.append(f"  Venue: {ss.get('venue', '?')}  (manual execution)")
            lines.append("")
    except Exception:
        pass

    # === Section 4.12.a: BTC Prediction Machine — multi-horizon forecast ===
    try:
        from core.btc_prediction import state_of_btc
        bp = state_of_btc()
        lines.append("BTC PREDICTION MACHINE — multi-horizon forecast (45+ signals)")
        lines.append("-" * 70)
        lines.append(f"  Current: ${bp['btc_price']:,.0f}   Regime: {bp['regime']}")
        lines.append("")
        for h, label in [("intraday", "Intraday (1d) "),
                          ("short_term", "Short (1-30d) "),
                          ("medium_term", "Medium (1-6m) "),
                          ("long_term", "Long (6m-2y)  ")]:
            hd = bp["horizons"][h]
            lines.append(f"    {label} {hd['interpretation']:<14s} "
                         f"score {hd['direction_score']:+.2f}  conf {hd['confidence']}")
        lines.append("")
        lines.append(f"  Price targets:")
        for h, label in [("short_term", "30d "), ("medium_term", "90d "),
                         ("long_term",  "180d")]:
            t = bp["price_targets"].get(h, {})
            if t:
                lines.append(f"    {label}  P25 ${t['p25']:>7,.0f}  "
                             f"median ${t['median']:>7,.0f}  P75 ${t['p75']:>7,.0f}")
        # Category breakdown (intraday horizon)
        lines.append("")
        lines.append(f"  Category sentiment (raw scores):")
        short_brk = bp["horizons"]["short_term"]["breakdown"]
        for cat in ("technical", "onchain", "sentiment", "derivatives",
                    "macro", "liquidations", "cycle", "flows", "options_adv",
                    "fundamentals", "regime_models"):
            br = short_brk.get(cat, {})
            score = br.get("score", 0)
            arrow = "++" if score > 0.4 else "+" if score > 0.1 else "=" if abs(score) <= 0.1 else "-" if score >= -0.4 else "--"
            lines.append(f"    {cat:<14s} {arrow:>3s} {score:+.2f}  "
                         f"({br.get('n_scored', 0)} signals)")
        # 3-lens ensemble
        ens = bp.get("ensemble", {})
        if ens:
            lines.append("")
            lines.append(f"  3-lens ensemble:  {ens.get('consensus', '?')}")
            for lens_name, ld in ens.get("lenses", {}).items():
                short_name = lens_name.replace("_lens", "")
                lines.append(f"    {short_name:<10s} {ld['score']:+.2f}  "
                             f"{ld['interpretation']:<14s} ({ld['n_signals']} signals)")
        # Hit rates if available
        hr = bp.get("hit_rates_by_horizon", {})
        if hr:
            lines.append("")
            lines.append(f"  Recent hit rates (rolling):")
            for h, d in hr.items():
                if d.get("n_observations", 0) >= 3:
                    lines.append(f"    {h:<14s} {d['n_correct']}/{d['n_observations']}  "
                                 f"({d['hit_rate']*100:.0f}% hit)")
        # Anomalies
        anom = bp.get("signal_anomalies", [])
        if anom:
            lines.append("")
            lines.append(f"  Signal anomalies (regime change candidates):")
            for a in anom[:3]:
                lines.append(f"    {a['signal']:<24s} z={a['z_score']:+.2f}  "
                             f"{a['interpretation']}")
        lines.append("")
    except Exception as _e:
        lines.append(f"BTC PREDICTION — failed: {_e}")
        lines.append("")

    # === Section 4.12.b: Cycle-top percentile detector ===
    # Backtest: fires 73 days before 2025-10-06 peak at 94.3% capture.
    # Suppressed when BTC > 12% below ATH (bear regime — false positives).
    try:
        from core.cycle_top_percentile import cycle_top_score
        cts = cycle_top_score()
        if not cts.get("error"):
            lines.append("CYCLE-TOP DETECTOR (percentile + rollover, backtest 94% capture)")
            lines.append("-" * 70)
            verdict = cts.get("verdict", "?")
            score = cts.get("score", 0)
            lines.append(f"  Composite: {score}/100  ->  {verdict}")
            lines.append(f"  Action:    {cts.get('action', '?')}")
            if verdict not in ("NOT_NEAR_ATH", "NORMAL"):
                # Only show details when there's signal
                for name, d in cts.get("indicators", {}).items():
                    if d.get("extreme") or d.get("rolled_over_from_high"):
                        marker = " EXTREME" if d.get("extreme") else " rolled"
                        lines.append(f"    {name:<22s} pct {d['percentile_rank']:>5.1f}%{marker}")
                if cts.get("weekly_macd_bear"):
                    lines.append(f"    Weekly MACD: BEAR cross active")
            lines.append("")
    except Exception:
        pass

    # === Section 4.13.a: Empirical regime gate (2026-06-01) ===
    try:
        from ops.regime_gate import current_regime, PAUSE_IN_BEAR, should_pause_sleeve
        r = current_regime()
        lines.append("EMPIRICAL REGIME GATE (backtest-calibrated)")
        lines.append("-" * 70)
        lines.append(f"  Current: {r['label']}")
        paused = []
        for s in sorted(PAUSE_IN_BEAR):
            p = should_pause_sleeve(s)
            if p["should_pause"]:
                paused.append(s)
        if paused:
            lines.append(f"  PAUSED sleeves (clear bear): {', '.join(paused)}")
            lines.append(f"    Rationale: backtest shows these strategies lose money in clear bear regimes.")
            lines.append(f"    Will auto-resume when BTC 30d > -8%.")
        else:
            lines.append(f"  All regime-gated sleeves ACTIVE (BTC 30d not in clear bear).")
        lines.append("")
    except Exception:
        pass

    # === Section 4.13.b: W16.G HRP weights snapshot ===
    try:
        from ops.portfolio_allocator import compute_and_persist as _hrp_compute, BASELINE_WEIGHTS
        hrp = _hrp_compute()
        if hrp.get("status") == "ok":
            lines.append("HRP SLEEVE ALLOCATION (W16.G, Lopez de Prado AFML 16)")
            lines.append("-" * 70)
            lines.append(f"  Computed from {hrp.get('n_observations', 0)} daily obs across {hrp.get('n_sleeves', 0)} sleeves")
            sorted_w = sorted(hrp["weights"].items(), key=lambda x: -x[1])
            for sleeve_name, hw in sorted_w[:8]:
                bw = BASELINE_WEIGHTS.get(sleeve_name, 0.0)
                scale = hw / bw if bw > 0 else 0.0
                tag = "" if 0.8 <= scale <= 1.25 else (" UPSIZE" if scale > 1.25 else " DOWNSIZE")
                lines.append(f"  {sleeve_name:<24s} HRP {hw*100:>5.1f}%   baseline {bw*100:>5.1f}%   {scale:.2f}x{tag}")
            lines.append("")
        elif hrp.get("status") == "waiting_for_data":
            lines.append("HRP SLEEVE ALLOCATION (W16.G)")
            lines.append("-" * 70)
            lines.append(f"  Waiting for >= 14 days of sleeve-level returns. Using hardcoded baseline weights.")
            lines.append("")
    except Exception:
        pass

    # === Section 4.13: W16.F VaR calibration warning ===
    # On Kupiec FAIL, also write .var_kupiec_breach.json flag for the gates
    # pipeline to soft-throttle all sleeves to 0.5x for 3 days while the
    # trader recalibrates the model.
    try:
        import json as _json
        from core.var_backtest import backtest_var
        from core.pnl_db import get_sleeve_returns
        # Aggregate across all sleeves for portfolio-level VaR check
        all_returns = []
        for sleeve_name in ("bah_btc", "xsmom", "pro_trend", "oversold_bounce",
                             "intraday_momentum", "basis_arb"):
            try:
                r = get_sleeve_returns(sleeve_name, days=90)
                if r:
                    all_returns.extend(r)
            except Exception:
                pass
        if len(all_returns) >= 30:
            vr = backtest_var(all_returns, var_pct=0.01, confidence_level=0.99)
            kupiec_verdict = vr.get("kupiec", {}).get("verdict", "")
            breach_flag = REPORT_DIR.parent / ".var_kupiec_breach.json"
            if "FAIL" in kupiec_verdict:
                lines.append("VaR CALIBRATION WARNING (W16.F)")
                lines.append("-" * 70)
                lines.append(f"  Kupiec test: {kupiec_verdict}")
                lines.append(f"  Observed breaches: {vr['kupiec']['n_breaches_observed']}  "
                             f"Expected: {vr['kupiec']['n_breaches_expected']:.1f}")
                lines.append(f"  -> Stated 1% VaR is functionally ~5x larger in fat tails.")
                lines.append(f"  -> Recalibrate or size for the empirical VaR, not the stated one.")
                lines.append(f"  -> Gates pipeline: all sleeves throttled to 0.5x for 3 days.")
                # Write breach flag for gates pipeline
                try:
                    breach_flag.write_text(_json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kupiec_verdict": kupiec_verdict,
                        "n_breaches_observed": vr["kupiec"]["n_breaches_observed"],
                        "n_breaches_expected": vr["kupiec"]["n_breaches_expected"],
                        "n_observations": vr["kupiec"]["n_observations"],
                    }))
                except Exception:
                    pass
                lines.append("")
            else:
                # PASS — clear any prior breach flag
                if breach_flag.exists():
                    try:
                        breach_flag.unlink()
                    except Exception:
                        pass
    except Exception:
        pass

    # === Section 5: footer ===
    lines.append("=" * 70)
    lines.append("Signal: MACD bear cross + close<EMA21 within 5 days")
    lines.append("Backtest n=5 (2018, 2022 bears): capture 67%, avoided 126% drawdown")
    lines.append("State file: btc_exit_signal_state.json  |  Dashboard: localhost:8510")
    lines.append("")

    body = "\n".join(lines)
    subject = f"Crypto Daily {today}"
    # Subject decorators based on urgency
    n_broken = sum(1 for s in current.values() if s.get("stop_alert") == "BROKEN")
    n_near = sum(1 for s in current.values() if s.get("stop_alert") == "NEAR")
    if n_broken:
        subject += f" — {n_broken} EMA21 BROKEN, ACTION REQUIRED"
    elif n_near:
        subject += f" — {n_near} NEAR exit"
    elif alert_msgs:
        subject += f" — {len(alert_msgs)} alerts"
    else:
        subject += " — all quiet"

    return subject, body


def main():
    prior = monitor.load_state()
    current = monitor.compute_status()
    alert_msgs = monitor.detect_alerts(current, prior)

    subject, body = build_report(current, alert_msgs)

    # Archive to disk
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = REPORT_DIR / f"{today}.txt"
    report_path.write_text(body, encoding="utf-8")

    # Email via SMTP
    email_status = alerts.alert_status()
    sent = False
    if email_status["email_configured"]:
        sent = alerts.send_email_report(subject, body)
        if sent:
            print(f"daily_report: emailed '{subject}' to configured recipient")
        else:
            print(f"daily_report: email send FAILED (creds set, but SMTP error). Report saved to {report_path}")
    else:
        print("daily_report: email NOT configured. Add EMAIL_FROM / EMAIL_TO / EMAIL_PASS to .env")
        print(f"daily_report: report saved to {report_path}")

    # Update state file (so detect_alerts in tomorrow's report compares against today)
    monitor.save_state(current)
    watchdog.beat()

    return {
        "subject": subject,
        "n_pairs": len(current),
        "n_alerts": len(alert_msgs),
        "emailed": sent,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    result = main()
    print(f"daily_report: {result}")
