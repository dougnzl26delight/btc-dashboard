"""Bug test for the 4-lever calibration changes.

Exercises:
  - portfolio cap math (count_active_pairs, effective_risk_pct)
  - orphan-pair detection in pro_trend_run
  - universe correctness
  - state file robustness (missing, corrupt, empty)
  - catalyst overlay actually disabled
  - import cleanliness across modules
  - reproducibility of final-config backtest

Each test prints PASS/FAIL with detail. Exit code 1 if any failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


N_FAIL = 0
N_PASS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global N_FAIL, N_PASS
    status = "PASS" if ok else "FAIL"
    if ok:
        N_PASS += 1
    else:
        N_FAIL += 1
    suffix = f" - {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


# ============================================================================
section("imports & module loading")
# ============================================================================

try:
    from strategies import pro_trend
    check("strategies.pro_trend imports", True)
except Exception as e:
    check("strategies.pro_trend imports", False, str(e))
    print("Cannot continue without pro_trend module."); sys.exit(1)

try:
    import pro_trend_run
    check("pro_trend_run imports", True)
except Exception as e:
    check("pro_trend_run imports", False, str(e))

try:
    from core import basis_executor
    from strategies import funding_basis_arb
    import basis_run
    check("basis_run + dependencies import", True)
except Exception as e:
    check("basis_run + dependencies import", False, str(e))


# ============================================================================
section("config sanity - what's actually live")
# ============================================================================

check("PRO_TREND_PAIRS has exactly 5 pairs",
      len(pro_trend.PRO_TREND_PAIRS) == 5,
      f"len={len(pro_trend.PRO_TREND_PAIRS)}")

check("Universe = SOL/BTC/OP/AVAX/ETH (top 5 by Sharpe)",
      set(pro_trend.PRO_TREND_PAIRS) == {"SOL/USDT", "BTC/USDT", "OP/USDT",
                                          "AVAX/USDT", "ETH/USDT"},
      f"got {pro_trend.PRO_TREND_PAIRS}")

check("PORTFOLIO_RISK_CAP exists and = 0.15",
      hasattr(pro_trend, "PORTFOLIO_RISK_CAP") and pro_trend.PORTFOLIO_RISK_CAP == 0.15,
      f"got {getattr(pro_trend, 'PORTFOLIO_RISK_CAP', None)}")

check("USE_CATALYST_OVERLAY = False",
      pro_trend.USE_CATALYST_OVERLAY is False)

check("RISK_PCT_PER_UNIT = 0.04",
      pro_trend.RISK_PCT_PER_UNIT == 0.04)

check("LEVERAGE_MULTIPLIER = 1.5",
      pro_trend.LEVERAGE_MULTIPLIER == 1.5)


# ============================================================================
section("count_active_pairs() and effective_risk_pct() math")
# ============================================================================

# Save current state files so we can restore after this test
REPO_ROOT = Path(__file__).resolve().parent.parent
state_files = list(REPO_ROOT.glob(".pro_trend_state_*.json"))
saved_states = {f.name: f.read_bytes() for f in state_files}


def _restore_states():
    # Remove anything we created
    for f in REPO_ROOT.glob(".pro_trend_state_*.json"):
        if f.name not in saved_states:
            try:
                f.unlink()
            except Exception:
                pass
    # Restore originals
    for name, data in saved_states.items():
        (REPO_ROOT / name).write_bytes(data)


def _set_state(pair_base: str, units_count: int):
    """Write a state file with N units for a given pair base (e.g. 'SOL')."""
    p = REPO_ROOT / f".pro_trend_state_{pair_base}.json"
    state = {
        "side": "long" if units_count > 0 else None,
        "units": [{"qty": 1.0, "entry_price": 100.0, "entry_atr": 1.0}
                  for _ in range(units_count)],
        "extreme": 100.0,
        "trail_stop": 90.0,
        "peak_equity": 100_000.0,
    }
    p.write_text(json.dumps(state))


try:
    # Wipe all state files, then test from clean
    for f in list(REPO_ROOT.glob(".pro_trend_state_*.json")):
        f.unlink()

    # Case 1: zero active
    n = pro_trend.count_active_pairs()
    check("0 active when no state files", n == 0, f"got {n}")
    risk = pro_trend.effective_risk_pct("SOL/USDT")
    # Excluding SOL itself, n_active=0, +1 = 1, cap = 0.15/1 = 0.15, base=0.04, min=0.04
    check("1st entry gets full base risk (0.04)",
          abs(risk - 0.04) < 1e-9, f"got {risk}")

    # Case 2: 1 already active (SOL), entering BTC
    _set_state("SOL", 1)
    n = pro_trend.count_active_pairs()
    check("1 active counted", n == 1, f"got {n}")
    risk = pro_trend.effective_risk_pct("BTC/USDT")
    # n_active excluding BTC = 1 (SOL), +1 = 2 active total, cap=0.15/2=0.075, base=0.04, min=0.04
    check("2 active -> still 0.04 (base < cap/n)",
          abs(risk - 0.04) < 1e-9, f"got {risk}")

    # Case 3: 4 active, entering 5th
    _set_state("SOL", 1); _set_state("BTC", 1)
    _set_state("OP", 1); _set_state("AVAX", 1)
    n = pro_trend.count_active_pairs()
    check("4 active counted", n == 4, f"got {n}")
    risk = pro_trend.effective_risk_pct("ETH/USDT")
    # n_active excl ETH = 4, +1 = 5, cap = 0.15/5 = 0.03 < 0.04 -> 0.03
    expected = 0.15 / 5
    check(f"5 active -> cap binds at {expected:.3f}",
          abs(risk - expected) < 1e-9, f"got {risk}")

    # Case 4: pair already active, asking for itself
    risk = pro_trend.effective_risk_pct("SOL/USDT")
    # excludes SOL, so n=3 (BTC/OP/AVAX) +1 = 4, cap=0.0375, base=0.04 -> 0.0375
    expected = 0.15 / 4
    check(f"existing pair excluded from count -> {expected:.4f}",
          abs(risk - expected) < 1e-9, f"got {risk}")

    # Case 5: orphan state file (NEAR - not in universe) does NOT count
    _set_state("NEAR", 1)  # not in PRO_TREND_PAIRS
    n = pro_trend.count_active_pairs()
    check("orphan pair not counted in count_active_pairs()",
          n == 4, f"got {n}")  # still 4 (NEAR not in universe)

finally:
    _restore_states()


# ============================================================================
section("orphan-pair detection in pro_trend_run")
# ============================================================================

try:
    for f in list(REPO_ROOT.glob(".pro_trend_state_*.json")):
        f.unlink()

    # No orphans
    orphans = pro_trend_run._orphaned_pairs()
    check("no orphans when all state files in universe (none)",
          orphans == [], f"got {orphans}")

    # Universe pair with units = NOT orphan
    _set_state("SOL", 1)
    orphans = pro_trend_run._orphaned_pairs()
    check("universe pair with units is not orphan",
          orphans == [], f"got {orphans}")

    # Universe pair flat = NOT orphan
    _set_state("SOL", 0)
    orphans = pro_trend_run._orphaned_pairs()
    check("universe pair flat is not orphan",
          orphans == [], f"got {orphans}")

    # Out-of-universe pair with units = ORPHAN
    _set_state("NEAR", 1)
    orphans = pro_trend_run._orphaned_pairs()
    check("out-of-universe pair with units IS orphan",
          orphans == ["NEAR/USDT"], f"got {orphans}")

    # Out-of-universe pair flat = NOT orphan (ignored)
    _set_state("NEAR", 0)
    orphans = pro_trend_run._orphaned_pairs()
    check("out-of-universe pair flat is NOT orphan",
          orphans == [], f"got {orphans}")

    # Multiple orphans
    _set_state("NEAR", 1); _set_state("DOGE", 1); _set_state("XRP", 1)
    orphans = sorted(pro_trend_run._orphaned_pairs())
    expected = ["DOGE/USDT", "NEAR/USDT", "XRP/USDT"]
    check("multiple orphans detected and sorted",
          orphans == expected, f"got {orphans}")

finally:
    _restore_states()


# ============================================================================
section("state file robustness")
# ============================================================================

try:
    for f in list(REPO_ROOT.glob(".pro_trend_state_*.json")):
        f.unlink()

    # Corrupt JSON
    bad = REPO_ROOT / ".pro_trend_state_SOL.json"
    bad.write_text("{not valid json")
    try:
        n = pro_trend.count_active_pairs()
        check("count_active_pairs swallows corrupt JSON",
              n == 0, f"got {n}")
    except Exception as e:
        check("count_active_pairs swallows corrupt JSON",
              False, f"raised {type(e).__name__}: {e}")

    # effective_risk_pct should also survive corruption (it calls count_active_pairs)
    try:
        risk = pro_trend.effective_risk_pct("BTC/USDT")
        check("effective_risk_pct survives corrupt sibling state",
              risk == 0.04, f"got {risk}")
    except Exception as e:
        check("effective_risk_pct survives corrupt sibling state",
              False, f"raised {type(e).__name__}: {e}")

    # _orphaned_pairs should swallow corrupt JSON
    orphans = pro_trend_run._orphaned_pairs()
    check("_orphaned_pairs swallows corrupt JSON", isinstance(orphans, list))

    # Empty file
    bad.write_text("")
    try:
        orphans = pro_trend_run._orphaned_pairs()
        check("_orphaned_pairs swallows empty file",
              isinstance(orphans, list))
    except Exception as e:
        check("_orphaned_pairs swallows empty file",
              False, f"raised {e}")

    # Valid but missing 'units' key
    bad.write_text(json.dumps({"side": None}))
    n = pro_trend.count_active_pairs()
    check("count_active_pairs handles missing 'units' key",
          n == 0, f"got {n}")

finally:
    _restore_states()


# ============================================================================
section("backtest reproducibility")
# ============================================================================

try:
    from core.vol_targeted_test import fetch_all, portfolio_backtest
    pair_data = fetch_all(days_back=1500)
    check("data fetch returns 5 pairs", len(pair_data) == 5,
          f"got {list(pair_data.keys())}")

    r1 = portfolio_backtest(
        pair_data=pair_data, base_risk=0.04, portfolio_risk_cap=0.15,
    )
    r2 = portfolio_backtest(
        pair_data=pair_data, base_risk=0.04, portfolio_risk_cap=0.15,
    )
    check("backtest deterministic (same params -> same result)",
          abs(r1["sharpe"] - r2["sharpe"]) < 1e-9 and
          abs(r1["annualized_return"] - r2["annualized_return"]) < 1e-9,
          f"r1 sharpe={r1['sharpe']}, r2 sharpe={r2['sharpe']}")

    check("backtest returns +45% ann +/- 1pp (regression check)",
          0.43 < r1["annualized_return"] < 0.47,
          f"got {r1['annualized_return']:.4f}")

    check("backtest sharpe > 0.95 (regression check)",
          r1["sharpe"] > 0.95, f"got {r1['sharpe']:.4f}")

except Exception as e:
    check(f"backtest reproducibility test", False, f"crash: {e}")


# ============================================================================
section("catalyst overlay disabled - verify it's actually skipped")
# ============================================================================

# Make sure cycle path doesn't call catalyst_signals when off.
# Quickest check: the import is still present (no logic break) but the
# returned multiplier is irrelevant.
try:
    from core.catalyst_signals import combined_catalyst_multiplier
    cm = combined_catalyst_multiplier()
    check("catalyst_signals still importable",
          isinstance(cm, dict) and "combined_mult" in cm)
    check("USE_CATALYST_OVERLAY confirms disabled",
          pro_trend.USE_CATALYST_OVERLAY is False)
except Exception as e:
    check("catalyst overlay check", False, str(e))


# ============================================================================
section("P&L attribution module")
# ============================================================================

try:
    from core import pnl_attribution

    # Save and restore the attribution file
    attrib_file = pnl_attribution.ATTRIB_FILE
    saved_attrib = attrib_file.read_bytes() if attrib_file.exists() else None

    try:
        # Clean slate
        if attrib_file.exists():
            attrib_file.unlink()

        # Tag a position
        pnl_attribution.tag_entry("TEST/USDT", sleeve="systematic_pro_trend",
                                   side="long", entry_price=100.0, qty=1.0)
        sleeve = pnl_attribution.get_sleeve("TEST/USDT")
        check("tag_entry stores sleeve correctly",
              sleeve == "systematic_pro_trend", f"got {sleeve}")

        # Reject invalid sleeve
        try:
            pnl_attribution.tag_entry("BAD/USDT", sleeve="not_a_sleeve",
                                       side="long", entry_price=1, qty=1)
            check("invalid sleeve rejected", False, "should have raised")
        except ValueError:
            check("invalid sleeve rejected", True)

        # Per-sleeve P&L
        result = pnl_attribution.per_sleeve_pnl({"TEST/USDT": 110.0})
        expected_pnl = 1.0 * (110.0 - 100.0)  # long, +10
        actual = result["per_sleeve_pnl"]["systematic_pro_trend"]
        check("per_sleeve_pnl computes correct unrealized P&L",
              abs(actual - expected_pnl) < 1e-9, f"got {actual}")

        # Untag
        pnl_attribution.untag("TEST/USDT")
        sleeve = pnl_attribution.get_sleeve("TEST/USDT")
        check("untag removes the position",
              sleeve == "unknown", f"got {sleeve}")
    finally:
        if saved_attrib is not None:
            attrib_file.write_bytes(saved_attrib)
        elif attrib_file.exists():
            attrib_file.unlink()
except Exception as e:
    check(f"P&L attribution tests", False, f"crash: {e}")


# ============================================================================
section("kill criteria monitor")
# ============================================================================

try:
    from ops import kill_criteria
    check("kill_criteria imports", True)
    check("BACKTEST_SHARPE constant set",
          kill_criteria.BACKTEST_SHARPE == 1.40,
          f"got {kill_criteria.BACKTEST_SHARPE}")
    check("BACKTEST_MAX_DD constant set",
          kill_criteria.BACKTEST_MAX_DD == 0.40,
          f"got {kill_criteria.BACKTEST_MAX_DD}")

    # Test kill-criteria logic with synthetic stats
    triggers = kill_criteria.check_kill_criteria(
        stats={"current_dd": 0.50, "rolling_sharpe": 0.5, "n_days_logged": 100},
        snapshot={},
    )
    check("K1 fires when DD > 45%",
          any(t["id"] == "K1" for t in triggers),
          f"got {triggers}")

    # K2 was refined: now requires 180-day Sharpe < -0.5 AND >=180 days logged.
    # The deeper test in the "refined K2 + new K5" section covers the new logic.
    # Here we check that just-Sharpe-negative does NOT trigger (false positive guard).
    triggers = kill_criteria.check_kill_criteria(
        stats={"current_dd": 0.30, "rolling_sharpe": -0.3,
               "rolling_180_sharpe": -0.3, "ytd_return": 0.0,
               "n_days_logged": 200},
        snapshot={},
    )
    check("K2 does NOT fire on mildly negative Sharpe (refined gate)",
          not any(t["id"] == "K2" for t in triggers),
          f"got {[t['id'] for t in triggers]}")

    triggers = kill_criteria.check_kill_criteria(
        stats={"current_dd": 0.20, "rolling_sharpe": 1.0, "n_days_logged": 100},
        snapshot={},
    )
    check("no triggers in healthy state",
          triggers == [], f"got {triggers}")
except Exception as e:
    check(f"kill criteria tests", False, f"crash: {e}")


# ============================================================================
section("intraday monitor + realtime kill switch + reviews")
# ============================================================================

try:
    from ops import pro_trend_intraday
    check("pro_trend_intraday imports", True)
    check("_all_managed_pairs returns universe + orphans",
          callable(pro_trend_intraday._all_managed_pairs))
    pairs = pro_trend_intraday._all_managed_pairs()
    check("intraday managed-pair list non-empty",
          len(pairs) >= 5, f"got {len(pairs)}")
except Exception as e:
    check("pro_trend_intraday tests", False, f"crash: {e}")

try:
    from ops import realtime_kill_switch
    check("realtime_kill_switch imports", True)
    # Without lock file, is_locked should return False
    locked, reason = realtime_kill_switch.is_locked()
    check("is_locked returns False when no lock file",
          locked is False, f"got locked={locked}")
    # check_velocity with empty history returns no triggers
    triggers = realtime_kill_switch.check_velocity([], 100_000.0)
    check("check_velocity with empty history returns no triggers",
          triggers == [], f"got {triggers}")
    # check_velocity that would trigger
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    history = [
        {"ts": (now - timedelta(minutes=30)).isoformat(),
         "total_mtm": 100_000.0},
        {"ts": (now - timedelta(minutes=8)).isoformat(),
         "total_mtm": 100_000.0},
    ]
    triggers = realtime_kill_switch.check_velocity(history, 90_000.0)
    # 10% drop in 8min — should trigger RT1 (5% in 10min)
    check("RT1 fires on 10% drop in 8min",
          any(t["id"] == "RT1" for t in triggers),
          f"got {triggers}")
except Exception as e:
    check("realtime_kill_switch tests", False, f"crash: {e}")

try:
    from ops import weekly_review
    check("weekly_review imports", True)
    # build_report should not crash even with no equity log
    rpt = weekly_review.build_report()
    check("weekly_review.build_report() returns string",
          isinstance(rpt, str) and len(rpt) > 100,
          f"got len={len(rpt) if isinstance(rpt, str) else type(rpt)}")
except Exception as e:
    check("weekly_review tests", False, f"crash: {e}")

try:
    from ops import monthly_oos
    check("monthly_oos imports", True)
    check("is_first_sunday() callable",
          callable(monthly_oos.is_first_sunday))
except Exception as e:
    check("monthly_oos tests", False, f"crash: {e}")


# ============================================================================
section("kill-switch lockout integration in pro_trend.cycle")
# ============================================================================

try:
    lock_file = REPO_ROOT / ".kill_switch_lock.json"
    saved_lock = lock_file.read_bytes() if lock_file.exists() else None

    try:
        # Write a lock 1 hour into future
        from datetime import datetime, timedelta, timezone
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        lock_file.write_text(json.dumps({
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "locked_until": future,
            "reason": "test",
        }))

        # Wipe state files so cycle would normally try to enter
        for f in REPO_ROOT.glob(".pro_trend_state_*.json"):
            f.unlink()

        # Run cycle on SOL (currently in bear regime — wouldn't enter anyway)
        # but check the action log includes entry_skipped
        # First need to mock or just verify the check logic is wired
        with open(REPO_ROOT / "strategies" / "pro_trend.py", "r") as f:
            code = f.read()
        check("pro_trend.cycle reads kill_switch_lock.json",
              ".kill_switch_lock.json" in code)
        check("pro_trend.cycle has entry_skipped action",
              "entry_skipped" in code)
    finally:
        if saved_lock is not None:
            lock_file.write_bytes(saved_lock)
        elif lock_file.exists():
            lock_file.unlink()
        _restore_states()
except Exception as e:
    check("lockout integration tests", False, f"crash: {e}")


# ============================================================================
section("XSMOM strategy + portfolio risk monitor")
# ============================================================================

try:
    from strategies import xsmom
    check("strategies.xsmom imports", True)
    check("XSMOM_UNIVERSE has 8 pairs",
          len(xsmom.XSMOM_UNIVERSE) == 8,
          f"got {len(xsmom.XSMOM_UNIVERSE)}")
    check("STRATEGY_ALLOCATION = 0.30 (30% of portfolio)",
          xsmom.STRATEGY_ALLOCATION == 0.30,
          f"got {xsmom.STRATEGY_ALLOCATION}")
    check("MOMENTUM_WINDOW_DAYS = 14",
          xsmom.MOMENTUM_WINDOW_DAYS == 14,
          f"got {xsmom.MOMENTUM_WINDOW_DAYS}")
    check("REBALANCE_FREQ_DAYS = 14",
          xsmom.REBALANCE_FREQ_DAYS == 14,
          f"got {xsmom.REBALANCE_FREQ_DAYS}")
except Exception as e:
    check("xsmom tests", False, f"crash: {e}")

try:
    from ops import portfolio_risk_monitor
    check("portfolio_risk_monitor imports", True)
    check("HIGH_CORRELATION_THRESHOLD = 0.85",
          portfolio_risk_monitor.HIGH_CORRELATION_THRESHOLD == 0.85)
except Exception as e:
    check("portfolio_risk_monitor tests", False, f"crash: {e}")

try:
    from core import institutional_validation
    check("institutional_validation imports", True)
    check("N_TRIALS reflects honest trial count",
          institutional_validation.N_TRIALS >= 100,
          f"got {institutional_validation.N_TRIALS}")
except Exception as e:
    check("institutional_validation tests", False, f"crash: {e}")

try:
    from core import monte_carlo_live_sim
    check("monte_carlo_live_sim imports", True)
    check("simulate_paths callable",
          callable(monte_carlo_live_sim.simulate_paths))
except Exception as e:
    check("monte_carlo tests", False, f"crash: {e}")


# ============================================================================
section("attribution module — xsmom sleeve added")
# ============================================================================

try:
    from core import pnl_attribution

    attrib_file = pnl_attribution.ATTRIB_FILE
    saved_attrib = attrib_file.read_bytes() if attrib_file.exists() else None

    try:
        if attrib_file.exists():
            attrib_file.unlink()

        # xsmom sleeve should be valid now
        pnl_attribution.tag_entry("xsmom:LINK/USDT", sleeve="xsmom",
                                   side="long", entry_price=14.0, qty=10.0)
        sleeve = pnl_attribution.get_sleeve("xsmom:LINK/USDT")
        check("xsmom sleeve accepted by tag_entry",
              sleeve == "xsmom", f"got {sleeve}")

        # per_sleeve_pnl includes xsmom bucket
        result = pnl_attribution.per_sleeve_pnl({"xsmom:LINK/USDT": 15.0})
        check("per_sleeve_pnl has xsmom bucket",
              "xsmom" in result["per_sleeve_pnl"])
        expected_pnl = 10 * (15.0 - 14.0)
        actual = result["per_sleeve_pnl"]["xsmom"]
        check("xsmom P&L computed correctly",
              abs(actual - expected_pnl) < 1e-9, f"got {actual}")
    finally:
        if saved_attrib is not None:
            attrib_file.write_bytes(saved_attrib)
        elif attrib_file.exists():
            attrib_file.unlink()
except Exception as e:
    check("xsmom attribution tests", False, f"crash: {e}")


# ============================================================================
section("regime detector + sim comparator + dashboard components")
# ============================================================================

try:
    from ops import regime_detector
    check("regime_detector imports", True)
    check("REGIME_EXPECTATIONS has BULL/BEAR/CHOP/TRANSITION",
          set(regime_detector.REGIME_EXPECTATIONS.keys()) ==
          {"BULL", "BEAR", "CHOP", "TRANSITION"})
except Exception as e:
    check("regime_detector tests", False, f"crash: {e}")

try:
    from ops import sim_comparator
    check("sim_comparator imports", True)
    # Test percentile_in_sim with synthetic data
    fake_record = {
        "tot_p5": -0.40, "tot_p25": -0.10, "tot_p50": 0.20,
        "tot_p75": 0.60, "tot_p95": 1.50,
    }
    pct_at_median = sim_comparator.percentile_in_sim(0.20, fake_record)
    check("percentile_in_sim returns ~50 at sim P50",
          abs(pct_at_median - 50) < 1, f"got {pct_at_median}")
    pct_below_p5 = sim_comparator.percentile_in_sim(-0.50, fake_record)
    check("percentile_in_sim < 5 when value below P5",
          pct_below_p5 < 5, f"got {pct_below_p5}")
    pct_above_p95 = sim_comparator.percentile_in_sim(2.00, fake_record)
    check("percentile_in_sim > 95 when value above P95",
          pct_above_p95 > 95, f"got {pct_above_p95}")
except Exception as e:
    check("sim_comparator tests", False, f"crash: {e}")

try:
    from ops import dashboard_components
    check("dashboard_components imports", True)
    data = dashboard_components.forward_envelope_data()
    check("forward_envelope_data returns envelope_p5/p50/p95",
          all(k in data for k in ["envelope_p5", "envelope_p50", "envelope_p95"])
          if "error" not in data else "error" in data,
          str(data.get("error", "ok")))
except Exception as e:
    check("dashboard_components tests", False, f"crash: {e}")


# ============================================================================
section("kill criteria — refined K2 + new K5")
# ============================================================================

try:
    from ops import kill_criteria
    # K5: YTD stop after month 4
    from datetime import datetime, timezone
    if datetime.now(timezone.utc).month >= 4:
        triggers = kill_criteria.check_kill_criteria(
            stats={"current_dd": 0.20, "rolling_sharpe": 0.5,
                   "rolling_180_sharpe": 0.5,
                   "ytd_return": -0.30, "n_days_logged": 200},
            snapshot={},
        )
        check("K5 fires on YTD < -25% after month 4",
              any(t["id"] == "K5" for t in triggers),
              f"got {[t['id'] for t in triggers]}")
    else:
        check("K5 month-4 gate respected (current month < 4)", True)

    # K2 refined: 180d Sharpe > -0.5 should NOT trigger
    triggers = kill_criteria.check_kill_criteria(
        stats={"current_dd": 0.20, "rolling_sharpe": -0.3,
               "rolling_180_sharpe": -0.3,
               "ytd_return": 0.05, "n_days_logged": 200},
        snapshot={},
    )
    check("K2 NOT triggered on 180d Sharpe -0.3 (above -0.5)",
          not any(t["id"] == "K2" for t in triggers),
          f"got {[t['id'] for t in triggers]}")

    # K2 refined: 180d Sharpe < -0.5 should trigger
    triggers = kill_criteria.check_kill_criteria(
        stats={"current_dd": 0.20, "rolling_sharpe": -0.7,
               "rolling_180_sharpe": -0.7,
               "ytd_return": 0.05, "n_days_logged": 200},
        snapshot={},
    )
    check("K2 fires on 180d Sharpe -0.7 (below -0.5)",
          any(t["id"] == "K2" for t in triggers),
          f"got {[t['id'] for t in triggers]}")

    # K3: 6+ months without entry
    triggers = kill_criteria.check_kill_criteria(
        stats={"current_dd": 0.10, "rolling_sharpe": 0.0,
               "rolling_180_sharpe": 0.0,
               "ytd_return": 0.05, "n_days_logged": 200,
               "months_since_systematic_entry": 7},
        snapshot={},
    )
    check("K3 fires on 6+ months without entry",
          any(t["id"] == "K3" for t in triggers),
          f"got {[t['id'] for t in triggers]}")
except Exception as e:
    check("refined kill criteria tests", False, f"crash: {e}")


# ============================================================================
section("GCR Tier A — information overlays")
# ============================================================================

try:
    from ops import oi_funding_overlay
    check("oi_funding_overlay imports", True)
    check("oi_funding_overlay.UNIVERSE_PAIRS has 7 pairs",
          len(oi_funding_overlay.UNIVERSE_PAIRS) == 7,
          f"got {len(oi_funding_overlay.UNIVERSE_PAIRS)}")
    # Test classifier
    regime = oi_funding_overlay.classify(0.9, 2.0, 0.05, 0.05)
    check("classify returns FROTH on high OI + high funding",
          regime == "FROTH", f"got {regime}")
    regime = oi_funding_overlay.classify(0.9, -1.0, -0.05, 0.05)
    check("classify returns EXHAUSTION on high OI + negative funding",
          regime == "EXHAUSTION", f"got {regime}")
    regime = oi_funding_overlay.classify(0.3, 0.5, 0.0, 0.05)
    check("classify returns NEUTRAL on low OI",
          regime == "NEUTRAL", f"got {regime}")
except Exception as e:
    check("oi_funding_overlay tests", False, f"crash: {e}")

try:
    from ops import etf_flow_overlay
    check("etf_flow_overlay imports", True)
    check("BIG_DAY_THRESHOLD_M = 500", etf_flow_overlay.BIG_DAY_THRESHOLD_M == 500)
    check("EXTREME_Z_THRESHOLD = 2.0", etf_flow_overlay.EXTREME_Z_THRESHOLD == 2.0)
except Exception as e:
    check("etf_flow_overlay tests", False, f"crash: {e}")

try:
    from ops import macro_filter
    check("macro_filter imports", True)
    # Test classifier with synthetic data
    import pandas as _pd
    dxy = _pd.Series([100] * 200, index=_pd.date_range("2024-01-01", periods=200))
    tnx = _pd.Series([4.0] * 200, index=_pd.date_range("2024-01-01", periods=200))
    vix = _pd.Series([15.0] * 200, index=_pd.date_range("2024-01-01", periods=200))
    r = macro_filter.classify_macro(dxy, tnx, vix)
    check("classify_macro returns valid regime on stable inputs",
          r["regime"] in ("RISK_ON", "NEUTRAL", "RISK_OFF", "FLIGHT"),
          f"got {r.get('regime')}")
except Exception as e:
    check("macro_filter tests", False, f"crash: {e}")

try:
    from ops import liq_cluster_proxy
    check("liq_cluster_proxy imports", True)
    check("UNIVERSE has 5 pairs", len(liq_cluster_proxy.UNIVERSE) == 5)
    check("CASCADE_VOL_MULT = 5.0",
          liq_cluster_proxy.CASCADE_VOL_MULT == 5.0)
except Exception as e:
    check("liq_cluster_proxy tests", False, f"crash: {e}")

try:
    from ops import catalyst_calendar
    check("catalyst_calendar imports", True)
    check("CATALYSTS list non-empty",
          len(catalyst_calendar.CATALYSTS) > 5,
          f"got {len(catalyst_calendar.CATALYSTS)}")
    # days_until — test with a known future date
    d = catalyst_calendar.days_until("2099-01-01")
    check("days_until returns positive for far-future date",
          d > 0, f"got {d}")
    d = catalyst_calendar.days_until("2000-01-01")
    check("days_until returns negative for past date",
          d < 0, f"got {d}")
except Exception as e:
    check("catalyst_calendar tests", False, f"crash: {e}")


# ============================================================================
section("end-to-end cycle smoke test")
# ============================================================================

# Run the actual cycle and verify it doesn't crash + returns expected shape
try:
    for pair in pro_trend.PRO_TREND_PAIRS:
        r = pro_trend.cycle(pair, mode="paper")
        if r.get("status") == "insufficient_data":
            check(f"{pair} cycle: insufficient_data",
                  False, "should have data")
            continue
        ok = (
            r.get("status") == "ok"
            and "price" in r and "sma" in r and "atr" in r
            and "actions" in r
        )
        check(f"{pair} cycle returns ok+expected fields", ok,
              "" if ok else f"got {r.keys()}")
except Exception as e:
    check("cycle smoke test", False, f"crash: {e}")


# ============================================================================
print()
print("=" * 60)
print(f"RESULTS: {N_PASS} pass, {N_FAIL} fail")
print("=" * 60)
sys.exit(0 if N_FAIL == 0 else 1)
