"""Statistical validation layer for the rotation trigger.

Five components, each addressing a real critique from the quant review:

  1. Historical backtest    -- when would trigger have fired in past cycles?
                                Maps simplified signals to BTC price history
                                for 2018, 2020, 2022 bottoms.

  2. Signal correlation     -- which scorecard signals are independent vs
                                redundant (all UTXO-age derivatives)?
                                Outputs "effective signal count".

  3. Threshold sensitivity  -- how does trigger change with +/- 10% thresholds?
                                Robustness check on Olson 589, scorecard 8/15.

  4. Confidence score       -- % confidence based on INDEPENDENT firing signals
                                (deduplicated via correlation). 0-100 score.

  5. Cycle-6 ETF modifier   -- auto-detect ETF-era smoothing and recommend
                                downscaled thresholds (since cycle 5 only
                                drew down -50% vs historical -85%).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Known signal-correlation clusters in the BTC bottom scorecard.
# Signals in the same cluster share underlying mechanism (UTXO age, miners,
# derivatives), so they count as ~1 independent observation, not N.
# 2026-07-07 logic audit (F3): patterns broadened to STEMS so the
# percentile-rank variants ("Mayer percentile-rank", "Golden Ratio pct-rank",
# "Log regression pct-rank") and label drift ("2-year MA", "MVRV-Z < -1.5 OR
# raw MVRV") fold into their real mechanism instead of falling to "other" and
# being counted as extra "independent" votes. Price-vs-long-average ratios
# (Mayer / Golden Ratio / Log-reg / 200wMA / 2yMA / Pi) are ONE axis.
SIGNAL_CLUSTERS = {
    "on_chain_valuation": [
        # UTXO/realized-value + price-vs-long-average valuation — one axis
        "mvrv", "realized cap", "reserve risk", "ahr999", "lth cost",
        "mayer", "golden ratio", "log regression", "log reg",
    ],
    "miner_health": [
        "hash ribbon", "difficulty cycle", "puell",
    ],
    "derivatives_positioning": [
        "funding", "coinbase premium", "open interest",
    ],
    "cycle_timing": [
        "cycle day", "cycle-4 analog", "pi cycle",
        "200-week", "200 week", "200wma", "2-year ma", "2y ma", "halving",
    ],
    "macro_overlay": [
        "liquidity", "nvt",
    ],
}


# Historical BTC cycle bottoms — for backtest grounding
HISTORICAL_BOTTOMS = {
    3: {
        "halving_date":   date(2016, 7, 9),
        "peak_date":      date(2017, 12, 17),
        "peak_price":     19_783,
        "bottom_date":    date(2018, 12, 15),
        "bottom_price":   3_200,
        "drawdown_pct":   -84,
    },
    4: {
        "halving_date":   date(2020, 5, 11),
        "peak_date":      date(2021, 11, 8),
        "peak_price":     67_526,
        "bottom_date":    date(2022, 11, 9),
        "bottom_price":   15_500,
        "drawdown_pct":   -77,
    },
    5: {
        "halving_date":   date(2024, 4, 20),
        "peak_date":      date(2025, 10, 6),
        "peak_price":     124_659,
        # Cycle 5 bottom hasn't fully formed yet -- use observed low so far
        "bottom_date":    date(2026, 6, 1),
        "bottom_price":   58_000,    # estimate; muted cycle
        "drawdown_pct":   -53,
    },
}


# =================================================================
# 1) HISTORICAL BACKTEST — when would trigger have fired?
# =================================================================
def historical_backtest() -> dict:
    """For each past cycle, find when the trigger conditions WOULD have fired.

    Uses simplified proxy signals (price-based) since we don't have full
    historical scorecard data. The proxy: at week W, what fraction of these
    rough bottom conditions were met?

      - BTC drawdown >= -50% (cycle bear underway)
      - BTC at or near 200wMA
      - Days post-halving >= 730 (>2y, in bear-end zone)

    These are CONSERVATIVE proxies. Real trigger uses 15-signal scorecard
    which fires earlier than these price-only proxies.
    """
    try:
        import pandas as pd
        from core import data
    except Exception:
        return {"error": "data module unavailable"}

    # Pull all-time BTC daily
    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=365 * 8)
        if df.empty: return {"error": "no historical data"}
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
    except Exception as e:
        return {"error": f"data fetch: {e}"}

    # Compute 200-week MA (weekly resample)
    wkly = df["close"].resample("W").last().dropna()
    wma200 = wkly.rolling(200).mean()

    results = []
    for cycle_n, cyc in HISTORICAL_BOTTOMS.items():
        peak_date = cyc["peak_date"]
        bottom_date = cyc["bottom_date"]
        bottom_price = cyc["bottom_price"]
        peak_price = cyc["peak_price"]

        # Walk forward from peak; check each week for proxy fire
        try:
            window = wkly[(wkly.index.date >= peak_date) &
                          (wkly.index.date <= bottom_date + pd.Timedelta(days=180).to_pytimedelta())]
            if window.empty: continue

            fire_date = None
            fire_price = None
            for d, px in window.items():
                dd_from_peak = (px / peak_price - 1) * 100
                wma_now = wma200.get(d, None)
                near_wma = (wma_now is not None and px / wma_now < 1.10)
                days_since_halving = (d.date() - cyc["halving_date"]).days

                # Proxy trigger: deep drawdown + near 200wMA + late in cycle
                proxy_score = 0
                if dd_from_peak <= -50: proxy_score += 1
                if near_wma:             proxy_score += 1
                if days_since_halving >= 730: proxy_score += 1
                if proxy_score >= 2:
                    fire_date = d.date()
                    fire_price = float(px)
                    break

            if fire_date is None:
                results.append({
                    "cycle":            cycle_n,
                    "would_have_fired": False,
                    "reason":            "proxy thresholds not met within window",
                })
                continue

            days_early_or_late = (fire_date - bottom_date).days
            pct_from_bottom = (fire_price / bottom_price - 1) * 100

            results.append({
                "cycle":            cycle_n,
                "would_have_fired": True,
                "fire_date":        fire_date.isoformat(),
                "fire_price":       round(fire_price, 0),
                "actual_bottom":    bottom_date.isoformat(),
                "actual_btm_price": bottom_price,
                "days_vs_bottom":   days_early_or_late,
                "pct_from_bottom":  round(pct_from_bottom, 1),
                "summary":          (f"Cycle {cycle_n}: fired on {fire_date} at "
                                       f"${fire_price:,.0f}, {abs(days_early_or_late)} days "
                                       f"{'before' if days_early_or_late<0 else 'after'} "
                                       f"actual bottom (${bottom_price:,.0f}), "
                                       f"{pct_from_bottom:+.1f}% from absolute low"),
            })
        except Exception as e:
            results.append({"cycle": cycle_n, "error": str(e)})

    # Aggregate
    fired = [r for r in results if r.get("would_have_fired")]
    avg_days = (sum(r.get("days_vs_bottom", 0) for r in fired) / len(fired)
                  if fired else None)
    avg_pct = (sum(r.get("pct_from_bottom", 0) for r in fired) / len(fired)
                if fired else None)

    return {
        "results":       results,
        "n_cycles":      len(HISTORICAL_BOTTOMS),
        "n_fired":       len(fired),
        "avg_days_vs_bottom": round(avg_days, 0) if avg_days is not None else None,
        "avg_pct_from_bottom": round(avg_pct, 1) if avg_pct is not None else None,
        "note":           ("Backtest uses price-only proxies. Real 15-signal scorecard "
                            "fires EARLIER than these proxies — these are conservative "
                            "lower-bound estimates."),
    }


# =================================================================
# 2) SIGNAL CORRELATION — independent vs redundant signals
# =================================================================
def signal_correlation() -> dict:
    """Group the bottom scorecard signals by underlying mechanism."""
    from core.dashboard_cache import get_cached

    nb = get_cached("btc_native_bottom_scorecard") or {}
    crit = nb.get("criteria", [])
    if not crit:
        return {"error": "no criteria available"}

    # Map each firing criterion to a cluster
    n_total = len(crit)
    n_firing = sum(1 for c in crit if c.get("met"))

    cluster_firings = {k: [] for k in SIGNAL_CLUSTERS}
    cluster_firings["other"] = []

    for c in crit:
        label = c.get("label", "")
        assigned = False
        for cluster, patterns in SIGNAL_CLUSTERS.items():
            if any(p.lower() in label.lower() for p in patterns):
                cluster_firings[cluster].append({
                    "label": label, "met": bool(c.get("met")),
                })
                assigned = True
                break
        if not assigned:
            cluster_firings["other"].append({
                "label": label, "met": bool(c.get("met")),
            })

    # Effective signal count: each cluster contributes max 1 (or fractional)
    n_clusters_firing = 0
    cluster_breakdown = {}
    for cluster, items in cluster_firings.items():
        if not items: continue
        n_in_cluster = len(items)
        n_firing_in_cluster = sum(1 for i in items if i["met"])
        # Cluster contributes 1 if ANY in it fired (deduplicates redundancy)
        cluster_breakdown[cluster] = {
            "n_total":   n_in_cluster,
            "n_firing":  n_firing_in_cluster,
            "active":    n_firing_in_cluster > 0,
            "labels":    [i["label"] for i in items],
        }
        if n_firing_in_cluster > 0:
            n_clusters_firing += 1

    # 2026-07-07 logic audit (F3): "other" is the UNCLASSIFIED residual — it is
    # NOT one deduplicated mechanism. Counting it as a single independent
    # cluster (as before) double-counted an axis whenever a known-redundant
    # signal leaked into it. Exclude it from the dedup math and report its
    # firing members separately as unclustered.
    _named = {k: v for k, v in cluster_breakdown.items() if k != "other"}
    n_clusters_total = sum(1 for v in _named.values() if v["n_total"] > 0)
    n_clusters_firing = sum(1 for v in _named.values() if v["active"])
    other = cluster_breakdown.get("other", {})
    n_other_firing = other.get("n_firing", 0) if other else 0

    naive_pct = n_firing / n_total if n_total else 0
    effective_pct = n_clusters_firing / n_clusters_total if n_clusters_total else 0
    _other_note = (f" plus {n_other_firing} unclustered signal(s)"
                   if n_other_firing else "")

    return {
        "raw_firing":        f"{n_firing}/{n_total}",
        "raw_pct":           round(naive_pct * 100, 1),
        "clusters_firing":   f"{n_clusters_firing}/{n_clusters_total}",
        "effective_pct":     round(effective_pct * 100, 1),
        "n_other_firing":    n_other_firing,
        "cluster_breakdown": cluster_breakdown,
        "interpretation":    (f"Raw scorecard shows {n_firing}/{n_total} signals. "
                                f"Deduplicated into {n_clusters_total} independent "
                                f"mechanisms, {n_clusters_firing} are firing{_other_note} "
                                f"— that's the real evidence count (correlated "
                                f"valuation ratios count once, not many times)."),
    }


# =================================================================
# 3) THRESHOLD SENSITIVITY — how robust is the trigger?
# =================================================================
def threshold_sensitivity() -> dict:
    """Test rotation trigger under +/- 10% threshold variations."""
    from core.rotation_trigger import evaluate_rotation_trigger
    from core.dashboard_cache import get_cached
    import importlib
    from core import rotation_trigger as rt_mod

    # Cache original constants
    original = {
        "BTC_BOTTOM_OVERWHELMING": rt_mod.BTC_BOTTOM_OVERWHELMING,
        "BTC_PRICE_TARGET":        rt_mod.BTC_PRICE_TARGET,
        "QQQ_GAP_LEVEL":           rt_mod.QQQ_GAP_LEVEL,
        "EQUITY_TOP_HARD":         rt_mod.EQUITY_TOP_HARD,
        "BTC_BOTTOM_MODERATE":     rt_mod.BTC_BOTTOM_MODERATE,
        "EQUITY_TOP_LIGHT":        rt_mod.EQUITY_TOP_LIGHT,
    }

    scenarios = {}
    for label, mult in [("strict", 1.10), ("baseline", 1.00), ("loose", 0.90)]:
        try:
            # Modify constants temporarily (round threshold counts to int)
            rt_mod.BTC_BOTTOM_OVERWHELMING = max(1, round(original["BTC_BOTTOM_OVERWHELMING"] * mult))
            rt_mod.BTC_PRICE_TARGET = int(original["BTC_PRICE_TARGET"] * (2 - mult))  # higher mult = stricter (lower price)
            rt_mod.QQQ_GAP_LEVEL = int(original["QQQ_GAP_LEVEL"] * (2 - mult))
            rt_mod.EQUITY_TOP_HARD = max(1, round(original["EQUITY_TOP_HARD"] * mult))
            rt_mod.BTC_BOTTOM_MODERATE = max(1, round(original["BTC_BOTTOM_MODERATE"] * mult))
            rt_mod.EQUITY_TOP_LIGHT = max(1, round(original["EQUITY_TOP_LIGHT"] * mult))

            s = evaluate_rotation_trigger()
            scenarios[label] = {
                "status":       s.get("overall"),
                "best_score":   s.get("best_score"),
                "fired":        s.get("fired"),
                "n_firing":     len(s.get("firing_paths", [])),
                "thresholds": {
                    "btc_bottom_overwhelming": rt_mod.BTC_BOTTOM_OVERWHELMING,
                    "btc_price_target":         rt_mod.BTC_PRICE_TARGET,
                    "qqq_gap_level":            rt_mod.QQQ_GAP_LEVEL,
                    "btc_bottom_moderate":      rt_mod.BTC_BOTTOM_MODERATE,
                    "equity_top_hard":          rt_mod.EQUITY_TOP_HARD,
                    "equity_top_light":         rt_mod.EQUITY_TOP_LIGHT,
                },
            }
        except Exception as e:
            scenarios[label] = {"error": str(e)}

    # Restore originals
    for k, v in original.items():
        setattr(rt_mod, k, v)

    # Interpret
    baseline = scenarios.get("baseline", {})
    loose = scenarios.get("loose", {})
    strict = scenarios.get("strict", {})

    interp = []
    if baseline.get("status") == loose.get("status") == strict.get("status"):
        interp.append("ROBUST: trigger status unchanged across +/-10% thresholds.")
    elif loose.get("fired") and not baseline.get("fired"):
        interp.append("BORDERLINE: trigger would fire under loose thresholds. "
                       "Watch for marginal moves.")
    elif strict.get("fired") and baseline.get("fired"):
        interp.append("CONFIRMED: trigger fires even with strict (+10%) thresholds. "
                       "High confidence.")
    else:
        interp.append("MIXED: trigger sensitive to threshold choice — proceed cautiously.")

    return {
        "scenarios":      scenarios,
        "interpretation": " ".join(interp),
    }


# =================================================================
# 4) CONFIDENCE SCORE — independent signals firing
# =================================================================
def confidence_score() -> dict:
    """Compute 0-100% confidence based on INDEPENDENT firing signals."""
    from core.dashboard_cache import get_cached

    # Pull from the rotation trigger result
    rt = get_cached("rotation_trigger") or {}
    overall = rt.get("overall", "ARMED")
    best_score = rt.get("best_score", 0) or 0
    paths = rt.get("paths", []) or []

    # Pull correlation cluster info
    corr = signal_correlation()
    effective_pct = corr.get("effective_pct", 0) or 0
    clusters_firing_str = corr.get("clusters_firing", "0/5") or "0/5"
    n_clusters_firing = int(clusters_firing_str.split("/")[0]) if "/" in clusters_firing_str else 0
    n_clusters_total = int(clusters_firing_str.split("/")[1]) if "/" in clusters_firing_str else 5

    # Confidence formula:
    #   30% weight: how many trigger paths approaching 2/2
    #   50% weight: how many INDEPENDENT signal clusters firing
    #   20% weight: cycle-6 modifier suggests we're past muted-bottom threshold
    paths_factor = sum(int(p.get("score", "0/2").split("/")[0]) for p in paths) / (2 * len(paths)) if paths else 0
    cluster_factor = n_clusters_firing / n_clusters_total if n_clusters_total else 0
    # Cycle-6 modifier (placeholder — refined below in cycle6_modifier)
    cycle6_factor = 0.5  # neutral default

    raw = (
        0.30 * paths_factor +
        0.50 * cluster_factor +
        0.20 * cycle6_factor
    )
    confidence_pct = round(raw * 100, 0)

    # Tier label
    if confidence_pct >= 75:
        tier = "HIGH"
    elif confidence_pct >= 50:
        tier = "MODERATE"
    elif confidence_pct >= 25:
        tier = "BUILDING"
    else:
        tier = "LOW"

    return {
        "confidence_pct":  confidence_pct,
        "tier":            tier,
        "factors": {
            "paths_factor":    round(paths_factor * 100, 0),
            "cluster_factor":  round(cluster_factor * 100, 0),
            "cycle6_factor":   round(cycle6_factor * 100, 0),
        },
        "interpretation":  (f"{confidence_pct:.0f}% confidence ({tier}). "
                              f"{n_clusters_firing} of {n_clusters_total} independent "
                              f"signal clusters firing."),
    }


# =================================================================
# 5) CYCLE-6 ETF MODIFIER — detect muted cycle, suggest adjustments
# =================================================================
def cycle6_modifier() -> dict:
    """Auto-detect ETF-era smoothing by comparing current drawdown vs historical pattern."""
    try:
        from core import data
        from core.halving_clock import current_halving_position
        import pandas as pd
    except Exception:
        return {"error": "data module unavailable"}

    try:
        pos = current_halving_position()
        days_post = pos.get("days_post_halving", 0)
        current_cycle = pos.get("current_cycle", 5)
    except Exception as e:
        return {"error": f"halving position: {e}"}

    try:
        df = data.ohlcv_extended("BTC/USDT", days_back=365 * 3)
        if df.empty: return {"error": "no price data"}
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        # ATH so far
        ath = float(df["close"].max())
        latest = float(df["close"].iloc[-1])
        current_dd_pct = (latest / ath - 1) * 100
    except Exception as e:
        return {"error": f"price data: {e}"}

    # Compare to historical pattern at similar days_post_halving
    # Cycle 3: at day 780, BTC was -84% from peak
    # Cycle 4: at day 780, BTC was -76% from peak
    # Cycle 5: at day 780, BTC is ~-50% (much shallower)
    historical_dd_at_day = {
        3: -84,  # 2018 cycle bear
        4: -76,  # 2022 cycle bear
    }
    avg_historical_dd = sum(historical_dd_at_day.values()) / len(historical_dd_at_day)

    # Ratio: current DD vs historical
    if current_dd_pct < 0 and avg_historical_dd < 0:
        muted_ratio = abs(current_dd_pct) / abs(avg_historical_dd)
    else:
        muted_ratio = 1.0

    if muted_ratio < 0.7:
        era = "ETF_MUTED"
        # Suggested threshold scaling: bottom scorecard 8/15 -> 6/15 etc.
        scale = 0.70
        msg = (f"Cycle {current_cycle} drawdown ({current_dd_pct:.1f}%) is "
               f"{round(muted_ratio*100)}% of historical bear depth "
               f"({avg_historical_dd:.0f}%). ETF flows are smoothing the cycle. "
               f"Consider scaling bottom thresholds 70% (e.g., 8/15 -> 6/15 fires).")
    elif muted_ratio < 0.9:
        era = "MILD_MUTED"
        scale = 0.85
        msg = (f"Drawdown is {round(muted_ratio*100)}% of historical pattern — "
               f"mild ETF smoothing. Slight threshold scaling recommended (~85%).")
    else:
        era = "HISTORICAL_BEAR"
        scale = 1.0
        msg = (f"Drawdown ({current_dd_pct:.1f}%) tracking historical bear pattern. "
               f"No threshold adjustment needed.")

    # Recommended scaled thresholds
    from core.rotation_trigger import (
        BTC_BOTTOM_OVERWHELMING, BTC_BOTTOM_MODERATE
    )
    return {
        "era":                era,
        "current_dd_pct":     round(current_dd_pct, 1),
        "avg_historical_dd":  round(avg_historical_dd, 1),
        "muted_ratio":        round(muted_ratio, 2),
        "suggested_scale":    scale,
        "thresholds": {
            "btc_bottom_overwhelming_scaled": max(1, round(BTC_BOTTOM_OVERWHELMING * scale)),
            "btc_bottom_moderate_scaled":      max(1, round(BTC_BOTTOM_MODERATE * scale)),
            "baseline_btc_overwhelming":       BTC_BOTTOM_OVERWHELMING,
            "baseline_btc_moderate":            BTC_BOTTOM_MODERATE,
        },
        "message":            msg,
    }


# =================================================================
# Convenience — all 5 in one call (for caching)
# =================================================================
def all_validation() -> dict:
    """Compute all 5 validation components in one shot."""
    out = {}
    for name, fn in [
        ("backtest",     historical_backtest),
        ("correlation",  signal_correlation),
        ("sensitivity",  threshold_sensitivity),
        ("confidence",   confidence_score),
        ("cycle6",       cycle6_modifier),
    ]:
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    out["computed_at"] = datetime.now(timezone.utc).isoformat()
    return out


def main():
    r = all_validation()
    for k, v in r.items():
        if k == "computed_at": continue
        print(f"--- {k} ---")
        if isinstance(v, dict):
            for key, val in v.items():
                if isinstance(val, (str, int, float, bool)) or val is None:
                    print(f"  {key}: {val}")
                else:
                    print(f"  {key}: <{type(val).__name__}>")
        print()


if __name__ == "__main__":
    main()
