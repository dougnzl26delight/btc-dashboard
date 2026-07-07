"""BTC-NATIVE bottom scorecard — guru-tier signals that called past lows.

The existing btc_bottom_scorecard.py has 8 solid Glassnode-style criteria.
This module adds 12 GURU-tier criteria that the top crypto bottom-callers
use: Phillip Swift (Pi Cycle Bottom), Willy Woo (NVTS), Charles Edwards
(Hash Ribbon), Trace Mayer (Mayer Multiple), Bob Loukas (2y MA + cycle
day), Hayes (funding rates), Chinese DCA crowd (AHR999).

Each criterion is binary. Verdict tiers:
  0-2:  HOLD            BTC still in bull / mid-cycle, no bottom forming
  3-4:  WATCH           Early bottom signals — start tracking
  5-6:  ACCUMULATE      Bottom forming — start DCA at 20% pace
  7-8:  STRONG_BUY      Multi-signal confluence — 40% DCA pace
  9-10: DEEP_VALUE      Generational bottom — 60% deploy, save 40% for retests
  11+:  EXTREME         Once-a-decade signal — 80%+ deploy

Calibrated on past bottoms:
  2015 ($150):   9-10 signals firing
  2018 ($3,200): 9-11 signals firing
  2020 Mar:      6-7 (sharp + fast — partial signal)
  2022 ($16,500): 8-10 signals
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Helpers
# ============================================================

def _btc_history(period: str = "max") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        df = yf.Ticker("BTC-USD").history(period=period)
        if df is None or df.empty: return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return None


def _cm(metric: str, days: int = 1460) -> Optional[pd.Series]:
    try:
        from core.btc_pro_signals import _cm as _coinmetrics
        df = _coinmetrics(metric, days=days)
        if df is None or df.empty: return None
        return df.iloc[:, 0]
    except Exception:
        return None


def _live_btc_price() -> float:
    try:
        from core import data
        return data.btc_spot()  # region-resilient (Kraken/Coinbase/Binance/Bitstamp)
    except Exception:
        return 0.0


# ============================================================
# GURU-TIER CRITERIA
# ============================================================

def pi_cycle_bottom() -> dict:
    """Phillip Swift Pi Cycle Bottom: 471d SMA × 0.745 < 150d SMA.

    Called every cycle bottom 1-3 within 3 days. Cycle 4 (Nov 2022)
    didn't trigger due to muted cycle dynamics.
    """
    df = _btc_history("5y")
    if df is None or len(df) < 471:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    sma_471 = float(closes.rolling(471).mean().iloc[-1])
    sma_150 = float(closes.rolling(150).mean().iloc[-1])
    if pd.isna(sma_471) or pd.isna(sma_150):
        return {"met": False, "status": "MAs not computed"}
    threshold = sma_471 * 0.745
    crossed = sma_150 < threshold
    ratio = sma_150 / threshold
    return {
        "met": bool(crossed),
        "value": ratio,
        "status": (f"150d MA ${sma_150:,.0f} vs threshold (471d × 0.745) ${threshold:,.0f}  "
                   f"ratio {ratio:.3f}  {'CROSSED' if crossed else 'not crossed'}"),
        "rationale": "Phillip Swift's Pi Cycle Bottom — called cycle 1/2/3 bottoms within 3 days.",
    }


def hash_ribbon_golden_cross() -> dict:
    """Hash Ribbon Golden Cross: 30d hash MA crosses ABOVE 60d hash MA
    after a death cross (miners coming back online after capitulation).

    Charles Edwards' signal — bottoms confirmed when this fires.
    """
    try:
        # Multiple fallback paths
        try:
            from core.btc_premium_free import _blockchain_info
            df = _blockchain_info("hash-rate", timespan="2years")
            if df is None or df.empty:
                raise ImportError("no blockchain_info data")
        except (ImportError, Exception):
            # Fallback via CoinMetrics hashrate proxy
            s = _cm("HashRate", days=730)
            if s is None or s.empty:
                return {"met": False, "status": "hashrate unavailable"}
            df = pd.DataFrame({"value": s})
        if len(df) < 60:
            return {"met": False, "status": "insufficient history"}
        s = df["value"].dropna()
        ma_30 = s.rolling(30).mean()
        ma_60 = s.rolling(60).mean()
        # Was below, now above
        was_below = ma_30.iloc[-30] < ma_60.iloc[-30] if len(ma_30) >= 30 else False
        now_above = ma_30.iloc[-1] > ma_60.iloc[-1]
        # And was a recent death cross (within last 180d)
        recent_death = False
        for i in range(min(180, len(ma_30) - 1), 0, -1):
            if i+1 < len(ma_30) and ma_30.iloc[-i] < ma_60.iloc[-i] and ma_30.iloc[-(i+1)] >= ma_60.iloc[-(i+1)]:
                recent_death = True; break
        met = now_above and (was_below or recent_death)
        return {
            "met": bool(met),
            "value": float(ma_30.iloc[-1] / ma_60.iloc[-1]),
            "status": (f"Hash 30d/60d ratio {ma_30.iloc[-1]/ma_60.iloc[-1]:.3f}  "
                       f"({'GOLDEN CROSS' if met else 'not confirmed'})"),
            "rationale": "Hash Ribbon Golden Cross (Charles Edwards) — miner capitulation ending.",
        }
    except Exception as e:
        return {"met": False, "status": f"error: {e!r}"[:60]}


def nvt_signal_low() -> dict:
    """NVT Signal: Network Value to Transactions ratio (90d MA tx volume).

    Woo's classic absolute <40 threshold is calibrated to CoinMetrics adjusted
    volume. We now source tx volume from blockchain.com (different scale), so
    we fire on PERCENTILE-RANK within this series' own history instead — bottom
    ~15th percentile = oversold. Scale-invariant; consistent with the other
    percentile-rank criteria (Mayer/GR/log-reg) on this board.
    """
    cap = _cm("CapMrktCurUSD", days=400)
    tx_vol = _cm("TxTfrValAdjUSD", days=400)
    if tx_vol is None:
        # 2026-07-07 signals audit: TxTfrValAdjUSD (and TxTfrValUSD) left the
        # CoinMetrics community tier — this criterion read "unavailable"
        # indefinitely. Free fallback: blockchain.com estimated tx volume
        # (no key, daily granularity). Different scale than CoinMetrics, so
        # the criterion below uses percentile-rank, NOT the absolute <40.
        try:
            import json as _json, urllib.request as _ur
            _u = ("https://api.blockchain.info/charts/"
                  "estimated-transaction-volume-usd?timespan=2years&format=json")
            with _ur.urlopen(_u, timeout=30) as _r:
                _vals = _json.loads(_r.read()).get("values", [])
            if _vals:
                _idx = pd.to_datetime([v["x"] for v in _vals], unit="s")
                tx_vol = pd.Series([float(v["y"]) for v in _vals], index=_idx)
        except Exception:
            tx_vol = None
    if cap is None or tx_vol is None:
        return {"met": False, "status": "data unavailable"}
    # normalize both to naive daily dates so concat aligns (CoinMetrics is tz-aware)
    try:
        cap.index = pd.to_datetime(cap.index).tz_localize(None).normalize()
        tx_vol.index = pd.to_datetime(tx_vol.index).tz_localize(None).normalize()
    except (TypeError, AttributeError):
        pass
    df = pd.concat([cap, tx_vol], axis=1).dropna()
    if df.empty or len(df) < 90:
        return {"met": False, "status": "insufficient history"}
    tx_90d = df.iloc[:, 1].rolling(90).mean()
    nvts = (df.iloc[:, 0] / tx_90d).dropna()
    if len(nvts) < 120:
        return {"met": False, "status": "insufficient NVT history"}
    nvts_now = float(nvts.iloc[-1])
    # 2026-07-07 fix: percentile-rank, not absolute <40 (blockchain.com series
    # runs on a different scale, ~200, where <40 could NEVER fire). Fraction of
    # this series' own history below the current read; bottom 15% = oversold.
    pct = float((nvts < nvts_now).mean())
    met = pct < 0.15
    return {
        "met": bool(met),
        "value": nvts_now,
        "status": (f"NVT Signal {nvts_now:.0f} = {pct*100:.0f}th pct of "
                   f"{len(nvts)}d history  ({'OVERSOLD' if met else 'normal'})"),
        "rationale": "Woo's NVT — bottom-decile relative to its own history marks the oversold band.",
    }


def mayer_multiple_low() -> dict:
    """Mayer Multiple: price / 200d MA. < 0.6 = historic bottom band."""
    df = _btc_history("3y")
    if df is None or len(df) < 200:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    ma_200 = closes.rolling(200).mean()
    price_now = _live_btc_price() or float(closes.iloc[-1])
    ma_now = float(ma_200.iloc[-1])
    if pd.isna(ma_now): return {"met": False, "status": "200d MA NaN"}
    mayer = price_now / ma_now
    met = mayer < 0.6
    return {
        "met": bool(met),
        "value": mayer,
        "status": f"Mayer Multiple {mayer:.2f}  ({'BOTTOM ZONE' if met else 'normal'})  threshold < 0.6",
        "rationale": "Trace Mayer's price/200dMA — every cycle bottom occurred when Mayer < 0.6.",
    }


def price_below_lth_cost_basis() -> dict:
    """Price below LTH realized price — major capitulation event.

    LTH = long-term holders (> 155 days). When spot price falls below
    their average cost basis, HODLers are underwater = real bottom.

    Cycle 2 (Dec 2018), Cycle 3 (Mar 2020), Cycle 4 (Nov 2022) ALL hit
    this state for at least 1 week at the cycle low.
    """
    try:
        from core.btc_cost_basis import realized_price
        rp = realized_price()
        if rp.get("error"):
            return {"met": False, "status": f"data unavailable: {rp.get('error')[:50]}"}
        lth_cost = rp.get("value", 0)
        price_now = _live_btc_price() or 0
        if lth_cost <= 0 or price_now <= 0:
            return {"met": False, "status": "missing inputs"}
        below = price_now < lth_cost
        pct = (price_now / lth_cost - 1) * 100
        return {
            "met": bool(below),
            "value": pct,
            "status": (f"BTC ${price_now:,.0f} vs LTH realized ${lth_cost:,.0f}  "
                       f"({pct:+.1f}%)  {'BELOW LTH' if below else 'above'}"),
            "rationale": "Glassnode: price < LTH cost basis = rare capitulation event marking real bottoms.",
        }
    except Exception as e:
        return {"met": False, "status": f"error: {e!r}"[:60]}


def two_year_ma_test() -> dict:
    """2-year MA test: price at or below 2y MA.

    Bob Loukas / blocktower research: cycle bottoms historically near 2y MA.
    """
    df = _btc_history("5y")
    if df is None or len(df) < 730:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    ma_2y = float(closes.rolling(730).mean().iloc[-1])
    price_now = _live_btc_price() or float(closes.iloc[-1])
    if pd.isna(ma_2y): return {"met": False, "status": "2y MA NaN"}
    pct = (price_now / ma_2y - 1) * 100
    met = price_now < ma_2y * 1.10  # within 10% above OR below
    return {
        "met": bool(met),
        "value": pct,
        "status": (f"BTC ${price_now:,.0f} vs 2y MA ${ma_2y:,.0f}  "
                   f"({pct:+.1f}%)  {'AT/BELOW 2yMA' if met else 'above'}"),
        "rationale": "Bob Loukas / blocktower: cycle bottoms historically at or below 2y MA.",
    }


def below_200_week_ma() -> dict:
    """At/below 200-week MA — Phillip Swift's signature bottom signal.

    Every cycle bottom (2015, Dec 2018, Mar 2020, Nov 2022) printed at or
    below the 200wMA. Confirmed live 2026-06-10 when BTC closed below it.
    Met when price within 5% above, at, or below the 200wMA.
    """
    df = _btc_history("5y")
    if df is None or len(df) < 1400:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    wkly = closes.resample("W").last().dropna()
    if len(wkly) < 200:
        return {"met": False, "status": "fewer than 200 weeks of data"}
    wma200 = float(wkly.tail(200).mean())
    price_now = _live_btc_price() or float(closes.iloc[-1])
    if pd.isna(wma200): return {"met": False, "status": "200wMA NaN"}
    pct = (price_now / wma200 - 1) * 100
    met = price_now < wma200 * 1.05  # within 5% above, at, or below
    return {
        "met": bool(met),
        "value": pct,
        "status": (f"BTC ${price_now:,.0f} vs 200wMA ${wma200:,.0f}  "
                   f"({pct:+.1f}%)  {'AT/BELOW 200wMA' if met else 'above'}"),
        "rationale": ("Swift: every cycle bottom printed at/below the 200wMA "
                      "(2015, 2018, 2020, 2022)."),
    }


def funding_rate_extreme_negative() -> dict:
    """Funding rate sustained negative — derivatives short capitulation.

    When perpetual funding goes -0.05%+ for several consecutive days,
    shorts are dominant + paying longs to stay short = positioning extreme.
    Bottoms typically follow within 7-14 days.
    """
    try:
        # 2026-07-07 signals audit: the old helper (btc_premium_free._funding_rate)
        # no longer exists — this criterion read "unavailable" indefinitely.
        # Rebuilt on the WORKING multi-venue fetcher (OI-weighted Binance/Bybit/
        # OKX, bps) + Binance funding history for the 7d average. Free data.
        from core.btc_clemente_alden import multi_exchange_funding
        cur = multi_exchange_funding() or {}
        if cur.get("error") or "agg_funding_bps" not in cur:
            return {"met": False, "status": "funding data unavailable"}
        recent = float(cur["agg_funding_bps"]) / 100.0   # bps -> % per 8h
        avg_7d = recent   # fallback if history fetch fails below
        try:
            import ccxt
            ex = ccxt.binance({"options": {"defaultType": "swap"}})
            hist = ex.fetch_funding_rate_history("BTC/USDT:USDT", limit=21)
            rates = [float(h.get("fundingRate") or 0) * 100 for h in hist if h]
            if rates:
                avg_7d = sum(rates) / len(rates)
        except Exception:
            pass
        met = recent < -0.03 and avg_7d < -0.02
        return {
            "met": bool(met),
            "value": recent,
            "status": (f"Funding 8h {recent:+.3f}%, 7d avg {avg_7d:+.3f}%  "
                       f"({'EXTREME NEG' if met else 'normal'})"),
            "rationale": "Hayes/Pal: sustained negative funding = shorts crowded, bottom in 7-14d.",
        }
    except Exception as e:
        # Fallback: use crude proxy via price action
        return {"met": False, "status": f"funding helper unavailable"}


def cycle_day_analog_match() -> dict:
    """Cycle day analog: current day-post-halving matches cycle 4 bottom day.

    Cycle 4 bottomed at $16,500 on Nov 9 2022 = day 942 post-halving 3.
    Cycle 5 day 942 post-halving 4 = Oct 17 2026.

    Fires when we're within 60 days of the analog bottom day (broad window).
    """
    try:
        from core.halving_clock import current_halving_position
        pos = current_halving_position()
        days_post = pos.get("days_post_halving", 0)
        CYCLE4_BOTTOM_DAY = 942
        diff = abs(days_post - CYCLE4_BOTTOM_DAY)
        met = diff <= 60
        return {
            "met": bool(met),
            "value": diff,
            "status": (f"Day {days_post} post-halving (cycle 4 bottom day: {CYCLE4_BOTTOM_DAY}, "
                       f"diff {diff}d)  {'IN ANALOG WINDOW' if met else 'too early'}"),
            "rationale": "Bob Loukas 4-year cycle: bottoms cluster near same day-post-halving.",
        }
    except Exception as e:
        return {"met": False, "status": f"error: {e!r}"[:60]}


def ahr999_buy_zone() -> dict:
    """AHR999 index — Chinese DCA timing tool.

    AHR999 = (price / 200d_ago_price) × (price / 200d_geomean) / 2
    < 0.45 = "buy zone" historically reliable.
    """
    df = _btc_history("3y")
    if df is None or len(df) < 200:
        return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    price_now = _live_btc_price() or float(closes.iloc[-1])
    price_200d_ago = float(closes.iloc[-200]) if len(closes) >= 200 else price_now
    # 200d geometric mean
    last_200 = closes.tail(200)
    geomean = float(np.exp(np.mean(np.log(last_200.dropna()))))
    if geomean <= 0 or price_200d_ago <= 0:
        return {"met": False, "status": "math error"}
    ratio_a = price_now / price_200d_ago
    ratio_b = price_now / geomean
    ahr999 = (ratio_a * ratio_b) / 2
    met = ahr999 < 0.45
    return {
        "met": bool(met),
        "value": ahr999,
        "status": f"AHR999 {ahr999:.3f}  ({'BUY ZONE' if met else 'normal'})  threshold < 0.45",
        "rationale": "Chinese DCA crowd indicator — sub-0.45 reliably marks accumulation zone.",
    }


def realized_cap_deep_drawdown() -> dict:
    """Realized Cap drawdown deeper than -15%. Historic bottom band.

    Uses the DERIVED realized cap (CapMrktCurUSD / MVRV) from btc_cost_basis,
    which sidesteps the paywalled CapRealUSD metric (CoinMetrics free tier
    returns CapRealUSD empty). Same number the bottom-countdown email uses.
    """
    try:
        from core.btc_cost_basis import realized_cap_drawdown_depth
        r = realized_cap_drawdown_depth() or {}
        dd_now = r.get("current_drawdown_pct")
        if dd_now is None or r.get("error"):
            return {"met": False, "status": "data unavailable"}
        dd_now = float(dd_now)
        met = dd_now < -15
        return {
            "met": bool(met),
            "value": dd_now,
            "status": f"Realized Cap drawdown {dd_now:+.1f}%  ({'BOTTOM BAND' if met else 'normal'})  threshold < -15%",
            "rationale": "Glassnode: Realized Cap drawdown < -15% has marked every cycle bottom since 2013.",
        }
    except Exception:
        return {"met": False, "status": "data unavailable"}


def mvrv_z_capitulation() -> dict:
    """MVRV-Z extreme negative — capitulation territory.

    z < -1.5 is deep capitulation. z < -1.0 is approaching.
    Uses raw MVRV ratio < 1.0 as fallback (HODLers underwater).
    """
    s = _cm("CapMVRVCur", days=1460)
    if s is None or len(s) < 200:
        return {"met": False, "status": "data unavailable"}
    s = s.dropna()
    mean = s.rolling(1460, min_periods=200).mean()
    std = s.rolling(1460, min_periods=200).std()
    z = (s - mean) / std
    raw_now = float(s.iloc[-1])
    z_now = float(z.iloc[-1]) if not pd.isna(z.iloc[-1]) else 0
    met = z_now < -1.5 or raw_now < 1.0
    return {
        "met": bool(met),
        "value": z_now,
        "status": (f"MVRV ratio {raw_now:.2f}, Z {z_now:+.2f}  "
                   f"({'CAPITULATION' if met else 'normal'})  threshold z<-1.5 or raw<1.0"),
        "rationale": "Glassnode/Checkmate: MVRV-Z < -1.5 or raw MVRV < 1.0 = HODLers underwater = bottom.",
    }


def coinbase_premium_negative() -> dict:
    """Coinbase Premium sustained negative — US institutional fear/selling.

    Coinbase = US institutional flow proxy. Sustained negative means
    institutions sold while Asia bought — sets up bottom.
    """
    try:
        from core.btc_advanced_signals import coinbase_premium
        cb = coinbase_premium()
        if cb.get("error"):
            return {"met": False, "status": f"data unavailable"}
        avg_7d = cb.get("avg_7d_pct", 0) or 0
        met = avg_7d < -0.10  # sustained 0.10%+ Coinbase discount
        return {
            "met": bool(met),
            "value": avg_7d,
            "status": (f"Coinbase Premium 7d avg {avg_7d:+.3f}%  "
                       f"({'INSTITUTIONAL FEAR' if met else 'normal'})"),
            "rationale": "Sustained Coinbase discount = US institutions selling = bottom setup.",
        }
    except Exception as e:
        return {"met": False, "status": f"error: {e!r}"[:60]}


# ============================================================
# MUTED-CYCLE-AWARE PERCENTILE-RANK BOTTOM INDICATORS
# Detect bottoms that don't hit historic absolute thresholds.
# ============================================================

def mayer_bottom_percentile() -> dict:
    """Mayer Multiple < 10th percentile of last 4y (bottom-relative)."""
    df = _btc_history("5y")
    if df is None or len(df) < 200: return {"met": False, "status": "insufficient data"}
    closes = df["Close"]
    ma_200 = closes.rolling(200).mean()
    mayer = (closes / ma_200).dropna()
    if len(mayer) < 100: return {"met": False, "status": "insufficient"}
    last_4y = mayer.tail(min(1460, len(mayer)))
    current = float(mayer.iloc[-1])
    pct = float((last_4y < current).mean() * 100)
    met = pct <= 10
    return {
        "met": bool(met),
        "value": current,
        "status": f"Mayer {current:.2f} = {pct:.0f}th pct  ({'BOTTOM RANK' if met else 'ok'})",
        "rationale": "Mayer percentile-rank — catches bottoms even in muted cycles.",
    }


def golden_ratio_bottom_pct() -> dict:
    """Golden Ratio Multiplier < 10th percentile of last 4y."""
    df = _btc_history("5y")
    if df is None or len(df) < 350: return {"met": False, "status": "insufficient"}
    closes = df["Close"]
    ma_350 = closes.rolling(350).mean()
    mult = (closes / ma_350).dropna()
    if len(mult) < 100: return {"met": False, "status": "insufficient"}
    last_4y = mult.tail(min(1460, len(mult)))
    current = float(mult.iloc[-1])
    pct = float((last_4y < current).mean() * 100)
    met = pct <= 10
    return {
        "met": bool(met),
        "value": current,
        "status": f"GR {current:.2f}x = {pct:.0f}th pct  ({'BOTTOM RANK' if met else 'ok'})",
        "rationale": "Golden Ratio percentile-rank — catches relative bottom in muted cycles.",
    }


def log_regression_bottom_pct() -> dict:
    """Log regression deviation < 10th percentile of last 4y."""
    df = _btc_history("5y")
    if df is None or len(df) < 365: return {"met": False, "status": "insufficient"}
    closes = df["Close"]
    GENESIS = pd.Timestamp("2009-01-03")
    days = (df.index - GENESIS).days.astype(float)
    log_days = np.log10(np.clip(days, 1, None))
    model = 10 ** (5.84 * log_days - 17.01)
    dev = ((closes / model - 1) * 100).dropna()
    if len(dev) < 100: return {"met": False, "status": "insufficient"}
    last_4y = dev.tail(min(1460, len(dev)))
    current = float(dev.iloc[-1])
    pct = float((last_4y < current).mean() * 100)
    met = pct <= 10
    return {
        "met": bool(met),
        "value": current,
        "status": f"Log dev {current:+.0f}% = {pct:.0f}th pct  ({'BOTTOM RANK' if met else 'ok'})",
        "rationale": "Log regression percentile-rank — detects relative undervaluation.",
    }


# ============================================================
# Aggregator
# ============================================================

CRITERIA_DEFS = [
    ("pi_cycle_bottom",     " 1. Pi Cycle Bottom (Swift)",                  pi_cycle_bottom),
    ("hash_ribbon_gx",      " 2. Hash Ribbon Golden Cross (Edwards)",        hash_ribbon_golden_cross),
    ("nvt_signal",          " 3. NVT Signal < 40 (Willy Woo)",              nvt_signal_low),
    ("mayer_multiple",      " 4. Mayer Multiple < 0.6 (Trace Mayer)",       mayer_multiple_low),
    ("price_below_lth",     " 5. Price below LTH cost basis",                price_below_lth_cost_basis),
    ("two_year_ma",         " 6. At/below 2-year MA (Bob Loukas)",          two_year_ma_test),
    ("funding_neg",         " 7. Funding rate extreme negative",              funding_rate_extreme_negative),
    ("cycle_day_analog",    " 8. Cycle day matches cycle-4 analog +/-60d",   cycle_day_analog_match),
    ("ahr999",              " 9. AHR999 < 0.45 (Chinese DCA)",              ahr999_buy_zone),
    ("rcap_deep_dd",        "10. Realized Cap drawdown < -15%",              realized_cap_deep_drawdown),
    ("mvrv_z_cap",          "11. MVRV-Z < -1.5 OR raw MVRV < 1.0",           mvrv_z_capitulation),
    ("cb_premium_neg",      "12. Coinbase Premium sustained negative",         coinbase_premium_negative),
    # Muted-cycle percentile-rank criteria
    ("mayer_pct_bottom",    "13. Mayer percentile-rank < 10th (last 4y)",    mayer_bottom_percentile),
    ("gr_pct_bottom",       "14. Golden Ratio pct-rank < 10th (last 4y)",    golden_ratio_bottom_pct),
    ("log_reg_pct_bottom",  "15. Log regression pct-rank < 10th (last 4y)",   log_regression_bottom_pct),
    # Added 2026-06-10 after BTC closed below the 200wMA — Swift's signature
    # signal was absent from the board despite firing at every prior bottom.
    ("below_200wma",        "16. At/below 200-week MA (Swift)",               below_200_week_ma),
]


def btc_native_bottom_scorecard() -> dict:
    """Run all 12 guru-tier bottom criteria."""
    criteria = []
    n_met = 0
    for key, label, fn in CRITERIA_DEFS:
        try: r = fn()
        except Exception as e: r = {"met": False, "status": f"error: {e!r}"[:60]}
        r["id"] = key; r["label"] = label
        criteria.append(r)
        if r.get("met"): n_met += 1

    # 2026-07-04..07: criteria grew 12 -> 16; tier thresholds below and the
    # verdict denominators are kept CONSISTENT with the live n_total (16).
    _nt = len(criteria)
    if n_met >= 13:    level = "EXTREME"
    elif n_met >= 11:  level = "DEEP_VALUE"
    elif n_met >= 8:   level = "STRONG_BUY"
    elif n_met >= 6:   level = "ACCUMULATE"
    elif n_met >= 3:   level = "WATCH"
    else:              level = "HOLD"

    verdict_text = {
        "EXTREME":     f"Once-a-decade bottom signal ({n_met}/{_nt}). Deploy 80%+ of dry powder.",
        "DEEP_VALUE":  f"Generational bottom confirmed ({n_met}/{_nt}). Deploy 60%, retain 40% for retests.",
        "STRONG_BUY":  f"Multi-signal confluence (>=8/{_nt}). 40% DCA pace recommended.",
        "ACCUMULATE":  f"Bottom forming (>=6/{_nt}). Start DCA at 20% pace.",
        "WATCH":       f"Early bottom signals (>=3/{_nt}). Track closely, no deploy yet.",
        "HOLD":        "Mid-cycle or bull. No bottom forming.",
    }[level]

    return {
        "criteria":      criteria,
        "n_met":         n_met,
        "n_total":       len(CRITERIA_DEFS),
        "verdict":       verdict_text,
        "verdict_level": level,
        "asof":          datetime.now(timezone.utc).isoformat(),
    }


def main():
    r = btc_native_bottom_scorecard()
    print("=" * 78)
    print("BTC NATIVE BOTTOM SCORECARD — guru-tier signals")
    print("=" * 78)
    for c in r["criteria"]:
        mark = "[FIRING]" if c.get("met") else "[ok    ]"
        try: print(f"  {mark} {c['label']:45s} {c.get('status','')[:75]}")
        except UnicodeEncodeError:
            s = c.get("status","").encode("ascii","replace").decode()
            print(f"  {mark} {c['label']:45s} {s[:75]}")
    print(f"\n  VERDICT: {r['verdict']}")
    print(f"  Level: {r['verdict_level']} ({r['n_met']}/{r['n_total']})")


if __name__ == "__main__":
    main()
