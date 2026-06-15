"""BTC-vs-equity relative attractiveness.

The rotation's whole premise is "BTC out-runs equities from here" — yet the
trigger only ever measures equity STRESS, never BTC's relative cheapness
directly. This makes the premise observable: how dislocated is BTC vs equities
(relative drawdown), and is the BTC/QQQ ratio stretched below its own trend.
Higher score = BTC more attractive vs equities. Free data only.
"""
from __future__ import annotations


def _close(ticker, period="2y"):
    import yfinance as yf
    return yf.Ticker(ticker).history(period=period, interval="1d")["Close"].dropna()


def _drawdown(s):
    return float(s.iloc[-1] / s.cummax().iloc[-1] - 1)


def btc_equity_relative_value() -> dict:
    out = {"score": None, "tier": "UNKNOWN", "components": {}, "detail": ""}
    try:
        btc, spx, qqq = _close("BTC-USD"), _close("^GSPC"), _close("QQQ")

        btc_dd, spx_dd = _drawdown(btc), _drawdown(spx)
        rel_dd = btc_dd - spx_dd                         # more negative = BTC much more drawn down

        rdf = btc.to_frame("btc").join(qqq.to_frame("qqq"), how="inner").dropna()
        ratio = (rdf["btc"] / rdf["qqq"]).dropna()
        sma200 = ratio.rolling(200).mean()
        sd = ratio.tail(200).std()
        ratio_z = float((ratio.iloc[-1] - sma200.iloc[-1]) / sd) if sd and sd > 0 else 0.0

        # 0..100, higher = BTC cheaper vs equities (deeper relative dd + ratio below trend)
        s_dd = min(1.0, max(0.0, (-rel_dd) / 0.40))      # BTC 40pp more drawn down -> 1.0
        s_ratio = min(1.0, max(0.0, (-ratio_z) / 2.0))   # ratio 2σ below trend -> 1.0
        score = round(100 * (0.6 * s_dd + 0.4 * s_ratio))

        out["score"] = score
        out["tier"] = ("BTC CHEAP vs equities" if score >= 66 else
                       "BTC neutral vs equities" if score >= 33 else
                       "BTC rich vs equities")
        out["components"] = {
            "btc_dd_pct": round(btc_dd * 100, 0),
            "spx_dd_pct": round(spx_dd * 100, 0),
            "rel_dd_pp": round(rel_dd * 100, 0),
            "btc_qqq_ratio_z": round(ratio_z, 2),
        }
        out["detail"] = (f"BTC {btc_dd*100:+.0f}% vs SPX {spx_dd*100:+.0f}% drawdown; "
                         f"BTC/QQQ ratio {ratio_z:+.1f}sigma vs 200d trend")
    except Exception as e:
        out["detail"] = f"unavailable ({type(e).__name__})"
    return out


if __name__ == "__main__":
    r = btc_equity_relative_value()
    print(f"BTC-vs-EQUITY: {r['score']} -> {r['tier']}")
    print(f"  {r['detail']}")
    print(f"  {r['components']}")
