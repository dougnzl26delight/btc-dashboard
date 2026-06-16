"""Per-cycle PEAK readings for the classic cycle metrics — MVRV, Pi Cycle ratio,
price/350d-MA, 2-yr-MA multiple, Puell — computed from our own data so the
"cycle amplitude compression" proof table on the Charts tab stays live each
refresh (the current cycle's row updates as it evolves).

Output is PURE PYTHON (floats / strs / None — no numpy arrays or pandas objects)
so the pickled panel is immune to numpy/plotly version drift on the cloud
(see the swift_charts numpy-pickle lesson). Sources: CoinMetrics PriceUSD +
CapMVRVCur (daily, back to 2010) and blockchain.com miner revenue (Puell).
"""
from __future__ import annotations

import pandas as pd
import requests


def _fmt_price(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "—"
    if v >= 1000:
        return f"${v / 1000:.1f}k"
    return f"${v:,.0f}"


def _round(v, n=2):
    """Coerce to a plain Python float (drops numpy types) or None."""
    try:
        f = float(v)
    except Exception:
        return None
    return round(f, n) if pd.notna(f) else None


def _puell_series():
    """Puell Multiple = daily miner revenue / its trailing 365d mean (blockchain.com)."""
    try:
        r = requests.get(
            "https://api.blockchain.info/charts/miners-revenue"
            "?timespan=all&format=json&sampled=false",
            timeout=40, headers={"User-Agent": "Mozilla/5.0"},
        )
        vals = r.json().get("values", [])
        s = pd.Series({pd.to_datetime(d["x"], unit="s"): float(d["y"]) for d in vals}).sort_index()
        s = s[s > 0]
        if s.empty:
            return None
        return s / s.rolling(365).mean()
    except Exception:
        return None


def cycle_peak_table() -> dict:
    """Return {'rows': [...], 'asof': 'YYYY-MM-DD'} — per-cycle peak metric readings.

    Each row: {cycle, price, price_fmt, mvrv, pi_cycle, p350, p730, puell}.
    """
    from core.btc_pro_signals import _cm

    def _series(metric):
        x = _cm(metric, days=6200)
        if x is None:
            return None
        s = x.iloc[:, 0] if isinstance(x, pd.DataFrame) else pd.Series(x)
        s = s.dropna()
        if s.empty:
            return None
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    price = _series("PriceUSD")
    if price is None or price.empty:
        return {"rows": [], "asof": None, "error": "no price data"}
    mvrv = _series("CapMVRVCur")

    df = pd.DataFrame({"price": price, "mvrv": mvrv}).sort_index()
    df["ma111"] = df["price"].rolling(111).mean()
    df["ma350"] = df["price"].rolling(350).mean()
    df["ma730"] = df["price"].rolling(730).mean()
    df["pi_ratio"] = df["ma111"] / (2 * df["ma350"])   # Pi Cycle Top: >=1 = top
    df["p350"] = df["price"] / df["ma350"]              # Golden-Ratio / Rainbow proxy
    df["p730"] = df["price"] / df["ma730"]              # 2-Yr MA multiplier

    puell = _puell_series()  # kept as its own series; sliced per-cycle below (avoids reindex misalignment)

    # Current-cycle label adapts to the latest peak year (e.g. "2024-25").
    cur_label = "2024-now"
    try:
        cur = df.loc["2024-01-01":, "price"]
        if not cur.empty:
            yr = cur.idxmax().year % 100
            cur_label = f"2024-{yr:02d}"
    except Exception:
        pass

    cycles = [
        ("2013",    "2013-01-01", "2014-02-28"),
        ("2017",    "2017-06-01", "2018-02-28"),
        ("2021",    "2021-01-01", "2021-12-31"),
        (cur_label, "2024-01-01", "2031-12-31"),
    ]

    rows = []
    for label, a, b in cycles:
        w = df.loc[a:b]
        if w.empty:
            continue
        ppk = w["price"].max()
        pv = None
        if puell is not None:
            try:
                pv = puell.loc[a:b].max()
            except Exception:
                pv = None
        rows.append({
            "cycle":     label,
            "price":     _round(ppk, 0),
            "price_fmt": _fmt_price(ppk),
            "mvrv":      _round(w["mvrv"].max()) if "mvrv" in w.columns else None,
            "pi_cycle":  _round(w["pi_ratio"].max()),
            "p350":      _round(w["p350"].max(), 1),
            "p730":      _round(w["p730"].max(), 1),
            "puell":     _round(pv, 1),
        })

    asof = df.index.max()
    return {"rows": rows, "asof": asof.strftime("%Y-%m-%d") if pd.notna(asof) else None}


if __name__ == "__main__":
    import json
    print(json.dumps(cycle_peak_table(), indent=2))
