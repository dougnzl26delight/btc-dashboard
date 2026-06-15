"""Prediction-market overlay — live crowd (real-money) odds from Polymarket on
questions relevant to the BTC / equity-rotation campaign.

Real money is often sharper than any single indicator, and *divergence* between
the crowd and our own model is the signal worth noticing. Public gamma API, no
auth. NOT investment advice — an external probability sanity-check.
"""
from __future__ import annotations
import json
import urllib.request

_GAMMA = "https://gamma-api.polymarket.com/markets"

# category -> keyword tuple (first match wins)
_CATEGORIES = [
    ("rates",     ("fed ", "interest rate", "rate cut", "rate hike", "fomc", "powell")),
    ("recession", ("recession", "gdp", "unemployment", "hard landing", "soft landing")),
    ("btc",       ("bitcoin", "btc")),
    ("crypto",    ("ethereum", " eth ", "crypto", "altcoin", "solana")),
    ("equities",  ("s&p", "nasdaq", "stock market", "dow ", "spx")),
]
_CAT_LABEL = {
    "rates": "Fed / rates", "recession": "Recession / macro", "btc": "Bitcoin price",
    "crypto": "Crypto (other)", "equities": "Equities",
}


def _fetch(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _parse_list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return v or []


def _isf(x) -> bool:
    try:
        float(x); return True
    except Exception:
        return False


def prediction_market_odds(max_items: int = 16) -> dict:
    """Return live, relevant Polymarket markets with implied Yes-probabilities,
    sorted by traded volume (liquidity ~ sharpness)."""
    data = []
    try:
        for _pg in range(6):  # up to 600 markets, most-liquid first
            page = _fetch(f"{_GAMMA}?closed=false&active=true&limit=100"
                          f"&offset={_pg * 100}&order=volume24hr&ascending=false")
            if not isinstance(page, list) or not page:
                break
            data.extend(page)
    except Exception as e:
        if not data:
            return {"error": f"{type(e).__name__}: {str(e)[:60]}", "markets": []}

    out = []
    for m in data:
        q = m.get("question") or ""
        ql = q.lower()
        cat = next((c for c, kws in _CATEGORIES if any(k in ql for k in kws)), None)
        if not cat:
            continue
        outs = _parse_list(m.get("outcomes"))
        prices = _parse_list(m.get("outcomePrices"))
        yes_p = None
        if outs and prices and len(outs) == len(prices):
            for o, p in zip(outs, prices):
                if str(o).strip().lower() == "yes" and _isf(p):
                    yes_p = float(p)
        try:
            vol = float(m.get("volume") or m.get("volumeNum") or 0)
        except Exception:
            vol = 0.0
        out.append({
            "question": q,
            "category": cat,
            "category_label": _CAT_LABEL.get(cat, cat),
            "yes_prob": yes_p,
            "outcomes": outs,
            "prices": [float(p) for p in prices if _isf(p)],
            "volume": vol,
            "end_date": (m.get("endDate") or "")[:10],
        })

    out.sort(key=lambda x: x["volume"], reverse=True)
    by_cat: dict[str, list] = {}
    for m in out:
        by_cat.setdefault(m["category"], []).append(m)
    return {
        "markets": out[:max_items],
        "by_category": {c: ms[:4] for c, ms in by_cat.items()},
        "n_total": len(out),
        "source": "Polymarket (live real-money crowd odds)",
    }


if __name__ == "__main__":
    r = prediction_market_odds()
    if r.get("error"):
        print("ERROR:", r["error"])
    else:
        print(f"{r['n_total']} relevant markets (source: {r['source']})\n")
        for m in r["markets"]:
            yp = f"{m['yes_prob']*100:.0f}% YES" if m["yes_prob"] is not None else "n/a"
            line = (f"[{m['category_label']:<16}] {yp:<9} "
                    f"vol ${m['volume']:,.0f}  ends {m['end_date']}  | {m['question'][:64]}")
            print(line.encode("ascii", "replace").decode())
