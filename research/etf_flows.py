"""BTC ETF flow signal — attempts to fetch daily total flows from Farside.

Farside has no API; HTML table is parseable via pandas.read_html. Best-effort —
if scraping breaks (site changes, blocks UA), the function returns an empty
series and the strategy logs a clear failure rather than silently using stale data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


FARSIDE_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"


def fetch_etf_flows() -> pd.Series:
    """Return daily total BTC ETF flow in USD millions, indexed by date.
    Returns empty series on any failure.
    """
    import urllib.request

    try:
        req = urllib.request.Request(
            FARSIDE_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[etf_flows] HTTP fetch failed: {e}")
        return pd.Series(dtype=float)

    from io import StringIO

    try:
        tables = pd.read_html(StringIO(html), attrs={"class": "etf"})
    except Exception as e:
        print(f"[etf_flows] read_html failed: {e}")
        return pd.Series(dtype=float)

    if not tables:
        print("[etf_flows] no etf-class tables found")
        return pd.Series(dtype=float)

    df = tables[0]
    # Farside columns: Date | IBIT | FBTC | ... | Total
    date_col = df.columns[0]
    total_candidates = [c for c in df.columns if "Total" in str(c)]
    if not total_candidates:
        print(f"[etf_flows] no Total column; cols: {list(df.columns)[:5]}...")
        return pd.Series(dtype=float)
    total_col = total_candidates[-1]

    work = df[[date_col, total_col]].copy()
    # Drop summary/footer rows (Date doesn't parse)
    work["ts"] = pd.to_datetime(work[date_col].astype(str), errors="coerce", utc=True)
    work = work.dropna(subset=["ts"]).set_index("ts").sort_index()

    # Farside uses parens for negatives ("(123.4)") and "-" for zero
    flows = (
        work[total_col].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .replace({"-": "0", "": "0", "nan": "0"})
    )
    return pd.to_numeric(flows, errors="coerce").dropna()


def etf_flow_signal(flows: pd.Series, ema_window: int = 7) -> pd.Series:
    """Convert daily ETF flows into a normalized signal in [-1, 1]."""
    if flows.empty:
        return pd.Series(dtype=float)
    smoothed = flows.ewm(span=ema_window).mean()
    rolling_std = smoothed.rolling(60).std()
    z = smoothed / rolling_std
    return z.clip(-2, 2) / 2.0


if __name__ == "__main__":
    flows = fetch_etf_flows()
    if flows.empty:
        print("ETF flow fetch failed — see error above")
    else:
        print(f"Fetched {len(flows)} days of ETF flows")
        print(f"Span: {flows.index[0].date()} -> {flows.index[-1].date()}")
        print(f"Last 5 flows (USD M): {flows.tail(5).round(1).to_dict()}")
