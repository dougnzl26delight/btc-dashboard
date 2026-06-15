"""Scan funding across the full basis universe + extended candidates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.funding_basis_arb import latest_funding_bps_8h


PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
    "MATIC/USDT", "TRX/USDT", "ATOM/USDT", "NEAR/USDT", "ARB/USDT",
    "OP/USDT", "APT/USDT", "SUI/USDT", "INJ/USDT", "TIA/USDT",
]


if __name__ == "__main__":
    print(f"{'Pair':<14s}  {'bps/8h':>8s}  {'Annual %':>9s}  {'Signal':>7s}")
    print("-" * 50)
    rows = []
    for p in PAIRS:
        bps = latest_funding_bps_8h(p)
        ann = bps * 3 * 365 / 100
        sig = "+1" if bps > 1.0 else ("-1" if bps < -1.0 else "0")
        rows.append((p, bps, ann, sig))
    rows.sort(key=lambda x: -x[1])
    for p, bps, ann, sig in rows:
        print(f"{p:<14s}  {bps:>+7.3f}   {ann:>+7.1f}%   {sig:>5s}")

    print()
    print("Threshold for entry: > 1.0 bps/8h (~11% annualized)")
    n_signaled = sum(1 for _, bps, _, _ in rows if bps > 1.0)
    print(f"Pairs signaling entry: {n_signaled}/{len(rows)}")
