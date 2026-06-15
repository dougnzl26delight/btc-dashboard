"""Migration: shared paper accounts -> per-sleeve sub-accounts.

Splits the current shared .paper_state.json and .paper_perp_state.json into
isolated sub-account state files per sleeve.

Allocations (matching strategy charters):
    Spot ($100k total)
      bah_btc         $10,000   ($10k notional BTC target = 10% of bankroll)
      oversold_bounce $15,000   ($3k each x 5 max positions = 15%)
      orchestrator    $30,000   (multi-strategy ensemble, 30%)
      reserve         $45,000   (dry powder + new sleeves)

    Perp ($100k total)
      xsmom           $30,000   (15% gross exposure across 4 perps)
      pro_trend       $30,000   (perp routing for shorts; longs go to spot)
      overbought_fade $10,000   (smaller short basket)
      basis_arb       $20,000   (market-neutral, capital-light)
      reserve         $10,000

Preserves existing open positions: each sleeve's known positions are migrated
into its sub-account. Any "orphan" positions (untagged) stay in the legacy
shared account until manually resolved.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


SPOT_ALLOCATIONS = {
    "bah_btc": 10_000.0,
    "oversold_bounce": 15_000.0,
    "orchestrator": 30_000.0,
    "spot_reserve": 45_000.0,
}

PERP_ALLOCATIONS = {
    "xsmom": 30_000.0,
    "pro_trend": 30_000.0,
    "overbought_fade": 10_000.0,
    "basis_arb": 20_000.0,
    "perp_reserve": 10_000.0,
}


def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def migrate_spot():
    """Split the current shared spot account.

    Strategy: each known sleeve gets its allocation in cash. Open positions
    tagged to a sleeve are transferred. Remainder stays in the legacy file
    (renamed to .paper_state_legacy_backup.json) for audit.
    """
    legacy = REPO_ROOT / ".paper_state.json"
    if not legacy.exists():
        print("No legacy .paper_state.json; nothing to migrate")
        return

    current = json.loads(legacy.read_text())
    current_cash = current.get("cash_quote", 0)
    current_positions = current.get("positions", {})

    # Filter out dust positions (< 1e-6 of base)
    significant_positions = {
        k: v for k, v in current_positions.items() if abs(v) > 1e-6
    }

    print(f"Current shared spot account: ${current_cash:,.2f} cash")
    print(f"  Significant positions: {significant_positions}")
    print()

    # Read sleeve state files to know what they "own"
    bah = _read_json(REPO_ROOT / ".bah_btc_state.json")
    bah_btc_qty = float(bah.get("btc_qty", 0))

    oversold = _read_json(REPO_ROOT / ".oversold_bounce_state.json")
    oversold_pos = oversold.get("open_positions", {})

    # Build per-sleeve sub-accounts
    sub_accounts = {}

    # ---- BAH BTC ----
    bah_alloc = SPOT_ALLOCATIONS["bah_btc"]
    bah_btc_actual = min(bah_btc_qty, current_positions.get("BTC", 0))
    bah_btc_value = bah_btc_actual * 73_414  # rough — uses last known price
    bah_cash = bah_alloc - bah_btc_value
    sub_accounts["bah_btc"] = {
        "cash_quote": max(0, bah_cash),
        "quote_currency": "USDT",
        "positions": {"BTC": bah_btc_actual} if bah_btc_actual > 0 else {},
    }
    print(f"  BAH BTC sub-account: ${bah_cash:,.2f} cash + {bah_btc_actual:.6f} BTC")

    # ---- Oversold bounce ----
    oversold_alloc = SPOT_ALLOCATIONS["oversold_bounce"]
    oversold_positions = {}
    oversold_value_held = 0.0
    for pair, info in oversold_pos.items():
        base = pair.split("/")[0]
        claimed_qty = float(info.get("qty", 0))
        # Use the strategy's claimed qty (not broker's — broker is shared)
        oversold_positions[base] = claimed_qty
        oversold_value_held += claimed_qty * float(info.get("entry_price", 0))
    oversold_cash = oversold_alloc - oversold_value_held
    sub_accounts["oversold_bounce"] = {
        "cash_quote": max(0, oversold_cash),
        "quote_currency": "USDT",
        "positions": oversold_positions,
    }
    print(f"  Oversold sub-account:  ${oversold_cash:,.2f} cash + "
          f"{len(oversold_positions)} positions worth ~${oversold_value_held:,.0f}")

    # ---- Orchestrator + reserve (cash-only sub-accounts) ----
    sub_accounts["orchestrator"] = {
        "cash_quote": SPOT_ALLOCATIONS["orchestrator"],
        "quote_currency": "USDT",
        "positions": {},
    }
    sub_accounts["spot_reserve"] = {
        "cash_quote": SPOT_ALLOCATIONS["spot_reserve"],
        "quote_currency": "USDT",
        "positions": {},
    }
    print(f"  Orchestrator sub-account: ${SPOT_ALLOCATIONS['orchestrator']:,.2f}")
    print(f"  Spot reserve:             ${SPOT_ALLOCATIONS['spot_reserve']:,.2f}")

    # Write all sub-account files
    for sleeve, data in sub_accounts.items():
        out = REPO_ROOT / f".paper_state_{sleeve}.json"
        _write_json(out, data)
        print(f"    wrote {out.name}")

    # Backup the legacy file (don't delete — keep for audit)
    backup = REPO_ROOT / f".paper_state_legacy_backup_{int(datetime.now().timestamp())}.json"
    legacy.rename(backup)
    print(f"\n  Legacy account backed up to: {backup.name}")

    return sub_accounts


def migrate_perp():
    """Same pattern for perp account."""
    legacy = REPO_ROOT / ".paper_perp_state.json"
    if not legacy.exists():
        print("No legacy .paper_perp_state.json; nothing to migrate")
        return

    current = json.loads(legacy.read_text())
    current_cash = current.get("cash_quote", 0)
    current_positions = current.get("positions", {})
    current_entries = current.get("entry_prices", {})
    sig = {k: v for k, v in current_positions.items() if abs(v) > 1e-9}

    print(f"\nCurrent shared perp account: ${current_cash:,.2f} cash")
    print(f"  Significant positions: {sig}")
    print()

    # Read sleeve state files
    xsmom = _read_json(REPO_ROOT / ".xsmom_state.json")
    xsmom_weights = xsmom.get("weights", {})

    # Pro_trend per-pair states
    pt_positions = {}
    pt_entries = {}
    for f in REPO_ROOT.glob(".pro_trend_state_*.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("side") and d.get("units"):
                base = f.stem.replace(".pro_trend_state_", "")
                qty_signed = sum(u.get("qty", 0) for u in d["units"])
                if d["side"] == "short":
                    qty_signed = -qty_signed
                pt_positions[base] = qty_signed
                if d["units"]:
                    pt_entries[base] = d["units"][0].get("entry_price", 0)
        except Exception:
            pass

    sub_accounts = {}

    # ---- XSMOM ----
    xsmom_alloc = PERP_ALLOCATIONS["xsmom"]
    xsmom_positions = {}
    xsmom_entries = {}
    for pair in xsmom_weights:
        base = pair.split("/")[0]
        if base in current_positions and abs(current_positions[base]) > 1e-9:
            xsmom_positions[base] = current_positions[base]
            xsmom_entries[base] = current_entries.get(base, 0)
    sub_accounts["xsmom"] = {
        "cash_quote": xsmom_alloc,
        "quote_currency": "USDT",
        "positions": xsmom_positions,
        "entry_prices": xsmom_entries,
        "accumulated_funding": {},
        "last_funding_ts": {},
    }
    print(f"  XSMOM sub-account: ${xsmom_alloc:,.2f} cash + {len(xsmom_positions)} positions")

    # ---- Pro_trend ----
    pt_alloc = PERP_ALLOCATIONS["pro_trend"]
    sub_accounts["pro_trend"] = {
        "cash_quote": pt_alloc,
        "quote_currency": "USDT",
        "positions": pt_positions,
        "entry_prices": pt_entries,
        "accumulated_funding": {},
        "last_funding_ts": {},
    }
    print(f"  Pro_trend sub-account: ${pt_alloc:,.2f} cash + {len(pt_positions)} positions")

    # ---- Other sleeves (cash only) ----
    for sleeve in ["overbought_fade", "basis_arb", "perp_reserve"]:
        sub_accounts[sleeve] = {
            "cash_quote": PERP_ALLOCATIONS[sleeve],
            "quote_currency": "USDT",
            "positions": {},
            "entry_prices": {},
            "accumulated_funding": {},
            "last_funding_ts": {},
        }
        print(f"  {sleeve} sub-account: ${PERP_ALLOCATIONS[sleeve]:,.2f}")

    for sleeve, data in sub_accounts.items():
        out = REPO_ROOT / f".paper_perp_state_{sleeve}.json"
        _write_json(out, data)
        print(f"    wrote {out.name}")

    backup = REPO_ROOT / f".paper_perp_state_legacy_backup_{int(datetime.now().timestamp())}.json"
    legacy.rename(backup)
    print(f"\n  Legacy perp account backed up to: {backup.name}")

    return sub_accounts


def main():
    print("=" * 80)
    print("PAPER SUB-ACCOUNT MIGRATION")
    print("=" * 80)
    print()
    migrate_spot()
    print()
    migrate_perp()
    print()
    print("=" * 80)
    print("MIGRATION COMPLETE")
    print("=" * 80)
    print()
    print("Each sleeve now has its own isolated paper sub-account.")
    print("No more cross-sleeve interference (orchestrator can't eat oversold positions).")
    print()
    print("Total spot allocated:", f"${sum(SPOT_ALLOCATIONS.values()):,.0f}")
    print("Total perp allocated:", f"${sum(PERP_ALLOCATIONS.values()):,.0f}")
    print("Total paper capital: ", f"${sum(SPOT_ALLOCATIONS.values()) + sum(PERP_ALLOCATIONS.values()):,.0f}")


if __name__ == "__main__":
    main()
