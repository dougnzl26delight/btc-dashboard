"""Binance broker adapter (Day 0 — buy-and-hold smoke test scope).

Modes:
  paper  Local fill simulator against live Binance ticks. No keys required.
  live   Real orders. If BINANCE_TESTNET=true, routed to the Binance Spot
         Testnet (testnet.binance.vision) — fully featured but uses test
         funds. Otherwise routed to mainnet (real money).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ccxt
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_STATE_LEGACY = REPO_ROOT / ".paper_state.json"  # shared (legacy)
PAPER_TRADES = REPO_ROOT / "paper_trades.csv"

DEFAULT_QUOTE = "USDT"
DEFAULT_PAIRS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")

Mode = Literal["paper", "live"]
Side = Literal["buy", "sell"]


def _state_file_for(sleeve: str | None) -> Path:
    """Return the paper state file path for a sleeve.

    None or "default" -> legacy shared .paper_state.json (preserved for old code)
    Other names       -> .paper_state_{sleeve}.json (isolated sub-account)
    """
    if not sleeve or sleeve == "default":
        return PAPER_STATE_LEGACY
    safe = sleeve.replace("/", "_").replace(":", "_")
    return REPO_ROOT / f".paper_state_{safe}.json"


@dataclass
class PaperState:
    cash_quote: float
    quote_currency: str = DEFAULT_QUOTE
    positions: dict[str, float] = field(default_factory=dict)
    state_file: Path = PAPER_STATE_LEGACY  # path for save()

    @classmethod
    def load(cls, start_quote: float, quote_currency: str = DEFAULT_QUOTE,
             sleeve: str | None = None) -> "PaperState":
        state_file = _state_file_for(sleeve)
        if state_file.exists():
            data = json.loads(state_file.read_text())
            if "cash_quote" not in data:
                raise RuntimeError(
                    f"{state_file.name} uses an old schema. "
                    "Delete it and the paper_trades.csv to start fresh on Binance."
                )
            return cls(
                cash_quote=data["cash_quote"],
                quote_currency=data.get("quote_currency", DEFAULT_QUOTE),
                positions=data.get("positions", {}),
                state_file=state_file,
            )
        return cls(cash_quote=start_quote, quote_currency=quote_currency,
                   state_file=state_file)

    def save(self) -> None:
        self.state_file.write_text(
            json.dumps(
                {
                    "cash_quote": self.cash_quote,
                    "quote_currency": self.quote_currency,
                    "positions": self.positions,
                },
                indent=2,
            )
        )


class Broker:
    def __init__(self, mode: Mode = "paper", long_only: bool = True,
                 sleeve: str | None = None,
                 paper_start_quote: float | None = None) -> None:
        """Per-sleeve isolated broker.

        sleeve: name of sub-account (e.g., "bah_btc", "oversold_bounce", "orchestrator").
                None or "default" -> shared legacy account (backward-compatible).
        paper_start_quote: initial cash for first-run of a NEW sub-account.
                Default $10k (env PAPER_START_USDT). Per-sleeve allocations are
                set by paper_subaccount_setup.py at migration time.

        In live mode, sleeve maps to a Binance sub-account (TODO: implement
        sub-account API routing; for now sleeve is ignored in live).
        """
        self.mode = mode
        self.long_only = long_only
        self.sleeve = sleeve
        self._ex = ccxt.binance(
            {
                "apiKey": os.getenv("BINANCE_API_KEY") or None,
                "secret": os.getenv("BINANCE_API_SECRET") or None,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )
        # Testnet only matters when actually placing live orders. Public market
        # data in paper mode comes from mainnet because testnet ticks are thin.
        if mode == "live" and os.getenv("BINANCE_TESTNET", "").lower() == "true":
            self._ex.set_sandbox_mode(True)
        if mode == "paper":
            start = paper_start_quote if paper_start_quote is not None else float(
                os.getenv("PAPER_START_USDT", "10000"))
            self._paper = PaperState.load(
                start_quote=start,
                quote_currency=DEFAULT_QUOTE,
                sleeve=sleeve,
            )

    def get_ticker(self, pair: str) -> dict:
        return self._ex.fetch_ticker(pair)

    def get_balance(self) -> dict[str, float]:
        if self.mode == "paper":
            return {self._paper.quote_currency: self._paper.cash_quote, **self._paper.positions}
        return {k: v for k, v in self._ex.fetch_balance()["total"].items() if v}

    def place_market_order(self, pair: str, side: Side, quote_amount: float) -> dict:
        ticker = self.get_ticker(pair)
        # Conservative paper fills: buy crosses the ask, sell hits the bid.
        price = ticker["ask"] if side == "buy" else ticker["bid"]
        base, quote = pair.split("/")
        qty = quote_amount / price

        if self.mode == "live":
            return self._ex.create_market_order(pair, side, qty)

        held = self._paper.positions.get(base, 0.0)
        if side == "buy":
            if quote_amount > self._paper.cash_quote:
                raise ValueError(
                    f"insufficient {quote}: need {quote_amount}, have {self._paper.cash_quote}"
                )
            # === W15.H: ANTI-AVERAGING-DOWN (Livermore: "Never average down") ===
            # If we already hold this asset AND it's currently underwater AND this
            # is a discretionary add (not a sleeve cycle entry), refuse.
            # Sleeves that explicitly need to top-up (BAH BTC monthly rebalance,
            # oversold_bounce) bypass via env var ALLOW_AVERAGE_DOWN=1.
            import os
            if held > 1e-9 and os.getenv("ALLOW_AVERAGE_DOWN", "0") != "1":
                # Check if currently underwater: read avg entry from trades log
                # Lightweight check: if current price < last buy price, we're averaging down
                from pathlib import Path as _P
                try:
                    trades_csv = _P(__file__).resolve().parent.parent / "paper_trades.csv"
                    if trades_csv.exists():
                        import csv as _csv
                        with trades_csv.open(encoding="utf-8") as f:
                            rows = [r for r in _csv.DictReader(f) if r.get("pair") == pair and r.get("side") == "buy"]
                        if rows:
                            last_buy_price = float(rows[-1]["price"])
                            if price < last_buy_price * 0.97:  # adding while 3%+ underwater
                                raise ValueError(
                                    f"AVERAGING-DOWN BLOCKED (Livermore rule): {pair} held qty {held:.4f}, "
                                    f"last buy ${last_buy_price:.2f}, current ${price:.2f} (-{(1-price/last_buy_price)*100:.1f}%). "
                                    f"To override, set env ALLOW_AVERAGE_DOWN=1 with documented reason."
                                )
                except ValueError:
                    raise
                except Exception:
                    pass  # log read failure should not block legitimate trades
            self._paper.cash_quote -= quote_amount
            self._paper.positions[base] = held + qty
        else:  # sell
            if qty > held and self.long_only:
                raise ValueError(
                    f"insufficient {base}: need {qty}, have {held} (long-only mode)"
                )
            # In short-allowed mode, position can go negative (paper-only short).
            # Live execution would route this to a perp broker.
            self._paper.positions[base] = held - qty
            self._paper.cash_quote += quote_amount

        self._paper.save()
        self._log_paper_trade(pair, side, qty, price)
        return {"pair": pair, "side": side, "qty": qty, "price": price, "mode": "paper"}

    def _log_paper_trade(self, pair: str, side: Side, qty: float, price: float) -> None:
        new = not PAPER_TRADES.exists()
        with PAPER_TRADES.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "sleeve", "pair", "side", "qty", "price", "value_quote"])
            w.writerow(
                [
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    self.sleeve or "default",
                    pair,
                    side,
                    f"{qty:.8f}",
                    f"{price:.4f}",
                    f"{qty * price:.2f}",
                ]
            )


def _cli() -> None:
    load_dotenv(REPO_ROOT / ".env")
    p = argparse.ArgumentParser(description="Binance broker adapter")
    p.add_argument(
        "--mode",
        default=os.getenv("TRADING_MODE", "paper"),
        choices=("paper", "live"),
    )
    p.add_argument("--status", action="store_true", help="Print tickers + balances and exit")
    p.add_argument("--buy", type=float, metavar="QUOTE", help="Buy this quote-currency value of --pair")
    p.add_argument("--sell", type=float, metavar="QUOTE", help="Sell this quote-currency value of --pair")
    p.add_argument("--pair", default="BTC/USDT")
    args = p.parse_args()

    b = Broker(mode=args.mode)

    if args.status:
        for pair in DEFAULT_PAIRS:
            t = b.get_ticker(pair)
            print(f"{pair:10s} bid={t['bid']:>12.2f}  ask={t['ask']:>12.2f}  last={t['last']:>12.2f}")
        live_target = "TESTNET" if os.getenv("BINANCE_TESTNET", "").lower() == "true" else "MAINNET"
        print(f"\nMode: {args.mode}" + (f" ({live_target})" if args.mode == "live" else ""))
        print("Balances:")
        for k, v in b.get_balance().items():
            print(f"  {k:6s} {v}")
        return

    if args.buy:
        print(b.place_market_order(args.pair, "buy", args.buy))
        return
    if args.sell:
        print(b.place_market_order(args.pair, "sell", args.sell))
        return

    print(f"broker ready (mode={args.mode}); use --status, --buy, or --sell")


if __name__ == "__main__":
    _cli()
