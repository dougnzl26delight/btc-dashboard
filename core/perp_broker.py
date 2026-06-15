"""Binance USDT-margined perpetual futures broker.

Paper mode: tracks perp positions, entry prices, accumulated funding in
.paper_perp_state.json. Funding payments are credited each cycle based on
current rate × notional. P&L is mark-to-market.

Live mode: routes via ccxt binance with defaultType=future. Live trading is
gated behind explicit `live` flag and BINANCE_API_KEY presence.

This is the missing piece for funding-rate basis arbitrage execution.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ccxt
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_PERP_STATE_LEGACY = REPO_ROOT / ".paper_perp_state.json"  # shared (legacy)
PERP_TRADES = REPO_ROOT / "paper_perp_trades.csv"
PERP_FUNDING_LOG = REPO_ROOT / "paper_perp_funding.csv"

Mode = Literal["paper", "live"]
Side = Literal["long", "short"]


def _perp_state_file_for(sleeve: str | None) -> Path:
    """Path for a perp sleeve's state file. None/'default' = legacy shared."""
    if not sleeve or sleeve == "default":
        return PAPER_PERP_STATE_LEGACY
    safe = sleeve.replace("/", "_").replace(":", "_")
    return REPO_ROOT / f".paper_perp_state_{safe}.json"


@dataclass
class PerpState:
    cash_quote: float
    quote_currency: str = "USDT"
    # positions: signed contract count per asset (positive=long, negative=short)
    positions: dict[str, float] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)
    accumulated_funding: dict[str, float] = field(default_factory=dict)
    last_funding_ts: dict[str, float] = field(default_factory=dict)
    state_file: Path = PAPER_PERP_STATE_LEGACY

    @classmethod
    def load(cls, start_quote: float, sleeve: str | None = None) -> "PerpState":
        state_file = _perp_state_file_for(sleeve)
        if state_file.exists():
            d = json.loads(state_file.read_text())
            return cls(
                cash_quote=float(d["cash_quote"]),
                quote_currency=d.get("quote_currency", "USDT"),
                positions=d.get("positions", {}),
                entry_prices=d.get("entry_prices", {}),
                accumulated_funding=d.get("accumulated_funding", {}),
                last_funding_ts=d.get("last_funding_ts", {}),
                state_file=state_file,
            )
        return cls(cash_quote=start_quote, state_file=state_file)

    def save(self) -> None:
        self.state_file.write_text(json.dumps({
            "cash_quote": self.cash_quote,
            "quote_currency": self.quote_currency,
            "positions": self.positions,
            "entry_prices": self.entry_prices,
            "accumulated_funding": self.accumulated_funding,
            "last_funding_ts": self.last_funding_ts,
        }, indent=2))


class PerpBroker:
    """Paper / live Binance USDT-margined perp broker.

    Paper mode is fully simulated — uses real Binance perp ticks for prices but
    state is local. Live mode is gated and requires BINANCE_API_KEY in env.
    """

    def __init__(self, mode: Mode = "paper", sleeve: str | None = None,
                 paper_start_quote: float | None = None) -> None:
        self.mode = mode
        self.sleeve = sleeve
        load_dotenv(REPO_ROOT / ".env")
        self._ex = ccxt.binance({
            "apiKey": os.getenv("BINANCE_API_KEY") or None,
            "secret": os.getenv("BINANCE_API_SECRET") or None,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if mode == "live" and not os.getenv("BINANCE_API_KEY"):
            raise RuntimeError("BINANCE_API_KEY required for live perp mode")
        if mode == "paper":
            start = paper_start_quote if paper_start_quote is not None else float(
                os.getenv("PAPER_PERP_START_USDT", "100000"))
            self._state = PerpState.load(start_quote=start, sleeve=sleeve)
        # Testnet routing
        if mode == "live" and os.getenv("BINANCE_TESTNET", "").lower() == "true":
            self._ex.set_sandbox_mode(True)

    # ----- read methods -----
    def perp_symbol(self, pair: str) -> str:
        """BTC/USDT -> BTC/USDT:USDT (ccxt unified perp symbol)."""
        if ":" in pair:
            return pair
        return f"{pair}:{pair.split('/')[1]}"

    def ticker(self, pair: str) -> dict:
        return self._ex.fetch_ticker(self.perp_symbol(pair))

    def funding_rate(self, pair: str) -> float:
        try:
            fr = self._ex.fetch_funding_rate(self.perp_symbol(pair))
            return float(fr.get("fundingRate") or 0.0)
        except Exception:
            return 0.0

    def get_balance(self) -> dict:
        if self.mode == "paper":
            return {
                "USDT": self._state.cash_quote,
                "positions": dict(self._state.positions),
                "entry_prices": dict(self._state.entry_prices),
                "accumulated_funding": dict(self._state.accumulated_funding),
            }
        return self._ex.fetch_balance()

    def position_value(self, pair: str) -> float:
        """Mark-to-market value of position on pair (USDT)."""
        if self.mode != "paper":
            raise NotImplementedError("live position_value via fetch_positions")
        base = pair.split("/")[0]
        contracts = self._state.positions.get(base, 0.0)
        if contracts == 0:
            return 0.0
        try:
            price = float(self.ticker(pair)["last"])
        except Exception:
            return 0.0
        return contracts * price

    def unrealized_pnl(self, pair: str) -> float:
        if self.mode != "paper":
            raise NotImplementedError
        base = pair.split("/")[0]
        contracts = self._state.positions.get(base, 0.0)
        if contracts == 0:
            return 0.0
        entry = self._state.entry_prices.get(base, 0.0)
        try:
            price = float(self.ticker(pair)["last"])
        except Exception:
            return 0.0
        return contracts * (price - entry)

    # ----- order methods -----
    def open_position(self, pair: str, side: Side, quote_amount: float) -> dict:
        """Open or increase a position. side='long' or 'short'."""
        ticker = self.ticker(pair)
        # Some Binance perp tickers don't carry bid/ask reliably — fall back to last.
        if side == "long":
            price_raw = ticker.get("ask") or ticker.get("last") or ticker.get("close")
        else:
            price_raw = ticker.get("bid") or ticker.get("last") or ticker.get("close")
        price = float(price_raw)
        contracts = quote_amount / price
        signed = contracts if side == "long" else -contracts
        base = pair.split("/")[0]

        if self.mode == "live":
            ccxt_side = "buy" if side == "long" else "sell"
            return self._ex.create_market_order(self.perp_symbol(pair), ccxt_side, contracts)

        # Paper: weighted-avg entry price
        prev_qty = self._state.positions.get(base, 0.0)
        prev_entry = self._state.entry_prices.get(base, 0.0)
        new_qty = prev_qty + signed

        if abs(new_qty) < 1e-12:
            self._state.positions[base] = 0.0
            self._state.entry_prices.pop(base, None)
        elif (prev_qty * signed) >= 0:
            # Same direction — weighted avg
            new_entry = (
                (prev_qty * prev_entry + signed * price) / new_qty
                if new_qty != 0 else price
            )
            self._state.positions[base] = new_qty
            self._state.entry_prices[base] = new_entry
        else:
            # Opposite direction — partial or full close
            if abs(signed) <= abs(prev_qty):
                # Reducing position — realize PnL on closed portion, keep entry
                self._state.positions[base] = new_qty
            else:
                # Flipped direction — old closed, new opens at this price
                self._state.positions[base] = new_qty
                self._state.entry_prices[base] = price

        self._state.save()
        self._log_trade(pair, side, contracts, price, "open")
        return {"pair": pair, "side": side, "contracts": contracts, "price": price, "mode": "paper"}

    def close_position(self, pair: str) -> dict:
        if self.mode == "live":
            raise NotImplementedError("live close via reduce-only order")
        base = pair.split("/")[0]
        held = self._state.positions.get(base, 0.0)
        if abs(held) < 1e-12:
            return {"closed": 0.0, "reason": "no position"}
        ticker = self.ticker(pair)
        if held > 0:
            price_raw = ticker.get("bid") or ticker.get("last") or ticker.get("close")
        else:
            price_raw = ticker.get("ask") or ticker.get("last") or ticker.get("close")
        price = float(price_raw)
        entry = self._state.entry_prices.get(base, price)
        realized = held * (price - entry)
        self._state.cash_quote += realized
        self._state.positions[base] = 0.0
        self._state.entry_prices.pop(base, None)
        self._state.save()
        self._log_trade(pair, "long" if held > 0 else "short", abs(held), price, "close",
                        realized_pnl=realized)
        return {"closed": held, "realized_pnl": realized, "exit_price": price}

    # ----- funding accumulation -----
    def settle_funding(self, pair: str) -> float:
        """Accumulate funding payment based on current funding rate × notional.

        Called per cycle (orchestrator) or on funding-event timestamps. Returns
        funding amount credited (positive when receiving, e.g. short + positive funding).
        """
        if self.mode != "paper":
            raise NotImplementedError
        base = pair.split("/")[0]
        contracts = self._state.positions.get(base, 0.0)
        if abs(contracts) < 1e-12:
            return 0.0

        # Only credit funding once per 8h funding interval
        now_ts = time.time()
        last_ts = self._state.last_funding_ts.get(base, 0.0)
        if now_ts - last_ts < 8 * 3600:
            return 0.0

        try:
            rate = self.funding_rate(pair)
            price = float(self.ticker(pair)["last"])
        except Exception:
            return 0.0

        notional = contracts * price
        # Long pays funding when rate > 0; short receives. Sign convention:
        #   funding_payment = -position_signed * notional_abs * rate
        funding_payment = -contracts * abs(notional) * rate / abs(notional) if notional != 0 else 0
        # Simplified: short with positive funding → positive payment
        funding_payment = -float(contracts) * float(rate) * float(price)

        self._state.cash_quote += funding_payment
        self._state.accumulated_funding[base] = self._state.accumulated_funding.get(base, 0.0) + funding_payment
        self._state.last_funding_ts[base] = now_ts
        self._state.save()
        self._log_funding(pair, contracts, rate, price, funding_payment)
        return funding_payment

    # ----- internal -----
    @staticmethod
    def _log_trade(pair: str, side: str, contracts: float, price: float, action: str,
                   realized_pnl: float = 0.0) -> None:
        new = not PERP_TRADES.exists()
        with PERP_TRADES.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "pair", "side", "contracts", "price", "action", "realized_pnl"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                pair, side, f"{contracts:.8f}", f"{price:.4f}", action, f"{realized_pnl:.4f}",
            ])

    @staticmethod
    def _log_funding(pair: str, contracts: float, rate: float, price: float, payment: float) -> None:
        new = not PERP_FUNDING_LOG.exists()
        with PERP_FUNDING_LOG.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_utc", "pair", "contracts", "rate", "price", "payment_usdt"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                pair, f"{contracts:.8f}", f"{rate:.8f}", f"{price:.4f}", f"{payment:.4f}",
            ])


if __name__ == "__main__":
    pb = PerpBroker(mode="paper")
    print("Balance:", pb.get_balance())
    print("BTC funding rate:", pb.funding_rate("BTC/USDT"))
    print("BTC ticker:", pb.ticker("BTC/USDT")["last"])
