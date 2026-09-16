"""Binance public WebSocket price feed with auto-reconnect + state cache.

Connects to wss://stream.binance.com:9443/stream and subscribes to bookTicker
streams for the trading universe. Maintains an in-memory cache of latest
best-bid/best-ask per pair, refreshed ~10x/second.

Persists the live cache to .live_prices.json every second so other processes
(e.g. dashboard) can read prices without their own WebSocket connection.

This is the SHARED real-time data layer. The realtime_monitor service
subscribes to events here for stop-loss execution. Dashboard reads the cache
file for live charts.

Auto-reconnect with exponential backoff. Disconnects logged. Will run forever.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import websocket

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LIVE_PRICES_FILE = REPO_ROOT / ".live_prices.json"
WS_LOG_FILE = REPO_ROOT / ".ws_feed.log"

# Default universe — mirrors orchestrator
DEFAULT_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "DOGE/USDT",
    "ATOM/USDT", "TAO/USDT", "ONDO/USDT",
]

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
RECONNECT_DELAYS = [1, 2, 4, 8, 16, 32, 60, 60, 60]  # seconds
CACHE_FLUSH_INTERVAL_SEC = 1.0

# Listener callbacks and the disk flush used to run INLINE on the WebSocket
# reader thread. Any slow listener therefore stalled frame reads, which delayed
# pong handling and tripped ping_timeout -> reconnect (the .ws_feed.log history
# shows ~150-200 reconnects/day, 75% of them "ping/pong timed out"). Both now
# run on their own threads so the reader only ever parses and caches.
EVENT_QUEUE_MAX = 10_000        # backpressure bound; oldest ticks dropped first
WS_LOG_MAX_BYTES = 2_000_000    # rotate at ~2MB (log was unbounded)


def _pair_to_stream(pair: str) -> str:
    """BTC/USDT -> btcusdt@bookTicker"""
    return pair.replace("/", "").lower() + "@bookTicker"


def _stream_to_pair(stream: str) -> str:
    """btcusdt@bookTicker -> BTC/USDT"""
    sym = stream.split("@")[0]
    base = sym[:-4].upper()  # remove "usdt"
    return f"{base}/USDT"


class PriceCache:
    """Thread-safe live price store, async listener dispatch + periodic JSON flush.

    The reader thread only does update() (cache write + enqueue). Listener
    callbacks run on a single dispatch thread, so tick ORDER is preserved while
    a slow listener can no longer block the socket.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        # Listeners: callbacks invoked on every update with (pair, bid, ask, ts)
        self._listeners: list[Callable] = []
        self._events: queue.Queue = queue.Queue(maxsize=EVENT_QUEUE_MAX)
        self._dropped = 0
        self._stop = threading.Event()
        self._dispatcher: Optional[threading.Thread] = None
        self._flusher: Optional[threading.Thread] = None

    # --- hot path (reader thread) -----------------------------------------
    def update(self, pair: str, bid: float, ask: float):
        ts = time.time()
        with self._lock:
            self._cache[pair] = {
                "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "ts": ts,
            }
        if not self._listeners:
            return
        try:
            self._events.put_nowait((pair, bid, ask, ts))
        except queue.Full:
            # Consumer is behind. Drop the OLDEST tick, not the newest — stale
            # prices are worthless to a stop-loss check, fresh ones are not.
            try:
                self._events.get_nowait()
                self._dropped += 1
                self._events.put_nowait((pair, bid, ask, ts))
            except (queue.Empty, queue.Full):
                self._dropped += 1

    def get(self, pair: str) -> Optional[dict]:
        with self._lock:
            return self._cache.get(pair)

    def all(self) -> dict:
        with self._lock:
            return dict(self._cache)

    @property
    def dropped_events(self) -> int:
        return self._dropped

    @property
    def queue_depth(self) -> int:
        return self._events.qsize()

    # --- listeners ---------------------------------------------------------
    def subscribe(self, listener: Callable) -> None:
        """Listener signature: (pair, bid, ask, timestamp_unix)."""
        self._listeners.append(listener)
        self.start_dispatcher()

    def start_dispatcher(self) -> None:
        if self._dispatcher is not None:
            return
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="ws-dispatch")
        self._dispatcher.start()

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pair, bid, ask, ts = self._events.get(timeout=1.0)
            except queue.Empty:
                continue
            for listener in self._listeners:
                try:
                    listener(pair, bid, ask, ts)
                except Exception as e:
                    _log(f"listener error on {pair}: {e}")

    # --- disk flush --------------------------------------------------------
    def start_flusher(self) -> None:
        if self._flusher is not None:
            return
        self._flusher = threading.Thread(
            target=self._flush_loop, daemon=True, name="ws-flush")
        self._flusher.start()

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(CACHE_FLUSH_INTERVAL_SEC):
                break
            self._flush()

    def _flush(self) -> None:
        try:
            snapshot = self.all()
            if not snapshot:
                return
            snapshot["_meta"] = {
                "flushed_at": datetime.now(timezone.utc).isoformat(),
                "n_pairs": len([k for k in snapshot if not k.startswith("_")]),
                "dropped_events": self._dropped,
                "queue_depth": self._events.qsize(),
            }
            # Write-then-rename: a reader (dashboard) never sees a half-written
            # file, which plain write_text allowed.
            tmp = LIVE_PRICES_FILE.parent / (LIVE_PRICES_FILE.name + ".tmp")
            tmp.write_text(json.dumps(snapshot, default=str))
            tmp.replace(LIVE_PRICES_FILE)
        except Exception as e:
            _log(f"flush error: {e}")

    def stop(self) -> None:
        self._stop.set()


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts}  {msg}\n"
    try:
        if WS_LOG_FILE.exists() and WS_LOG_FILE.stat().st_size > WS_LOG_MAX_BYTES:
            WS_LOG_FILE.replace(WS_LOG_FILE.parent / (WS_LOG_FILE.name + ".1"))
    except Exception:
        pass
    try:
        with WS_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _build_url(pairs: list[str]) -> str:
    streams = [_pair_to_stream(p) for p in pairs]
    return f"{BINANCE_WS_BASE}?streams={'/'.join(streams)}"


class BinanceFeed:
    """Persistent WebSocket connection with reconnect-on-disconnect."""

    def __init__(self, pairs: list[str] | None = None, cache: PriceCache | None = None):
        self.pairs = pairs or DEFAULT_PAIRS
        self.cache = cache or PriceCache()
        self._stop = False
        self._ws: Optional[websocket.WebSocketApp] = None
        self._reconnect_idx = 0

    def _on_message(self, _ws, message: str):
        try:
            msg = json.loads(message)
            stream = msg.get("stream", "")
            payload = msg.get("data") or msg
            if "@bookTicker" not in stream:
                return
            pair = _stream_to_pair(stream)
            bid = float(payload.get("b") or 0)
            ask = float(payload.get("a") or 0)
            if bid > 0 and ask > 0:
                self.cache.update(pair, bid, ask)
        except Exception as e:
            _log(f"message parse error: {e}")

    def _on_error(self, _ws, error):
        _log(f"WS error: {error}")

    def _on_close(self, _ws, code, reason):
        _log(f"WS closed: code={code} reason={reason}")

    def _on_open(self, _ws):
        self._reconnect_idx = 0
        _log(f"WS opened — subscribed to {len(self.pairs)} pairs")

    def run_forever(self):
        """Persistent connection with exponential-backoff reconnect."""
        url = _build_url(self.pairs)
        _log(f"Starting feed on {url[:80]}...")
        while not self._stop:
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                _log(f"WS run_forever exception: {e}")

            if self._stop:
                break
            delay = RECONNECT_DELAYS[min(self._reconnect_idx, len(RECONNECT_DELAYS) - 1)]
            self._reconnect_idx += 1
            _log(f"Reconnecting in {delay}s (attempt {self._reconnect_idx})")
            time.sleep(delay)

    def stop(self):
        self._stop = True
        self.cache.stop()
        if self._ws:
            self._ws.close()


# Singleton accessors
_feed_singleton: Optional[BinanceFeed] = None
_feed_thread: Optional[threading.Thread] = None


def start_background_feed(pairs: list[str] | None = None) -> PriceCache:
    """Start the WS feed in a background thread. Returns the shared PriceCache.

    Safe to call multiple times — subsequent calls return the existing cache.
    """
    global _feed_singleton, _feed_thread
    if _feed_singleton is not None:
        return _feed_singleton.cache
    _feed_singleton = BinanceFeed(pairs)
    _feed_singleton.cache.start_flusher()
    _feed_thread = threading.Thread(target=_feed_singleton.run_forever,
                                     daemon=True, name="binance-ws-feed")
    _feed_thread.start()
    return _feed_singleton.cache


def read_live_prices() -> dict:
    """Read the latest live price cache from disk (for other processes)."""
    if not LIVE_PRICES_FILE.exists():
        return {}
    try:
        return json.loads(LIVE_PRICES_FILE.read_text())
    except Exception:
        return {}


def main():
    """Run as foreground process — prints live prices every 5s."""
    print(f"Starting Binance WS feed for {len(DEFAULT_PAIRS)} pairs...")
    cache = start_background_feed(DEFAULT_PAIRS)
    print("Connecting...")
    time.sleep(3)
    try:
        while True:
            snapshot = cache.all()
            if snapshot:
                print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Live prices:")
                for pair in sorted(snapshot.keys()):
                    d = snapshot[pair]
                    px = d["mid"]
                    age = time.time() - d["ts"]
                    fmt = f"${px:,.4f}" if px < 100 else f"${px:,.2f}"
                    print(f"  {pair:<12s} {fmt:>14s}  age={age:.2f}s")
            else:
                print(".", end="", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping...")
        if _feed_singleton:
            _feed_singleton.stop()


if __name__ == "__main__":
    main()
