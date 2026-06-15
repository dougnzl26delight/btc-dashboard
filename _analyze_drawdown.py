"""One-off analysis of current down positions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import data

# NEAR full regime context
df = data.ohlcv_extended("NEAR/USDT", days_back=250)
last = df["close"].iloc[-1]
sma200 = df["close"].rolling(200).mean().iloc[-1]
sma50 = df["close"].rolling(50).mean().iloc[-1]
high60 = df["high"].tail(60).max()
low60 = df["low"].tail(60).min()
print("NEAR/USDT regime:")
print(f"  current:   ${last:.4f}")
print(f"  SMA200:    ${sma200:.4f}  (price/SMA200 = {last/sma200-1:+.1%})")
print(f"  SMA50:     ${sma50:.4f}  (price/SMA50 = {last/sma50-1:+.1%})")
print(f"  60d high:  ${high60:.4f}  ({(last-high60)/high60:+.1%} from peak)")
print(f"  60d low:   ${low60:.4f}  ({(last-low60)/low60:+.1%} from trough)")
print()

print("XSMOM short-leg post-mortem (1d returns since rebalance):")
for pair in ["LINK/USDT", "SOL/USDT", "ATOM/USDT", "ETH/USDT"]:
    df = data.ohlcv_extended(pair, days_back=5)
    ret_1d = df["close"].iloc[-1] / df["close"].iloc[-2] - 1
    print(f"  {pair:<12s} 1d ret: {ret_1d:+.2%}")
print()
print("For XSMOM to win: longs (LINK, SOL) must outperform shorts (ATOM, ETH).")
print("Check the spread:")
links_avg = 0
shorts_avg = 0
for pair in ["LINK/USDT", "SOL/USDT"]:
    df = data.ohlcv_extended(pair, days_back=2)
    links_avg += df["close"].iloc[-1] / df["close"].iloc[-2] - 1
links_avg /= 2
for pair in ["ATOM/USDT", "ETH/USDT"]:
    df = data.ohlcv_extended(pair, days_back=2)
    shorts_avg += df["close"].iloc[-1] / df["close"].iloc[-2] - 1
shorts_avg /= 2
spread = links_avg - shorts_avg
print(f"  Long basket 1d:  {links_avg:+.2%}")
print(f"  Short basket 1d: {shorts_avg:+.2%}")
print(f"  Long-Short spread: {spread:+.2%}  ({'WIN' if spread > 0 else 'LOSS'})")
