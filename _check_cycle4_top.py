"""Find the actual cycle 4 top in our data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import data

df = data.ohlcv_extended("BTC/USDT", days_back=3000)
recent = df[df.index >= "2024-01-01"]

print("Highest CLOSE in cycle 4 (since Jan 2024):")
idx = recent["close"].idxmax()
print(f"  Date: {idx.date()}")
print(f"  Close: ${recent.loc[idx, 'close']:,.0f}")
print(f"  Same-day high: ${recent.loc[idx, 'high']:,.0f}")
print()

print("Highest INTRADAY HIGH in cycle 4:")
idx = recent["high"].idxmax()
print(f"  Date: {idx.date()}")
print(f"  High: ${recent.loc[idx, 'high']:,.0f}")
print(f"  Close that day: ${recent.loc[idx, 'close']:,.0f}")
print()

print("Top 15 highest CLOSE days in cycle 4 (Jan 2024+):")
top = recent.nlargest(15, "close")[["close", "high", "low"]]
for ts, row in top.iterrows():
    print(f"  {ts.date()}   close ${row['close']:>8,.0f}   high ${row['high']:>8,.0f}   low ${row['low']:>8,.0f}")

print()
print(f"Current BTC price: ${df['close'].iloc[-1]:,.0f}")
print(f"As of date: {df.index[-1].date()}")
