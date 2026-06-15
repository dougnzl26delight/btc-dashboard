"""Find recent BTC lows since Oct 2025 peak."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import data

df = data.ohlcv_extended("BTC/USDT", days_back=300)
recent = df[df.index >= "2025-10-06"]

print(f"Range since Oct 6, 2025 (cycle 4 top): {recent.index[0].date()} to {recent.index[-1].date()}")
print(f"Days: {len(recent)}")
print()
print("Top 15 LOWEST closes since cycle top:")
low15 = recent.nsmallest(15, "close")[["close", "high", "low"]]
for ts, row in low15.iterrows():
    print(f"  {ts.date()}   close ${row['close']:>8,.0f}   high ${row['high']:>8,.0f}   low ${row['low']:>8,.0f}")

print()
print("Drawdown stats:")
peak = 124659  # close on 2025-10-06
recent_min = recent["close"].min()
recent_min_date = recent["close"].idxmin().date()
current = df["close"].iloc[-1]
print(f"  Peak (Oct 6, 2025):    ${peak:,.0f}")
print(f"  Lowest close so far:   ${recent_min:,.0f}  (on {recent_min_date})  = {(recent_min/peak - 1):+.1%} from peak")
print(f"  Current price:         ${current:,.0f}  = {(current/peak - 1):+.1%} from peak")
print(f"  Days since peak:       {(df.index[-1] - df.index[df.index.get_loc('2025-10-06', method='nearest')]).days if '2025-10-06' in df.index.strftime('%Y-%m-%d').tolist() else 'n/a'}")
