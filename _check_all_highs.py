"""Show all major BTC highs in our 8yr dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import data

df = data.ohlcv_extended("BTC/USDT", days_back=3000)

print(f"Data window: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total bars: {len(df)}")
print()

print("=" * 80)
print("ALL-TIME HIGH (intraday high)")
print("=" * 80)
idx = df["high"].idxmax()
print(f"  Date: {idx.date()}")
print(f"  High: ${df.loc[idx, 'high']:,.0f}")
print(f"  Close that day: ${df.loc[idx, 'close']:,.0f}")
print()

print("=" * 80)
print("Top 30 highest INTRADAY HIGHS in the 8yr data")
print("=" * 80)
top30 = df.nlargest(30, "high")[["close", "high"]]
for ts, row in top30.iterrows():
    print(f"  {ts.date()}   high ${row['high']:>10,.0f}   close ${row['close']:>10,.0f}")

print()
print("=" * 80)
print("MAJOR PEAKS BY MONTH (highest each calendar month)")
print("=" * 80)
df_with_month = df.copy()
df_with_month["ym"] = df_with_month.index.to_period("M")
monthly_max = df_with_month.groupby("ym")["high"].max().sort_values(ascending=False).head(30)
for ym, hi in monthly_max.items():
    print(f"  {ym}: ${hi:,.0f}")

print()
print(f"Current BTC: ${df['close'].iloc[-1]:,.0f}  (as of {df.index[-1].date()})")
