import json
from pathlib import Path

root = Path(__file__).resolve().parent
files = sorted(root.glob(".pro_trend_state_*.json"))
print(f"{'State file':<40s}  {'Side':<6s}  {'Units':>5s}  {'TrailStop':>12s}")
for f in files:
    s = json.loads(f.read_text())
    side = s.get("side") or "flat"
    n_units = len(s.get("units", []))
    trail = float(s.get("trail_stop", 0))
    print(f"{f.name:<40s}  {side:<6s}  {n_units:>5d}  ${trail:>10,.4f}")
