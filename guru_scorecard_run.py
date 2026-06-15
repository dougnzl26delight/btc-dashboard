"""Daily guru signal-scorecard updater (generic) — capture new directional calls
from each configured guru's tweets + grade any that have matured (30d), persist
a per-guru log. Mirrors olson_scorecard_run.py but for the non-Olson gurus in
core.guru_scorecard.GURU_CFGS (Benjamin Cowen, + any added later).
Scheduled task: Crypto_guru_scorecard_daily.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from core.guru_scorecard import GURU_CFGS, update_and_grade
    for key, cfg in GURU_CFGS.items():
        try:
            r = update_and_grade(cfg)
            msg = (f"[guru_scorecard:{key}] logged={r['n_logged']} scored={r['n_scored']} "
                   f"pending={r['n_pending']} hit={r['hit_rate_pct']}% "
                   f"payoff_R={r['payoff_R']} expectancy={r['expectancy_pct']}")
        except Exception as e:
            msg = f"[guru_scorecard:{key}] ERROR {type(e).__name__}: {e}"
        try:
            print(msg)
        except Exception:
            print(msg.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
