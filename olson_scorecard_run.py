"""Daily Olson signal-scorecard updater — capture new directional calls from his
tweets + grade any that have matured (30d), persist to .olson_scorecard_log.json.
Scheduled task: Crypto_olson_scorecard_daily.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from core.olson_scorecard import update_and_grade
    r = update_and_grade()
    msg = (f"[olson_scorecard] logged={r['n_logged']} scored={r['n_scored']} "
           f"pending={r['n_pending']} hit={r['hit_rate_pct']}% "
           f"payoff_R={r['payoff_R']} expectancy={r['expectancy_pct']}")
    try:
        print(msg)
    except Exception:
        print(msg.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
