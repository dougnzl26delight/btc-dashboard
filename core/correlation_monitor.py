"""Signal concordance monitor.

When many independent strategies agree on direction, two things could be
happening:
  1. A real regime change — strong, broad signal alignment is meaningful
  2. Strategy collapse — your strategies became redundant and lost diversity

Either way, when concordance gets high, the position SIZE matters more than
usual: you're either right and want to be in size, or wrong and about to
take a correlated hit. Alert so you can look at it.
"""

from __future__ import annotations

CONCORDANCE_ALERT_THRESHOLD = 0.85   # alert if 85%+ of strategies agree
CONCORDANCE_ZERO_THRESHOLD = 0.10    # signals below this magnitude treated as flat


def signal_concordance(signals: dict[str, float]) -> dict:
    """Concordance score in [0, 1]. 1 = unanimous, 0 = perfectly split."""
    if not signals:
        return {"score": 0.0, "n_signaling": 0, "direction": "flat"}

    signs = []
    for v in signals.values():
        try:
            f = float(v)
        except Exception:
            continue
        if f > CONCORDANCE_ZERO_THRESHOLD:
            signs.append(1)
        elif f < -CONCORDANCE_ZERO_THRESHOLD:
            signs.append(-1)

    n = len(signs)
    if n < 2:
        return {"score": 0.0, "n_signaling": n, "direction": "flat"}

    pos = sum(1 for s in signs if s > 0)
    neg = sum(1 for s in signs if s < 0)
    score = abs(pos - neg) / n
    direction = "bullish" if pos > neg else ("bearish" if neg > pos else "split")
    return {
        "score": float(score),
        "n_signaling": n,
        "n_pos": pos,
        "n_neg": neg,
        "direction": direction,
    }


def check_concordance(signals: dict[str, float], pair: str = "") -> dict:
    """Compute and (if extreme) alert. Returns the concordance dict."""
    from ops.alerts import alert
    c = signal_concordance(signals)
    if c["score"] >= CONCORDANCE_ALERT_THRESHOLD and c["n_signaling"] >= 3:
        alert(
            f"Signal concordance HIGH on {pair or 'portfolio'}: "
            f"{c['n_signaling']} strategies, {c['score']:.0%} agreement, "
            f"{c['direction']}. Inspect for regime change or strategy collapse.",
            level="warning",
        )
    return c
