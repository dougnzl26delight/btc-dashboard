"""Weekly red-team — the in-house devil's advocate.

A green dashboard is an echo chamber. This compiles the STRONGEST bear case
AGAINST the rotation campaign from live signals + structural risks, ranked by
current relevance, so conviction is stress-tested every week instead of
reinforced. Pairs with campaign_kill_criteria (what would prove me wrong).

NOT investment advice — an adversarial sanity check on your own thesis.
"""
from __future__ import annotations
from datetime import datetime, timezone

_SEV_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def red_team_report() -> dict:
    from core.dashboard_cache import get_cached as g
    from core.campaign_kill_criteria import campaign_thesis_check

    th = campaign_thesis_check()
    nb = g("btc_native_bottom_scorecard") or {}
    bot_n, bot_tot = nb.get("n_met") or 0, nb.get("n_total") or 16
    rt = g("rotation_trigger") or {}
    btc = rt.get("btc_price") or 0
    rt_overall = rt.get("overall", "?")
    rp = g("realized_price") or {}
    rpx = rp.get("value")

    args = []

    def arg(title, severity, evidence, why):
        args.append({"title": title, "severity": severity, "evidence": evidence, "why": why})

    # Promote any live kill-criteria warnings/trips to bear arguments first.
    for c in th.get("criteria", []):
        if c["status"] in ("WARNING", "TRIPPED"):
            arg(f"[Kill-criterion] {c['name']}",
                "HIGH" if c["status"] == "TRIPPED" else "MEDIUM",
                c.get("current", ""), c["detail"])

    # ── Standing structural bear arguments (always argued, ranked by relevance) ──
    if rpx and btc and btc > rpx:
        arg("The bottom is NOT confirmed — you may be early", "MEDIUM",
            f"Bottom scorecard {bot_n}/{bot_tot}; BTC {((btc/rpx-1)*100):+.0f}% above cost basis "
            f"(realized ${rpx:,.0f}).",
            "Valuation gauges (Swift MAX BUY, Cowen BUY) flash 'cheap' weeks-to-months BEFORE the "
            "actual low. The pull to deploy early is the trap — the real bottom could be far lower.")

    arg("The target could overshoot well below $52k", "MEDIUM",
        "Olson: bearish-W, bottom not in; his realized-cost anchor has drifted toward ~$52.8k.",
        "Targets are bands, not points. Capitulations routinely wick far below the 'expected' floor; "
        "anchoring on $52-57k risks catching a falling knife.")

    arg("Equities may melt up for years — your exit is hostage to a crash", "MEDIUM",
        f"Rotation status: {rt_overall}; equity-priority needs stocks to crack (ERP negative, but no "
        "actual stress yet).",
        "If the AI bull runs another 1-2 years you hold equities far longer than planned. The new "
        "BTC-LED BACKSTOP stops you missing the BTC bottom, but the equity-EXIT timing still depends "
        "on a decline that may not come when you expect.")

    arg("All-in on one volatile asset is structurally fragile", "MEDIUM",
        "The plan rotates ~93% of the book into a single asset at the bottom.",
        "Even a CORRECT bottom call can sit through a -40% post-entry drawdown. Concentration is how "
        "multi-year plans blow up — laddered entries, position sizing and a funded hedge are the antidote.")

    arg("Stocks AND Bitcoin could crash together (Burry/Hussman)", "MEDIUM",
        "Tail hedge currently OPTIONAL; BTC-equity correlation rises in stress events.",
        "The rotation assumes you exit stocks BEFORE the deep crash. In a liquidity event both fall at "
        "once — you'd rotate from one falling asset into another. Keep the tail hedge alive.")

    arg("Your edge is backtested on ~3 cycles — a tiny sample", "LOW",
        "45+ signals tuned on 2018/2022; this cycle is structurally different (ETF era, your own "
        "muting-detector confirms it).",
        "'Marked every bottom since 2013' is n≈3. The ETF regime may break the very relationships the "
        "scorecard relies on — the map may no longer match the territory.")

    args.sort(key=lambda a: _SEV_ORDER.get(a["severity"], 3))

    n_high = sum(1 for a in args if a["severity"] == "HIGH")
    headline = (
        "Thesis is BROKEN on a pre-registered criterion — stop and re-evaluate."
        if th.get("verdict") == "THESIS BROKEN" else
        "Conviction check: the direction holds, but here are the strongest arguments you're wrong."
        if th.get("verdict") in ("WATCH", "THESIS INTACT") else
        "Weekly conviction check.")

    return {
        "thesis_verdict": th.get("verdict"),
        "thesis_color": th.get("color"),
        "thesis_criteria": th.get("criteria", []),
        "headline": headline,
        "n_arguments": len(args),
        "n_high": n_high,
        "arguments": args,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def red_team_email_body() -> tuple[str, str]:
    """(subject, body) for the weekly red-team email — plain text, ASCII-safe."""
    r = red_team_report()
    try:
        from core.plain_email import plain_lead
        lead = plain_lead(
            r["headline"],
            [f"Campaign thesis health: {r['thesis_verdict']}.",
             f"{r['n_arguments']} bear arguments on the table ({r['n_high']} high-severity).",
             "This email exists to argue AGAINST your plan — read it as the opposition would."],
            "Read the bear case below. If none of it changes your mind, that's earned conviction. "
            "If one does, you found it here instead of in a drawdown.",
            mood=("act" if r["thesis_verdict"] == "THESIS BROKEN" else "watch"))
    except Exception:
        lead = f"WEEKLY RED-TEAM — {r['headline']}\n"

    lines = [lead, "=== THE BEAR CASE (strongest arguments you're wrong) ==="]
    for i, a in enumerate(r["arguments"], 1):
        lines.append("")
        lines.append(f"  {i}. [{a['severity']}] {a['title']}")
        if a.get("evidence"):
            lines.append(f"      Evidence: {a['evidence']}")
        lines.append(f"      Why it matters: {a['why']}")
    lines += ["", "--- This is a deliberate devil's advocate, not advice. ---",
              "Open dashboard: https://btcdelight.com"]
    body = "\n".join(lines)
    subject = (f"[RED-TEAM] Thesis {r['thesis_verdict']} — {r['n_arguments']} arguments you're wrong")
    return subject, body


if __name__ == "__main__":
    subj, body = red_team_email_body()
    print("SUBJECT:", subj)
    print(body.encode("ascii", "replace").decode())
