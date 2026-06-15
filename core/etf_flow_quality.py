"""ETF flow QUALITY — directional demand vs carry/basis.

A large spot-ETF 'inflow' that's really a basis trade (long ETF / short CME
futures, delta-neutral) is NOT bullish spot demand — it's yield harvesting.
Treating it as demand is a classic ETF-era trap. This crosses ETF net flow
against the CME basis / funding to label the flow's true character, refining
(not replacing) the existing ETF-flow signal.

Best-effort over the existing cached ETF / OI-funding panels; degrades to a
DATA GAP label if the feeds aren't present, rather than guessing.
"""
from __future__ import annotations

CARRY_BASIS_THRESHOLD = 0.10   # >10% annualised basis => carry-rich, demand suspect


def _first_num(d, keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def etf_flow_quality() -> dict:
    out = {"label": "UNKNOWN", "net_flow": None, "basis": None, "detail": ""}
    try:
        from core.dashboard_cache import get_cached
        ef = get_cached("etf_flow_overlay") or get_cached("etf_regime") or {}
        cb = get_cached("oi_funding_overlay") or {}

        nf = _first_num(ef, ("flows_5d_M", "flows_30d_M", "flows_60d_M",
                             "net_flow_5d", "net_flow", "flow_5d", "trailing_flow", "net_usd"))
        basis = _first_num(cb, ("cme_basis_annualized", "basis_annualized",
                                "funding_annualized", "basis", "annualized_basis"))
        out["net_flow"], out["basis"] = nf, basis

        if nf is None:
            out["label"] = "DATA GAP"
            out["detail"] = "no ETF net-flow read available"
            return out
        if nf < 0:
            out["label"] = "DISTRIBUTION (net outflow)"
        elif basis is not None and basis > CARRY_BASIS_THRESHOLD:
            out["label"] = "CARRY/BASIS-DRIVEN (not real spot demand)"
        elif basis is None:
            # honest: a positive flow without a basis read can't be confirmed as
            # real demand vs carry — don't assert demand we haven't verified.
            out["label"] = "INFLOW — quality unverified (no basis read)"
        else:
            out["label"] = "REAL DIRECTIONAL DEMAND"
        b = f"{basis*100:.0f}%" if isinstance(basis, (int, float)) else "n/a"
        out["detail"] = f"net flow ${nf:+.0f}M; annualised basis {b}"
    except Exception as e:
        out["detail"] = f"unavailable ({type(e).__name__})"
    return out


if __name__ == "__main__":
    r = etf_flow_quality()
    print(f"ETF FLOW QUALITY: {r['label']}")
    print(f"  {r['detail']}")
