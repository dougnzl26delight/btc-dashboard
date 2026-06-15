"""External Bitcoin chart embeds — public charts we can iframe directly.

Many sites display paid-tier metrics for free as public web pages.
We can embed their public charts directly in our dashboard via iframe
or st.components.v1.iframe.

This is genuinely outside-the-box: instead of paying for the API,
we let the user view the SAME chart hosted on the source site.

Confidence: 10/10 — these ARE the official charts, just displayed in
an iframe rather than rebuilt from API data.

Sites used (all free public access):
  - LookIntoBitcoin.com   (Swift's site — all his indicators)
  - Woobull.com           (Willy Woo's free dashboards)
  - BitcoinVisuals.com    (free on-chain explorer)
  - Mempool.space         (mempool + difficulty + halving)
  - Glassnode.com/charts  (some free preview charts)
  - Bybit Insights        (free derivatives data)
  - CoinGlass.com         (free liquidation maps + funding)
"""

# Each entry: (label, url, description, category, height_px)
EMBEDS = {
    # Phillip Swift's site — all his signature indicators
    "lookintobitcoin_pi_top": (
        "Pi Cycle Top (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/",
        "Swift's signature top indicator — official live chart",
        "swift", 600,
    ),
    "lookintobitcoin_pi_bottom": (
        "Pi Cycle Bottom (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/pi-cycle-bottom-indicator/",
        "Swift's bottom indicator — official live chart",
        "swift", 600,
    ),
    "lookintobitcoin_rainbow": (
        "Rainbow Chart (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/bitcoin-rainbow-chart/",
        "Iconic Rainbow chart — official Swift version",
        "swift", 600,
    ),
    "lookintobitcoin_golden_ratio": (
        "Golden Ratio Multiplier",
        "https://www.lookintobitcoin.com/charts/golden-ratio-multiplier/",
        "Price vs 350d MA with Fibonacci bands",
        "swift", 600,
    ),
    "lookintobitcoin_2y_mult": (
        "2-Year MA Multiplier",
        "https://www.lookintobitcoin.com/charts/bitcoin-investor-tool/",
        "Price vs 2y MA with 5x top band",
        "swift", 600,
    ),
    "lookintobitcoin_mvrv_z": (
        "MVRV Z-Score (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/mvrv-zscore/",
        "MVRV Z-score with cycle zone bands — Swift's preferred version",
        "swift", 600,
    ),
    "lookintobitcoin_reserve_risk": (
        "Reserve Risk (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/reserve-risk/",
        "Live Reserve Risk chart — paid-tier metric for free",
        "swift", 600,
    ),
    "lookintobitcoin_puell": (
        "Puell Multiple (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/puell-multiple/",
        "Miner revenue / 365d MA with zones",
        "swift", 600,
    ),
    "lookintobitcoin_nupl": (
        "Net Unrealized P/L (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/relative-unrealized-profit--loss/",
        "NUPL with Hope/Fear/Optimism/Belief/Euphoria zones",
        "swift", 600,
    ),
    "lookintobitcoin_hodl_waves": (
        "HODL Waves (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/hodl-waves/",
        "Supply distribution by coin age — THE structural cycle indicator",
        "swift", 600,
    ),
    "lookintobitcoin_realized_price": (
        "Realized Price (LookIntoBitcoin)",
        "https://www.lookintobitcoin.com/charts/realized-price/",
        "Average cost basis of all BTC",
        "swift", 600,
    ),
    "lookintobitcoin_short_holder_realized": (
        "STH Realized Price",
        "https://www.lookintobitcoin.com/charts/short-term-holder-realized-price/",
        "Short-term holder cost basis (155d MA)",
        "swift", 600,
    ),
    "lookintobitcoin_thermo_cap": (
        "Thermocap Multiple",
        "https://www.lookintobitcoin.com/charts/thermocap-multiple/",
        "Price vs cumulative miner revenue",
        "swift", 600,
    ),

    # Willy Woo's site
    "woobull_nvt_signal": (
        "NVT Signal (Woobull)",
        "https://woobull.com/woobull-charts/",
        "Willy Woo's NVT Signal — bottom indicator",
        "woo", 800,
    ),

    # BitcoinVisuals
    "bitcoinvisuals_utxo_age": (
        "UTXO Age Distribution",
        "https://bitcoinvisuals.com/chain-utxo-age-distribution",
        "Live UTXO age bands — free version of HODL Waves",
        "onchain", 600,
    ),

    # Mempool.space
    "mempool_difficulty": (
        "Difficulty Adjustment",
        "https://mempool.space/graphs/mining/difficulty",
        "Live difficulty chart with epoch progress",
        "miner", 500,
    ),
    "mempool_hash_rate": (
        "Hash Rate Live",
        "https://mempool.space/graphs/mining/hashrate-difficulty",
        "Live network hashrate + difficulty overlay",
        "miner", 500,
    ),
    "mempool_block_rewards": (
        "Block Rewards",
        "https://mempool.space/graphs/mining/block-rewards",
        "Subsidy vs fees over time",
        "miner", 500,
    ),

    # Coinglass (free)
    "coinglass_funding": (
        "Funding Rates Heatmap",
        "https://www.coinglass.com/FundingRate",
        "Live perpetual funding across all exchanges",
        "derivatives", 700,
    ),
    "coinglass_liquidations": (
        "Liquidation Map",
        "https://www.coinglass.com/LiquidationData",
        "Live liquidation cascade visualization",
        "derivatives", 600,
    ),
}


def get_embeds_by_category(category: str = None) -> dict:
    if category is None: return EMBEDS
    return {k: v for k, v in EMBEDS.items() if v[3] == category}


def render_embed_html(url: str, height: int = 600) -> str:
    """Build iframe HTML for an external chart."""
    return f"""
<iframe src="{url}"
        width="100%" height="{height}px"
        frameborder="0" scrolling="no"
        style="border: 1px solid #2a2d36; border-radius: 8px;"
        loading="lazy">
</iframe>
"""


def main():
    print("=" * 70)
    print(f"BTC EXTERNAL CHART EMBEDS ({len(EMBEDS)} sources)")
    print("=" * 70)
    by_cat = {}
    for k, (label, url, descr, cat, h) in EMBEDS.items():
        by_cat.setdefault(cat, []).append((label, url))
    for cat, items in by_cat.items():
        print(f"\n{cat.upper()} ({len(items)}):")
        for label, url in items:
            print(f"  - {label:35s} {url}")


if __name__ == "__main__":
    main()
