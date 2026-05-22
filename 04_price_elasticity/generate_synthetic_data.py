"""
Synthetic Price Elasticity Data Generator
==========================================

Generates 104 weeks of SKU-level pricing and sales data for a healthcare
retail (hearing aid) portfolio. The latent demand model is log-linear with
own-price elasticity, within-brand cross-elasticity, promotional lift,
and mild seasonality.

Portfolio:
    Phonak Audéo Lumity      — L30 (entry), L50 (mid), L70 (high), L90 (premium)
    Signia Pure Charge&Go    — 1X (entry), 3X (mid), 5X (high), 7X (premium)
    Oticon Real series       — 1 (entry), 2 (mid), 3 (high), 1MN (premium)

True elasticities are baked in by tier (more elastic for entry-level SKUs,
less elastic for premium) so the model's recovered estimates can be
validated against ground truth.

Run:
    python generate_synthetic_data.py

Outputs:
    data/transactions.csv    — weekly SKU-level pricing and sales
    data/true_elasticities.csv — ground-truth elasticities for validation
"""

from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================
N_WEEKS = 104  # 24 months of weekly data
START_DATE = pd.Timestamp("2024-01-01")

# Portfolio definition
SKUS = [
    # Phonak Audéo Lumity (Sonova)
    {"sku": "PH-AUD-L30",  "brand": "Phonak", "tier": "entry",   "regular_price": 1499, "base_weekly_qty": 30},
    {"sku": "PH-AUD-L50",  "brand": "Phonak", "tier": "mid",     "regular_price": 2799, "base_weekly_qty": 22},
    {"sku": "PH-AUD-L70",  "brand": "Phonak", "tier": "high",    "regular_price": 4199, "base_weekly_qty": 14},
    {"sku": "PH-AUD-L90",  "brand": "Phonak", "tier": "premium", "regular_price": 5499, "base_weekly_qty": 9},
    # Signia Pure Charge&Go (WS Audiology)
    {"sku": "SI-PUR-1X",   "brand": "Signia", "tier": "entry",   "regular_price": 1399, "base_weekly_qty": 26},
    {"sku": "SI-PUR-3X",   "brand": "Signia", "tier": "mid",     "regular_price": 2599, "base_weekly_qty": 19},
    {"sku": "SI-PUR-5X",   "brand": "Signia", "tier": "high",    "regular_price": 3899, "base_weekly_qty": 12},
    {"sku": "SI-PUR-7X",   "brand": "Signia", "tier": "premium", "regular_price": 5199, "base_weekly_qty": 8},
    # Oticon Real (Demant)
    {"sku": "OT-REA-1",    "brand": "Oticon", "tier": "entry",   "regular_price": 1599, "base_weekly_qty": 22},
    {"sku": "OT-REA-2",    "brand": "Oticon", "tier": "mid",     "regular_price": 2899, "base_weekly_qty": 17},
    {"sku": "OT-REA-3",    "brand": "Oticon", "tier": "high",    "regular_price": 4099, "base_weekly_qty": 11},
    {"sku": "OT-REA-1MN",  "brand": "Oticon", "tier": "premium", "regular_price": 5399, "base_weekly_qty": 7},
]

# True elasticities by tier (more negative = more elastic = more price-sensitive)
# Entry-level buyers are price-sensitive; premium buyers care less about price.
TIER_ELASTICITY = {
    "entry":    -1.85,
    "mid":      -1.40,
    "high":     -1.05,
    "premium":  -0.70,
}

# Within-brand cross-price elasticity (substitutes — sister SKUs of same brand)
WITHIN_BRAND_CROSS_ELASTICITY = 0.28  # if Phonak L50 gets cheaper, L70 demand falls

# Promotional lift (additional log-demand boost during promotional weeks
# beyond what the price drop alone explains, e.g., advertising amplification)
PROMO_LIFT_COEF = 0.10

# Noise scale on log demand
NOISE_SD = 0.16

OUTPUT_DIR = Path(__file__).parent / "data"


# ============================================================================
# PRICE TRAJECTORY GENERATION
# ============================================================================
def generate_price_path(regular_price: float, n_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate weekly actual prices for a SKU.
    Most weeks at regular price; ~12-15% of weeks see promotional discounts.

    Returns:
        actual_price : np.ndarray of shape (n_weeks,)
        promo_flag   : np.ndarray of shape (n_weeks,), 0/1 indicator
    """
    price = np.full(n_weeks, regular_price, dtype=float)
    promo = np.zeros(n_weeks, dtype=int)

    # Sprinkle promotional events through the year
    week = 0
    while week < n_weeks:
        gap = np.random.randint(5, 11)
        if week + gap >= n_weeks:
            break
        promo_length = np.random.choice([1, 2, 3])
        discount = np.random.uniform(0.05, 0.25)  # 5–25% off
        end = min(week + gap + promo_length, n_weeks)
        price[week + gap:end] = regular_price * (1 - discount)
        promo[week + gap:end] = 1
        week = end

    # Add small random price jitter on non-promo weeks (regional pricing differences)
    non_promo = promo == 0
    price[non_promo] *= 1 + 0.015 * np.random.randn(non_promo.sum())

    return price, promo


# ============================================================================
# DEMAND GENERATION
# ============================================================================
def generate_demand(
    sku_records: dict[str, dict],
    n_weeks: int,
) -> dict[str, np.ndarray]:
    """
    Generate weekly quantity sold per SKU using the latent log-linear demand model:

        log Q_i,t = log(base_qty_i)
                    + elasticity_i * log(P_i,t / P_i,ref)
                    + cross_elasticity * log(P_avg_sister_brand,t / P_avg_sister_brand,ref)
                    + promo_lift * promo_indicator
                    + seasonality(t)
                    + noise

    where sister brand = average of other SKUs in the same brand.
    """
    n_skus = len(sku_records)
    # Pre-compute reference prices (the time-averaged price) for each SKU
    for sku_id, record in sku_records.items():
        record["ref_price"] = np.mean(record["price"])

    # Seasonality — mild Q4 lift (year-end insurance benefits)
    weeks_arr = np.arange(n_weeks)
    seasonality = 0.06 * np.sin(2 * np.pi * weeks_arr / 52 - np.pi / 2)

    quantity = {}
    for sku_id, record in sku_records.items():
        sku_meta = record["meta"]
        elasticity = TIER_ELASTICITY[sku_meta["tier"]]
        base_qty = sku_meta["base_weekly_qty"]

        # Own price effect
        log_price_ratio = np.log(record["price"] / record["ref_price"])
        own_price_effect = elasticity * log_price_ratio

        # Within-brand cross-price effect (avg price of other SKUs in same brand)
        sister_skus = [
            other for other, other_rec in sku_records.items()
            if other != sku_id and other_rec["meta"]["brand"] == sku_meta["brand"]
        ]
        if sister_skus:
            sister_prices = np.array(
                [sku_records[s]["price"] / sku_records[s]["ref_price"]
                 for s in sister_skus]
            )
            sister_log_ratio = np.log(sister_prices.mean(axis=0))
            cross_effect = WITHIN_BRAND_CROSS_ELASTICITY * sister_log_ratio
        else:
            cross_effect = 0

        promo_effect = PROMO_LIFT_COEF * record["promo"]
        noise = NOISE_SD * np.random.randn(n_weeks)

        log_qty = (
            np.log(base_qty)
            + own_price_effect
            + cross_effect
            + promo_effect
            + seasonality
            + noise
        )
        qty = np.exp(log_qty)
        # Round to whole units and ensure >= 1
        quantity[sku_id] = np.clip(np.round(qty).astype(int), 1, None)

    return quantity


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    print(f"Generating {N_WEEKS} weeks of synthetic SKU-level pricing data "
          f"across {len(SKUS)} SKUs ({len(set(s['brand'] for s in SKUS))} brands)...")

    dates = pd.date_range(start=START_DATE, periods=N_WEEKS, freq="W")

    # Generate price paths for each SKU
    sku_records: dict[str, dict] = {}
    for sku in SKUS:
        price, promo = generate_price_path(sku["regular_price"], N_WEEKS)
        sku_records[sku["sku"]] = {
            "meta": sku,
            "price": price,
            "promo": promo,
        }

    # Generate quantity from latent demand model
    quantity = generate_demand(sku_records, N_WEEKS)

    # Assemble long-format DataFrame
    rows = []
    for sku_id, record in sku_records.items():
        meta = record["meta"]
        for t in range(N_WEEKS):
            rows.append({
                "date":            dates[t],
                "week":            t + 1,
                "sku":             sku_id,
                "brand":           meta["brand"],
                "tier":            meta["tier"],
                "regular_price":   meta["regular_price"],
                "actual_price":    round(record["price"][t], 2),
                "promo_flag":      int(record["promo"][t]),
                "discount_pct":    round(
                    100 * (1 - record["price"][t] / meta["regular_price"]), 2
                ),
                "quantity":        int(quantity[sku_id][t]),
                "revenue":         round(record["price"][t] * quantity[sku_id][t], 2),
            })
    df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "transactions.csv", index=False)

    # Save true elasticities for later validation
    true_rows = []
    for sku in SKUS:
        true_rows.append({
            "sku":             sku["sku"],
            "brand":           sku["brand"],
            "tier":            sku["tier"],
            "true_elasticity": TIER_ELASTICITY[sku["tier"]],
            "regular_price":   sku["regular_price"],
            "base_weekly_qty": sku["base_weekly_qty"],
        })
    pd.DataFrame(true_rows).to_csv(
        OUTPUT_DIR / "true_elasticities.csv", index=False
    )

    print(f"\n✅ Saved {len(df):,} weekly observations")
    print(f"   → {OUTPUT_DIR / 'transactions.csv'}")
    print(f"   → {OUTPUT_DIR / 'true_elasticities.csv'}\n")

    # Diagnostics
    print("Portfolio summary:")
    summary = df.groupby(["brand", "tier", "sku"], as_index=False).agg(
        avg_price=("actual_price", "mean"),
        avg_qty=("quantity", "mean"),
        promo_weeks=("promo_flag", "sum"),
        total_revenue=("revenue", "sum"),
    )
    summary["promo_pct"] = (summary["promo_weeks"] / N_WEEKS * 100).round(1)
    summary = summary.drop(columns="promo_weeks")
    print(summary.to_string(index=False))

    print("\nTier-level conversion summary:")
    print(f"  {'tier':<10} {'avg price':>12} {'avg weekly qty':>15} "
          f"{'true elasticity':>16}")
    for tier in ["entry", "mid", "high", "premium"]:
        tier_df = df[df["tier"] == tier]
        ap = tier_df["actual_price"].mean()
        aq = tier_df["quantity"].mean()
        print(f"  {tier:<10} ${ap:>11,.0f}  {aq:>15.1f}  "
              f"{TIER_ELASTICITY[tier]:>16.2f}")


if __name__ == "__main__":
    main()
