"""
Run Price Elasticity Analysis
==============================

Loads SKU-level transaction data, fits per-SKU log-log elasticity models
and within-brand cross-price elasticity matrices, then produces five
visualizations:

    1. Own-price elasticity by SKU (with 95% CIs, grouped by brand)
    2. Demand curves for selected SKUs (scatter + fitted curve + CI band)
    3. Within-brand cross-price elasticity heatmap (Phonak)
    4. Revenue impact of a hypothetical 10% price increase
    5. Promotional lift analysis (observed vs counterfactual)

Run:
    python run_analysis.py

Requires: data/transactions.csv (run generate_synthetic_data.py first)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from elasticity_model import (
    fit_all_skus,
    fit_cross_price_matrix,
    fit_own_price_elasticity,
    demand_curve,
    pricing_recommendation,
    revenue_change_estimate,
)


DATA_PATH = Path(__file__).parent / "data" / "transactions.csv"
TRUE_PARAMS_PATH = Path(__file__).parent / "data" / "true_elasticities.csv"
OUTPUT_DIR = Path(__file__).parent / "output"

BRAND_COLORS = {
    "Phonak": "#1a73e8",
    "Signia": "#34a853",
    "Oticon": "#ea4335",
}
TIER_ORDER = ["entry", "mid", "high", "premium"]


# ============================================================================
# VISUALIZATIONS
# ============================================================================
def plot_elasticity_comparison(
    summary: pd.DataFrame,
    true_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar chart of own-price elasticity per SKU with 95% CIs, grouped by brand."""
    # Order: brand then tier
    summary = summary.copy()
    summary["tier_rank"] = summary["tier"].map(
        {t: i for i, t in enumerate(TIER_ORDER)}
    )
    summary = summary.sort_values(["brand", "tier_rank"]).reset_index(drop=True)
    summary = summary.merge(true_df[["sku", "true_elasticity"]], on="sku")

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(summary))
    err_low = summary["elasticity"] - summary["ci_low"]
    err_high = summary["ci_high"] - summary["elasticity"]

    bar_colors = [BRAND_COLORS[b] for b in summary["brand"]]
    ax.bar(x, summary["elasticity"], color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.5, label="Estimated elasticity")
    ax.errorbar(x, summary["elasticity"], yerr=[err_low, err_high],
                fmt="none", ecolor="#1f2937", elinewidth=1.2, capsize=4)
    # Truth markers
    ax.scatter(x, summary["true_elasticity"], color="#1f2937", marker="D",
               s=55, zorder=5, label="True elasticity (ground truth)",
               edgecolor="white", linewidth=1)

    # Unit-elastic reference line
    ax.axhline(-1.0, color="#9aa3af", linestyle="--", linewidth=1,
               alpha=0.7, label="Unit-elastic (ε = −1)")

    # Tier labels under bars
    labels = [f"{row['sku']}\n({row['tier']})"
              for _, row in summary.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Own-price elasticity (β)", fontsize=11)
    ax.set_title("Own-price elasticity by SKU — estimated vs ground truth",
                 fontsize=13, pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Custom legend with brand colors
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BRAND_COLORS["Phonak"], alpha=0.85, label="Phonak"),
        plt.Rectangle((0, 0), 1, 1, color=BRAND_COLORS["Signia"], alpha=0.85, label="Signia"),
        plt.Rectangle((0, 0), 1, 1, color=BRAND_COLORS["Oticon"], alpha=0.85, label="Oticon"),
        plt.Line2D([0], [0], marker="D", color="#1f2937", linewidth=0,
                   markersize=8, label="True elasticity"),
        plt.Line2D([0], [0], color="#9aa3af", linestyle="--", label="Unit-elastic (ε = −1)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_demand_curves(
    df: pd.DataFrame,
    skus_to_show: list[str],
    output_path: Path,
) -> None:
    """2x3 grid of demand curves with scatter, fitted curve, and CI band."""
    n_cols = 3
    n_rows = (len(skus_to_show) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, sku in enumerate(skus_to_show):
        ax = axes[i]
        fit = fit_own_price_elasticity(df, sku, include_sister=True)
        sku_df = df[df["sku"] == sku]
        brand = sku_df["brand"].iloc[0]
        tier = sku_df["tier"].iloc[0]
        color = BRAND_COLORS[brand]

        # Scatter of observed (price, quantity)
        ax.scatter(sku_df["actual_price"], sku_df["quantity"],
                   color=color, alpha=0.45, s=35, edgecolor="white", linewidth=0.5,
                   label="Observed weekly")

        # Fitted demand curve over price grid
        p_min = sku_df["actual_price"].min() * 0.92
        p_max = sku_df["actual_price"].max() * 1.08
        grid = np.linspace(p_min, p_max, 80)
        curve = demand_curve(fit, grid)
        ax.plot(grid, curve["predicted"], color=color, linewidth=2.5,
                label=f"Fitted: ε = {fit['elasticity']:.2f}")
        ax.fill_between(grid, curve["lower"], curve["upper"],
                         color=color, alpha=0.18, label="95% CI")

        ax.set_xlabel("Price ($)", fontsize=10)
        ax.set_ylabel("Quantity / week", fontsize=10)
        ax.set_title(f"{sku}  —  {brand} {tier}", fontsize=11, pad=10)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    # Hide unused subplots
    for i in range(len(skus_to_show), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Estimated demand curves — observed weekly data and log-log fit",
                  fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_cross_elasticity_heatmap(
    matrix: pd.DataFrame,
    brand: str,
    output_path: Path,
) -> None:
    """Heatmap of within-brand cross-price elasticity matrix."""
    # Diverging colormap centered at 0
    abs_max = max(abs(matrix.values.min()), abs(matrix.values.max()))
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rwb", ["#ea4335", "#ffffff", "#1a73e8"]
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto")

    skus = matrix.index.tolist()
    ax.set_xticks(np.arange(len(skus)))
    ax.set_yticks(np.arange(len(skus)))
    ax.set_xticklabels(skus, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(skus, fontsize=10)
    ax.set_xlabel("Cross-price SKU (j)", fontsize=11, labelpad=10)
    ax.set_ylabel("Demand SKU (i)", fontsize=11, labelpad=10)
    ax.set_title(
        f"Cross-price elasticity matrix — {brand}\n"
        f"Diagonal = own-price (negative); off-diagonal = cross-price (positive → substitutes)",
        fontsize=12, pad=15,
    )

    # Annotate cells
    for i in range(len(skus)):
        for j in range(len(skus)):
            val = matrix.values[i, j]
            if pd.isna(val):
                continue
            color = "white" if abs(val) > abs_max * 0.5 else "#1f2937"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    color=color, fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elasticity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_revenue_impact(
    summary: pd.DataFrame,
    pct_change: float,
    output_path: Path,
) -> None:
    """Bar chart of expected % revenue change from a hypothetical price change."""
    summary = summary.copy()
    summary["tier_rank"] = summary["tier"].map(
        {t: i for i, t in enumerate(TIER_ORDER)}
    )
    summary = summary.sort_values(["brand", "tier_rank"]).reset_index(drop=True)

    # % change in revenue ≈ (1 + ε) × % change in price
    summary["pct_revenue_change"] = (1 + summary["elasticity"]) * pct_change

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(summary))
    bar_colors = [
        "#1a73e8" if v > 0 else "#ea4335" for v in summary["pct_revenue_change"]
    ]
    bars = ax.bar(x, summary["pct_revenue_change"], color=bar_colors,
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, summary["pct_revenue_change"]):
        offset = 0.4 if val >= 0 else -0.4
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.1f}%", ha="center", fontsize=9,
                va="bottom" if val >= 0 else "top", color="#1f2937")

    ax.axhline(0, color="#1f2937", linewidth=1)
    labels = [f"{row['sku']}\n({row['tier']})"
              for _, row in summary.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"Expected % change in revenue from a +{pct_change:.0f}% price move",
                  fontsize=11)
    ax.set_title(
        f"Revenue impact of a hypothetical +{pct_change:.0f}% price increase by SKU\n"
        "Blue (positive) = inelastic SKUs where the price increase grows revenue. "
        "Red (negative) = elastic SKUs where revenue falls.",
        fontsize=12, pad=15,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_promotional_lift(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    For each SKU, compare actual avg weekly quantity during promo weeks vs
    non-promo weeks, alongside the implied lift from observed price drop alone
    (via own-price elasticity).
    """
    rows = []
    for _, row in summary.iterrows():
        sku = row["sku"]
        sku_df = df[df["sku"] == sku]
        promo_qty    = sku_df.loc[sku_df["promo_flag"] == 1, "quantity"].mean()
        nonpromo_qty = sku_df.loc[sku_df["promo_flag"] == 0, "quantity"].mean()
        promo_price    = sku_df.loc[sku_df["promo_flag"] == 1, "actual_price"].mean()
        nonpromo_price = sku_df.loc[sku_df["promo_flag"] == 0, "actual_price"].mean()
        if pd.isna(promo_qty) or pd.isna(nonpromo_qty) or nonpromo_qty == 0:
            continue

        observed_lift = (promo_qty / nonpromo_qty - 1) * 100
        # Lift implied by elasticity from observed price change alone
        pct_price_change = (promo_price / nonpromo_price - 1) * 100
        implied_lift = row["elasticity"] * pct_price_change

        rows.append({
            "sku":            sku,
            "brand":          row["brand"],
            "tier":           row["tier"],
            "observed_lift":  observed_lift,
            "implied_lift":   implied_lift,
            "incremental":    observed_lift - implied_lift,
        })
    promo_df = pd.DataFrame(rows)
    promo_df["tier_rank"] = promo_df["tier"].map(
        {t: i for i, t in enumerate(TIER_ORDER)}
    )
    promo_df = promo_df.sort_values(["brand", "tier_rank"]).reset_index(drop=True)

    x = np.arange(len(promo_df))
    bar_width = 0.4
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - bar_width / 2, promo_df["observed_lift"], bar_width,
           color="#1a73e8", alpha=0.85, edgecolor="white",
           label="Observed lift during promotional weeks")
    ax.bar(x + bar_width / 2, promo_df["implied_lift"], bar_width,
           color="#9aa3af", alpha=0.85, edgecolor="white",
           label="Lift implied by own-price elasticity alone")

    labels = [f"{row['sku']}\n({row['tier']})"
              for _, row in promo_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% change in weekly quantity (promo vs non-promo)", fontsize=11)
    ax.set_title(
        "Promotional lift — observed vs implied by price-elasticity alone\n"
        "Gap = incremental lift from promotional effects beyond pure price (advertising, halo, etc.)",
        fontsize=12, pad=15,
    )
    ax.axhline(0, color="#1f2937", linewidth=1)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")
    return promo_df


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    if not DATA_PATH.exists():
        print(f"❌ Data file not found at {DATA_PATH}")
        print(f"   Run: python generate_synthetic_data.py")
        sys.exit(1)

    print(f"Loading transactions from {DATA_PATH.name}...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"   Loaded {len(df):,} weekly observations across "
          f"{df['sku'].nunique()} SKUs and {df['brand'].nunique()} brands\n")

    true_df = pd.read_csv(TRUE_PARAMS_PATH) if TRUE_PARAMS_PATH.exists() \
              else None

    # ----------------------------------------------------------------
    # Fit own-price elasticity for each SKU
    # ----------------------------------------------------------------
    print("Fitting per-SKU own-price elasticity (log-log OLS)...")
    summary = fit_all_skus(df, include_sister=True)
    print("\nElasticity summary:")
    cols_to_show = ["sku", "brand", "tier", "elasticity",
                    "ci_low", "ci_high", "r_squared"]
    print(summary[cols_to_show].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    # ----------------------------------------------------------------
    # Cross-price elasticity matrices per brand
    # ----------------------------------------------------------------
    print("\nFitting within-brand cross-price elasticity matrices...")
    cross_matrices: dict[str, pd.DataFrame] = {}
    for brand in df["brand"].unique():
        cross_matrices[brand] = fit_cross_price_matrix(df, brand)
        print(f"   ✓ {brand}")

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "elasticity_summary.csv", index=False)
    for brand, mat in cross_matrices.items():
        mat.to_csv(OUTPUT_DIR / f"cross_elasticity_{brand.lower()}.csv")

    # ----------------------------------------------------------------
    # Visualizations
    # ----------------------------------------------------------------
    print("\nSaving visualizations...")
    if true_df is not None:
        plot_elasticity_comparison(
            summary, true_df, OUTPUT_DIR / "elasticity_comparison.png"
        )

    # Pick 6 representative SKUs for demand curve grid:
    # entry + premium for each brand to contrast elastic vs inelastic
    selected = ["PH-AUD-L30", "PH-AUD-L90",
                "SI-PUR-1X",  "SI-PUR-7X",
                "OT-REA-1",   "OT-REA-1MN"]
    plot_demand_curves(df, selected, OUTPUT_DIR / "demand_curves.png")

    plot_cross_elasticity_heatmap(
        cross_matrices["Phonak"], "Phonak",
        OUTPUT_DIR / "cross_elasticity_phonak.png",
    )

    plot_revenue_impact(
        summary, pct_change=10.0,
        output_path=OUTPUT_DIR / "revenue_impact.png",
    )

    promo_df = plot_promotional_lift(
        df, summary, OUTPUT_DIR / "promotional_lift.png"
    )
    promo_df.to_csv(OUTPUT_DIR / "promotional_lift.csv", index=False)

    # ----------------------------------------------------------------
    # Pricing recommendations summary
    # ----------------------------------------------------------------
    print("\nPricing recommendations by SKU:")
    print(f"  {'sku':<14} {'tier':<10} {'elasticity':>11} "
          f"{'+10% rev impact':>17}  recommendation")
    print("  " + "-" * 110)
    for _, row in summary.sort_values(
        ["brand", "tier"], key=lambda c: c.map({t: i for i, t in enumerate(TIER_ORDER)})
        if c.name == "tier" else c
    ).iterrows():
        rev_change = (1 + row["elasticity"]) * 10
        rec = pricing_recommendation(row["elasticity"])
        print(f"  {row['sku']:<14} {row['tier']:<10} "
              f"{row['elasticity']:>11.2f} {rev_change:>+15.1f}%   {rec}")

    print("\n" + "=" * 70)
    print("Key takeaway")
    print("=" * 70)
    inelastic = summary[summary["elasticity"] > -1.0]
    print(f"  {len(inelastic)} of {len(summary)} SKUs are inelastic "
          f"(|ε| < 1) — these are candidates for price increases.")
    print(f"  Premium SKUs show the lowest elasticity, consistent with the")
    print(f"  classic finding that premium buyers are less price-sensitive.")


if __name__ == "__main__":
    main()
