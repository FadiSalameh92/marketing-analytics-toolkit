"""
Run Bayesian Marketing Mix Model Analysis
==========================================

Loads the synthetic weekly spend + sales data, fits the Bayesian MMM,
and produces visualizations:

    - Saturation curves per channel (response vs spend, diminishing returns)
    - Sales decomposition over time (base + channel contributions)
    - ROI estimates per channel (posterior mean vs ground truth)
    - Adstock decay curves per channel
    - Posterior parameter recovery vs true parameters

Run:
    python run_analysis.py

Requires: data/mmm_weekly_data.csv (run generate_synthetic_data.py first)

Note: MCMC sampling takes ~1-3 minutes on a typical laptop.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mmm_model import (
    BayesianMMM,
    geometric_adstock_np,
    hill_saturation_np,
)


DATA_PATH = Path(__file__).parent / "data" / "mmm_weekly_data.csv"
TRUE_PARAMS_PATH = Path(__file__).parent / "data" / "true_parameters.csv"
OUTPUT_DIR = Path(__file__).parent / "output"

CHANNELS = ["paid_search", "paid_social", "direct_mail", "tv", "radio"]
CHANNEL_COLORS = {
    "paid_search": "#1a73e8",
    "paid_social": "#34a853",
    "direct_mail": "#fbbc04",
    "tv":          "#ea4335",
    "radio":       "#9c27b0",
}


# ============================================================================
# VISUALIZATIONS
# ============================================================================
def plot_saturation_curves(
    mmm: BayesianMMM,
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot saturation response curves per channel, with annotated avg spend."""
    fig, ax = plt.subplots(figsize=(10, 6))
    means = mmm.posterior_means()
    max_spend_K = max(data[f"spend_{ch}"].max() for ch in CHANNELS) * 1.2
    spend_grid = np.linspace(0.01, max_spend_K, 200)

    for ch in CHANNELS:
        p = means[ch]
        response = p["beta"] * hill_saturation_np(
            spend_grid, p["half_sat"], p["slope"]
        )
        ax.plot(spend_grid, response, color=CHANNEL_COLORS[ch],
                linewidth=2.2, label=ch)

        # Marker at the avg weekly spend
        avg_spend = data[f"spend_{ch}"].mean()
        avg_response = p["beta"] * hill_saturation_np(
            np.array([avg_spend]), p["half_sat"], p["slope"]
        )[0]
        ax.scatter([avg_spend], [avg_response],
                   color=CHANNEL_COLORS[ch], s=70, zorder=5,
                   edgecolor="white", linewidth=1.5)

    ax.set_xlabel("Weekly spend ($K)", fontsize=11)
    ax.set_ylabel("Sales response ($K) — saturation effect only", fontsize=11)
    ax.set_title("Channel saturation curves with diminishing returns",
                 fontsize=13, pad=15)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.98,
            "● marker = avg weekly spend per channel",
            transform=ax.transAxes, fontsize=9, color="#5f6b7a",
            verticalalignment="top")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_sales_decomposition(
    mmm: BayesianMMM,
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Stacked area chart of sales decomposition over time."""
    contrib = mmm.channel_contributions()

    # Build the base layer: total observed sales minus channel contributions
    channel_total = sum(contrib[ch] for ch in CHANNELS)
    base_layer = data["sales"].values - channel_total.values

    fig, ax = plt.subplots(figsize=(12, 6))
    dates = data["date"]

    # Stacked layers
    layers = [base_layer]
    layer_labels = ["base + trend + seasonality"]
    layer_colors = ["#cfd4dc"]
    for ch in CHANNELS:
        layers.append(contrib[ch].values)
        layer_labels.append(ch)
        layer_colors.append(CHANNEL_COLORS[ch])

    ax.stackplot(dates, *layers,
                 labels=layer_labels,
                 colors=layer_colors,
                 alpha=0.85,
                 edgecolor="white",
                 linewidth=0.3)

    # Overlay observed sales
    ax.plot(dates, data["sales"], color="#1f2937",
            linewidth=1.2, label="observed sales", linestyle="--")

    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Weekly sales ($K)", fontsize=11)
    ax.set_title("Sales decomposition: base + channel contributions over time",
                 fontsize=13, pad=15)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_roi_comparison(
    roi_estimates: pd.DataFrame,
    true_params: pd.DataFrame,
    output_path: Path,
) -> None:
    """Side-by-side bar comparison of posterior ROI vs true ROI."""
    merged = roi_estimates.merge(
        true_params[["channel", "implied_roi"]].rename(
            columns={"implied_roi": "true_roi"}
        ),
        on="channel",
    )
    merged = merged.sort_values("true_roi", ascending=True)

    channels = merged["channel"].tolist()
    posterior = merged["roi"].values
    truth = merged["true_roi"].values
    y = np.arange(len(channels))
    height = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(y - height/2, posterior, height,
            color="#1a73e8", label="Posterior estimate",
            edgecolor="white", linewidth=0.5)
    ax.barh(y + height/2, truth, height,
            color="#9aa3af", label="True ROI",
            edgecolor="white", linewidth=0.5)

    # Annotations
    for i, (p, t) in enumerate(zip(posterior, truth)):
        ax.text(p + 0.1, i - height/2, f"{p:.2f}x",
                va="center", fontsize=10, color="#1f2937")
        ax.text(t + 0.1, i + height/2, f"{t:.2f}x",
                va="center", fontsize=10, color="#5f6b7a")

    ax.set_yticks(y)
    ax.set_yticklabels(channels)
    ax.set_xlabel("ROI (sales contribution per dollar spent)", fontsize=11)
    ax.set_title("ROI by channel: posterior estimate vs ground truth",
                 fontsize=13, pad=15)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_adstock_curves(
    mmm: BayesianMMM,
    output_path: Path,
) -> None:
    """Plot adstock decay curves: response to a $1 spend pulse at week 0."""
    means = mmm.posterior_means()
    lags = np.arange(mmm.max_lag)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for ch in CHANNELS:
        p = means[ch]
        # Geometric adstock weights
        weights = p["decay"] ** lags
        weights = weights / weights.sum()
        ax.plot(lags, weights, marker="o", color=CHANNEL_COLORS[ch],
                linewidth=2, markersize=7, label=ch)

    ax.set_xlabel("Weeks since spend", fontsize=11)
    ax.set_ylabel("Adstock weight (fraction of total carryover)", fontsize=11)
    ax.set_title("Adstock decay curves — how spend carries forward over time",
                 fontsize=13, pad=15)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(lags)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    if not DATA_PATH.exists():
        print(f"❌ Data file not found at {DATA_PATH}")
        print(f"   Run: python generate_synthetic_data.py")
        sys.exit(1)

    print(f"Loading weekly MMM data from {DATA_PATH.name}...")
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"   Loaded {len(data)} weeks of data\n")

    # ----------------------------------------------------------------
    # Fit Bayesian MMM
    # ----------------------------------------------------------------
    print("Building Bayesian MMM...")
    mmm = BayesianMMM(data, CHANNELS, max_lag=8)

    print("Running MCMC sampling (this takes 1-3 minutes)...")
    t0 = time.time()
    mmm.fit(draws=1000, tune=1000, chains=2, cores=1, progressbar=False)
    elapsed = time.time() - t0
    print(f"   Sampling completed in {elapsed:.1f} seconds\n")

    # ----------------------------------------------------------------
    # Posterior summary
    # ----------------------------------------------------------------
    print("Posterior parameter summary (94% HDI):")
    summary = mmm.parameter_summary()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # ----------------------------------------------------------------
    # ROI estimates
    # ----------------------------------------------------------------
    print("ROI estimates from posterior means:")
    roi = mmm.roi_estimates()
    print(roi.to_string(index=False))
    print()

    # ----------------------------------------------------------------
    # Compare to ground truth
    # ----------------------------------------------------------------
    if TRUE_PARAMS_PATH.exists():
        true_params = pd.read_csv(TRUE_PARAMS_PATH)
        print("Parameter recovery vs ground truth:")
        posterior = mmm.posterior_means()
        recovery_rows = []
        for ch in CHANNELS:
            true_row = true_params[true_params["channel"] == ch].iloc[0]
            for param_name, true_col in [
                ("decay", "true_decay"),
                ("half_sat", "true_half_sat"),
                ("slope", "true_slope"),
                ("beta", "true_beta"),
            ]:
                recovery_rows.append({
                    "channel": ch,
                    "parameter": param_name,
                    "posterior_mean": round(posterior[ch][param_name], 3),
                    "true": round(true_row[true_col], 3),
                    "abs_pct_error": round(
                        abs(posterior[ch][param_name] - true_row[true_col])
                        / abs(true_row[true_col]) * 100, 1
                    ),
                })
        recovery_df = pd.DataFrame(recovery_rows)
        print(recovery_df.to_string(index=False))
        print()

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving CSV outputs...")
    summary.to_csv(OUTPUT_DIR / "posterior_summary.csv", index=False)
    roi.to_csv(OUTPUT_DIR / "roi_estimates.csv", index=False)
    if TRUE_PARAMS_PATH.exists():
        recovery_df.to_csv(OUTPUT_DIR / "parameter_recovery.csv", index=False)
    mmm.channel_contributions().to_csv(
        OUTPUT_DIR / "channel_contributions.csv", index=False
    )

    print("\nSaving visualizations...")
    plot_saturation_curves(mmm, data, OUTPUT_DIR / "saturation_curves.png")
    plot_sales_decomposition(mmm, data, OUTPUT_DIR / "sales_decomposition.png")
    plot_adstock_curves(mmm, OUTPUT_DIR / "adstock_decay.png")
    if TRUE_PARAMS_PATH.exists():
        plot_roi_comparison(
            roi, true_params, OUTPUT_DIR / "roi_comparison.png"
        )

    # ----------------------------------------------------------------
    # Interpretive summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Key takeaways from this MMM run")
    print("=" * 70)
    top_roi = roi.iloc[0]
    bottom_roi = roi.iloc[-1]
    print(f"  Highest ROI channel:   {top_roi['channel']:<14} "
          f"({top_roi['roi']:.2f}x — every $1 returned ${top_roi['roi']:.2f})")
    print(f"  Lowest ROI channel:    {bottom_roi['channel']:<14} "
          f"({bottom_roi['roi']:.2f}x)")

    total_spend = sum(data[f'spend_{ch}'].sum() for ch in CHANNELS)
    total_contribution = roi['total_contribution_K'].sum()
    print(f"  Total marketing spend: ${total_spend:,.0f}K")
    print(f"  Total contribution:    ${total_contribution:,.0f}K")
    print(f"  Blended ROI:           {total_contribution / total_spend:.2f}x")


if __name__ == "__main__":
    main()
