"""
Run Markov Chain Attribution Analysis
======================================

Loads the synthetic journey data, fits the Markov attribution model,
compares against four heuristic baselines, and produces visualizations.

Run:
    python run_analysis.py

Requires: data/journeys.csv (run generate_synthetic_data.py first)

Outputs:
    output/attribution_comparison.png  — bar chart comparing all 5 methods
    output/transition_matrix.png       — heatmap of channel-to-channel transitions
    output/attribution_results.csv     — full numerical results
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from markov_attribution import (
    MarkovAttribution,
    compare_methods,
    journeys_to_paths,
)


DATA_PATH = Path(__file__).parent / "data" / "journeys.csv"
OUTPUT_DIR = Path(__file__).parent / "output"


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================
def plot_attribution_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart comparing attribution shares across methods."""
    methods = ["markov", "first_touch", "last_touch", "linear", "time_decay"]
    method_labels = {
        "markov": "Markov (removal effect)",
        "first_touch": "First-touch",
        "last_touch": "Last-touch",
        "linear": "Linear",
        "time_decay": "Time-decay",
    }
    method_colors = {
        "markov": "#1a73e8",
        "first_touch": "#9aa3af",
        "last_touch": "#5f6b7a",
        "linear": "#cfd4dc",
        "time_decay": "#7e8a9a",
    }

    channels = comparison["channel"].tolist()
    n_channels = len(channels)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(11, 6))
    bar_width = 0.16
    x = np.arange(n_channels)

    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            comparison[method],
            bar_width,
            label=method_labels[method],
            color=method_colors[method],
            edgecolor="white",
            linewidth=0.5,
        )
        # Highlight Markov bars
        if method == "markov":
            for bar in bars:
                bar.set_edgecolor("#0b3d91")
                bar.set_linewidth(1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(channels, fontsize=11)
    ax.set_ylabel("Attribution Share", fontsize=11)
    ax.set_title(
        "Attribution share by channel: Markov vs heuristic methods",
        fontsize=13, pad=15,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_transition_matrix(
    matrix_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap of the channel-to-channel transition matrix."""
    # Hide pseudo-states from heatmap for readability, keep only channels
    channels = [s for s in matrix_df.index
                if s not in {"(start)", "(conversion)", "(null)"}]
    sub = matrix_df.loc[channels, channels]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blues_clean", ["#f4f6f8", "#1a73e8"]
    )
    im = ax.imshow(sub.values, cmap=cmap, vmin=0, vmax=sub.values.max())

    ax.set_xticks(np.arange(len(channels)))
    ax.set_yticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(channels, fontsize=10)
    ax.set_xlabel("To channel", fontsize=11, labelpad=10)
    ax.set_ylabel("From channel", fontsize=11, labelpad=10)
    ax.set_title("Channel-to-channel transition probabilities",
                 fontsize=13, pad=15)

    # Annotate cells
    for i in range(len(channels)):
        for j in range(len(channels)):
            val = sub.values[i, j]
            color = "white" if val > sub.values.max() * 0.5 else "#1f2937"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Transition probability")
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

    print(f"Loading journey data from {DATA_PATH.name}...")
    touchpoints = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(touchpoints):,} touchpoint rows")

    paths = journeys_to_paths(touchpoints)
    print(f"   Collapsed to {len(paths):,} journeys "
          f"({paths['converted'].sum():,} conversions)\n")

    # ----------------------------------------------------------------
    # Fit Markov attribution
    # ----------------------------------------------------------------
    print("Fitting Markov chain attribution model...")
    model = MarkovAttribution(paths).fit()
    print(f"   Base conversion probability: {model.base_conversion_prob_:.4f}")
    print(f"   Channels detected: {', '.join(model.channels)}\n")

    markov_result = model.attribution()
    print("Markov attribution (removal-effect method):")
    print(markov_result.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print()

    # ----------------------------------------------------------------
    # Compare to heuristic methods
    # ----------------------------------------------------------------
    print("Comparing to heuristic baselines...")
    comparison = compare_methods(paths)
    print("\nAttribution share comparison across methods:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.1%}" if isinstance(x, float) else x,
        )
    )

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "attribution_results.csv", index=False)
    markov_result.to_csv(OUTPUT_DIR / "markov_attribution.csv", index=False)
    model.transition_matrix_df().to_csv(OUTPUT_DIR / "transition_matrix.csv")

    print(f"\nSaving visualizations...")
    plot_attribution_comparison(
        comparison, OUTPUT_DIR / "attribution_comparison.png"
    )
    plot_transition_matrix(
        model.transition_matrix_df(),
        OUTPUT_DIR / "transition_matrix.png",
    )

    # ----------------------------------------------------------------
    # Interpretive summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Key takeaway: how Markov differs from heuristic methods")
    print("=" * 70)
    for _, row in comparison.iterrows():
        ch = row["channel"]
        markov_share = row["markov"]
        last_share = row["last_touch"]
        diff = markov_share - last_share
        direction = "more" if diff > 0 else "less"
        print(
            f"  {ch:<14} Markov: {markov_share:>6.1%}  vs  "
            f"Last-touch: {last_share:>6.1%}   →  "
            f"Markov gives {abs(diff):.1%} {direction} credit"
        )


if __name__ == "__main__":
    main()
