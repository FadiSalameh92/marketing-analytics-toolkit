"""
Run XGBoost Lead Scoring Analysis
==================================

Trains the XGBoost lead scoring model on synthetic data, compares against
a logistic regression baseline, and produces visualizations:

    - Feature importance bar chart
    - ROC curve (XGBoost vs baseline)
    - Precision-Recall curve
    - Calibration curve (predicted vs observed probability)
    - Lift curve and capacity analysis chart
    - Score distribution by outcome

Run:
    python run_analysis.py

Requires: data/leads.csv (run generate_synthetic_data.py first)
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

from lead_scoring_model import (
    XGBoostLeadScorer,
    LogisticBaseline,
    lift_curve_data,
    capacity_analysis,
)


DATA_PATH = Path(__file__).parent / "data" / "leads.csv"
OUTPUT_DIR = Path(__file__).parent / "output"


# ============================================================================
# VISUALIZATIONS
# ============================================================================
def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Horizontal bar chart of XGBoost feature importance."""
    df = importance_df.sort_values("importance_normalized", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(df["feature"], df["importance_normalized"],
                    color="#1a73e8", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, df["importance_normalized"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10, color="#1f2937")

    ax.set_xlabel("Relative importance (normalized gain)", fontsize=11)
    ax.set_title("Feature importance — what XGBoost uses to score leads",
                 fontsize=13, pad=15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(0, df["importance_normalized"].max() * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    xgb_proba: np.ndarray,
    log_proba: np.ndarray,
    xgb_auc: float,
    log_auc: float,
    output_path: Path,
) -> None:
    """ROC curves comparing XGBoost vs logistic baseline."""
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_proba)
    fpr_log, tpr_log, _ = roc_curve(y_true, log_proba)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(fpr_xgb, tpr_xgb, color="#1a73e8", linewidth=2.4,
            label=f"XGBoost  (AUC = {xgb_auc:.3f})")
    ax.plot(fpr_log, tpr_log, color="#9aa3af", linewidth=2,
            label=f"Logistic  (AUC = {log_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#cfd4dc", linewidth=1.2,
            linestyle="--", label="Random (AUC = 0.5)")

    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("ROC curve — XGBoost vs logistic regression baseline",
                 fontsize=13, pad=15)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_lift_curve(
    lift_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Lift curve showing capture rate vs % of leads called."""
    fig, ax = plt.subplots(figsize=(9, 6))
    # Reference: random calling
    ax.plot([0, 100], [0, 100], color="#cfd4dc", linewidth=1.5,
            linestyle="--", label="Random (no model)")

    # Lift curve
    ax.plot(lift_df["bin_pct"], lift_df["capture_pct"],
            color="#1a73e8", linewidth=2.6, marker="o", markersize=5,
            label="XGBoost-ranked")
    ax.fill_between(lift_df["bin_pct"], lift_df["bin_pct"],
                     lift_df["capture_pct"],
                     color="#1a73e8", alpha=0.10)

    # Annotate key points
    for target_pct in [10, 20, 30]:
        row = lift_df.iloc[(lift_df["bin_pct"] - target_pct).abs().argmin()]
        ax.annotate(
            f"Top {target_pct:.0f}% → {row['capture_pct']:.0f}% captured\n"
            f"({row['lift']:.2f}x lift)",
            xy=(row["bin_pct"], row["capture_pct"]),
            xytext=(row["bin_pct"] + 10, row["capture_pct"] - 13),
            fontsize=9, color="#1f2937",
            arrowprops=dict(arrowstyle="->", color="#5f6b7a", lw=1),
        )

    ax.set_xlabel("% of leads called (sorted by model score, descending)",
                  fontsize=11)
    ax.set_ylabel("% of total conversions captured", fontsize=11)
    ax.set_title("Lift curve — model-ranked calling vs random",
                 fontsize=13, pad=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    n_bins: int = 15,
) -> None:
    """Reliability diagram: predicted probability vs observed conversion rate."""
    # Equal-frequency binning
    order = np.argsort(y_proba)
    y_sorted = y_true[order]
    proba_sorted = y_proba[order]
    bin_edges = np.linspace(0, len(y_proba), n_bins + 1, dtype=int)
    bins = []
    for i in range(n_bins):
        s, e = bin_edges[i], bin_edges[i + 1]
        if e > s:
            bins.append({
                "predicted": proba_sorted[s:e].mean(),
                "observed":  y_sorted[s:e].mean(),
                "count":     e - s,
            })
    cal_df = pd.DataFrame(bins)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0, 1], [0, 1], color="#cfd4dc", linewidth=1.5,
            linestyle="--", label="Perfect calibration")
    ax.plot(cal_df["predicted"], cal_df["observed"],
            color="#1a73e8", marker="o", markersize=8, linewidth=2,
            label="XGBoost model")

    # Scatter sizes proportional to bin count
    ax.scatter(cal_df["predicted"], cal_df["observed"],
                s=cal_df["count"] / cal_df["count"].max() * 200,
                color="#1a73e8", alpha=0.35, edgecolor="white", linewidth=1)

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed conversion rate", fontsize=11)
    ax.set_title("Calibration curve — are predicted probabilities reliable?",
                 fontsize=13, pad=15)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(cal_df["predicted"].max(), cal_df["observed"].max()) * 1.1)
    ax.set_ylim(0, max(cal_df["predicted"].max(), cal_df["observed"].max()) * 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   → {output_path}")


def plot_score_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Histogram of predicted probabilities, split by true outcome."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(0, max(y_proba.max(), 1.0), 35)

    ax.hist(y_proba[y_true == 0], bins=bins, color="#9aa3af", alpha=0.75,
            label=f"Did not convert (n = {(y_true == 0).sum():,})",
            edgecolor="white", linewidth=0.3)
    ax.hist(y_proba[y_true == 1], bins=bins, color="#1a73e8", alpha=0.75,
            label=f"Converted (n = {(y_true == 1).sum():,})",
            edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Predicted probability of conversion", fontsize=11)
    ax.set_ylabel("Number of leads", fontsize=11)
    ax.set_title("Score distribution by actual outcome — does the model separate classes?",
                 fontsize=13, pad=15)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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

    print(f"Loading leads from {DATA_PATH.name}...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df):,} leads "
          f"({df['converted'].sum():,} conversions, "
          f"{df['converted'].mean():.1%} rate)\n")

    # ----------------------------------------------------------------
    # Train / validation / test split (60 / 20 / 20)
    # ----------------------------------------------------------------
    train_df, temp_df = train_test_split(
        df, test_size=0.40, stratify=df["converted"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["converted"], random_state=42
    )
    print(f"Split — train: {len(train_df):,} | "
          f"val: {len(val_df):,} | test: {len(test_df):,}\n")

    # ----------------------------------------------------------------
    # Train XGBoost
    # ----------------------------------------------------------------
    print("Training XGBoost with early stopping...")
    scorer = XGBoostLeadScorer()
    scorer.fit(train_df, val_df=val_df)
    xgb_metrics = scorer.evaluate(test_df)

    # ----------------------------------------------------------------
    # Train logistic baseline
    # ----------------------------------------------------------------
    print("Training logistic regression baseline...")
    baseline = LogisticBaseline()
    baseline.fit(train_df)
    log_metrics = baseline.evaluate(test_df)

    # ----------------------------------------------------------------
    # Print metrics comparison
    # ----------------------------------------------------------------
    print("\nTest set metrics:")
    print(f"  {'metric':<14} {'XGBoost':>10} {'Logistic':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    for metric in ["auc_roc", "auc_pr", "brier_score"]:
        print(f"  {metric:<14} {xgb_metrics[metric]:>10.4f} "
              f"{log_metrics[metric]:>10.4f}")

    # ----------------------------------------------------------------
    # Feature importance
    # ----------------------------------------------------------------
    importance_df = scorer.feature_importance()
    print(f"\nTop features (XGBoost gain importance):")
    print(importance_df.head(10).to_string(index=False))

    # ----------------------------------------------------------------
    # Capacity analysis at multiple thresholds
    # ----------------------------------------------------------------
    y_test = test_df["converted"].values
    y_proba = scorer.predict_proba(test_df)

    print(f"\nCapacity analysis — if call-center can call top X% of leads:")
    print(f"  {'capacity':<12} {'leads':>8} {'captures':>10} {'pct':>8} {'lift':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
    for pct in [5, 10, 20, 30, 50, 100]:
        cap = capacity_analysis(y_test, y_proba, capacity_pct=float(pct))
        print(f"  top {pct:>2}%      {cap['leads_called']:>8,} "
              f"{cap['conversions_captured']:>10,} "
              f"{cap['capture_pct']:>7.1f}% "
              f"{cap['lift_vs_random']:>5.2f}x")

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([xgb_metrics, log_metrics],
                 index=["xgboost", "logistic"]).to_csv(
        OUTPUT_DIR / "model_metrics.csv"
    )
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    lift_df = lift_curve_data(y_test, y_proba)
    lift_df.to_csv(OUTPUT_DIR / "lift_curve.csv", index=False)

    print("\nSaving visualizations...")
    plot_feature_importance(importance_df,
                             OUTPUT_DIR / "feature_importance.png")
    plot_roc_curves(y_test, y_proba, baseline.predict_proba(test_df),
                    xgb_metrics["auc_roc"], log_metrics["auc_roc"],
                    OUTPUT_DIR / "roc_curves.png")
    plot_lift_curve(lift_df, OUTPUT_DIR / "lift_curve.png")
    plot_calibration_curve(y_test, y_proba,
                            OUTPUT_DIR / "calibration_curve.png")
    plot_score_distribution(y_test, y_proba,
                             OUTPUT_DIR / "score_distribution.png")

    # ----------------------------------------------------------------
    # Interpretive summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Operational interpretation — how this model would be used")
    print("=" * 72)
    cap_20 = capacity_analysis(y_test, y_proba, capacity_pct=20.0)
    print(f"  If the call-center has capacity for the top 20% of inbound leads")
    print(f"  ({cap_20['leads_called']:,} of {len(test_df):,} test leads),")
    print(f"  ranking by XGBoost score captures "
          f"{cap_20['capture_pct']:.0f}% of all conversions —")
    print(f"  a {cap_20['lift_vs_random']:.1f}x lift vs calling leads in a random order.")
    print()
    print(f"  Top feature drivers: "
          f"{', '.join(importance_df['feature'].head(3).tolist())}.")


if __name__ == "__main__":
    main()
