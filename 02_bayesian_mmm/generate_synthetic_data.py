"""
Synthetic Marketing Mix Model Data Generator
=============================================

Generates 104 weeks of weekly marketing spend and sales data with:
    - 5 channels (paid_search, paid_social, direct_mail, tv, radio)
    - Realistic spend patterns (continuous vs. flighted vs. campaign-based)
    - Known true parameters for adstock, saturation, and channel effects
    - Underlying base demand, trend, and seasonality

The true parameters are saved to data/true_parameters.csv so the MMM
model's posterior estimates can be validated against ground truth.

Run:
    python generate_synthetic_data.py

Outputs:
    data/mmm_weekly_data.csv     — weekly spend + sales
    data/true_parameters.csv     — ground truth parameters per channel
"""

from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================
N_WEEKS = 104                       # ~2 years of weekly data
START_DATE = pd.Timestamp("2024-01-01")

CHANNELS = ["paid_search", "paid_social", "direct_mail", "tv", "radio"]

# True parameters (the MMM should recover these from the data)
TRUE_PARAMS = {
    "paid_search": {
        "decay":      0.10,  # quick burn — search clicks act fast
        "half_sat":   18.0,  # in $K of weekly spend
        "slope":      1.6,
        "beta":     220.0,   # contribution coefficient
        "base_spend": 15.0,  # avg weekly spend in $K
        "spend_pattern": "continuous_stable",
    },
    "paid_social": {
        "decay":      0.30,
        "half_sat":   12.0,
        "slope":      1.8,
        "beta":     150.0,
        "base_spend": 10.0,
        "spend_pattern": "continuous_seasonal",
    },
    "direct_mail": {
        "decay":      0.45,  # slow burn — physical mail lingers
        "half_sat":    8.0,
        "slope":      1.4,
        "beta":     110.0,
        "base_spend":  6.0,
        "spend_pattern": "campaign_based",
    },
    "tv": {
        "decay":      0.55,  # long carryover
        "half_sat":   25.0,
        "slope":      2.0,
        "beta":     180.0,
        "base_spend": 18.0,
        "spend_pattern": "flighted",
    },
    "radio": {
        "decay":      0.40,
        "half_sat":   10.0,
        "slope":      1.7,
        "beta":      85.0,
        "base_spend":  7.0,
        "spend_pattern": "flighted",
    },
}

BASE_SALES_WEEKLY = 400.0  # in $K — sales with no marketing
TREND_PER_WEEK = 0.8       # gentle organic trend in $K
SALES_NOISE_SD = 20.0      # in $K — Gaussian noise on observed sales

OUTPUT_DIR = Path(__file__).parent / "data"


# ============================================================================
# SPEND PATTERN GENERATORS
# ============================================================================
def generate_continuous_stable(base_spend: float, n_weeks: int) -> np.ndarray:
    """Paid search: continuous, fairly stable with small weekly variation."""
    spend = base_spend * (1 + 0.15 * np.random.randn(n_weeks))
    return np.clip(spend, 0.1, None)


def generate_continuous_seasonal(base_spend: float, n_weeks: int) -> np.ndarray:
    """Paid social: continuous with seasonal amplification."""
    weeks = np.arange(n_weeks)
    seasonal = 1.0 + 0.35 * np.sin(2 * np.pi * weeks / 52 - np.pi / 2)
    noise = 1 + 0.10 * np.random.randn(n_weeks)
    spend = base_spend * seasonal * noise
    return np.clip(spend, 0.1, None)


def generate_campaign_based(base_spend: float, n_weeks: int) -> np.ndarray:
    """Direct mail: sporadic campaigns ~once every 6-10 weeks."""
    spend = np.zeros(n_weeks)
    pos = 0
    while pos < n_weeks:
        gap = np.random.randint(6, 11)
        if pos + gap >= n_weeks:
            break
        # 2-3 week campaign at elevated spend
        campaign_length = np.random.choice([2, 3])
        campaign_spend = base_spend * np.random.uniform(3.0, 4.5)
        end = min(pos + gap + campaign_length, n_weeks)
        spend[pos + gap:end] = campaign_spend * (1 + 0.10 * np.random.randn(end - pos - gap))
        pos = end
    # Add light always-on baseline
    spend = spend + base_spend * 0.4 * (1 + 0.10 * np.random.randn(n_weeks))
    return np.clip(spend, 0.1, None)


def generate_flighted(base_spend: float, n_weeks: int) -> np.ndarray:
    """TV/Radio: alternating heavy and light weeks (flighting strategy)."""
    spend = np.zeros(n_weeks)
    in_flight = False
    weeks_in_state = 0
    for t in range(n_weeks):
        if weeks_in_state == 0:
            in_flight = not in_flight
            duration = np.random.randint(4, 9) if in_flight else np.random.randint(2, 5)
            weeks_in_state = duration
        if in_flight:
            spend[t] = base_spend * np.random.uniform(1.8, 2.8)
        else:
            spend[t] = base_spend * np.random.uniform(0.15, 0.45)
        weeks_in_state -= 1
    spend = spend * (1 + 0.10 * np.random.randn(n_weeks))
    return np.clip(spend, 0.1, None)


PATTERN_FUNCTIONS = {
    "continuous_stable":   generate_continuous_stable,
    "continuous_seasonal": generate_continuous_seasonal,
    "campaign_based":      generate_campaign_based,
    "flighted":            generate_flighted,
}


# ============================================================================
# ADSTOCK + SATURATION (used to compute true sales contribution)
# ============================================================================
def geometric_adstock(spend: np.ndarray, decay: float, max_lag: int = 8) -> np.ndarray:
    """
    Geometric adstock as a weighted sum of recent spend with decay rate.
    adstocked_t = sum_{i=0}^{max_lag-1} (decay^i * spend_{t-i}) / normalizer
    """
    weights = decay ** np.arange(max_lag)
    weights = weights / weights.sum()
    n = len(spend)
    adstocked = np.zeros(n)
    for lag in range(max_lag):
        if lag == 0:
            adstocked += weights[lag] * spend
        else:
            lagged = np.concatenate([np.zeros(lag), spend[:-lag]])
            adstocked += weights[lag] * lagged
    return adstocked


def hill_saturation(x: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    """
    Hill saturation function — bounded between 0 and 1.
    Captures diminishing returns: response rises quickly at low spend,
    plateaus as spend grows.
    """
    return x ** slope / (half_sat ** slope + x ** slope)


# ============================================================================
# MAIN GENERATION
# ============================================================================
def main() -> None:
    print(f"Generating {N_WEEKS} weeks of synthetic MMM data across "
          f"{len(CHANNELS)} channels...")

    # Build the date axis
    dates = pd.date_range(start=START_DATE, periods=N_WEEKS, freq="W")

    # Generate spend for each channel
    spend_data = {}
    for channel in CHANNELS:
        params = TRUE_PARAMS[channel]
        generator = PATTERN_FUNCTIONS[params["spend_pattern"]]
        spend_data[channel] = generator(params["base_spend"], N_WEEKS)

    # Compute true sales contribution for each channel
    contributions = {}
    for channel in CHANNELS:
        params = TRUE_PARAMS[channel]
        adstocked = geometric_adstock(spend_data[channel], params["decay"])
        saturated = hill_saturation(adstocked, params["half_sat"], params["slope"])
        contributions[channel] = params["beta"] * saturated

    # Build sales: base + trend + seasonality + channel contributions + noise
    weeks = np.arange(N_WEEKS)
    trend = TREND_PER_WEEK * weeks
    seasonality = 30.0 * np.sin(2 * np.pi * weeks / 52 - np.pi / 4)
    base_with_trend = BASE_SALES_WEEKLY + trend + seasonality
    total_contribution = sum(contributions.values())
    noise = SALES_NOISE_SD * np.random.randn(N_WEEKS)
    sales = base_with_trend + total_contribution + noise

    # Assemble main dataset
    df = pd.DataFrame({"date": dates})
    for channel in CHANNELS:
        df[f"spend_{channel}"] = spend_data[channel].round(2)
    df["sales"] = sales.round(2)
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["trend_week"] = weeks

    # Save data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "mmm_weekly_data.csv", index=False)

    # Save true parameters for later validation
    true_param_rows = []
    for channel in CHANNELS:
        p = TRUE_PARAMS[channel]
        total_spend = spend_data[channel].sum()
        total_contribution = contributions[channel].sum()
        roi = total_contribution / total_spend if total_spend > 0 else 0
        true_param_rows.append({
            "channel": channel,
            "true_decay": p["decay"],
            "true_half_sat": p["half_sat"],
            "true_slope": p["slope"],
            "true_beta": p["beta"],
            "total_spend_K": round(total_spend, 1),
            "total_contribution_K": round(total_contribution, 1),
            "implied_roi": round(roi, 2),
        })
    true_df = pd.DataFrame(true_param_rows)
    true_df.to_csv(OUTPUT_DIR / "true_parameters.csv", index=False)

    print(f"\n✅ Saved {len(df)} weeks of MMM data")
    print(f"   → {OUTPUT_DIR / 'mmm_weekly_data.csv'}")
    print(f"   → {OUTPUT_DIR / 'true_parameters.csv'}\n")

    print("Dataset summary:")
    print(f"  Date range:        {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Avg weekly sales:  ${df['sales'].mean():.1f}K")
    print(f"  Min / max sales:   ${df['sales'].min():.1f}K / ${df['sales'].max():.1f}K\n")

    print("Channel spend summary (in $K):")
    for channel in CHANNELS:
        col = f"spend_{channel}"
        total = df[col].sum()
        weekly_avg = df[col].mean()
        print(f"  {channel:<14} total ${total:>7.1f}K   avg ${weekly_avg:>5.1f}K/wk")

    print("\nTrue ROI (sales contribution / spend) by channel:")
    for row in true_param_rows:
        print(f"  {row['channel']:<14} ROI = {row['implied_roi']:>5.2f}x   "
              f"(contributed ${row['total_contribution_K']:>6.1f}K from ${row['total_spend_K']:>6.1f}K spend)")


if __name__ == "__main__":
    main()
