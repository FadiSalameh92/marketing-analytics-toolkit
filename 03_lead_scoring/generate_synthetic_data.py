"""
Synthetic Lead Scoring Data Generator
======================================

Generates ~25,000 synthetic inbound leads with features and conversion outcomes
for a healthcare retail (hearing aid) context. Features include:

    - lead_source            categorical  — channel that originated the lead
    - age                    integer      — lead age in years
    - gender                 categorical  — M / F / Other
    - region                 categorical  — Canadian province
    - audiology_assessment   categorical  — hearing loss severity (or none)
    - prior_consultations    integer      — number of past consultations
    - prior_purchases        integer      — number of prior product purchases
    - days_since_first_touch integer      — recency of lead
    - touchpoint_count       integer      — engagement depth
    - call_time_of_day       categorical  — morning / afternoon / evening
    - day_of_week            categorical  — weekday / weekend

The 'converted' outcome is generated from a latent log-odds model so the
underlying signal structure is known — XGBoost should learn to recover it.

Run:
    python generate_synthetic_data.py

Outputs:
    data/leads.csv — one row per lead
"""

from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================
N_LEADS = 25_000

LEAD_SOURCES = ["paid_search", "paid_social", "direct_mail", "tv",
                "organic", "referral"]
LEAD_SOURCE_WEIGHTS = [0.28, 0.22, 0.15, 0.10, 0.18, 0.07]

GENDERS = ["M", "F", "Other"]
GENDER_WEIGHTS = [0.46, 0.51, 0.03]

REGIONS = ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB"]
REGION_WEIGHTS = [0.38, 0.23, 0.13, 0.11, 0.04, 0.03, 0.05, 0.03]

AUDIOLOGY_LEVELS = ["none", "mild", "moderate", "severe", "profound"]
AUDIOLOGY_WEIGHTS = [0.18, 0.27, 0.32, 0.17, 0.06]

CALL_TIMES = ["morning", "afternoon", "evening"]
CALL_TIME_WEIGHTS = [0.42, 0.43, 0.15]

OUTPUT_DIR = Path(__file__).parent / "data"


# ============================================================================
# LATENT CONVERSION MODEL
# ============================================================================
# Generates conversion probability from features via a log-odds model.
# This is the "true" signal structure XGBoost will try to recover.

AUDIOLOGY_EFFECT = {
    "none":     -0.40,
    "mild":      0.20,
    "moderate":  1.10,
    "severe":    2.00,
    "profound":  2.80,
}

SOURCE_EFFECT = {
    "paid_search":  0.55,
    "paid_social":  0.10,
    "direct_mail":  0.30,
    "tv":          -0.10,
    "organic":      0.45,
    "referral":     0.90,
}

CALL_TIME_EFFECT = {
    "morning":    0.15,
    "afternoon":  0.20,
    "evening":   -0.05,
}


def latent_log_odds(row: dict) -> float:
    """Compute the latent log-odds of conversion for a lead."""
    score = -5.0  # baseline log-odds — calibrated to produce ~12% overall conversion

    # Audiology severity — strongest signal
    score += AUDIOLOGY_EFFECT[row["audiology_assessment"]]

    # Prior purchases — very strong (already a customer)
    score += 1.4 * min(row["prior_purchases"], 2)

    # Age — peaks around 70-75 for hearing aids
    age = row["age"]
    score += 0.045 * (age - 50) - 0.0015 * (age - 72) ** 2

    # Lead source
    score += SOURCE_EFFECT[row["lead_source"]]

    # Prior consultations — interest signal
    score += 0.35 * min(row["prior_consultations"], 3)

    # Touchpoint count — engagement
    score += 0.13 * min(row["touchpoint_count"], 10)

    # Recency — stale leads convert less
    days = row["days_since_first_touch"]
    if days > 30:
        score -= 0.04 * (days - 30)

    # Call time
    score += CALL_TIME_EFFECT[row["call_time_of_day"]]

    # Weekend leads slightly lower (less follow-up)
    if row["day_of_week"] == "weekend":
        score -= 0.20

    # Mild gender effect (audiology demographic skew)
    if row["gender"] == "M":
        score += 0.10

    # Noise
    score += np.random.normal(0, 0.55)

    return score


def log_odds_to_prob(x: float) -> float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# DATA GENERATION
# ============================================================================
def generate_lead(lead_id: int) -> dict:
    """Generate a single synthetic lead with realistic feature distributions."""
    # Age — skewed toward older for hearing aid retail
    age = int(np.clip(np.random.normal(67, 13), 35, 95))

    # Touchpoint count — geometric distribution
    touchpoints = int(np.clip(np.random.geometric(p=0.30), 1, 15))

    # Recency — exponential
    days_since = int(np.clip(np.random.exponential(scale=22), 0, 180))

    # Prior consultations and purchases — Poisson, prior_purchases ≤ prior_consultations
    prior_consultations = int(np.random.poisson(0.5))
    prior_purchases = int(np.clip(np.random.poisson(0.15), 0, prior_consultations))

    row = {
        "lead_id": lead_id,
        "lead_source": np.random.choice(LEAD_SOURCES, p=LEAD_SOURCE_WEIGHTS),
        "age": age,
        "gender": np.random.choice(GENDERS, p=GENDER_WEIGHTS),
        "region": np.random.choice(REGIONS, p=REGION_WEIGHTS),
        "audiology_assessment": np.random.choice(AUDIOLOGY_LEVELS, p=AUDIOLOGY_WEIGHTS),
        "prior_consultations": prior_consultations,
        "prior_purchases": prior_purchases,
        "days_since_first_touch": days_since,
        "touchpoint_count": touchpoints,
        "call_time_of_day": np.random.choice(CALL_TIMES, p=CALL_TIME_WEIGHTS),
        "day_of_week": np.random.choice(["weekday", "weekend"], p=[0.78, 0.22]),
    }

    # Generate conversion outcome from the latent model
    log_odds = latent_log_odds(row)
    prob = log_odds_to_prob(log_odds)
    row["converted"] = int(np.random.random() < prob)

    return row


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    print(f"Generating {N_LEADS:,} synthetic leads for healthcare retail "
          "(hearing aids)...")

    rows = [generate_lead(i + 1) for i in range(N_LEADS)]
    df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "leads.csv", index=False)
    print(f"\n✅ Saved {len(df):,} leads to {OUTPUT_DIR / 'leads.csv'}\n")

    # Diagnostics
    conv_rate = df["converted"].mean()
    print(f"Dataset summary:")
    print(f"  Total leads:         {len(df):,}")
    print(f"  Converted leads:     {df['converted'].sum():,}")
    print(f"  Overall conversion:  {conv_rate:.2%}\n")

    print("Conversion rate by lead source:")
    by_source = (df.groupby("lead_source")["converted"]
                   .agg(["mean", "count"])
                   .sort_values("mean", ascending=False))
    for source, row in by_source.iterrows():
        print(f"  {source:<14} {row['mean']:.2%}  (n = {int(row['count']):,})")

    print("\nConversion rate by audiology severity:")
    audiology_order = ["none", "mild", "moderate", "severe", "profound"]
    by_audiology = df.groupby("audiology_assessment")["converted"].mean()
    for level in audiology_order:
        if level in by_audiology.index:
            print(f"  {level:<10} {by_audiology[level]:.2%}")

    print("\nConversion rate by prior purchases:")
    for n_prior in sorted(df["prior_purchases"].unique()):
        rate = df.loc[df["prior_purchases"] == n_prior, "converted"].mean()
        count = (df["prior_purchases"] == n_prior).sum()
        print(f"  {n_prior} prior purchase(s):  {rate:.2%}  (n = {count:,})")

    print("\nConversion rate by age bracket:")
    df["age_bracket"] = pd.cut(df["age"],
                                bins=[0, 50, 60, 70, 80, 100],
                                labels=["<50", "50-59", "60-69", "70-79", "80+"])
    by_age = df.groupby("age_bracket", observed=True)["converted"].mean()
    for bracket, rate in by_age.items():
        print(f"  {bracket:<6}  {rate:.2%}")


if __name__ == "__main__":
    main()
