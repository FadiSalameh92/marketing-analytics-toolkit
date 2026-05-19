"""
Synthetic Customer Journey Data Generator
==========================================

Generates realistic multi-touch customer journey data for testing the
Markov chain attribution model. Each journey is a sequence of channel
touchpoints terminating in either a conversion or a drop-off.

The five channels mimic a typical consumer marketing mix:
    - paid_search  — high intent, late-funnel
    - paid_social  — mid-funnel discovery, broad reach
    - direct_mail  — slow-burn discovery, regional targeting
    - tv           — pure awareness, broad reach
    - call_center  — high intent, very late funnel (inbound calls)

Run:
    python generate_synthetic_data.py

Outputs:
    data/journeys.csv         — one row per touchpoint
    data/journeys_summary.csv — one row per journey
"""

from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================
N_JOURNEYS = 50_000

CHANNELS = ["paid_search", "paid_social", "direct_mail", "tv", "call_center"]

# Channel characteristics
# - discovery_weight: how often the channel starts a journey
# - conversion_weight: how much the channel pushes toward conversion when present
# - position_pref: 0 = early, 1 = late (used to bias channel order within journeys)
CHANNEL_PROFILES = {
    "paid_search": {"discovery_weight": 0.20, "conversion_weight": 1.8, "position_pref": 0.75},
    "paid_social": {"discovery_weight": 0.35, "conversion_weight": 0.6, "position_pref": 0.35},
    "direct_mail": {"discovery_weight": 0.15, "conversion_weight": 0.4, "position_pref": 0.30},
    "tv":          {"discovery_weight": 0.25, "conversion_weight": 0.3, "position_pref": 0.20},
    "call_center": {"discovery_weight": 0.05, "conversion_weight": 2.5, "position_pref": 0.90},
}

# Base conversion probability (before channel effects)
BASE_CONVERSION_PROB = 0.015

OUTPUT_DIR = Path(__file__).parent / "data"


# ============================================================================
# JOURNEY GENERATION
# ============================================================================
def sample_journey_length() -> int:
    """Path length follows a geometric-like distribution; median ~3-4 touches."""
    length = np.random.geometric(p=0.28)
    return int(np.clip(length, 1, 12))


def sample_first_channel() -> str:
    """Choose the entry channel weighted by discovery propensity."""
    weights = np.array([CHANNEL_PROFILES[c]["discovery_weight"] for c in CHANNELS])
    weights = weights / weights.sum()
    return np.random.choice(CHANNELS, p=weights)


def sample_next_channel(current_position: float) -> str:
    """
    Choose the next channel based on relative position in the journey
    (0 = beginning, 1 = end). Channels with a position_pref closer to the
    current position are weighted higher.
    """
    weights = []
    for c in CHANNELS:
        pref = CHANNEL_PROFILES[c]["position_pref"]
        # Higher weight when journey position aligns with channel's typical position
        distance = abs(current_position - pref)
        weight = np.exp(-3.0 * distance)  # exponential falloff
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.random.choice(CHANNELS, p=weights)


def determine_conversion(touchpoints: list[str]) -> bool:
    """
    Probability of conversion depends on which channels appeared in the
    journey. Late-funnel channels (paid_search, call_center) drive most lift.
    """
    # Aggregate conversion weight from unique channels in the journey
    unique_channels = set(touchpoints)
    total_weight = sum(CHANNEL_PROFILES[c]["conversion_weight"] for c in unique_channels)

    # Logistic-style conversion probability, anchored to BASE_CONVERSION_PROB
    # when no high-converting channels are present
    logit_base = np.log(BASE_CONVERSION_PROB / (1 - BASE_CONVERSION_PROB))
    logit_adjusted = logit_base + 0.45 * total_weight
    prob = 1 / (1 + np.exp(-logit_adjusted))

    # Cap at a realistic ceiling
    prob = min(prob, 0.45)
    return np.random.random() < prob


def generate_single_journey(journey_id: int) -> list[dict]:
    """Generate one customer journey as a list of touchpoint records."""
    length = sample_journey_length()
    touchpoints = []

    # First touch
    current = sample_first_channel()
    touchpoints.append(current)

    # Subsequent touches
    for step in range(1, length):
        position = step / max(length - 1, 1)
        next_channel = sample_next_channel(position)
        touchpoints.append(next_channel)

    converted = determine_conversion(touchpoints)

    return [
        {
            "journey_id": journey_id,
            "step": step + 1,
            "channel": ch,
            "converted": converted,
            "journey_length": length,
        }
        for step, ch in enumerate(touchpoints)
    ]


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    print(f"Generating {N_JOURNEYS:,} synthetic customer journeys across "
          f"{len(CHANNELS)} channels...")

    all_rows = []
    for journey_id in range(1, N_JOURNEYS + 1):
        all_rows.extend(generate_single_journey(journey_id))

    df = pd.DataFrame(all_rows)

    # Journey-level summary
    summary = (df.groupby("journey_id")
                 .agg(touches=("step", "max"),
                      converted=("converted", "first"),
                      path=("channel", lambda x: " > ".join(x)))
                 .reset_index())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "journeys.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "journeys_summary.csv", index=False)

    print(f"\n✅ Saved {len(df):,} touchpoints across {N_JOURNEYS:,} journeys")
    print(f"   → {OUTPUT_DIR / 'journeys.csv'}")
    print(f"   → {OUTPUT_DIR / 'journeys_summary.csv'}")

    # Diagnostics
    conv_rate = summary["converted"].mean()
    avg_length = summary["touches"].mean()
    median_length = summary["touches"].median()

    print(f"\nDataset summary:")
    print(f"  Overall conversion rate:  {conv_rate:.2%}")
    print(f"  Avg journey length:       {avg_length:.2f} touchpoints")
    print(f"  Median journey length:    {median_length:.0f} touchpoints")

    print(f"\nChannel touch frequency:")
    ch_freq = df["channel"].value_counts(normalize=True).round(3)
    for ch, pct in ch_freq.items():
        print(f"  {ch:<14} {pct:.1%}")

    print(f"\nConversion rate by journey containing channel:")
    for ch in CHANNELS:
        contains = summary["path"].str.contains(ch)
        rate = summary.loc[contains, "converted"].mean()
        share = contains.mean()
        print(f"  {ch:<14} {rate:.2%} (in {share:.1%} of journeys)")


if __name__ == "__main__":
    main()
