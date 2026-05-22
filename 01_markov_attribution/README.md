# Markov Chain Attribution

Multi-touch attribution using the **removal-effect method** on a first-order Markov chain over customer journey sequences. Originally built to attribute revenue across paid search, paid social, direct mail, TV, and call-center inbound channels for a healthcare retail business.

This portfolio version uses synthetic data with the same five-channel structure.

## What this answers

> "Last-touch attribution says call_center drives 52% of conversions and TV drives 6%. Is that real, or an artifact of how the channels are sequenced in the funnel?"

Heuristic attribution methods (first-touch, last-touch, linear, time-decay) assign credit based on *where in the path* a channel appears. Markov attribution assigns credit based on *how indispensable* each channel is to the overall conversion graph — measured by how much conversion probability collapses if you remove the channel entirely. This reveals upper-funnel channels that don't close but make closing possible.

## Sample outputs

![Attribution share by channel](docs/attribution_comparison.png)
*The five attribution methods disagree sharply. Last-touch credits call_center with 52% of conversions; Markov's removal-effect method assigns it 24.5% — correcting the late-funnel overcounting. The discovery channels (paid_social, TV) that last-touch nearly ignores recover meaningful credit under Markov.*

![Channel-to-channel transition probabilities](docs/transition_matrix.png)
*The transition matrix underlying the model: the probability of moving from each channel to the next in a journey. These transition structures — not journey position — are what the removal-effect method uses to value each channel.*

## Setup

```bash
# From the repo root, install dependencies
pip install -r ../requirements.txt

# Generate the synthetic journey dataset
python generate_synthetic_data.py

# Run the attribution analysis
python run_analysis.py
```

Outputs land in `output/` — attribution comparison chart, transition matrix heatmap, and CSV result files.

## File structure

```
01_markov_attribution/
├── README.md                       # This file
├── generate_synthetic_data.py      # Synthetic journey generator (50K journeys)
├── markov_attribution.py           # Core attribution module (importable)
├── run_analysis.py                 # Executable analysis pipeline
├── data/                           # Generated locally — not committed
│   ├── journeys.csv                # Touchpoint-level data
│   └── journeys_summary.csv        # Journey-level summary
├── docs/                           # Showcase images for README
│   ├── attribution_comparison.png
│   └── transition_matrix.png
└── output/                         # Analysis outputs — not committed
    ├── attribution_comparison.png
    ├── transition_matrix.png
    ├── attribution_results.csv
    ├── markov_attribution.csv
    └── transition_matrix.csv
```

## Detailed results

From a representative run on 50,000 synthetic journeys with ~3,200 conversions:

| channel     | Markov | First-touch | Last-touch | Linear | Time-decay |
|-------------|--------|-------------|------------|--------|------------|
| call_center | 24.5%  | 5.4%        | 52.0%      | 23.2%  | 31.6%      |
| paid_search | 22.2%  | 23.2%       | 27.1%      | 24.4%  | 27.0%      |
| paid_social | 20.8%  | 33.0%       | 8.8%       | 21.3%  | 17.2%      |
| tv          | 16.9%  | 24.0%       | 6.2%       | 16.2%  | 12.3%      |
| direct_mail | 15.6%  | 14.3%       | 5.8%       | 14.8%  | 11.9%      |

**What this tells you:**

- Last-touch overstates call_center by ~27 percentage points and understates TV / paid_social / direct_mail by 10+ points each — exactly the bias you'd expect when high-intent channels appear late in funnels.
- First-touch overstates paid_social and TV (the discovery channels) by the symmetric amount.
- Markov gives a balanced view, recognizing that closing channels and discovery channels both contribute.
- For budget decisions, the gap between Markov and last-touch can materially change reallocation logic — moving spend off discovery channels because last-touch says they don't convert tends to collapse the funnel.

## Methodology

### 1. Build the transition matrix

Each journey is converted to a sequence:
```
(start) → channel_A → channel_B → channel_C → (conversion | null)
```

Transitions across all journeys are counted and normalized into a row-stochastic transition probability matrix. `(start)`, `(conversion)`, and `(null)` are added as absorbing states (`conversion` and `null` are terminal).

### 2. Compute base conversion probability

Using the canonical form of the absorbing Markov chain:

$$ P = \begin{bmatrix} Q & R \\ 0 & I \end{bmatrix} $$

where `Q` is the transient-to-transient submatrix and `R` is the transient-to-absorbing submatrix. The fundamental matrix `N = (I − Q)⁻¹` lets us compute the absorption probability:

$$ B = N \cdot R $$

The `(start, conversion)` entry of `B` is the **baseline conversion probability** — what the model predicts conversion rate to be given the observed channel transitions.

### 3. Removal effects

For each channel `c`:

1. Modify the transition matrix so that all probability mass that previously went *into* `c` is rerouted to `(null)`.
2. Recompute the absorption probability with the modified matrix.
3. The **removal effect** is `1 − (new_prob / base_prob)` — the proportional drop in conversion when `c` is removed from the funnel.

### 4. Normalize to attribution credits

Each channel's attribution share is its removal effect divided by the sum of all removal effects. Multiplied by total observed conversions, this gives **attributed conversions** that sum to the observed total.

## Why this beats heuristic methods

| Aspect | Heuristic methods | Markov |
|---|---|---|
| Captures channel interaction | No — credit depends only on position | Yes — channels are valued by how much the chain depends on them |
| Reveals upper-funnel value | No — first-touch only credits the entry channel | Yes — discovery channels show meaningful attribution even when they don't close |
| Stable across path lengths | No — long paths dilute credit under linear | Yes — based on transition structure, not journey length |
| Reproducible | Yes | Yes |
| Computationally cheap | Trivial | Modest — matrix inversion on a small state space |

## Limitations to be aware of

- **First-order Markov assumption.** Channel-to-channel transitions depend only on the current channel, not on the full history. Higher-order Markov chains can capture more nuance but require dramatically more data and become unstable with sparse paths.
- **No timing information.** This implementation treats journey position as discrete steps, not calendar time. A separate time-decay weighting could be added if recency effects matter.
- **No incrementality testing.** Attribution describes *contribution*, not *incrementality*. The [Bayesian Marketing Mix Model](../02_bayesian_mmm/) in this repo addresses the spend-vs-outcome question more directly with adstock and saturation curves.
- **Sensitive to absorbing-state definitions.** Treating `null` as a single state aggregates all non-conversion outcomes (drop-off, indefinite delay, returned and converted via untracked channel). Real implementations may need more nuanced absorbing-state design.

## Reference reading

- Anderl, Becker, von Wangenheim, Schumann (2016). "Mapping the customer journey: Lessons learned from graph-based online attribution modeling." *International Journal of Research in Marketing*, 33(3), 457–474. — the canonical academic treatment of removal-effect Markov attribution.
- Shao & Li (2011). "Data-driven multi-touch attribution models." KDD 2011. — earlier foundational treatment.

## About this implementation

This portfolio version uses synthetic data generated by `generate_synthetic_data.py`. The data has no relationship to any real company — distributions are randomly generated to mimic plausible consumer marketing journey patterns. The attribution algorithm, methodology, and analytical framework reflect a production deployment built to inform marketing budget allocation in a healthcare retail context.
