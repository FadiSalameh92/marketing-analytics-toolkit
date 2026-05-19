"""
Markov Chain Attribution Module
================================

Implements the removal-effect method for multi-touch attribution using a
first-order Markov chain over channel transitions.

Core algorithm:
    1. Build a transition probability matrix from observed customer journeys,
       with START as the entry state and CONVERSION/NULL as absorbing states.
    2. Compute the baseline probability of reaching CONVERSION from START
       via the fundamental matrix of the absorbing Markov chain.
    3. For each channel, compute the "removal effect" — the proportional drop
       in conversion probability when transitions through that channel are
       redirected to NULL.
    4. Normalize removal effects across channels to produce attribution
       credits that sum to the baseline conversion count.

Also provides four heuristic baselines for comparison:
    - First-touch attribution
    - Last-touch attribution
    - Linear (equal credit across touches)
    - Time-decay (exponential weighting toward last touch)
"""

from __future__ import annotations
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================================
# DATA PREP
# ============================================================================
def journeys_to_paths(touchpoints: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse touchpoint-level data into one row per journey.

    Expects columns: journey_id, step, channel, converted.
    Returns columns: journey_id, path (list[str]), converted.
    """
    required = {"journey_id", "step", "channel", "converted"}
    missing = required - set(touchpoints.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = touchpoints.sort_values(["journey_id", "step"])
    paths = (df.groupby("journey_id")
               .agg(path=("channel", list),
                    converted=("converted", "first"))
               .reset_index())
    return paths


# ============================================================================
# MARKOV ATTRIBUTION (removal-effect method)
# ============================================================================
class MarkovAttribution:
    """
    Multi-touch attribution using removal-effect analysis on a first-order
    Markov chain over channel sequences.

    Parameters
    ----------
    paths : pd.DataFrame
        DataFrame with columns ['path', 'converted']. 'path' is a list of
        channel strings; 'converted' is boolean.
    """

    START = "(start)"
    CONVERSION = "(conversion)"
    NULL = "(null)"

    def __init__(self, paths: pd.DataFrame):
        self.paths = paths.copy()
        self.channels: list[str] = sorted(
            {c for p in paths["path"] for c in p}
        )
        self.states: list[str] = [self.START] + self.channels + [
            self.CONVERSION, self.NULL,
        ]
        self.transition_matrix_: np.ndarray | None = None
        self.base_conversion_prob_: float | None = None

    # ------------------------------------------------------------------
    # Step 1 — Build transition matrix
    # ------------------------------------------------------------------
    def _build_transition_matrix(self) -> np.ndarray:
        """Construct the transition probability matrix from observed paths."""
        n = len(self.states)
        idx = {s: i for i, s in enumerate(self.states)}
        transition_counts = np.zeros((n, n))

        for _, row in self.paths.iterrows():
            path = [self.START] + list(row["path"])
            path.append(self.CONVERSION if row["converted"] else self.NULL)

            for from_state, to_state in zip(path[:-1], path[1:]):
                transition_counts[idx[from_state], idx[to_state]] += 1

        # Absorbing states stay absorbed
        for absorbing in (self.CONVERSION, self.NULL):
            transition_counts[idx[absorbing], :] = 0
            transition_counts[idx[absorbing], idx[absorbing]] = 1

        # Normalize rows to probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid divide-by-zero
        transition_matrix = transition_counts / row_sums
        return transition_matrix

    # ------------------------------------------------------------------
    # Step 2 — Compute absorption probability via fundamental matrix
    # ------------------------------------------------------------------
    def _absorption_probability(self, matrix: np.ndarray) -> float:
        """
        Probability of reaching CONVERSION from START in the absorbing
        Markov chain represented by `matrix`.

        Uses the canonical form decomposition:
            P = [[Q, R], [0, I]]
        where Q is the transient-to-transient submatrix and R is the
        transient-to-absorbing submatrix. The fundamental matrix is
            N = (I - Q)^-1
        and the probability of absorbing into each absorbing state from
        each transient state is N @ R.
        """
        idx = {s: i for i, s in enumerate(self.states)}
        absorbing = {idx[self.CONVERSION], idx[self.NULL]}
        transient = [i for i in range(len(self.states)) if i not in absorbing]

        Q = matrix[np.ix_(transient, transient)]
        R = matrix[np.ix_(transient, list(absorbing))]
        I = np.eye(len(transient))
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse on singular matrices
            N = np.linalg.pinv(I - Q)
        absorption = N @ R

        # Absorbing state column order matches the iteration order in `absorbing`
        absorbing_order = sorted(absorbing)
        conv_col = absorbing_order.index(idx[self.CONVERSION])
        start_row = transient.index(idx[self.START])
        return float(absorption[start_row, conv_col])

    # ------------------------------------------------------------------
    # Step 3 — Removal effects
    # ------------------------------------------------------------------
    def _removal_effect(self, channel: str) -> float:
        """
        Removal effect of `channel`: relative drop in conversion probability
        when all transitions through `channel` are redirected to NULL.
        """
        idx = {s: i for i, s in enumerate(self.states)}
        modified = self.transition_matrix_.copy()

        # Send everything that was going INTO `channel` to NULL instead
        ch_idx = idx[channel]
        null_idx = idx[self.NULL]
        # Add channel's incoming probability mass to NULL column
        modified[:, null_idx] += modified[:, ch_idx]
        modified[:, ch_idx] = 0
        # Channel can no longer be transitioned to; treat it as a dead-end
        # (its row no longer matters because incoming prob is zero)
        modified[ch_idx, :] = 0
        modified[ch_idx, null_idx] = 1

        new_prob = self._absorption_probability(modified)
        return 1.0 - (new_prob / self.base_conversion_prob_)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self) -> "MarkovAttribution":
        """Build the transition matrix and base conversion probability."""
        self.transition_matrix_ = self._build_transition_matrix()
        self.base_conversion_prob_ = self._absorption_probability(
            self.transition_matrix_
        )
        return self

    def attribution(self) -> pd.DataFrame:
        """
        Return per-channel attribution credits.

        Each channel's credit is its removal effect normalized by the sum of
        all removal effects, then scaled to the total observed conversions.
        """
        if self.transition_matrix_ is None:
            self.fit()

        total_conversions = int(self.paths["converted"].sum())
        removal_effects = {
            ch: self._removal_effect(ch) for ch in self.channels
        }
        total_re = sum(removal_effects.values())

        rows = []
        for ch, re in removal_effects.items():
            share = re / total_re if total_re > 0 else 0
            rows.append({
                "channel": ch,
                "removal_effect": re,
                "attribution_share": share,
                "attributed_conversions": share * total_conversions,
            })

        return (pd.DataFrame(rows)
                .sort_values("attribution_share", ascending=False)
                .reset_index(drop=True))

    def transition_matrix_df(self) -> pd.DataFrame:
        """Return the transition matrix as a labeled DataFrame."""
        if self.transition_matrix_ is None:
            self.fit()
        return pd.DataFrame(
            self.transition_matrix_,
            index=self.states,
            columns=self.states,
        )


# ============================================================================
# HEURISTIC ATTRIBUTION BASELINES (for comparison)
# ============================================================================
def _heuristic_attribution(
    paths: pd.DataFrame,
    weighting: str,
) -> pd.DataFrame:
    """
    Generic helper for first_touch, last_touch, linear, time_decay attribution.
    Only credits journeys where converted == True.
    """
    if weighting not in {"first", "last", "linear", "time_decay"}:
        raise ValueError(f"Unknown weighting: {weighting}")

    credits: dict[str, float] = defaultdict(float)
    converted_paths = paths.loc[paths["converted"], "path"]

    for path in converted_paths:
        n = len(path)
        if n == 0:
            continue
        if weighting == "first":
            credits[path[0]] += 1.0
        elif weighting == "last":
            credits[path[-1]] += 1.0
        elif weighting == "linear":
            for ch in path:
                credits[ch] += 1.0 / n
        elif weighting == "time_decay":
            # Exponential weighting toward last touch, half-life = 2 positions
            half_life = 2.0
            weights = np.array([
                0.5 ** ((n - 1 - i) / half_life) for i in range(n)
            ])
            weights = weights / weights.sum()
            for ch, w in zip(path, weights):
                credits[ch] += w

    df = pd.DataFrame(
        [(ch, credit) for ch, credit in credits.items()],
        columns=["channel", "attributed_conversions"],
    )
    total = df["attributed_conversions"].sum()
    df["attribution_share"] = df["attributed_conversions"] / total if total > 0 else 0
    return df.sort_values("attribution_share", ascending=False).reset_index(drop=True)


def first_touch_attribution(paths: pd.DataFrame) -> pd.DataFrame:
    return _heuristic_attribution(paths, "first")


def last_touch_attribution(paths: pd.DataFrame) -> pd.DataFrame:
    return _heuristic_attribution(paths, "last")


def linear_attribution(paths: pd.DataFrame) -> pd.DataFrame:
    return _heuristic_attribution(paths, "linear")


def time_decay_attribution(paths: pd.DataFrame) -> pd.DataFrame:
    return _heuristic_attribution(paths, "time_decay")


def compare_methods(paths: pd.DataFrame) -> pd.DataFrame:
    """
    Return a wide-format DataFrame comparing attribution shares across
    Markov, first-touch, last-touch, linear, and time-decay methods.
    """
    markov = (MarkovAttribution(paths).fit().attribution()
              [["channel", "attribution_share"]]
              .rename(columns={"attribution_share": "markov"}))
    ft = (first_touch_attribution(paths)[["channel", "attribution_share"]]
          .rename(columns={"attribution_share": "first_touch"}))
    lt = (last_touch_attribution(paths)[["channel", "attribution_share"]]
          .rename(columns={"attribution_share": "last_touch"}))
    lin = (linear_attribution(paths)[["channel", "attribution_share"]]
           .rename(columns={"attribution_share": "linear"}))
    td = (time_decay_attribution(paths)[["channel", "attribution_share"]]
          .rename(columns={"attribution_share": "time_decay"}))

    comparison = (markov.merge(ft, on="channel", how="outer")
                        .merge(lt, on="channel", how="outer")
                        .merge(lin, on="channel", how="outer")
                        .merge(td, on="channel", how="outer")
                        .fillna(0)
                        .sort_values("markov", ascending=False)
                        .reset_index(drop=True))
    return comparison
