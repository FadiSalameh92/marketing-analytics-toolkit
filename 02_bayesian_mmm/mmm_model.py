"""
Bayesian Marketing Mix Model (PyMC)
====================================

A media mix model that estimates the contribution of each marketing channel
to sales, with adstock (carryover) and Hill saturation (diminishing returns)
applied to each channel's spend.

Model specification:

    sales_t = intercept
              + trend * t
              + seasonality(t)
              + sum_c [ beta_c * Hill(Adstock(spend_{c,t}; decay_c); half_sat_c, slope_c) ]
              + noise_t

Where:
    Adstock(spend; decay) = (1/Z) * sum_{l=0}^{L-1} decay^l * spend_{t-l}
    Hill(x; half_sat, slope) = x^slope / (half_sat^slope + x^slope)

Priors:
    decay        ~ Beta(2, 2)              # carryover rate in [0, 1]
    half_sat     ~ HalfNormal(sigma=20)    # saturation half-point in spend units
    slope        ~ HalfNormal(sigma=2)     # Hill curve steepness
    beta         ~ HalfNormal(sigma=150)   # channel effect coefficient
    intercept    ~ Normal(mu=400, sigma=100)
    trend_coef   ~ Normal(mu=0, sigma=2)
    season_coef  ~ Normal(mu=0, sigma=40)
    sigma        ~ HalfNormal(sigma=50)

Inference uses NUTS (No-U-Turn Sampler) with default settings tuned for
modest runtime (~1-3 minutes on a typical laptop).
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az


# ============================================================================
# UTILITY: ADSTOCK AND SATURATION (NumPy versions for plotting/diagnostics)
# ============================================================================
def geometric_adstock_np(
    spend: np.ndarray,
    decay: float,
    max_lag: int = 8,
) -> np.ndarray:
    """NumPy implementation of geometric adstock — for plotting and diagnostics."""
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


def hill_saturation_np(x: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    """NumPy Hill saturation — for plotting saturation curves."""
    return x ** slope / (half_sat ** slope + x ** slope)


# ============================================================================
# MODEL CLASS
# ============================================================================
class BayesianMMM:
    """
    Bayesian Media Mix Model with adstock and Hill saturation per channel.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns 'sales' and 'spend_<channel>' for each channel
        in `channels`, plus 'trend_week' (integer week index) and
        'week_of_year' (1-52).
    channels : list[str]
        Channel names. Spend columns must be named 'spend_<channel>'.
    max_lag : int
        Maximum lag for the adstock window (in weeks). Default 8.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        channels: Sequence[str],
        max_lag: int = 8,
    ):
        self.data = data.copy().reset_index(drop=True)
        self.channels = list(channels)
        self.max_lag = max_lag
        self.n_weeks = len(self.data)
        self.n_channels = len(self.channels)
        self.model: pm.Model | None = None
        self.trace: az.InferenceData | None = None
        self._lag_matrices: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Pre-computation: lagged spend matrices
    # ------------------------------------------------------------------
    def _build_lag_matrices(self) -> None:
        """Pre-compute one (n_weeks, max_lag) matrix per channel."""
        for ch in self.channels:
            spend = self.data[f"spend_{ch}"].values.astype(float)
            mat = np.zeros((self.n_weeks, self.max_lag))
            for lag in range(self.max_lag):
                if lag == 0:
                    mat[:, lag] = spend
                else:
                    mat[lag:, lag] = spend[:-lag]
            self._lag_matrices[ch] = mat

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def build(self) -> pm.Model:
        """Construct the PyMC model graph."""
        self._build_lag_matrices()
        sales = self.data["sales"].values.astype(float)
        weeks = self.data["trend_week"].values.astype(float)

        coords = {"channel": self.channels, "week": np.arange(self.n_weeks)}

        with pm.Model(coords=coords) as model:
            # Channel-level priors
            decay     = pm.Beta("decay",        alpha=2, beta=2,   dims="channel")
            half_sat  = pm.HalfNormal("half_sat", sigma=20,        dims="channel")
            slope     = pm.HalfNormal("slope",    sigma=2,         dims="channel")
            beta      = pm.HalfNormal("beta",     sigma=150,       dims="channel")

            # Base, trend, seasonality
            intercept    = pm.Normal("intercept",      mu=400, sigma=100)
            trend_coef   = pm.Normal("trend_coef",     mu=0,   sigma=2)
            season_coef  = pm.Normal("season_coef",    mu=0,   sigma=40)
            season_phase = pm.Normal("season_phase",   mu=0,   sigma=1)

            # Build per-channel saturated, adstocked contribution
            media_contrib = pt.zeros(self.n_weeks)
            for i, ch in enumerate(self.channels):
                lag_mat = self._lag_matrices[ch]

                lags = pt.arange(self.max_lag, dtype="float64")
                weights = decay[i] ** lags
                weights = weights / pt.sum(weights)

                adstocked = pt.dot(lag_mat, weights)
                saturated = adstocked ** slope[i] / (
                    half_sat[i] ** slope[i] + adstocked ** slope[i]
                )
                media_contrib = media_contrib + beta[i] * saturated

            # Trend and seasonality
            t = pt.as_tensor_variable(weeks)
            trend = trend_coef * t
            seasonality = season_coef * pt.sin(
                2 * np.pi * t / 52 + season_phase
            )

            mu = intercept + trend + seasonality + media_contrib

            # Likelihood
            sigma = pm.HalfNormal("sigma", sigma=50)
            pm.Normal("y", mu=mu, sigma=sigma, observed=sales)

        self.model = model
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def fit(
        self,
        draws: int = 1000,
        tune: int = 500,
        chains: int = 2,
        cores: int | None = None,
        target_accept: float = 0.95,
        random_seed: int = 42,
        progressbar: bool = True,
    ) -> az.InferenceData:
        """
        Run NUTS sampling and return the InferenceData.

        Parameters
        ----------
        cores : int | None
            Number of CPU cores to use for parallel chains. None lets PyMC
            choose, but on some environments (single-core machines, BLAS
            without OMP) this can fail with a divide-by-zero error during
            setup. If you hit that, pass cores=1 explicitly.
        """
        if self.model is None:
            self.build()
        sample_kwargs = dict(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            idata_kwargs={"log_likelihood": False},
        )
        if cores is not None:
            sample_kwargs["cores"] = cores
        with self.model:
            self.trace = pm.sample(**sample_kwargs)
        return self.trace

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------
    def parameter_summary(self) -> pd.DataFrame:
        """Return posterior mean, standard deviation, and credible interval
        for each channel parameter. Handles both old and new arviz APIs."""
        if self.trace is None:
            raise RuntimeError("Call fit() before parameter_summary().")
        var_names = ["decay", "half_sat", "slope", "beta"]
        try:
            # Newer arviz (>= 0.20) uses ci_prob
            summary = az.summary(
                self.trace,
                var_names=var_names,
                kind="stats",
                ci_prob=0.94,
            )
        except TypeError:
            # Older arviz uses hdi_prob
            summary = az.summary(
                self.trace,
                var_names=var_names,
                kind="stats",
                hdi_prob=0.94,
            )
        return summary.reset_index()

    def posterior_means(self) -> dict[str, dict[str, float]]:
        """Return per-channel posterior means as a nested dict."""
        if self.trace is None:
            raise RuntimeError("Call fit() before posterior_means().")
        out: dict[str, dict[str, float]] = {}
        for i, ch in enumerate(self.channels):
            out[ch] = {
                "decay":    float(self.trace.posterior["decay"].sel(channel=ch).mean()),
                "half_sat": float(self.trace.posterior["half_sat"].sel(channel=ch).mean()),
                "slope":    float(self.trace.posterior["slope"].sel(channel=ch).mean()),
                "beta":     float(self.trace.posterior["beta"].sel(channel=ch).mean()),
            }
        return out

    # ------------------------------------------------------------------
    # Channel contributions and ROI
    # ------------------------------------------------------------------
    def channel_contributions(self) -> pd.DataFrame:
        """
        Compute the expected weekly contribution of each channel using
        posterior mean parameters.
        """
        if self.trace is None:
            raise RuntimeError("Call fit() before channel_contributions().")
        means = self.posterior_means()

        contributions = {"date": self.data["date"]}
        for ch in self.channels:
            spend = self.data[f"spend_{ch}"].values
            p = means[ch]
            adstocked = geometric_adstock_np(spend, p["decay"], self.max_lag)
            saturated = hill_saturation_np(adstocked, p["half_sat"], p["slope"])
            contributions[ch] = p["beta"] * saturated
        return pd.DataFrame(contributions)

    def roi_estimates(self) -> pd.DataFrame:
        """ROI = total contribution / total spend per channel (posterior mean)."""
        contrib = self.channel_contributions()
        rows = []
        for ch in self.channels:
            total_contrib = contrib[ch].sum()
            total_spend = self.data[f"spend_{ch}"].sum()
            roi = total_contrib / total_spend if total_spend > 0 else 0
            rows.append({
                "channel": ch,
                "total_spend_K": round(total_spend, 1),
                "total_contribution_K": round(total_contrib, 1),
                "roi": round(roi, 2),
            })
        return (pd.DataFrame(rows)
                .sort_values("roi", ascending=False)
                .reset_index(drop=True))

    # ------------------------------------------------------------------
    # Saturation curves (for plotting)
    # ------------------------------------------------------------------
    def saturation_curves(
        self,
        max_spend_K: float = 50.0,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """
        Compute the response curve for each channel using posterior means.
        Returns a long-format DataFrame: (channel, spend, response).
        """
        if self.trace is None:
            raise RuntimeError("Call fit() before saturation_curves().")
        means = self.posterior_means()
        spend_grid = np.linspace(0.01, max_spend_K, n_points)

        rows = []
        for ch in self.channels:
            p = means[ch]
            # Saturated response (no adstock; we visualize the saturation curve only)
            response = p["beta"] * hill_saturation_np(
                spend_grid, p["half_sat"], p["slope"]
            )
            for s, r in zip(spend_grid, response):
                rows.append({"channel": ch, "spend_K": s, "response_K": r})
        return pd.DataFrame(rows)
