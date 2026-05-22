"""
Price Elasticity Model
=======================

Estimates own-price elasticity per SKU using log-log OLS regression, plus
within-brand cross-price elasticity matrices for substitution analysis.

The log-log specification:

    log Q_i,t = α_i + β_i · log(P_i,t)
                    + γ_i · log(P_sister_avg,t)
                    + δ_i · promo_i,t
                    + season(t)
                    + ε_i,t

where the coefficient β_i is directly interpretable as the price elasticity
of demand: a 1% increase in price leads to β_i% change in quantity demanded.

Provides three core analyses:
    1. Per-SKU own-price elasticity (with 95% confidence intervals)
    2. Within-brand cross-price elasticity matrix
    3. Promotional lift quantification
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_modeling_frame(
    df: pd.DataFrame,
    target_sku: str,
) -> pd.DataFrame:
    """
    Build a per-SKU modeling frame with:
        - log_qty
        - log_price (own price)
        - log_sister_price (avg of within-brand competitor SKUs)
        - promo_flag
        - month_sin, month_cos (for seasonality)
    """
    own = df[df["sku"] == target_sku].copy().sort_values("week").reset_index(drop=True)
    target_brand = own["brand"].iloc[0]

    # Within-brand sister SKUs (substitutes)
    sister = (df[(df["brand"] == target_brand) & (df["sku"] != target_sku)]
                .groupby("week")["actual_price"].mean()
                .reset_index()
                .rename(columns={"actual_price": "sister_price"}))

    merged = own.merge(sister, on="week", how="left")
    merged["log_qty"]          = np.log(merged["quantity"])
    merged["log_price"]        = np.log(merged["actual_price"])
    merged["log_sister_price"] = np.log(merged["sister_price"])

    # Seasonality terms — Fourier basis for week-of-year
    week_of_year = pd.to_datetime(merged["date"]).dt.isocalendar().week.astype(float)
    merged["season_sin"] = np.sin(2 * np.pi * week_of_year / 52)
    merged["season_cos"] = np.cos(2 * np.pi * week_of_year / 52)

    return merged.dropna()


# ============================================================================
# OWN-PRICE ELASTICITY (per SKU)
# ============================================================================
def fit_own_price_elasticity(
    df: pd.DataFrame,
    sku: str,
    include_sister: bool = True,
) -> dict:
    """
    Estimate the own-price elasticity for one SKU via log-log OLS.

    Returns:
        elasticity, ci_low, ci_high, std_err, p_value, r_squared, n_obs,
        sister_elasticity (if include_sister), promo_lift, params, model
    """
    frame = prepare_modeling_frame(df, sku)
    if frame.empty or len(frame) < 30:
        raise ValueError(
            f"SKU {sku}: not enough rows ({len(frame)}) for elasticity fit"
        )

    feature_cols = ["log_price"]
    if include_sister:
        feature_cols.append("log_sister_price")
    feature_cols += ["promo_flag", "season_sin", "season_cos"]

    X = sm.add_constant(frame[feature_cols])
    y = frame["log_qty"]
    model = sm.OLS(y, X).fit()

    own_elasticity = model.params["log_price"]
    own_se = model.bse["log_price"]
    conf_int = model.conf_int(alpha=0.05).loc["log_price"]

    out = {
        "sku":              sku,
        "elasticity":       float(own_elasticity),
        "std_err":          float(own_se),
        "ci_low":           float(conf_int[0]),
        "ci_high":          float(conf_int[1]),
        "p_value":          float(model.pvalues["log_price"]),
        "r_squared":        float(model.rsquared),
        "n_obs":            int(len(frame)),
        "promo_lift_log":   float(model.params.get("promo_flag", np.nan)),
        "model":            model,
    }
    if include_sister and "log_sister_price" in model.params.index:
        out["sister_elasticity"] = float(model.params["log_sister_price"])
        out["sister_se"]         = float(model.bse["log_sister_price"])
    return out


def fit_all_skus(df: pd.DataFrame, include_sister: bool = True) -> pd.DataFrame:
    """Fit own-price elasticity for every SKU and return a summary DataFrame."""
    skus = df["sku"].unique().tolist()
    rows = []
    for sku in skus:
        result = fit_own_price_elasticity(df, sku, include_sister=include_sister)
        meta = df[df["sku"] == sku].iloc[0]
        rows.append({
            "sku":               sku,
            "brand":             meta["brand"],
            "tier":              meta["tier"],
            "elasticity":        result["elasticity"],
            "std_err":           result["std_err"],
            "ci_low":            result["ci_low"],
            "ci_high":           result["ci_high"],
            "p_value":           result["p_value"],
            "r_squared":         result["r_squared"],
            "n_obs":             result["n_obs"],
            "promo_lift_log":    result["promo_lift_log"],
            "sister_elasticity": result.get("sister_elasticity", np.nan),
        })
    summary = (pd.DataFrame(rows)
                 .sort_values(["brand", "tier"])
                 .reset_index(drop=True))
    return summary


# ============================================================================
# CROSS-PRICE ELASTICITY MATRIX (within-brand)
# ============================================================================
def fit_cross_price_matrix(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    For each ordered pair (i, j) of SKUs within the given brand, estimate
    the cross-price elasticity: how does demand for SKU i respond to
    price changes in SKU j?

        log Q_i,t = α + β_ii · log(P_i,t) + β_ij · log(P_j,t)
                        + δ · promo_i,t + season(t) + ε

    Returns a wide matrix (rows = SKU i, cols = SKU j) of β_ij values.
    """
    brand_df = df[df["brand"] == brand]
    skus = sorted(brand_df["sku"].unique())
    matrix = pd.DataFrame(np.nan, index=skus, columns=skus, dtype=float)

    for target_sku in skus:
        own_df = (brand_df[brand_df["sku"] == target_sku]
                    .sort_values("week").reset_index(drop=True))
        for cross_sku in skus:
            if cross_sku == target_sku:
                # Own-price elasticity on the diagonal — refit clean own-price model
                X_cols = ["log_own", "promo_flag", "season_sin", "season_cos"]
                frame = own_df.copy()
                frame["log_own"]    = np.log(frame["actual_price"])
                frame["log_qty"]    = np.log(frame["quantity"])
                week_of_year = pd.to_datetime(frame["date"]).dt.isocalendar().week.astype(float)
                frame["season_sin"] = np.sin(2 * np.pi * week_of_year / 52)
                frame["season_cos"] = np.cos(2 * np.pi * week_of_year / 52)
                X = sm.add_constant(frame[X_cols])
                model = sm.OLS(frame["log_qty"], X).fit()
                matrix.loc[target_sku, target_sku] = float(model.params["log_own"])
            else:
                # Cross elasticity: regress log Q_i on log P_i and log P_j
                cross_prices = (brand_df[brand_df["sku"] == cross_sku]
                                 [["week", "actual_price"]]
                                 .rename(columns={"actual_price": "cross_price"}))
                merged = own_df.merge(cross_prices, on="week", how="inner")
                merged["log_own"]   = np.log(merged["actual_price"])
                merged["log_cross"] = np.log(merged["cross_price"])
                merged["log_qty"]   = np.log(merged["quantity"])
                week_of_year = pd.to_datetime(merged["date"]).dt.isocalendar().week.astype(float)
                merged["season_sin"] = np.sin(2 * np.pi * week_of_year / 52)
                merged["season_cos"] = np.cos(2 * np.pi * week_of_year / 52)

                X_cols = ["log_own", "log_cross", "promo_flag",
                          "season_sin", "season_cos"]
                X = sm.add_constant(merged[X_cols])
                try:
                    model = sm.OLS(merged["log_qty"], X).fit()
                    matrix.loc[target_sku, cross_sku] = float(
                        model.params["log_cross"]
                    )
                except Exception:
                    matrix.loc[target_sku, cross_sku] = np.nan
    return matrix


# ============================================================================
# DEMAND CURVE GENERATION (for visualization)
# ============================================================================
def demand_curve(
    fit_result: dict,
    price_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Given an OLS fit result and a grid of prices, compute predicted quantities
    (with 95% confidence bands) by holding other regressors at their mean.

    Returns dict with keys: price, predicted, lower, upper.
    """
    model = fit_result["model"]
    log_grid = np.log(price_grid)

    # Build prediction frame with covariates held at mean
    mean_values = dict(zip(model.model.exog_names,
                            model.model.exog.mean(axis=0)))
    rows = []
    for log_p in log_grid:
        row = mean_values.copy()
        row["log_price"] = log_p
        rows.append(row)
    pred_df = pd.DataFrame(rows)[model.model.exog_names]

    pred = model.get_prediction(pred_df)
    summary = pred.summary_frame(alpha=0.05)
    return {
        "price":     price_grid,
        "predicted": np.exp(summary["mean"]).values,
        "lower":     np.exp(summary["mean_ci_lower"]).values,
        "upper":     np.exp(summary["mean_ci_upper"]).values,
    }


# ============================================================================
# REVENUE / PRICING RECOMMENDATION
# ============================================================================
def pricing_recommendation(elasticity: float) -> str:
    """
    Translate an estimated elasticity into a directional pricing recommendation
    based on the standard result:
        |ε| > 1 (elastic)  → price cut increases revenue
        |ε| < 1 (inelastic) → price increase increases revenue
        |ε| = 1 (unit)     → revenue is invariant to price (locally)
    """
    if elasticity >= -0.20:
        return "Highly inelastic — strong case for price increase"
    if elasticity > -1.0:
        return "Inelastic — price increase likely raises revenue"
    if -1.10 <= elasticity <= -0.90:
        return "Near unit-elastic — revenue stable across price changes"
    if elasticity > -1.5:
        return "Elastic — price cuts likely raise revenue"
    return "Highly elastic — strong case for price reduction"


def revenue_change_estimate(
    current_price: float,
    pct_price_change: float,
    elasticity: float,
) -> dict[str, float]:
    """
    Estimate the % change in revenue from a small change in price.

    R = P × Q
    log R = log P + log Q
    log R = log P + (α + ε · log P) + ...
    dlogR/dlogP = 1 + ε
    so % change in revenue ≈ (1 + ε) × % change in price
    """
    pct_revenue_change = (1 + elasticity) * pct_price_change
    new_price = current_price * (1 + pct_price_change / 100)
    return {
        "current_price":       current_price,
        "new_price":           new_price,
        "pct_price_change":    pct_price_change,
        "pct_revenue_change":  pct_revenue_change,
    }
