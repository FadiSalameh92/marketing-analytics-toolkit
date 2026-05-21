"""
XGBoost Lead Scoring Model
===========================

Trains and evaluates an XGBoost classifier to predict the probability that
an inbound lead will convert, for the purpose of call-center prioritization.

Provides a logistic regression baseline for comparison so the marginal value
of the gradient-boosting approach is explicit.

Core API:

    scorer = XGBoostLeadScorer(features, target='converted')
    scorer.fit(train_df)
    probs = scorer.predict_proba(test_df)
    metrics = scorer.evaluate(test_df)
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
)
import xgboost as xgb


# Categorical and numeric feature columns used by both models
CATEGORICAL_FEATURES = [
    "lead_source", "gender", "region",
    "audiology_assessment", "call_time_of_day", "day_of_week",
]
NUMERIC_FEATURES = [
    "age", "prior_consultations", "prior_purchases",
    "days_since_first_touch", "touchpoint_count",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ============================================================================
# XGBOOST CLASSIFIER (with native categorical support)
# ============================================================================
class XGBoostLeadScorer:
    """
    XGBoost classifier with native categorical feature handling.

    Parameters
    ----------
    target : str
        Name of the binary outcome column in the training data.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, target: str = "converted", random_state: int = 42):
        self.target = target
        self.random_state = random_state
        self.model: xgb.XGBClassifier | None = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to pandas category dtype for XGBoost."""
        out = df[ALL_FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            out[col] = out[col].astype("category")
        return out

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "XGBoostLeadScorer":
        """Fit the XGBoost classifier with sensible defaults."""
        X = self._prepare(train_df)
        y = train_df[self.target].values

        # Note: we do NOT use scale_pos_weight here. With ~12% positive class,
        # the imbalance isn't extreme enough to justify it, and using it tends
        # to push predictions higher than calibrated. We prefer well-calibrated
        # probabilities for the capacity analysis downstream.

        self.model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            tree_method="hist",
            enable_categorical=True,
            random_state=self.random_state,
            eval_metric="logloss",
            early_stopping_rounds=25 if val_df is not None else None,
            n_jobs=-1,
        )

        eval_set = None
        if val_df is not None:
            X_val = self._prepare(val_df)
            y_val = val_df[self.target].values
            eval_set = [(X_val, y_val)]

        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted probability of conversion."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = self._prepare(df)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given probability threshold."""
        return (self.predict_proba(df) >= threshold).astype(int)

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return feature importance as a DataFrame, sorted descending."""
        if self.model is None:
            raise RuntimeError("Call fit() before feature_importance().")
        booster = self.model.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        # Map XGBoost's f0, f1, ... back to original feature names
        feature_map = dict(zip(self.model.feature_names_in_,
                                self.model.feature_names_in_))
        df = pd.DataFrame([
            {"feature": feature_map.get(k, k), "importance": v}
            for k, v in scores.items()
        ])
        if df.empty:
            return pd.DataFrame(columns=["feature", "importance",
                                          "importance_normalized"])
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["importance_normalized"] = df["importance"] / df["importance"].sum()
        return df

    def evaluate(self, test_df: pd.DataFrame) -> dict[str, float]:
        """Return standard classification metrics on the test set."""
        if self.model is None:
            raise RuntimeError("Call fit() before evaluate().")
        y_true = test_df[self.target].values
        y_proba = self.predict_proba(test_df)
        y_pred = (y_proba >= 0.5).astype(int)

        return {
            "auc_roc":          roc_auc_score(y_true, y_proba),
            "auc_pr":           average_precision_score(y_true, y_proba),
            "brier_score":      brier_score_loss(y_true, y_proba),
            "accuracy":         (y_pred == y_true).mean(),
            "positive_rate":    y_true.mean(),
            "predicted_positive_rate": y_pred.mean(),
        }


# ============================================================================
# LOGISTIC REGRESSION BASELINE
# ============================================================================
class LogisticBaseline:
    """
    Logistic regression baseline with one-hot encoding for categoricals
    and standardization for numerics. Used to quantify the marginal value
    of the XGBoost model.
    """

    def __init__(self, target: str = "converted", random_state: int = 42):
        self.target = target
        self.random_state = random_state
        self.model: LogisticRegression | None = None
        self.scaler: StandardScaler | None = None
        self.feature_columns_: list[str] | None = None

    def _prepare(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """One-hot encode categoricals and standardize numerics."""
        df_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True)
        df_num = df[NUMERIC_FEATURES].copy()

        if fit:
            self.feature_columns_ = list(df_cat.columns)
            self.scaler = StandardScaler().fit(df_num.values)
        else:
            # Align categorical dummies to training columns
            for col in self.feature_columns_ or []:
                if col not in df_cat.columns:
                    df_cat[col] = 0
            df_cat = df_cat[self.feature_columns_]

        scaled_num = self.scaler.transform(df_num.values)
        return np.hstack([df_cat.values.astype(float), scaled_num])

    def fit(self, train_df: pd.DataFrame) -> "LogisticBaseline":
        X = self._prepare(train_df, fit=True)
        y = train_df[self.target].values
        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = self._prepare(df, fit=False)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, test_df: pd.DataFrame) -> dict[str, float]:
        y_true = test_df[self.target].values
        y_proba = self.predict_proba(test_df)
        return {
            "auc_roc":     roc_auc_score(y_true, y_proba),
            "auc_pr":      average_precision_score(y_true, y_proba),
            "brier_score": brier_score_loss(y_true, y_proba),
        }


# ============================================================================
# LIFT CURVE / GAINS ANALYSIS
# ============================================================================
def lift_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Compute lift-curve data: if we sort leads by predicted probability descending
    and take the top X%, what fraction of total conversions do we capture?

    Returns columns:
        bin_pct        — cumulative % of leads called (0–100)
        capture_pct    — cumulative % of total conversions captured
        lift           — ratio vs random calling (capture_pct / bin_pct)
        avg_score      — mean predicted probability in this bin
    """
    n = len(y_true)
    # Sort by predicted probability descending
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    proba_sorted = y_proba[order]
    total_conversions = y_sorted.sum()

    rows = []
    for i in range(1, n_bins + 1):
        cutoff = int(n * i / n_bins)
        captured = y_sorted[:cutoff].sum()
        avg_score = proba_sorted[:cutoff].mean()
        bin_pct = 100.0 * cutoff / n
        capture_pct = 100.0 * captured / max(total_conversions, 1)
        rows.append({
            "bin_pct": bin_pct,
            "capture_pct": capture_pct,
            "lift": capture_pct / bin_pct if bin_pct > 0 else 0,
            "avg_score": avg_score,
        })
    return pd.DataFrame(rows)


def capacity_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    capacity_pct: float = 20.0,
) -> dict[str, float]:
    """
    If the call-center can only handle the top `capacity_pct` of leads,
    how many conversions are captured vs random calling?
    """
    n = len(y_true)
    cutoff = int(n * capacity_pct / 100)
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    total_conversions = y_sorted.sum()

    captured = y_sorted[:cutoff].sum()
    capture_pct = 100.0 * captured / max(total_conversions, 1)
    lift = capture_pct / capacity_pct if capacity_pct > 0 else 0

    return {
        "capacity_pct":         capacity_pct,
        "leads_called":         cutoff,
        "conversions_captured": int(captured),
        "total_conversions":    int(total_conversions),
        "capture_pct":          capture_pct,
        "lift_vs_random":       lift,
    }
