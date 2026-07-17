"""Statistical diagnostics shared by reports and the dashboard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wind_forecast.features import TARGET


def newey_west_t_stat(values: pd.Series, max_lag: int = 24) -> float:
    """Return a Newey-West t-statistic for the mean of a serially correlated series."""

    clean = values.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 3:
        return float("nan")

    mean = float(clean.mean())
    centered = clean - mean
    long_run_var = float(np.mean(centered**2))
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        long_run_var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance

    if long_run_var <= 0:
        return float("nan")
    return mean / math.sqrt(long_run_var / n)


def build_forecast_diagnostics(predictions: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build compact, source-controlled diagnostics for reporting surfaces."""

    prediction_columns = {TARGET, "smard_forecast", "xgboost_residual"}
    fold_columns = {"model", "fold", "skill_vs_smard_mae_pct"}
    missing_predictions = prediction_columns.difference(predictions.columns)
    missing_folds = fold_columns.difference(fold_metrics.columns)
    if missing_predictions:
        raise ValueError(f"Predictions are missing columns: {sorted(missing_predictions)}")
    if missing_folds:
        raise ValueError(f"Fold metrics are missing columns: {sorted(missing_folds)}")

    smard_abs_error = (predictions[TARGET] - predictions["smard_forecast"]).abs()
    xgb_abs_error = (predictions[TARGET] - predictions["xgboost_residual"]).abs()
    loss_diff_t = newey_west_t_stat(smard_abs_error - xgb_abs_error, max_lag=24)

    xgb_folds = fold_metrics.loc[fold_metrics["model"] == "xgboost_residual"].copy()
    if xgb_folds.empty:
        raise ValueError("Fold metrics contain no xgboost_residual rows.")

    best = xgb_folds.sort_values("skill_vs_smard_mae_pct", ascending=False).iloc[0]
    worst = xgb_folds.sort_values("skill_vs_smard_mae_pct", ascending=True).iloc[0]
    return pd.DataFrame(
        [
            {
                "newey_west_loss_diff_t_stat": loss_diff_t,
                "xgboost_fold_wins": int((xgb_folds["skill_vs_smard_mae_pct"] > 0).sum()),
                "fold_count": int(len(xgb_folds)),
                "mean_fold_skill_pct": float(xgb_folds["skill_vs_smard_mae_pct"].mean()),
                "best_fold": int(best["fold"]),
                "best_fold_skill_pct": float(best["skill_vs_smard_mae_pct"]),
                "worst_fold": int(worst["fold"]),
                "worst_fold_skill_pct": float(worst["skill_vs_smard_mae_pct"]),
            }
        ]
    )
