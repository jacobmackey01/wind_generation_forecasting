"""Statistical diagnostics shared by reports and the dashboard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wind_forecast.features import TARGET


NORMAL_95_Z = 1.959963984540054
DEFAULT_BOOTSTRAP_REPETITIONS = 2_000
DEFAULT_BOOTSTRAP_SEED = 42


def newey_west_mean_inference(values: pd.Series, max_lag: int = 24) -> dict[str, float | int]:
    """Return HAC inference for the mean of a serially correlated series."""

    clean = values.dropna().to_numpy(dtype=float)
    n = len(clean)
    mean = float(clean.mean()) if n else float("nan")
    if n < 3:
        return {
            "observations": n,
            "mean": mean,
            "standard_error": float("nan"),
            "t_stat": float("nan"),
            "ci_95_lower": float("nan"),
            "ci_95_upper": float("nan"),
        }

    centered = clean - mean
    long_run_var = float(np.mean(centered**2))
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        long_run_var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance

    if long_run_var <= 0:
        standard_error = float("nan")
    else:
        standard_error = math.sqrt(long_run_var / n)

    if not math.isfinite(standard_error) or standard_error == 0:
        t_stat = float("nan")
        ci_lower = float("nan")
        ci_upper = float("nan")
    else:
        t_stat = mean / standard_error
        ci_lower = mean - NORMAL_95_Z * standard_error
        ci_upper = mean + NORMAL_95_Z * standard_error

    return {
        "observations": n,
        "mean": mean,
        "standard_error": standard_error,
        "t_stat": t_stat,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }


def newey_west_t_stat(values: pd.Series, max_lag: int = 24) -> float:
    """Return the Newey-West t-statistic retained for backward compatibility."""

    return float(newey_west_mean_inference(values, max_lag=max_lag)["t_stat"])


def delivery_day_block_bootstrap_mean_ci(
    values: pd.Series,
    delivery_timestamps: pd.Series,
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Return a deterministic 95% CI from resampled local delivery-day blocks."""

    if repetitions < 100:
        raise ValueError("Block-bootstrap repetitions must be at least 100.")
    timestamps = pd.to_datetime(delivery_timestamps, errors="coerce", utc=True).dt.tz_convert("Europe/Berlin")
    frame = pd.DataFrame({"value": values, "delivery_day": timestamps.dt.date}).dropna()
    grouped = frame.groupby("delivery_day", sort=True)["value"]
    day_sums = grouped.sum().to_numpy(dtype=float)
    day_counts = grouped.count().to_numpy(dtype=float)
    block_count = len(day_sums)
    if block_count < 2:
        return {
            "block_count": block_count,
            "repetitions": repetitions,
            "ci_95_lower": float("nan"),
            "ci_95_upper": float("nan"),
        }

    rng = np.random.default_rng(seed)
    sampled_days = rng.integers(0, block_count, size=(repetitions, block_count))
    sampled_means = day_sums[sampled_days].sum(axis=1) / day_counts[sampled_days].sum(axis=1)
    ci_lower, ci_upper = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "block_count": block_count,
        "repetitions": repetitions,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
    }


def build_forecast_diagnostics(predictions: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build compact, source-controlled diagnostics for reporting surfaces."""

    prediction_columns = {"timestamp_local", TARGET, "smard_forecast", "xgboost_residual"}
    fold_columns = {"model", "fold", "skill_vs_smard_mae_pct"}
    missing_predictions = prediction_columns.difference(predictions.columns)
    missing_folds = fold_columns.difference(fold_metrics.columns)
    if missing_predictions:
        raise ValueError(f"Predictions are missing columns: {sorted(missing_predictions)}")
    if missing_folds:
        raise ValueError(f"Fold metrics are missing columns: {sorted(missing_folds)}")

    smard_abs_error = (predictions[TARGET] - predictions["smard_forecast"]).abs()
    xgb_abs_error = (predictions[TARGET] - predictions["xgboost_residual"]).abs()
    loss_diff = smard_abs_error - xgb_abs_error
    smard_mae = float(smard_abs_error.mean())
    nw = newey_west_mean_inference(loss_diff, max_lag=24)
    bootstrap = delivery_day_block_bootstrap_mean_ci(loss_diff, predictions["timestamp_local"])

    def as_skill_pct(value: float | int) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or smard_mae == 0:
            return float("nan")
        return numeric / smard_mae * 100.0

    xgb_folds = fold_metrics.loc[fold_metrics["model"] == "xgboost_residual"].copy()
    if xgb_folds.empty:
        raise ValueError("Fold metrics contain no xgboost_residual rows.")

    best = xgb_folds.sort_values("skill_vs_smard_mae_pct", ascending=False).iloc[0]
    worst = xgb_folds.sort_values("skill_vs_smard_mae_pct", ascending=True).iloc[0]
    return pd.DataFrame(
        [
            {
                "loss_diff_observations": int(nw["observations"]),
                "loss_diff_mean_mw": float(nw["mean"]),
                "loss_diff_skill_pct": as_skill_pct(nw["mean"]),
                "newey_west_max_lag": 24,
                "newey_west_loss_diff_se_mw": float(nw["standard_error"]),
                "newey_west_loss_diff_t_stat": float(nw["t_stat"]),
                "newey_west_loss_diff_ci_95_lower_mw": float(nw["ci_95_lower"]),
                "newey_west_loss_diff_ci_95_upper_mw": float(nw["ci_95_upper"]),
                "newey_west_skill_ci_95_lower_pct": as_skill_pct(nw["ci_95_lower"]),
                "newey_west_skill_ci_95_upper_pct": as_skill_pct(nw["ci_95_upper"]),
                "delivery_day_block_count": int(bootstrap["block_count"]),
                "day_block_bootstrap_repetitions": int(bootstrap["repetitions"]),
                "day_block_bootstrap_loss_diff_ci_95_lower_mw": float(bootstrap["ci_95_lower"]),
                "day_block_bootstrap_loss_diff_ci_95_upper_mw": float(bootstrap["ci_95_upper"]),
                "day_block_bootstrap_skill_ci_95_lower_pct": as_skill_pct(bootstrap["ci_95_lower"]),
                "day_block_bootstrap_skill_ci_95_upper_pct": as_skill_pct(bootstrap["ci_95_upper"]),
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
