"""XGBoost modelling and walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from wind_forecast.features import SMARD_BASELINE, TARGET, feature_columns


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_days: int = 180
    fold_days: int = 30
    step_days: int = 30


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    error = y_pred.to_numpy(dtype=float) - y_true.to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def _clean_model_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    needed = ["timestamp_utc", "timestamp_local", TARGET, SMARD_BASELINE, "actual_lag_168h", *features]
    needed = list(dict.fromkeys(needed))
    clean = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return clean.sort_values("timestamp_utc").reset_index(drop=True)


def _fold_boundaries(df: pd.DataFrame, config: WalkForwardConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    min_ts = df["timestamp_utc"].min()
    max_ts = df["timestamp_utc"].max()
    fold_start = min_ts + pd.Timedelta(days=config.initial_train_days)
    boundaries: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while fold_start <= max_ts:
        fold_end = min(fold_start + pd.Timedelta(days=config.fold_days), max_ts + pd.Timedelta(hours=1))
        if fold_end > fold_start:
            boundaries.append((fold_start, fold_end))
        fold_start = fold_start + pd.Timedelta(days=config.step_days)
    return boundaries


def fit_xgboost_residual(train: pd.DataFrame, features: list[str]) -> XGBRegressor:
    x = train[features]
    y = train[TARGET] - train[SMARD_BASELINE]
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=2,
        tree_method="hist",
        eval_metric="rmse",
    )
    model.fit(x, y)
    return model


def _predict_xgb(model: XGBRegressor, valid: pd.DataFrame, train: pd.DataFrame, features: list[str]) -> np.ndarray:
    residual = model.predict(valid[features])
    upper_clip = max(float(train[TARGET].quantile(0.999) * 1.1), float(train[TARGET].max()))
    return np.clip(valid[SMARD_BASELINE].to_numpy(dtype=float) + residual, 0.0, upper_clip)


def _hour_month_climatology_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    train_keyed = train[["timestamp_local", TARGET]].copy()
    train_keyed["month"] = train_keyed["timestamp_local"].dt.month
    train_keyed["hour"] = train_keyed["timestamp_local"].dt.hour

    valid_keyed = valid[["timestamp_local"]].copy()
    valid_keyed["month"] = valid_keyed["timestamp_local"].dt.month
    valid_keyed["hour"] = valid_keyed["timestamp_local"].dt.hour

    month_hour = train_keyed.groupby(["month", "hour"])[TARGET].mean().rename("month_hour_mean")
    hour = train_keyed.groupby("hour")[TARGET].mean().rename("hour_mean")
    global_mean = float(train_keyed[TARGET].mean())

    out = valid_keyed.join(month_hour, on=["month", "hour"]).join(hour, on="hour")
    return out["month_hour_mean"].fillna(out["hour_mean"]).fillna(global_mean).to_numpy(dtype=float)


def _predictions_for_fold(
    model: XGBRegressor,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    fold_id: int,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
) -> pd.DataFrame:
    preds = valid[["timestamp_utc", "timestamp_local", TARGET, SMARD_BASELINE, "actual_lag_168h"]].copy()
    preds["fold"] = fold_id
    preds["fold_start"] = fold_start
    preds["fold_end"] = fold_end
    preds["persistence_prev_week"] = preds["actual_lag_168h"]
    preds["hour_month_climatology"] = _hour_month_climatology_predict(train, valid)
    preds["smard_forecast"] = preds[SMARD_BASELINE]
    preds["xgboost_residual"] = _predict_xgb(model, valid, train, features)
    preds["id"] = preds["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return preds


def _metrics_table(predictions: pd.DataFrame, by_fold: bool) -> pd.DataFrame:
    prediction_columns = [
        ("persistence_prev_week", "persistence_prev_week"),
        ("hour_month_climatology", "hour_month_climatology"),
        ("smard_forecast", "smard_forecast"),
        ("xgboost_residual", "xgboost_residual"),
    ]
    groups = predictions.groupby("fold", sort=True) if by_fold else [(None, predictions)]

    rows = []
    for fold, frame in groups:
        smard_mae = metrics(frame[TARGET], frame["smard_forecast"])["mae"]
        for model_name, column in prediction_columns:
            row = {"model": model_name}
            if by_fold:
                row["fold"] = int(fold)
                row["fold_start"] = str(frame["fold_start"].iloc[0])
                row["fold_end"] = str(frame["fold_end"].iloc[0])
            row.update(metrics(frame[TARGET], frame[column]))
            row["skill_vs_smard_mae_pct"] = float((smard_mae - row["mae"]) / smard_mae * 100.0) if smard_mae else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _feature_importance(models: list[XGBRegressor], features: list[str]) -> pd.DataFrame:
    rows = []
    for fold_id, model in enumerate(models, start=1):
        for feature, importance in zip(features, model.feature_importances_):
            rows.append({"fold": fold_id, "feature": feature, "importance": float(importance)})
    if not rows:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    raw = pd.DataFrame(rows)
    return (
        raw.groupby("feature", as_index=False)["importance"]
        .agg(importance_mean="mean", importance_std="std")
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def validate_walk_forward(
    frame: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run expanding walk-forward validation and return predictions and metrics."""

    config = config or WalkForwardConfig()
    features = feature_columns(frame)
    clean = _clean_model_frame(frame, features)
    folds = _fold_boundaries(clean, config)
    if not folds:
        raise RuntimeError("Not enough clean data for the requested walk-forward validation setup.")

    all_predictions: list[pd.DataFrame] = []
    models: list[XGBRegressor] = []

    for fold_id, (fold_start, fold_end) in enumerate(folds, start=1):
        train = clean[clean["timestamp_utc"] < fold_start].copy()
        valid = clean[(clean["timestamp_utc"] >= fold_start) & (clean["timestamp_utc"] < fold_end)].copy()
        if len(train) < 24 * 90 or valid.empty:
            continue
        model = fit_xgboost_residual(train, features)
        models.append(model)
        all_predictions.append(_predictions_for_fold(model, train, valid, features, fold_id, fold_start, fold_end))

    if not all_predictions:
        raise RuntimeError("No valid walk-forward folds were produced.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    overall_metrics = _metrics_table(predictions, by_fold=False)
    fold_metrics = _metrics_table(predictions, by_fold=True)
    importance = _feature_importance(models, features)
    return predictions, overall_metrics, fold_metrics, importance
