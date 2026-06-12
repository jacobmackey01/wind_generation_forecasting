"""Feature engineering for hourly wind generation forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "actual_wind_total_mw"
SMARD_BASELINE = "forecast_wind_total_mw"


BASE_FEATURE_COLUMNS = [
    "forecast_wind_onshore_mw",
    "forecast_wind_offshore_mw",
    "forecast_wind_total_mw",
    "forecast_wind_total_ramp_1h",
    "forecast_wind_total_ramp_24h",
    "forecast_offshore_share",
    "actual_lag_24h",
    "actual_lag_48h",
    "actual_lag_168h",
    "actual_roll_24h_mean",
    "actual_roll_168h_mean",
    "forecast_error_lag_24h",
    "forecast_error_lag_48h",
    "forecast_error_lag_168h",
    "forecast_error_roll_168h_mean",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]


def _add_direction_features(out: pd.DataFrame) -> pd.DataFrame:
    direction_columns = [column for column in out.columns if column.endswith("_wind_direction_100m_deg")]
    for column in direction_columns:
        radians = np.deg2rad(out[column])
        prefix = column.removesuffix("_deg")
        out[f"{prefix}_sin"] = np.sin(radians)
        out[f"{prefix}_cos"] = np.cos(radians)
    return out


def _add_weather_aggregates(out: pd.DataFrame) -> pd.DataFrame:
    speed_columns = [column for column in out.columns if column.endswith("_wind_speed_100m_ms")]
    gust_columns = [column for column in out.columns if column.endswith("_wind_gusts_10m_ms")]
    temp_columns = [column for column in out.columns if column.endswith("_temperature_2m_c")]
    pressure_columns = [column for column in out.columns if column.endswith("_pressure_msl_hpa")]

    if speed_columns:
        out["weather_wind_speed_100m_mean_ms"] = out[speed_columns].mean(axis=1)
        out["weather_wind_speed_100m_max_ms"] = out[speed_columns].max(axis=1)
        out["weather_wind_speed_100m_cubed_mean"] = (out[speed_columns] ** 3).mean(axis=1)
    if gust_columns:
        out["weather_wind_gusts_10m_mean_ms"] = out[gust_columns].mean(axis=1)
    if temp_columns:
        out["weather_temperature_2m_mean_c"] = out[temp_columns].mean(axis=1)
    if pressure_columns:
        out["weather_pressure_msl_mean_hpa"] = out[pressure_columns].mean(axis=1)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, rolling, forecast, and weather features."""

    out = df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    local = out["timestamp_local"]

    out["hour"] = local.dt.hour
    out["day_of_week"] = local.dt.dayofweek
    out["month"] = local.dt.month
    out["dayofyear"] = local.dt.dayofyear
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dayofyear_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366)
    out["dayofyear_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366)

    out["forecast_wind_total_ramp_1h"] = out[SMARD_BASELINE].diff()
    out["forecast_wind_total_ramp_24h"] = out[SMARD_BASELINE] - out[SMARD_BASELINE].shift(24)
    out["forecast_offshore_share"] = out["forecast_wind_offshore_mw"] / out[SMARD_BASELINE].replace(0, np.nan)

    forecast_error = out[TARGET] - out[SMARD_BASELINE]
    out["actual_lag_24h"] = out[TARGET].shift(24)
    out["actual_lag_48h"] = out[TARGET].shift(48)
    out["actual_lag_168h"] = out[TARGET].shift(168)
    out["actual_roll_24h_mean"] = out[TARGET].shift(24).rolling(24, min_periods=12).mean()
    out["actual_roll_168h_mean"] = out[TARGET].shift(24).rolling(168, min_periods=48).mean()
    out["forecast_error_lag_24h"] = forecast_error.shift(24)
    out["forecast_error_lag_48h"] = forecast_error.shift(48)
    out["forecast_error_lag_168h"] = forecast_error.shift(168)
    out["forecast_error_roll_168h_mean"] = forecast_error.shift(24).rolling(168, min_periods=48).mean()

    out = _add_direction_features(out)
    out = _add_weather_aggregates(out)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric model feature columns present in a feature frame."""

    weather_feature_columns = [
        column
        for column in df.columns
        if (
            column.endswith("_wind_speed_100m_ms")
            or column.endswith("_wind_gusts_10m_ms")
            or column.endswith("_temperature_2m_c")
            or column.endswith("_pressure_msl_hpa")
            or column.endswith("_wind_direction_100m_sin")
            or column.endswith("_wind_direction_100m_cos")
            or column.startswith("weather_")
        )
    ]
    columns = [column for column in BASE_FEATURE_COLUMNS if column in df.columns]
    columns.extend(column for column in weather_feature_columns if column not in columns)
    return columns
