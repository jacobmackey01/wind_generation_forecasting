from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.features import add_features  # noqa: E402


def test_actual_lag_24h_uses_previous_day_value() -> None:
    hours = pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": hours,
            "timestamp_local": hours.tz_convert("Europe/Berlin"),
            "actual_wind_onshore_mw": range(72),
            "actual_wind_offshore_mw": [10] * 72,
            "forecast_wind_onshore_mw": [20] * 72,
            "forecast_wind_offshore_mw": [5] * 72,
        }
    )
    frame["actual_wind_total_mw"] = frame["actual_wind_onshore_mw"] + frame["actual_wind_offshore_mw"]
    frame["forecast_wind_total_mw"] = frame["forecast_wind_onshore_mw"] + frame["forecast_wind_offshore_mw"]

    features = add_features(frame)

    assert features.loc[24, "actual_lag_24h"] == frame.loc[0, "actual_wind_total_mw"]
    assert features.loc[48, "actual_lag_24h"] == frame.loc[24, "actual_wind_total_mw"]
