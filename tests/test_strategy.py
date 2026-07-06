from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.strategy import build_public_forecast_ramp_trade_log, build_trade_log, summarize_strategy  # noqa: E402


def test_wind_edge_position_direction() -> None:
    hours = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    feature_frame = pd.DataFrame({"timestamp_utc": hours, "price_da_eur_mwh": range(30)})
    predictions = pd.DataFrame(
        {
            "timestamp_utc": hours,
            "timestamp_local": hours.tz_convert("Europe/Berlin"),
            "actual_wind_total_mw": [0.0] * 30,
            "forecast_wind_total_mw": [10_000.0] * 30,
            "xgboost_residual": [10_000.0] * 30,
            "fold": [1] * 30,
            "fold_start": [hours[0]] * 30,
            "fold_end": [hours[-1]] * 30,
        }
    )
    predictions.loc[24, "xgboost_residual"] = 12_000.0
    predictions.loc[25, "xgboost_residual"] = 8_000.0

    trade_log = build_trade_log(feature_frame, predictions, threshold_mw=1500.0, transaction_cost_eur_mwh=0.0)

    assert int(trade_log.loc[trade_log["timestamp_utc"] == hours[24], "position"].iloc[0]) == -1
    assert int(trade_log.loc[trade_log["timestamp_utc"] == hours[25], "position"].iloc[0]) == 1


def test_strategy_summary_includes_significance_fields() -> None:
    trade_log = pd.DataFrame(
        {
            "position": [1, -1, 1, -1],
            "gross_pnl_eur_mwh": [2.0, 1.0, -1.0, 3.0],
            "net_pnl_eur_mwh": [1.5, 0.5, -1.5, 2.5],
            "fold": [1, 1, 2, 2],
            "fold_start": ["2025-01-01"] * 4,
            "fold_end": ["2025-01-02"] * 4,
        }
    )

    overall, _ = summarize_strategy(trade_log)

    assert "hit_rate_z_stat" in overall.columns
    assert "net_pnl_t_stat" in overall.columns
    assert float(overall["hit_rate_z_stat"].iloc[0]) == 1.0


def test_public_forecast_ramp_benchmark_position_direction() -> None:
    hours = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    feature_frame = pd.DataFrame(
        {
            "timestamp_utc": hours,
            "price_da_eur_mwh": range(30),
            "forecast_wind_total_ramp_24h": [0.0] * 30,
        }
    )
    feature_frame.loc[24, "forecast_wind_total_ramp_24h"] = 2_000.0
    feature_frame.loc[25, "forecast_wind_total_ramp_24h"] = -2_000.0
    predictions = pd.DataFrame(
        {
            "timestamp_utc": hours,
            "fold": [1] * 30,
            "fold_start": [hours[0]] * 30,
            "fold_end": [hours[-1]] * 30,
        }
    )

    trade_log = build_public_forecast_ramp_trade_log(
        feature_frame,
        predictions,
        threshold_mw=1500.0,
        transaction_cost_eur_mwh=0.0,
    )

    assert int(trade_log.loc[trade_log["timestamp_utc"] == hours[24], "position"].iloc[0]) == -1
    assert int(trade_log.loc[trade_log["timestamp_utc"] == hours[25], "position"].iloc[0]) == 1
