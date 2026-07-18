from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.diagnostics import (  # noqa: E402
    build_forecast_diagnostics,
    delivery_day_block_bootstrap_mean_ci,
    newey_west_mean_inference,
    newey_west_t_stat,
)
from wind_forecast.features import TARGET  # noqa: E402


def test_forecast_diagnostics_summarize_loss_and_fold_stability() -> None:
    loss_diff = [1.0, -0.5, 2.0, -1.0, 0.5, -0.25] * 10
    predictions = pd.DataFrame(
        {
            "timestamp_local": pd.date_range("2026-01-01", periods=len(loss_diff), freq="h", tz="Europe/Berlin"),
            TARGET: [0.0] * len(loss_diff),
            "smard_forecast": [5.0 + value for value in loss_diff],
            "xgboost_residual": [5.0] * len(loss_diff),
        }
    )
    fold_metrics = pd.DataFrame(
        {
            "model": ["xgboost_residual", "xgboost_residual", "smard_forecast"],
            "fold": [1, 2, 1],
            "skill_vs_smard_mae_pct": [5.0, -3.0, 0.0],
        }
    )

    diagnostics = build_forecast_diagnostics(predictions, fold_metrics).iloc[0]

    assert math.isfinite(float(diagnostics["newey_west_loss_diff_t_stat"]))
    assert float(diagnostics["loss_diff_mean_mw"]) == sum(loss_diff) / len(loss_diff)
    assert float(diagnostics["newey_west_loss_diff_ci_95_lower_mw"]) < float(diagnostics["loss_diff_mean_mw"])
    assert float(diagnostics["newey_west_loss_diff_ci_95_upper_mw"]) > float(diagnostics["loss_diff_mean_mw"])
    assert int(diagnostics["delivery_day_block_count"]) == 3
    assert int(diagnostics["day_block_bootstrap_repetitions"]) == 2_000
    assert float(diagnostics["day_block_bootstrap_loss_diff_ci_95_lower_mw"]) <= float(
        diagnostics["day_block_bootstrap_loss_diff_ci_95_upper_mw"]
    )
    assert int(diagnostics["xgboost_fold_wins"]) == 1
    assert int(diagnostics["fold_count"]) == 2
    assert float(diagnostics["mean_fold_skill_pct"]) == 1.0
    assert int(diagnostics["best_fold"]) == 1
    assert int(diagnostics["worst_fold"]) == 2


def test_newey_west_wrapper_matches_full_inference() -> None:
    values = pd.Series([1.0, 0.5, -0.25, 1.5, -0.5] * 20)

    inference = newey_west_mean_inference(values, max_lag=4)

    assert math.isclose(newey_west_t_stat(values, max_lag=4), float(inference["t_stat"]))
    assert math.isclose(
        float(inference["t_stat"]),
        float(inference["mean"]) / float(inference["standard_error"]),
    )


def test_delivery_day_bootstrap_is_reproducible() -> None:
    values = pd.Series(range(72), dtype=float)
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=72, freq="h", tz="Europe/Berlin"))

    first = delivery_day_block_bootstrap_mean_ci(values, timestamps, repetitions=500, seed=7)
    second = delivery_day_block_bootstrap_mean_ci(values, timestamps, repetitions=500, seed=7)

    assert first == second
    assert first["block_count"] == 3


def test_delivery_day_bootstrap_handles_mixed_dst_offsets() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    timestamps = pd.Series(
        [
            "2026-03-28 23:00:00+01:00",
            "2026-03-29 00:00:00+01:00",
            "2026-03-29 03:00:00+02:00",
            "2026-03-30 00:00:00+02:00",
        ]
    )

    result = delivery_day_block_bootstrap_mean_ci(values, timestamps, repetitions=100, seed=1)

    assert result["block_count"] == 3
