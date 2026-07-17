from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.diagnostics import build_forecast_diagnostics  # noqa: E402
from wind_forecast.features import TARGET  # noqa: E402


def test_forecast_diagnostics_summarize_loss_and_fold_stability() -> None:
    loss_diff = [1.0, -0.5, 2.0, -1.0, 0.5, -0.25] * 10
    predictions = pd.DataFrame(
        {
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
    assert int(diagnostics["xgboost_fold_wins"]) == 1
    assert int(diagnostics["fold_count"]) == 2
    assert float(diagnostics["mean_fold_skill_pct"]) == 1.0
    assert int(diagnostics["best_fold"]) == 1
    assert int(diagnostics["worst_fold"]) == 2
