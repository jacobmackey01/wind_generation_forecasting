from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

from wind_forecast.features import add_features  # noqa: E402
from wind_forecast.models import WalkForwardConfig, validate_walk_forward  # noqa: E402
from wind_forecast.qa import qa_checks, write_qa_report  # noqa: E402
from wind_forecast.report import write_case_study, write_figures  # noqa: E402
from wind_forecast.smard import build_smard_dataset  # noqa: E402
from wind_forecast.strategy import backtest_wind_price_signal  # noqa: E402
from wind_forecast.weather import build_weather_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Germany wind generation forecasting pipeline.")
    parser.add_argument("--start", default="2025-01-01", help="Start date, local Germany date, YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-05-31", help="End date, local Germany date, YYYY-MM-DD.")
    parser.add_argument("--initial-train-days", type=int, default=180)
    parser.add_argument("--fold-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--signal-threshold-mw", type=float, default=1500.0)
    parser.add_argument("--transaction-cost-eur-mwh", type=float, default=0.5)
    parser.add_argument("--no-cache", action="store_true", help="Force re-download of raw public data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cache = not args.no_cache

    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    outputs_dir = ROOT / "outputs"
    docs_dir = ROOT / "docs"
    figures_dir = outputs_dir / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching SMARD data...")
    smard = build_smard_dataset(args.start, args.end, raw_dir / "smard", use_cache=use_cache)
    print("Fetching Open-Meteo historical forecast data...")
    weather_start = (pd.Timestamp(args.start) - pd.Timedelta(days=1)).date().isoformat()
    weather = build_weather_dataset(weather_start, args.end, raw_dir / "open_meteo", use_cache=use_cache)

    dataset = smard.merge(weather, on="timestamp_utc", how="left").sort_values("timestamp_utc").reset_index(drop=True)
    dataset.to_csv(processed_dir / "germany_wind_dataset.csv", index=False)

    checks, qa_summary = qa_checks(dataset)
    checks.to_csv(outputs_dir / "qa_checks.csv", index=False)
    write_qa_report(checks, qa_summary, outputs_dir / "qa_report.md")

    feature_frame = add_features(dataset)
    config = WalkForwardConfig(
        initial_train_days=args.initial_train_days,
        fold_days=args.fold_days,
        step_days=args.step_days,
    )
    predictions, metrics, fold_metrics, feature_importance = validate_walk_forward(feature_frame, config)

    metrics.to_csv(outputs_dir / "metrics.csv", index=False)
    fold_metrics.to_csv(outputs_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(outputs_dir / "predictions.csv", index=False)
    feature_importance.to_csv(outputs_dir / "feature_importance.csv", index=False)

    submission = predictions[["id", "xgboost_residual"]].rename(columns={"xgboost_residual": "y_pred"})
    submission.to_csv(outputs_dir / "submission.csv", index=False)

    trade_log, strategy_metrics, strategy_fold_metrics, strategy_thresholds = backtest_wind_price_signal(
        feature_frame,
        predictions,
        threshold_mw=args.signal_threshold_mw,
        transaction_cost_eur_mwh=args.transaction_cost_eur_mwh,
    )
    trade_log.to_csv(outputs_dir / "strategy_trade_log.csv", index=False)
    strategy_metrics.to_csv(outputs_dir / "strategy_metrics.csv", index=False)
    strategy_fold_metrics.to_csv(outputs_dir / "strategy_fold_metrics.csv", index=False)
    strategy_thresholds.to_csv(outputs_dir / "strategy_threshold_sensitivity.csv", index=False)

    write_figures(predictions, fold_metrics, feature_importance, figures_dir)
    write_case_study(
        dataset=dataset,
        qa_summary=qa_summary,
        metrics=metrics,
        fold_metrics=fold_metrics,
        predictions=predictions,
        feature_importance=feature_importance,
        strategy_metrics=strategy_metrics,
        strategy_fold_metrics=strategy_fold_metrics,
        strategy_thresholds=strategy_thresholds,
        path=docs_dir / "wind_forecast_case_study.md",
    )

    xgb = metrics.loc[metrics["model"] == "xgboost_residual"].iloc[0]
    smard = metrics.loc[metrics["model"] == "smard_forecast"].iloc[0]
    print("Done.")
    print(f"SMARD forecast MAE: {smard['mae']:.2f} MW")
    print(f"XGBoost residual MAE: {xgb['mae']:.2f} MW")
    print(f"XGBoost skill vs SMARD MAE: {xgb['skill_vs_smard_mae_pct']:.2f}%")


if __name__ == "__main__":
    main()
