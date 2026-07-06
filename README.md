# Germany Wind Generation Forecasting With XGBoost

Prototype forecasting hourly German wind generation with public data, XGBoost, walk-forward validation, and a simple wind-to-price trading-signal backtest.

The project is deliberately built as a candidate-facing case-study repo: reproducible ingestion, QA checks, feature engineering, baselines, an XGBoost improved model, honest time-series validation, a trading-signal research layer, and a short written interpretation of where the model helps or fails.

## Market And Target

- Market: Germany (`DE`)
- Target: hourly actual wind generation, onshore plus offshore, in MW
- Forecast horizon framing: rolling 24-hour-ahead calibration of a public wind forecast
- Strategy framing: paper long/short German day-ahead price-surprise signal driven by wind forecast residuals, benchmarked against a public SMARD forecast-ramp rule

## Public Sources

- SMARD / Bundesnetzagentur API: actual onshore/offshore wind generation and SMARD forecast onshore/offshore wind generation.
  - API documentation: https://github.com/bundesAPI/smard-api
  - Filters used: `4067` actual wind onshore, `1225` actual wind offshore, `123` forecast wind onshore, `3791` forecast wind offshore.
- Open-Meteo Historical Forecast API: archived weather forecast model features at representative German wind locations.
  - Documentation: https://open-meteo.com/en/docs/historical-forecast-api
  - Variables used: 100m wind speed/direction, 10m gusts, temperature, and sea-level pressure.
- SMARD day-ahead price series for Germany/Luxembourg, used for the strategy research layer.
  - Filter used: `4169` day-ahead auction price (`DE-LU`).

## Models

Baselines:

- Previous-week same-hour persistence.
- Train-fold hour/month climatology.
- SMARD published total wind forecast.

Improved model:

- XGBoost residual calibrator.
- The model predicts the residual between actual wind generation and the SMARD forecast, then adds the residual back to the SMARD forecast.
- Features include wind forecast level/ramp/share, lagged actuals, lagged forecast errors, rolling means, calendar seasonality, and weather forecast covariates.

## Validation

Validation is expanding walk-forward time-series cross-validation. No random split is used.

Default setup:

- Initial training window: 180 days
- Fold length: 30 days
- Step: 30 days
- Metrics: MAE, RMSE, bias, and skill versus the SMARD forecast baseline

The July rerun pins the direct package versions in `requirements.txt`, including `xgboost==3.3.0`, because earlier broad version ranges moved the fitted XGBoost result slightly. That version drift did not change the conclusion, but it should be controlled in a case-study repo.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --start 2025-01-01 --end 2026-05-31
```

## Outputs

Running the pipeline writes:

- `data/processed/germany_wind_dataset.csv`
- `outputs/qa_checks.csv`
- `outputs/qa_report.md`
- `outputs/metrics.csv`
- `outputs/fold_metrics.csv`
- `outputs/predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/strategy_trade_log.csv`
- `outputs/strategy_metrics.csv`
- `outputs/strategy_fold_metrics.csv`
- `outputs/strategy_threshold_sensitivity.csv`
- `outputs/submission.csv`
- `outputs/figures/*.png`
- `docs/wind_forecast_case_study.md`

The GitHub repository tracks the reproducible code, docs, and compact QA/metric outputs. Large generated files such as the processed hourly dataset, full prediction table, submission CSV, full trade log, and figures are intentionally left as pipeline outputs rather than source-controlled assets.

## Notes On Honesty

The headline model uses SMARD's public wind forecast as the strongest baseline and main ex-ante driver. This project should be read as a rolling 24-hour-ahead calibration model because it uses 24-hour actual and forecast-error lags. It is not a true prompt/intraday model; once recent metered actuals are available, previous-hour persistence should be tested and would likely be difficult to beat. It is also not a strict day-ahead auction signal: because the German day-ahead auction clears around D-1 noon, 24-hour lagged actuals would be unavailable for some later delivery hours. Weather inputs come from Open-Meteo's archived forecast endpoint rather than fixed D-1 noon model-run snapshots, so the current weather rows should also be treated as post-auction for strict DA trading. For a strict D-1 noon gate-closure forecast, lagged features and weather rows should be recomputed relative to the issue timestamp. Production strategy validation should use executable DA-to-intraday or imbalance settlement marks, not the paper DA(t) minus DA(t-24) proxy used here.
