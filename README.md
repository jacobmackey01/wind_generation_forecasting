# Germany Wind Generation Forecasting With XGBoost

Prototype forecasting hourly German wind generation with public data, XGBoost, walk-forward validation, and a simple wind-to-price trading-signal backtest.

The project is deliberately built as a candidate-facing case-study repo: reproducible ingestion, QA checks, feature engineering, baselines, an XGBoost improved model, honest time-series validation, a trading-signal research layer, and a short written interpretation of where the model helps or fails.

## Frozen D-1 Specification

[`docs/forecast_issue_time_specification.md`](docs/forecast_issue_time_specification.md) freezes the proposed production-oriented forecast before any parameter tuning: 11:00 Europe/Berlin D-1 issue time, exact weather-model run, conservative lag cutoffs, grouped walk-forward validation, inference outputs, prospective holdout, and model-versioning requirements.

The existing May and June results do not comply with that stricter information set. They remain historical rolling-calibration research outputs and must not be presented as strict day-ahead auction forecasts. No XGBoost parameter search should begin until the specification's mandatory issue-time QA gates pass.

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

`outputs/forecast_diagnostics.csv` now records the mean hourly loss difference, lag-24 Newey-West standard error and 95% confidence interval, the corresponding approximate skill interval, and a delivery-day block-bootstrap interval. For the June reference run, both interval estimates span zero; the small pooled improvement is therefore not distinguishable from no incremental edge.

## Prospective Holdout Protection

The 2026-07-01 to 2026-09-30 target period is mechanically locked. `scripts/run_pipeline.py` rejects any overlapping date range immediately after parsing arguments, before data ingestion, model fitting, target metrics, or an OpenAI call. This remains true after the embargo ends: the historical research pipeline is not the route for scoring the holdout.

A separate release-seal utility is ready for the future strict D-1 model:

```powershell
python scripts/holdout_manifest.py freeze --model-manifest manifests/promoted_model.json --predictions outputs/holdout_predictions.csv
python scripts/holdout_manifest.py verify --manifest manifests/holdout_release_manifest.json
```

The freeze command requires a `WG-D1-001` model promoted before the holdout, rejects prediction CSVs containing target columns, and writes a manifest plus SHA-256 sidecar over the model manifest and predictions. Verification rejects scoring before 2026-10-01, a late-frozen manifest, or any changed artifact. No real release manifest is created yet because the strict issue-time pipeline has not produced a promoted model or prospective predictions. When it does, both seal files must be committed before target scoring; the Git history supplies the external timestamp that a locally regenerated hash cannot prove by itself.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --start 2025-01-01 --end 2026-05-31
streamlit run dashboard.py
```

By default, each full pipeline run generates a fresh downstream AI review after all deterministic outputs are complete. Use `--skip-llm-review` for an offline or zero-cost run. To refresh only the AI review from existing CSV outputs:

```powershell
python scripts/run_llm_review.py
```

The local key is read from `.env.local`, which is ignored by Git; `.env.example` documents the expected variables without containing a credential. The legacy-explicit `--run-llm-review` flag is also accepted, although the LLM step is now the default.

## AI/LLM Integration

The programmatic OpenAI component reduces the manual work of turning validation outputs into a concise trader/reviewer memo. It uses `gpt-5.6-luna` with low reasoning through the Responses API and parses a strict Pydantic schema.

The LLM receives a compact evidence package derived from the QA, model, fold, strategy, and threshold CSVs. It does not receive raw credentials, change forecasts, calculate P&L, choose trades, or feed back into XGBoost. Narrative findings must cite supplied evidence IDs; an unknown or missing evidence ID causes the response to be rejected. Exact metric values are attached by deterministic code after generation. A SHA-256 evidence fingerprint ties the saved review to the current CSV snapshot, and Streamlit refuses to render a stale review after a different pipeline run.

The component writes:

- `outputs/llm/analyst_review.md`: human-readable review
- `outputs/llm/analyst_review.json`: structured dashboard artifact
- `outputs/llm/prompt.md`: exact system and user prompts
- `outputs/llm/run_log.json`: prompts, evidence, structured response, model, request metadata, and token usage

Streamlit displays the latest saved review as a downstream interpretation panel. Page refreshes never call the API, avoiding accidental spend and keeping the API key out of the frontend.

## Validation Dashboard

The Streamlit dashboard is deliberately built around the negative result rather than a flattering model headline. It shows fold-level MAE against SMARD, the forecast significance diagnostic, cumulative strategy proxy P&L, the November concentration, threshold sensitivity, and the public forecast-ramp benchmark on the same evidence base. The large public-ramp result is presented as a diagnostic that the DA(t) minus DA(t-24) proxy rewards public-information repricing, not as evidence of a free executable edge.

It runs from the compact tracked CSVs in `outputs/`, so a fresh clone can open the dashboard immediately. Re-running the pipeline refreshes those inputs before the dashboard is launched.

The tracked `outputs/` directory contains the June-extended rerun. The static Netlify bundle in `site/data/` deliberately preserves the original May case-study snapshot, including its matching LLM review, so the public dashboard remains a reproducible record of that release rather than silently changing when local outputs are regenerated. See `docs/run_comparison_may_vs_june.md` for a like-for-like comparison.

## Outputs

Running the pipeline writes:

- `data/processed/germany_wind_dataset.csv`
- `outputs/qa_checks.csv`
- `outputs/qa_report.md`
- `outputs/metrics.csv`
- `outputs/forecast_diagnostics.csv`
- `outputs/fold_metrics.csv`
- `outputs/predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/strategy_trade_log.csv`
- `outputs/strategy_metrics.csv`
- `outputs/strategy_fold_metrics.csv`
- `outputs/strategy_threshold_sensitivity.csv`
- `outputs/llm/analyst_review.md`
- `outputs/llm/analyst_review.json`
- `outputs/llm/prompt.md`
- `outputs/llm/run_log.json`
- `outputs/submission.csv`
- `outputs/figures/*.png`
- `docs/forecast_issue_time_specification.md`
- `docs/wind_forecast_case_study.md`

The GitHub repository tracks the reproducible code, docs, and compact QA/metric outputs. Large generated files such as the processed hourly dataset, full prediction table, submission CSV, full trade log, and figures are intentionally left as pipeline outputs rather than source-controlled assets.

## Notes On Honesty

Strategy average P&L is reported in EUR/MWh per traded hour. Strategy total P&L is reported as EUR per 1 MW hourly clip, so the public-ramp benchmark and residual signal should be compared on per-trade economics before comparing aggregate totals with very different trade counts.

The headline model uses SMARD's public wind forecast as the strongest baseline and main ex-ante driver. This project should be read as a rolling 24-hour-ahead calibration model because it uses 24-hour actual and forecast-error lags. It is not a true prompt/intraday model; once recent metered actuals are available, previous-hour persistence should be tested and would likely be difficult to beat. It is also not a strict day-ahead auction signal: because the German day-ahead auction clears around D-1 noon, 24-hour lagged actuals would be unavailable for some later delivery hours. Weather inputs come from Open-Meteo's archived forecast endpoint rather than fixed D-1 noon model-run snapshots, so the current weather rows should also be treated as post-auction for strict DA trading. For a strict D-1 noon gate-closure forecast, lagged features and weather rows should be recomputed relative to the issue timestamp. Production strategy validation should use executable DA-to-intraday or imbalance settlement marks, not the paper DA(t) minus DA(t-24) proxy used here.
