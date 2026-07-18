# Germany Wind Generation Forecasting Case Study

Jacob Mackey

## Objective

Build a clean, trading-relevant forecasting pipeline for German hourly wind generation. The model predicts actual onshore plus offshore wind generation and is evaluated against simple and market-relevant baselines using walk-forward validation.

The intended use case is rolling 24-hour-ahead calibration of the public wind forecast. The information set is published forecasts plus observations and forecast errors that are at least 24 hours old relative to delivery. At true prompt or intraday horizons, recent metered actuals become available and a last-observation style benchmark would dominate this feature set.

## Data

Sample: 2024-12-31 23:00:00+00:00 to 2026-06-30 21:00:00+00:00 UTC, 13103 hourly rows.

- SMARD / Bundesnetzagentur: actual wind generation and published wind generation forecasts for Germany.
- SMARD / Bundesnetzagentur: German/Luxembourg day-ahead auction prices, series 4169, for the strategy research layer.
- Open-Meteo Historical Forecast API: archived forecast-model weather covariates at four representative German wind locations.

QA passed: True. Checks cover timestamp uniqueness/continuity, local power-day hour counts, missing values, and plausible generation, weather, and day-ahead price ranges.

## Features

The feature set includes SMARD wind forecast levels and ramps, previous-day and previous-week actual wind lags, rolling means, lagged forecast errors, calendar seasonality, and weather forecast variables such as 100m wind speed/direction, gusts, temperature, and pressure.

## Models And Validation

Baselines are previous-week same-hour persistence, train-fold hour/month climatology, and the SMARD published wind forecast. The improved model is an XGBoost residual calibrator: it predicts actual minus SMARD forecast, then adds the correction back to the SMARD baseline. Validation uses expanding walk-forward folds from 2025-07-06 23:00:00+00:00 to 2026-06-30 22:00:00+00:00, not a random split.

| model | mae | rmse | bias | skill_vs_smard_mae_pct |
| --- | --- | --- | --- | --- |
| persistence_prev_week | 11,311.50 | 14,565.58 | 97.20 | -714.10 |
| hour_month_climatology | 8,695.84 | 10,898.42 | -742.13 | -525.85 |
| smard_forecast | 1,389.44 | 2,004.11 | -133.54 | 0.00 |
| xgboost_residual | 1,384.67 | 1,968.31 | -36.94 | 0.34 |

## Interpretation

XGBoost does not show a reliable edge over the SMARD forecast in this backtest: pooled MAE is only 0.34% better, while average fold skill is -0.49%.

The mean hourly absolute-error improvement is 4.77 MW. Its lag-24 Newey-West standard error is 23.08 MW, giving t = 0.21 and a normal 95% interval of [-40.47, 50.00] MW. Dividing by the fixed SMARD MAE gives an approximate skill interval of [-2.91, 3.60]%. A delivery-day block bootstrap gives [-40.91, 50.27] MW. Both intervals span zero, so the data do not distinguish the small pooled improvement from no incremental edge.

The previous-week persistence MAE was 11,311.50 MW and the hour/month climatology MAE was 8,695.84 MW. The serious benchmark is SMARD: MAE 1,389.44 MW versus XGBoost 1,384.67 MW. XGBoost beat SMARD in 7 of 12 folds, but the fold dispersion is large enough that I would not claim a production edge from this evidence alone.

Bias is also worth watching. SMARD bias was -133.54 MW and XGBoost bias was -36.94 MW overall, but several folds show over-correction. The strongest relative fold was fold 6 (Dec 2025), where XGBoost improved MAE versus SMARD by 13.3%. The weakest relative fold was fold 4 (Oct 2025), where XGBoost lost by 26.4%. That October failure is the clearest stress case in the backtest and should be investigated around sharp ramps, storm regimes, curtailment/congestion, or weather-regime changes where lagged errors stop being stable.

## Top Features

| feature | importance_mean | importance_std |
| --- | --- | --- |
| weather_wind_gusts_10m_mean_ms | 0.0842 | 0.0140 |
| weather_wind_speed_100m_cubed_mean | 0.0519 | 0.0184 |
| schleswig_holstein_wind_gusts_10m_ms | 0.0296 | 0.0187 |
| dayofyear_cos | 0.0259 | 0.0066 |
| lower_saxony_wind_speed_100m_ms | 0.0256 | 0.0048 |
| lower_saxony_pressure_msl_hpa | 0.0244 | 0.0048 |
| schleswig_holstein_temperature_2m_c | 0.0227 | 0.0046 |
| hour_cos | 0.0214 | 0.0039 |
| month_cos | 0.0210 | 0.0052 |
| weather_wind_speed_100m_max_ms | 0.0207 | 0.0056 |

## How This Would Be Used

For trading or dispatch analysis, the useful signal is the rolling 24-hour-ahead calibrated deviation from the public wind forecast. A positive model residual says actual wind is expected above the published forecast, which is bearish for residual load and power prices all else equal. A negative residual says the opposite. The signal should be invalidated or down-weighted when fresh TSO/weather updates materially change the forecast, when observed wind errors diverge from lagged error patterns, or when grid constraints/curtailment dominate weather-driven production.

## Strategy Backtest

The strategy converts the wind residual into a paper price-surprise signal. If XGBoost forecasts wind at least 1.5 GW above the public SMARD forecast, the signal is short German day-ahead price; if it is at least 1.5 GW below, the signal is long. The benchmark applies the same rule to the public SMARD 24-hour forecast ramp alone. P&L is measured against a previous-day same-hour day-ahead price baseline with 0.5 EUR/MWh transaction cost.

Average P&L is reported in EUR/MWh per traded hour. Total P&L is the sum of those hourly price-surprise outcomes for 1 MW clips, so it should be read as EUR per MW of fixed clip size rather than as a standalone EUR/MWh price.

This is a research backtest, not an executable exchange P&L claim: the entry price is a transparent persistence proxy, not a historical traded forward quote. Economically, a production version would monetize a forecast-error edge through day-ahead to intraday or imbalance settlement, not through DA(t) minus DA(t-24). The goal here is narrower: test whether the wind residual contains directionally useful price information beyond a public forecast-ramp rule after costs.

| strategy | hours | trades | trade_rate | gross_hit_rate | avg_net_pnl_eur_mwh | total_net_pnl_eur_per_mw_clip | sharpe_like_per_trade | hit_rate_z_stat | net_pnl_t_stat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_wind_residual_signal | 8615 | 771 | 0.089 | 0.543 | 1.811 | 1,396.660 | 0.038 | 2.413 | 1.066 |
| public_smard_forecast_ramp_signal | 8615 | 7400 | 0.859 | 0.801 | 20.791 | 153,852.240 | 0.546 | 51.823 | 46.991 |

The residual strategy looks positive on the simple hourly statistics, but it still needs clustered and out-of-sample validation: the 1.5 GW rule trades 8.9% of hours, has a 54.3% gross hit rate, earns 1.811 EUR/MWh after costs, and has a Sharpe-like score of 0.038. Per trade, the public SMARD forecast-ramp benchmark earns 20.791 EUR/MWh versus 1.811 EUR/MWh for the residual strategy. Total P&L over 1 MW hourly clips is 153,852.24 EUR versus 1,396.66 EUR, so the residual correction does not beat the public-only rule here. Simple hourly statistics are gross-hit-rate z-stat 2.41 and per-trade net-P&L t-stat 1.07, before clustered adjustment. Fold-level P&L is uneven: 6 of 12 folds are positive, with the best fold in Nov 2025 and the worst in Jan 2026. The full-sample 1 MW-clip P&L is 1,396.66 EUR, but Nov 2025 alone contributes 2,810.89 EUR; excluding that fold, the strategy loses 1,414.23 EUR over 679 trades (-2.083 EUR/MWh per trade).

Threshold sensitivity is included to avoid pretending one hand-picked trigger tells the whole story.

| strategy | threshold_mw | trades | gross_hit_rate | avg_net_pnl_eur_mwh | total_net_pnl_eur_per_mw_clip | sharpe_like_per_trade | net_pnl_t_stat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_wind_residual_signal | 500.0 | 4429 | 0.527 | 1.115 | 4,937.320 | 0.027 | 1.827 |
| public_smard_forecast_ramp_signal | 500.0 | 8187 | 0.781 | 19.244 | 157,552.440 | 0.490 | 44.331 |
| xgboost_wind_residual_signal | 1000.0 | 1923 | 0.551 | 2.551 | 4,904.850 | 0.062 | 2.721 |
| public_smard_forecast_ramp_signal | 1000.0 | 7770 | 0.791 | 19.935 | 154,892.190 | 0.522 | 46.027 |
| xgboost_wind_residual_signal | 1500.0 | 771 | 0.543 | 1.811 | 1,396.660 | 0.038 | 1.066 |
| public_smard_forecast_ramp_signal | 1500.0 | 7400 | 0.801 | 20.791 | 153,852.240 | 0.546 | 46.991 |
| xgboost_wind_residual_signal | 2000.0 | 346 | 0.462 | -7.178 | -2,483.640 | -0.131 | -2.443 |
| public_smard_forecast_ramp_signal | 2000.0 | 7021 | 0.808 | 21.381 | 150,114.440 | 0.559 | 46.803 |
| xgboost_wind_residual_signal | 3000.0 | 109 | 0.239 | -34.439 | -3,753.830 | -0.599 | -6.259 |
| public_smard_forecast_ramp_signal | 3000.0 | 6301 | 0.826 | 23.044 | 145,202.600 | 0.593 | 47.094 |

The best residual threshold in this five-point grid is 500 MW, but that is a multiple-testing maximum with unadjusted net-P&L t-stat 1.83, so it supports further research rather than deployment.

## AI/LLM Integration

A programmatic OpenAI step reduces the manual work of translating deterministic validation outputs into a concise analyst review. It reads a compact evidence package from the QA, forecast, fold, strategy, and threshold CSVs, then uses the Responses API with a strict Pydantic output schema. Exact prompts, evidence, output, request metadata, and token usage are logged under `outputs/llm/`.

The LLM is deliberately downstream-only: it cannot change the data, XGBoost forecast, signal, trade log, P&L, or validation metrics. Each generated finding must cite a supplied evidence ID, unknown IDs are rejected, and the displayed metric values are attached by deterministic code after generation. A SHA-256 fingerprint ties the review to the exact evidence snapshot. The resulting review is shown in Streamlit only when that fingerprint matches the current CSVs; the dashboard never calls the API on refresh.

## Limitations

The validation now spans summer, autumn, winter, and spring folds, but it is still only one annual cycle. I would not treat the result as seasonally robust until it is repeated over multiple years and distinct weather regimes.

The current feature set is valid for a rolling 24-hour-ahead information set. It is not a true prompt/intraday model; once recent metered actuals are available, a previous-hour persistence benchmark should be tested and would likely be hard to beat. It is also not a strict day-ahead auction signal: the German day-ahead auction clears around D-1 noon, so 24-hour lagged actuals would be unavailable for some later delivery hours. The weather covariates are archived near-delivery forecast values rather than D-1 noon model-run snapshots, so they should be treated as post-auction for strict day-ahead trading. For a strict D-1 noon forecast, all lagged features and weather rows should be recomputed relative to the issue timestamp.

The 2026-07-01 to 2026-09-30 prospective holdout remains unscored. The research CLI rejects any overlapping date range before ingestion, and a future strict-model scoring route must verify a promoted-model and prediction manifest whose SHA-256 seal predates target scoring.

The XGBoost hyperparameters were kept fixed for the reported rerun, but they were chosen during prototyping on this same history. A production study should tune on a separate period or use nested time-series validation.

Overall, this should be read as a validation-first research prototype rather than a production signal. The pipeline now closes the loop from forecast to backtest to trading-strategy support, but the trading result is threshold-sensitive, fold-concentrated, and benchmarked only with a paper price-surprise proxy. A deployable version would need a longer multi-year backtest, forecast-run/lead-time pinned weather inputs, explicit issue-time feature cuts, regional generation constraints, historical executable DA-to-intraday or imbalance price marks, realistic transaction costs, and hyperparameter tuning isolated from the evaluation window.
