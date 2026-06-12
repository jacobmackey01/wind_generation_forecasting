# Germany Wind Generation Forecasting Case Study

Jacob Mackey

## Objective

Build a clean, trading-relevant forecasting pipeline for German hourly wind generation. The model predicts actual onshore plus offshore wind generation and is evaluated against simple and market-relevant baselines using walk-forward validation.

The intended use case is rolling 24-hour-ahead calibration of the public wind forecast. The information set is published forecasts plus observations and forecast errors that are at least 24 hours old relative to delivery. At true prompt or intraday horizons, recent metered actuals become available and a last-observation style benchmark would dominate this feature set.

## Data

Sample: 2024-12-31 23:00:00+00:00 to 2026-05-31 21:00:00+00:00 UTC, 12383 hourly rows.

- SMARD / Bundesnetzagentur: actual wind generation and published wind generation forecasts for Germany.
- Open-Meteo Historical Forecast API: archived forecast-model weather covariates at four representative German wind locations.

QA passed: True. Checks cover timestamp uniqueness/continuity, local power-day hour counts, missing values, and plausible generation/weather ranges.

## Features

The feature set includes SMARD wind forecast levels and ramps, previous-day and previous-week actual wind lags, rolling means, lagged forecast errors, calendar seasonality, and weather forecast variables such as 100m wind speed/direction, gusts, temperature, and pressure.

## Models And Validation

Baselines are previous-week same-hour persistence, train-fold hour/month climatology, and the SMARD published wind forecast. The improved model is an XGBoost residual calibrator: it predicts actual minus SMARD forecast, then adds the correction back to the SMARD baseline. Validation uses expanding walk-forward folds from 2025-07-06 23:00:00+00:00 to 2026-05-31 22:00:00+00:00, not a random split.

| model | mae | rmse | bias | skill_vs_smard_mae_pct |
| --- | --- | --- | --- | --- |
| persistence_prev_week | 11,430.50 | 14,695.62 | 60.08 | -717.20 |
| hour_month_climatology | 8,707.26 | 11,014.03 | -1,097.14 | -522.51 |
| smard_forecast | 1,398.74 | 2,029.81 | -108.85 | 0.00 |
| xgboost_residual | 1,388.77 | 1,971.50 | 2.68 | 0.71 |

## Interpretation

XGBoost does not show a reliable edge over the SMARD forecast in this backtest: pooled MAE is only 0.71% better, while average fold skill is -0.12%.

A simple Newey-West lag-24 t-stat on the hourly absolute-error loss differential is 0.40. This is included only as a sanity check, but it supports the same conclusion: the observed pooled improvement is noise, not a statistically robust edge.

The previous-week persistence MAE was 11,430.50 MW and the hour/month climatology MAE was 8,707.26 MW. The serious benchmark is SMARD: MAE 1,398.74 MW versus XGBoost 1,388.77 MW. XGBoost beat SMARD in 7 of 11 folds, but the fold dispersion is large enough that I would not claim a production edge from this evidence alone.

Bias is also worth watching. SMARD bias was -108.85 MW and XGBoost bias was 2.68 MW overall, but several folds show over-correction. The strongest relative fold was fold 6 (Dec 2025), where XGBoost improved MAE versus SMARD by 13.0%. The weakest relative fold was fold 4 (Oct 2025), where XGBoost lost by 25.9%. That October failure is the clearest stress case in the backtest and should be investigated around sharp ramps, storm regimes, curtailment/congestion, or weather-regime changes where lagged errors stop being stable.

## Top Features

| feature | importance_mean | importance_std |
| --- | --- | --- |
| weather_wind_gusts_10m_mean_ms | 0.0842 | 0.0122 |
| weather_wind_speed_100m_cubed_mean | 0.0521 | 0.0146 |
| dayofyear_cos | 0.0267 | 0.0074 |
| lower_saxony_pressure_msl_hpa | 0.0253 | 0.0046 |
| schleswig_holstein_wind_gusts_10m_ms | 0.0252 | 0.0139 |
| lower_saxony_wind_speed_100m_ms | 0.0252 | 0.0048 |
| month_cos | 0.0246 | 0.0055 |
| schleswig_holstein_temperature_2m_c | 0.0233 | 0.0036 |
| north_sea_wind_gusts_10m_ms | 0.0227 | 0.0122 |
| hour_cos | 0.0207 | 0.0032 |

## How This Would Be Used

For trading or dispatch analysis, the useful signal is the rolling 24-hour-ahead calibrated deviation from the public wind forecast. A positive model residual says actual wind is expected above the published forecast, which is bearish for residual load and power prices all else equal. A negative residual says the opposite. The signal should be invalidated or down-weighted when fresh TSO/weather updates materially change the forecast, when observed wind errors diverge from lagged error patterns, or when grid constraints/curtailment dominate weather-driven production.

## Limitations

The validation now spans summer, autumn, winter, and spring folds, but it is still only one annual cycle. I would not treat the result as seasonally robust until it is repeated over multiple years and distinct weather regimes.

The current feature set is valid for a rolling 24-hour-ahead information set. It is not a true prompt/intraday model; once recent metered actuals are available, a previous-hour persistence benchmark should be tested and would likely be hard to beat. For a strict D-1 noon forecast, all lagged features should be recomputed relative to the issue timestamp.

The XGBoost hyperparameters were kept fixed for the reported rerun, but they were chosen during prototyping on this same history. A production study should tune on a separate period or use nested time-series validation.

Overall, this should be read as a validation-first prototype rather than a production signal. The pipeline demonstrates the right data QA, walk-forward discipline, and benchmark framing, but a deployable version would need a longer multi-year backtest, forecast-run/lead-time pinned weather inputs, explicit issue-time feature cuts, regional generation constraints, and hyperparameter tuning isolated from the evaluation window.
