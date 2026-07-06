# Germany Wind Generation Forecasting Case Study

Jacob Mackey

## Objective

Build a clean, trading-relevant forecasting pipeline for German hourly wind generation. The model predicts actual onshore plus offshore wind generation and is evaluated against simple and market-relevant baselines using walk-forward validation.

The intended use case is rolling 24-hour-ahead calibration of the public wind forecast. The information set is published forecasts plus observations and forecast errors that are at least 24 hours old relative to delivery. At true prompt or intraday horizons, recent metered actuals become available and a last-observation style benchmark would dominate this feature set.

## Data

Sample: 2024-12-31 23:00:00+00:00 to 2026-05-31 21:00:00+00:00 UTC, 12383 hourly rows.

- SMARD / Bundesnetzagentur: actual wind generation and published wind generation forecasts for Germany.
- SMARD / Bundesnetzagentur: German/Luxembourg day-ahead auction prices, series 4169, for the strategy research layer.
- Open-Meteo Historical Forecast API: archived forecast-model weather covariates at four representative German wind locations.

QA passed: True. Checks cover timestamp uniqueness/continuity, local power-day hour counts, missing values, and plausible generation, weather, and day-ahead price ranges.

## Features

The feature set includes SMARD wind forecast levels and ramps, previous-day and previous-week actual wind lags, rolling means, lagged forecast errors, calendar seasonality, and weather forecast variables such as 100m wind speed/direction, gusts, temperature, and pressure.

## Models And Validation

Baselines are previous-week same-hour persistence, train-fold hour/month climatology, and the SMARD published wind forecast. The improved model is an XGBoost residual calibrator: it predicts actual minus SMARD forecast, then adds the correction back to the SMARD baseline. Validation uses expanding walk-forward folds from 2025-07-06 23:00:00+00:00 to 2026-05-31 22:00:00+00:00, not a random split.

| model | mae | rmse | bias | skill_vs_smard_mae_pct |
| --- | --- | --- | --- | --- |
| persistence_prev_week | 11,430.50 | 14,695.62 | 60.08 | -717.20 |
| hour_month_climatology | 8,707.26 | 11,014.03 | -1,097.14 | -522.51 |
| smard_forecast | 1,398.74 | 2,029.81 | -108.85 | 0.00 |
| xgboost_residual | 1,396.43 | 1,988.55 | -0.18 | 0.17 |

## Interpretation

XGBoost does not show a reliable edge over the SMARD forecast in this backtest: pooled MAE is only 0.17% better, while average fold skill is -0.64%.

A simple Newey-West lag-24 t-stat on the hourly absolute-error loss differential is 0.09. This is included only as a sanity check, but it supports the same conclusion: the observed pooled improvement is noise, not a statistically robust edge.

The previous-week persistence MAE was 11,430.50 MW and the hour/month climatology MAE was 8,707.26 MW. The serious benchmark is SMARD: MAE 1,398.74 MW versus XGBoost 1,396.43 MW. XGBoost beat SMARD in 7 of 11 folds, but the fold dispersion is large enough that I would not claim a production edge from this evidence alone.

Bias is also worth watching. SMARD bias was -108.85 MW and XGBoost bias was -0.18 MW overall, but several folds show over-correction. The strongest relative fold was fold 6 (Dec 2025), where XGBoost improved MAE versus SMARD by 13.4%. The weakest relative fold was fold 4 (Oct 2025), where XGBoost lost by 22.9%. That October failure is the clearest stress case in the backtest and should be investigated around sharp ramps, storm regimes, curtailment/congestion, or weather-regime changes where lagged errors stop being stable.

## Top Features

| feature | importance_mean | importance_std |
| --- | --- | --- |
| weather_wind_gusts_10m_mean_ms | 0.0814 | 0.0144 |
| weather_wind_speed_100m_cubed_mean | 0.0566 | 0.0166 |
| dayofyear_cos | 0.0267 | 0.0065 |
| schleswig_holstein_wind_gusts_10m_ms | 0.0263 | 0.0140 |
| lower_saxony_wind_speed_100m_ms | 0.0254 | 0.0050 |
| lower_saxony_pressure_msl_hpa | 0.0254 | 0.0052 |
| schleswig_holstein_temperature_2m_c | 0.0223 | 0.0057 |
| hour_cos | 0.0213 | 0.0040 |
| month_cos | 0.0209 | 0.0057 |
| north_sea_wind_gusts_10m_ms | 0.0209 | 0.0100 |

## How This Would Be Used

For trading or dispatch analysis, the useful signal is the rolling 24-hour-ahead calibrated deviation from the public wind forecast. A positive model residual says actual wind is expected above the published forecast, which is bearish for residual load and power prices all else equal. A negative residual says the opposite. The signal should be invalidated or down-weighted when fresh TSO/weather updates materially change the forecast, when observed wind errors diverge from lagged error patterns, or when grid constraints/curtailment dominate weather-driven production.

## Strategy Backtest

The strategy converts the wind residual into a paper price-surprise signal. If XGBoost forecasts wind at least 1.5 GW above the public SMARD forecast, the signal is short German day-ahead price; if it is at least 1.5 GW below, the signal is long. The benchmark applies the same rule to the public SMARD 24-hour forecast ramp alone. P&L is measured against a previous-day same-hour day-ahead price baseline with 0.5 EUR/MWh transaction cost.

This is a research backtest, not an executable exchange P&L claim: the entry price is a transparent persistence proxy, not a historical traded forward quote. Economically, a production version would monetize a forecast-error edge through day-ahead to intraday or imbalance settlement, not through DA(t) minus DA(t-24). The goal here is narrower: test whether the wind residual contains directionally useful price information beyond a public forecast-ramp rule after costs.

| strategy | hours | trades | trade_rate | gross_hit_rate | avg_net_pnl_eur_mwh | total_net_pnl_eur_mwh | sharpe_like_per_trade | hit_rate_z_stat | net_pnl_t_stat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_wind_residual_signal | 7895 | 787 | 0.100 | 0.529 | 0.240 | 188.720 | 0.005 | 1.604 | 0.143 |
| public_smard_forecast_ramp_signal | 7895 | 6822 | 0.864 | 0.800 | 20.110 | 137,188.390 | 0.553 | 49.615 | 45.670 |

Read honestly, the residual strategy is a weak research signal rather than a tradable edge: the 1.5 GW rule trades 10.0% of hours, has a 52.9% gross hit rate, earns 0.240 EUR/MWh after costs, and has a Sharpe-like score of only 0.005. The public SMARD forecast-ramp benchmark earns 137,188.39 EUR/MWh versus 188.72 for the residual strategy, so the residual correction does not beat the public-only rule here. Simple strategy-side significance checks are also weak: unclustered gross-hit-rate z-stat 1.60 and per-trade net-P&L t-stat 0.14. Fold-level clustering would make this evidence weaker, not stronger. Fold-level P&L is uneven: 4 of 11 folds are positive, with the best fold in Nov 2025 and the worst in Apr 2026. The full-sample P&L is 188.72 EUR/MWh, but Nov 2025 alone contributes 2,554.32; excluding that fold, the strategy loses 2,365.60 over 706 trades (-3.351 EUR/MWh per trade).

Threshold sensitivity is included to avoid pretending one hand-picked trigger tells the whole story.

| strategy | threshold_mw | trades | gross_hit_rate | avg_net_pnl_eur_mwh | total_net_pnl_eur_mwh | sharpe_like_per_trade | net_pnl_t_stat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost_wind_residual_signal | 500.0 | 4076 | 0.529 | 0.738 | 3,007.230 | 0.019 | 1.189 |
| public_smard_forecast_ramp_signal | 500.0 | 7524 | 0.782 | 18.730 | 140,927.000 | 0.499 | 43.271 |
| xgboost_wind_residual_signal | 1000.0 | 1855 | 0.544 | 1.659 | 3,077.460 | 0.041 | 1.762 |
| public_smard_forecast_ramp_signal | 1000.0 | 7149 | 0.790 | 19.303 | 138,000.520 | 0.528 | 44.641 |
| xgboost_wind_residual_signal | 1500.0 | 787 | 0.529 | 0.240 | 188.720 | 0.005 | 0.143 |
| public_smard_forecast_ramp_signal | 1500.0 | 6822 | 0.800 | 20.110 | 137,188.390 | 0.553 | 45.670 |
| xgboost_wind_residual_signal | 2000.0 | 346 | 0.462 | -6.064 | -2,098.270 | -0.113 | -2.098 |
| public_smard_forecast_ramp_signal | 2000.0 | 6494 | 0.807 | 20.645 | 134,069.650 | 0.565 | 45.521 |
| xgboost_wind_residual_signal | 3000.0 | 90 | 0.189 | -39.339 | -3,540.530 | -0.801 | -7.596 |
| public_smard_forecast_ramp_signal | 3000.0 | 5849 | 0.824 | 22.250 | 130,137.390 | 0.600 | 45.893 |

The best residual threshold in this five-point grid is 1,000 MW, but that is a multiple-testing maximum with unadjusted net-P&L t-stat 1.76, so it supports further research rather than deployment.

## Limitations

The validation now spans summer, autumn, winter, and spring folds, but it is still only one annual cycle. I would not treat the result as seasonally robust until it is repeated over multiple years and distinct weather regimes.

The current feature set is valid for a rolling 24-hour-ahead information set. It is not a true prompt/intraday model; once recent metered actuals are available, a previous-hour persistence benchmark should be tested and would likely be hard to beat. It is also not a strict day-ahead auction signal: the German day-ahead auction clears around D-1 noon, so 24-hour lagged actuals would be unavailable for some later delivery hours. The weather covariates are archived near-delivery forecast values rather than D-1 noon model-run snapshots, so they should be treated as post-auction for strict day-ahead trading. For a strict D-1 noon forecast, all lagged features and weather rows should be recomputed relative to the issue timestamp.

The XGBoost hyperparameters were kept fixed for the reported rerun, but they were chosen during prototyping on this same history. A production study should tune on a separate period or use nested time-series validation.

Overall, this should be read as a validation-first research prototype rather than a production signal. The pipeline now closes the loop from forecast to backtest to trading-strategy support, but the trading result is threshold-sensitive, fold-concentrated, and benchmarked only with a paper price-surprise proxy. A deployable version would need a longer multi-year backtest, forecast-run/lead-time pinned weather inputs, explicit issue-time feature cuts, regional generation constraints, historical executable DA-to-intraday or imbalance price marks, realistic transaction costs, and hyperparameter tuning isolated from the evaluation window.
