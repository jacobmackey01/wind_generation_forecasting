# May Snapshot vs June-Extended Rerun

## Technical summary

Extending the pipeline from 31 May to 30 June 2026 added 720 validation hours, but it did not strengthen the forecasting claim. During the added June hours, XGBoost MAE was 1,300.29 MW versus 1,261.12 MW for the public SMARD forecast, equivalent to -3.11% skill. The full rerun still showed only +0.34% pooled skill with a Newey-West t-statistic of 0.21, so the forecast edge remains indistinguishable from noise.

The residual paper strategy looked better in the full rerun, rising from EUR 0.240 to EUR 1.811 net P&L per trade. June itself contributed only EUR 149.94 from 20 trades. Most of the full-sample change came from historical SMARD actual-generation revisions and refitting XGBoost, which changed 244 earlier position decisions. The result remains statistically weak (net P&L t-statistic 1.07), concentrated in November, and based on a non-executable DA(t) minus DA(t-24) proxy.

## Forecast comparison

| Measure | May snapshot | June-extended rerun | Change |
|---|---:|---:|---:|
| Validation hours | 7,895 | 8,615 | +720 |
| Walk-forward folds | 11 | 12 | +1 |
| SMARD MAE | 1,398.74 MW | 1,389.44 MW | -9.30 MW |
| XGBoost MAE | 1,396.43 MW | 1,384.67 MW | -11.75 MW |
| XGBoost skill vs SMARD | +0.17% | +0.34% | +0.18 pp |
| Newey-West loss-difference t-statistic | 0.09 | 0.21 | +0.11 |
| Folds won by XGBoost | 7 / 11 | 7 / 12 | No additional win |

The lower full-sample MAE does not mean June improved the model. June was an easier absolute-error month for both forecasts, and XGBoost underperformed SMARD during the added 720 hours:

| Added June 2026 hours | SMARD | XGBoost |
|---|---:|---:|
| MAE | 1,261.12 MW | 1,300.29 MW |
| RMSE | 1,822.68 MW | 1,948.81 MW |
| Bias | -246.52 MW | -432.91 MW |
| Relative XGBoost skill |  | -3.11% |

## Residual paper-trade comparison

The rule takes a long power position when the XGBoost wind estimate is at least 1,500 MW below the public forecast and a short position when it is at least 1,500 MW above it. A 0.5 EUR/MWh transaction-cost assumption is applied.

| Measure | May snapshot | June-extended rerun | Change |
|---|---:|---:|---:|
| Suggested trades | 787 | 771 | -16 |
| Long positions | 327 | 329 | +2 |
| Short positions | 460 | 442 | -18 |
| Trade rate | 9.97% | 8.95% | -1.02 pp |
| Gross hit rate | 52.86% | 54.35% | +1.49 pp |
| Net P&L per trade | +0.240 EUR/MWh | +1.811 EUR/MWh | +1.572 EUR/MWh |
| Total proxy P&L per 1 MW clips | +188.72 EUR | +1,396.66 EUR | +1,207.94 EUR |
| Net P&L t-statistic | 0.14 | 1.07 | Still statistically weak |
| Positive folds | 4 / 11 | 6 / 12 | +2 folds |
| P&L excluding November | -2,365.60 EUR | -1,414.23 EUR | Still negative |

June alone generated 20 residual paper trades: 17 long and 3 short. Sixteen were gross winners, producing an 80.0% gross hit rate, +7.497 EUR/MWh net per trade, and +149.94 EUR total proxy P&L. The June-only P&L t-statistic was 0.69, so the 20-trade result is too small and noisy to establish an edge.

## Why the historical result changed

This was not a pure one-month append. Extending the requested range forced the cached public series to be downloaded again. Across the common historical data:

- 1,464 actual-wind rows changed, with a mean absolute revision of 26.00 MW over the common validation hours and a maximum revision of 1,525.71 MW.
- The public SMARD forecast and day-ahead price series did not change.
- Refitting the residual model changed its prediction on 7,890 of 7,895 common validation hours.
- The residual strategy changed position on 244 common hours: 140 previous trades became flat and 104 previous flat hours became trades. No long position directly reversed to short or vice versa.
- On the same 7,895 timestamps, residual proxy P&L increased from +188.72 EUR to +1,246.72 EUR. The added June hours contributed only +149.94 EUR of the new +1,396.66 EUR total.

The public-ramp benchmark remained unchanged on the common hours. Across the extended sample it produced +20.791 EUR/MWh per trade, versus +1.811 for the residual rule. This continues to diagnose the structure of the DA(t) minus DA(t-24) proxy rather than an executable free edge.

## Interpretation

The conclusion remains unchanged: XGBoost does not reliably beat the public forecast, and the residual strategy does not demonstrate statistically robust or executable P&L. June is useful because it adds another negative forecast fold and shows that a better-looking aggregate strategy result can be driven mostly by data revisions and refitting rather than new out-of-sample evidence.

For reproducible comparisons, future runs should preserve immutable dated raw-data snapshots and report both a fixed-overlap comparison and an appended-period comparison. A production trading test would also need issue-time-valid features and auction-to-intraday or imbalance settlement prices.

## Evidence

- May snapshot: packaged project dated 7 July 2026; it matches the compact CSV files deployed to Netlify.
- June rerun: local pipeline outputs generated on 15 July 2026 with `--start 2025-01-01 --end 2026-06-30`.
- Forecast metrics are calculated at hourly validation grain. Trading metrics use 1 MW hourly paper clips and the research proxy documented in the project.
