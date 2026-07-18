# German Wind Forecast Issue-Time Specification

Specification ID: `WG-D1-001`

Version: `0.1.0`

Status: Frozen for implementation

Owner: Jacob Mackey
Freeze date: 2026-07-16

## 1. Purpose

This document freezes the forecast question and information set before any XGBoost hyperparameter tuning. It prevents model-selection decisions from depending on data that would not have been available when a trading decision was made.

The existing pipeline remains a useful rolling-calibration research result, but it does not comply with this specification. Its May and June outputs are retained as historical reference runs rather than evidence for the D-1 model defined here.

No parameter search may begin until every mandatory issue-time QA check in section 9 passes.

## 2. Decision And Forecast Contract

The production-oriented research question is:

> At 11:00 Europe/Berlin on D-1, what will total German onshore plus offshore wind generation be for each local delivery hour on day D?

The contract is fixed as follows:

| Field | Frozen definition |
|---|---|
| Market | Germany, with DE-LU used only for downstream price analysis |
| Issue time | 11:00 Europe/Berlin on D-1 |
| Delivery window | Every local delivery hour on day D |
| Target | Realised German onshore plus offshore wind generation in MW |
| Forecast grain | Hourly |
| Forecast horizon | Approximately 13 to 37 hours from issue time; exact lead stored per row |
| Primary key | UTC delivery timestamp |
| Model family | XGBoost regression |
| Model target | Direct actual wind generation, not a residual to SMARD |
| Trading status | Forecast input only until executable settlement data are added |

The one-hour buffer before the coupled day-ahead auction is intentional. A later issue time requires a new specification version.

## 3. Time And Calendar Rules

- `issue_time_local` is always 11:00 in `Europe/Berlin` on D-1.
- `issue_time_utc` is derived with timezone-aware conversion and is never hard-coded.
- `delivery_timestamp_utc` is the unique model row key.
- `delivery_date_local`, local hour, UTC offset, and daylight-saving flag are stored as features or audit fields.
- Spring clock-change delivery days contain 23 hourly rows; autumn clock-change days contain 25. The repeated local hour remains unique in UTC.
- A fold boundary may split only between issue dates, never between hours belonging to the same D-1 forecast batch.

## 4. Information Set

Every feature row must include `available_at_utc`. The pipeline must assert:

```text
available_at_utc <= issue_time_utc
```

If a source does not expose a publication timestamp, the specification uses a conservative cutoff. A less conservative assumption requires documentary evidence and a version change.

### 4.1 Weather

The current stitched Open-Meteo Historical Forecast series is forbidden for this D-1 model. It combines the first hours of successive model runs and therefore does not preserve the forecast that existed at 11:00 D-1.

The frozen weather source is:

- Open-Meteo Single Runs API.
- Fixed model: ECMWF IFS HRES 9 km.
- Fixed initialisation: 00:00 UTC on D-1.
- Conservative availability timestamp: 06:30 UTC on D-1.
- Locations: North Sea, Schleswig-Holstein, Lower Saxony, and Brandenburg coordinates already documented by the project.
- Variables: 100 m wind speed and direction, 10 m gusts, 2 m temperature, and mean sea-level pressure.
- Derived wind direction features use sine and cosine components rather than raw circular degrees alone.

The availability timestamp is not inferred from model initialisation. ECMWF's dissemination schedule publishes the 0-90 hour fields from the 00 UTC run by approximately 06:12 UTC; `06:30 UTC` adds a fixed buffer and still precedes the 11:00 Europe/Berlin issue time in both winter and summer.

Each weather row stores the model name, run initialisation timestamp, conservative availability timestamp, delivery timestamp, and lead hours. The availability timestamp must precede issue time, and the run should produce lead times of roughly 22 to 46 hours across the delivery day.

`best_match`, reanalysis, stitched historical forecasts, and later model runs are prohibited.

### 4.2 Metered Wind And Lag Features

Because the current SMARD ingestion does not retain publication and revision timestamps, the conservative metered-data cutoff is the end of D-2 in local market time.

Allowed lagged generation features are:

- D-2 same-hour actual wind, equivalent to a 48-hour calendar lag in normal days.
- D-7 same-hour actual wind.
- Rolling statistics calculated only from observations at or before the D-2 cutoff.
- Calendar and seasonal features derived from the delivery timestamp.

The following current features are prohibited:

- `actual_lag_24h`.
- Any rolling feature whose source window crosses the issue-time cutoff.
- Any lag implemented only by row count without explicit DST-safe timestamp matching.

### 4.3 SMARD Forecasts

SMARD documents that the four German TSOs submit following-day forecasts by 18:00. The public SMARD forecast is therefore not assumed to be available at the 11:00 auction decision time.

For specification `WG-D1-001`:

- The current SMARD forecast level, ramp, and residual are excluded from model features.
- XGBoost predicts actual wind directly.
- SMARD remains a labelled post-hoc operational comparator.
- Performance against SMARD is not an equal-information comparison and is not the model-selection objective.
- Historical SMARD forecast-error features are excluded until their publication and revision history is explicitly audited.

## 5. Baselines

All selection baselines must use exactly the same information cutoff as XGBoost.

The required baselines are:

1. D-2 same-hour persistence.
2. Train-only hour/month climatology.
3. Regularised linear regression using the same issue-time-valid weather and calendar features as XGBoost.

The regularised linear model is the primary ex-ante baseline. SMARD is reported separately as a post-hoc operational benchmark.

## 6. Model And Feature Contract

The XGBoost model predicts hourly actual wind generation directly. Predictions are clipped only to physically defensible limits derived from training data and documented installed capacity.

The feature schema is ordered, versioned, and saved with the model. Features may include:

- Fixed-run weather values at the four representative locations.
- Weather aggregates and spatial gradients.
- Sine/cosine wind direction components.
- D-2 and D-7 actual generation lags.
- Cutoff-safe rolling generation summaries.
- Hour, weekday, month, daylight-saving flag, and cyclical calendar encodings.

No feature is accepted merely because it is present in the historical merged table. It must have a documented source timestamp and availability rule.

The existing XGBoost settings are an initial candidate only. Results from the rolling-calibration model do not justify carrying those parameters into this specification.

## 7. Validation Design

Validation is grouped expanding walk-forward evaluation by issue date.

- Training for each outer fold contains only delivery dates whose outcomes were available before the first validation issue time.
- Validation batches contain complete delivery days.
- Outer validation windows are 30 consecutive issue dates.
- No random split is permitted.
- Hyperparameter selection, feature selection, early stopping, and threshold selection occur only inside each outer training window.
- A separate tuning protocol must freeze the candidate configurations and selection rule before the first tuned run.

All data through 2026-06-30 are development history because their outcomes have already been inspected. They may be used inside nested walk-forward validation, but they are not described as a pristine final holdout.

The period from 2026-07-01 through 2026-09-30 is a locked prospective holdout. No aggregate forecast or trading performance for that period is to be calculated or inspected before 2026-10-01. Partial-period operational QA may check schema, timestamps, and missingness only, without target-based metrics.

The repository enforces this mechanically in two stages:

1. The existing research pipeline fails before ingestion whenever its requested dates overlap the holdout. It cannot become a holdout scorer merely because the calendar passes 2026-10-01.
2. A future dedicated scoring route must first verify a hash-sealed release manifest containing a model manifest promoted before 2026-07-01 and a target-free prediction CSV. The manifest and its SHA-256 sidecar must be frozen before scoring, and target access remains embargoed until 2026-10-01 Europe/Berlin.

No holdout release manifest exists at this specification stage because no `WG-D1-001` model has passed the issue-time gates or been promoted. Once one exists, the manifest and sidecar must be committed before target scoring. The repository timestamp then provides external audit evidence; a local timestamp or regenerable hash alone is not immutable proof of when the decision was frozen.

## 8. Metrics And Inference

Primary model-selection metric:

- Hourly MAE versus the regularised linear ex-ante baseline.

Required secondary metrics:

- RMSE and bias.
- Mean and median outer-fold skill.
- Number and proportion of positive-skill folds.
- Daily MAE distribution.
- Error during high-wind hours and large wind ramps.
- D-2 persistence and climatology comparisons.
- Post-hoc comparison with SMARD, clearly labelled as later-information evidence.

For each baseline loss differential, the pipeline must save:

- Mean hourly absolute-error difference in MW.
- Newey-West lag-24 standard error.
- Newey-West t-statistic.
- Normal-approximation 95% confidence interval in MW.
- Approximate skill interval in percent, explicitly treating baseline MAE as fixed.
- Delivery-day block-bootstrap confidence interval as a sensitivity check.

The shared diagnostics implementation now emits these fields for the historical residual-versus-SMARD reference run. That implementation work does not make the historical model compliant with the stricter `WG-D1-001` information set or equal-information baseline contract.

A reliable incremental-edge claim requires all of the following:

- Positive pooled skill versus the primary ex-ante baseline.
- Positive median outer-fold skill.
- Positive skill in at least two thirds of outer folds.
- A 95% loss-difference interval whose lower bound is above zero.
- No single fold contributing more than half of the aggregate improvement.
- Confirmation on the locked prospective holdout.

Failure to meet these gates is reported as no reliable edge, not followed by an adaptive search for a more favourable specification.

## 9. Mandatory Issue-Time QA Gates

The implementation must fail before model fitting if any gate fails:

1. Every feature has a source and `available_at_utc` rule.
2. Every feature timestamp is no later than its batch issue timestamp.
3. Every delivery day has 23, 24, or 25 unique UTC rows as appropriate.
4. Every weather row comes from the fixed 00:00 UTC D-1 ECMWF IFS HRES run.
5. Weather model name, run time, 06:30 UTC availability time, valid time, and lead hours are non-null.
6. No `best_match`, stitched historical forecast, reanalysis, or later weather run enters the model frame.
7. No current SMARD forecast level, ramp, or residual enters the feature frame.
8. No 24-hour actual or forecast-error lag enters the feature frame.
9. Rolling features end at or before the conservative D-2 cutoff.
10. Training rows precede validation issue dates without overlap.
11. Raw files are immutable, dated snapshots with SHA-256 hashes.
12. Package versions, code commit, specification version, and feature schema hash are recorded.

## 10. Trading Translation

Higher forecast wind is directionally bearish for residual load and power prices, all else equal; lower wind is directionally bullish. This relationship is not itself a trade.

An executable strategy test must specify:

- Decision time: 11:00 Europe/Berlin D-1.
- Entry: actual day-ahead auction clearing price or an auditable order assumption.
- Exit or settlement: a defined intraday VWAP, imbalance price, or other executable mark.
- Position size, fees, bid/ask cost, liquidity, and limits.
- The pre-auction expectation against which the wind surprise is measured.

Until those data exist, the current `DA(t) - DA(t-24)` result remains a structural relationship diagnostic. It must not be presented as executable P&L. The post-auction SMARD forecast ramp cannot be used to justify a pre-auction trade.

## 11. Model Artifact And Run Manifest

After the tuning protocol is frozen and nested validation is complete, the selected candidate is refit on the permitted development history and saved in XGBoost native JSON or UBJ format.

Each model version must include:

- Model artifact and SHA-256 hash.
- Ordered feature schema and transformations.
- XGBoost parameters and random seed.
- Training start and end issue dates.
- Raw-data snapshot hashes.
- Weather model and run rule.
- Code commit and dependency lockfile.
- Validation metrics and confidence intervals.
- Specification and tuning-protocol versions.
- Creation timestamp and promotion status.

New retraining runs create new immutable versions. They do not overwrite the approved model, and rollback remains possible.

## 12. AI Analyst Review Boundary

The OpenAI review remains downstream-only. It may summarise a run only when:

- The run manifest declares `WG-D1-001`.
- All mandatory issue-time QA gates passed.
- The evidence fingerprint matches the deterministic outputs.

The LLM cannot select features, tune XGBoost, choose the promoted model, calculate metrics, or change a trading decision.

## 13. Current Reference Runs

These runs predate this specification and are explicitly non-compliant with its information set:

| Snapshot | Validation folds | XGBoost skill vs SMARD | NW t-stat | Status |
|---|---:|---:|---:|---|
| Through 2026-05-31 | 11 | +0.165% | 0.092 | Historical rolling-calibration reference |
| Through 2026-06-30 | 12 | +0.343% | 0.207 | Historical rolling-calibration reference |

The earlier five-fold `+1.01%` result is superseded and must not be used as the current headline.

## 14. Change Control

Any change to issue time, weather model, weather run, target, delivery grain, feature availability, baseline, holdout, or edge-claim rule requires:

1. A new specification version.
2. A written reason recorded before results are calculated.
3. Re-execution of all issue-time QA checks.
4. Separate reporting from prior versions rather than silent replacement.

## References

- SMARD forecast data: https://www.smard.de/page/en/wiki-article/5884/206318/forecast-data
- EPEX SPOT market timing factsheet: https://www.epexspot.com/sites/default/files/download_center_files/Factsheet%20EU%20IEM-2211.pdf
- Open-Meteo Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api
- Open-Meteo Previous Runs API: https://open-meteo.com/en/docs/previous-runs-api
- Open-Meteo Single Runs API: https://open-meteo.com/en/docs/single-runs-api
- ECMWF dissemination schedule: https://www.ecmwf.int/en/forecasts/datasets/set-i
