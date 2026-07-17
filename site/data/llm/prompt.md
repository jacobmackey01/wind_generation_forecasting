# System Prompt

You are a European power-market research reviewer.
Draft a concise, evidence-bounded review of a wind forecast and its paper trading-signal backtest.

Rules:
- Use only the supplied evidence. Do not invent, recalculate, or extrapolate facts.
- Do not quote metric values in prose. Cite evidence IDs; the deterministic renderer attaches exact values.
- Treat an absolute t-statistic below 1.96 as lacking conventional statistical support, while noting that a t-statistic is a diagnostic rather than proof.
- Distinguish the public SMARD baseline from any incremental XGBoost signal.
- Never describe the DA(t) minus DA(t-24) proxy as executable P&L or trading advice.
- State a null or weak result plainly when the evidence supports it.
- Every forecast assessment, strategy assessment, and risk flag must cite one or more exact evidence IDs.
- Keep the writing suitable for a trader or hiring reviewer and use ASCII characters only.

The LLM is a downstream summarizer. It does not alter forecasts, trades, QA checks, or metrics.


# User Prompt

Write the structured analyst review from the evidence package below. Use evidence IDs exactly as supplied. Keep risk flags, invalidation conditions, and next steps concise.

{
  "evidence": [
    {
      "id": "qa.checks",
      "label": "Pipeline QA checks",
      "value": "50/50 passed; 12383 hourly rows; failures: none",
      "interpretation": "The review input passed the pipeline's deterministic data checks."
    },
    {
      "id": "forecast.validation_design",
      "label": "Forecast validation design",
      "value": "11 expanding walk-forward folds, 2025-07-06 to 2026-05-31; no random split",
      "interpretation": "The reported forecast result is out-of-sample in temporal order."
    },
    {
      "id": "forecast.mae_comparison",
      "label": "XGBoost versus public SMARD MAE",
      "value": "XGBoost 1,396.43 MW; SMARD 1,398.74 MW; skill +0.17%",
      "interpretation": "The pooled MAE difference is small relative to the public operational forecast."
    },
    {
      "id": "forecast.significance",
      "label": "Forecast loss-difference diagnostic",
      "value": "Newey-West t=0.09; mean fold skill -0.64%",
      "interpretation": "The forecast result lacks conventional statistical support and is not consistently positive by fold."
    },
    {
      "id": "forecast.fold_dispersion",
      "label": "Forecast fold dispersion",
      "value": "XGBoost wins 7/11; best +13.4%; worst -22.9%",
      "interpretation": "Average performance hides material regime variation."
    },
    {
      "id": "forecast.worst_regime",
      "label": "Worst forecast fold",
      "value": "Oct 2025; skill -22.9%",
      "interpretation": "This is the clearest observed forecast stress regime."
    },
    {
      "id": "forecast.bias",
      "label": "Forecast bias",
      "value": "XGBoost -0.2 MW; SMARD -108.8 MW",
      "interpretation": "Bias should be monitored separately from absolute error."
    },
    {
      "id": "strategy.residual_result",
      "label": "Residual strategy result",
      "value": "787 trades; +0.240 EUR/MWh per trade; t=0.14",
      "interpretation": "The model-derived residual signal does not have conventional statistical support."
    },
    {
      "id": "strategy.fold_concentration",
      "label": "Residual strategy fold concentration",
      "value": "4/11 positive folds; best Nov 2025 +2,554.32 EUR; excluding it -2,365.60 EUR",
      "interpretation": "A large share of the aggregate result depends on one validation month."
    },
    {
      "id": "strategy.public_benchmark",
      "label": "Public SMARD ramp benchmark",
      "value": "6822 trades; +20.110 EUR/MWh per trade; 83.9x the residual signal",
      "interpretation": "The proxy strongly rewards public forecast repricing, so aggregate proxy P&L is not proof of private edge."
    },
    {
      "id": "strategy.price_proxy",
      "label": "Price-settlement limitation",
      "value": "DA(t) minus DA(t-24), not an executable day-ahead-to-intraday or imbalance spread",
      "interpretation": "Neither strategy's proxy P&L should be presented as an executable trading return."
    },
    {
      "id": "strategy.reported_threshold",
      "label": "Reported residual trigger",
      "value": "absolute signal >= 1,500 MW",
      "interpretation": "Hours below the trigger are no-trade observations."
    },
    {
      "id": "strategy.threshold_robustness",
      "label": "Residual threshold sensitivity",
      "value": "best of 5 tested points: 1,000 MW, +1.659 EUR/MWh per trade, unadjusted t=1.76",
      "interpretation": "The best grid point is an in-sample multiple-choice maximum, not holdout evidence."
    },
    {
      "id": "design.information_set",
      "label": "Auction information-set limitation",
      "value": "24-hour lagged actual/error features and archived weather are not pinned to strict D-1 noon availability",
      "interpretation": "The current signal is rolling 24-hour-ahead research, not a strict day-ahead auction implementation."
    },
    {
      "id": "design.sample_scope",
      "label": "Seasonal scope",
      "value": "summer, autumn, winter, and spring folds within one annual cycle; no multi-year validation",
      "interpretation": "One pass through the seasons is insufficient evidence of regime robustness."
    },
    {
      "id": "design.llm_boundary",
      "label": "LLM role",
      "value": "downstream summarization only; no influence on QA, features, XGBoost, signals, trades, or metrics",
      "interpretation": "The AI component reduces manual reporting work without contaminating the quantitative pipeline."
    }
  ]
}
