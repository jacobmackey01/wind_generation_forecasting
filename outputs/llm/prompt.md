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
      "value": "50/50 passed; 13103 hourly rows; failures: none",
      "interpretation": "The review input passed the pipeline's deterministic data checks."
    },
    {
      "id": "forecast.validation_design",
      "label": "Forecast validation design",
      "value": "12 expanding walk-forward folds, 2025-07-06 to 2026-06-30; no random split",
      "interpretation": "The reported forecast result is out-of-sample in temporal order."
    },
    {
      "id": "forecast.mae_comparison",
      "label": "XGBoost versus public SMARD MAE",
      "value": "XGBoost 1,384.67 MW; SMARD 1,389.44 MW; skill +0.34%",
      "interpretation": "The pooled MAE difference is small relative to the public operational forecast."
    },
    {
      "id": "forecast.significance",
      "label": "Forecast loss-difference diagnostic",
      "value": "Newey-West t=0.21; mean fold skill -0.49%",
      "interpretation": "The forecast result lacks conventional statistical support and is not consistently positive by fold."
    },
    {
      "id": "forecast.fold_dispersion",
      "label": "Forecast fold dispersion",
      "value": "XGBoost wins 7/12; best +13.3%; worst -26.4%",
      "interpretation": "Average performance hides material regime variation."
    },
    {
      "id": "forecast.worst_regime",
      "label": "Worst forecast fold",
      "value": "Oct 2025; skill -26.4%",
      "interpretation": "This is the clearest observed forecast stress regime."
    },
    {
      "id": "forecast.bias",
      "label": "Forecast bias",
      "value": "XGBoost -36.9 MW; SMARD -133.5 MW",
      "interpretation": "Bias should be monitored separately from absolute error."
    },
    {
      "id": "strategy.residual_result",
      "label": "Residual strategy result",
      "value": "771 trades; +1.811 EUR/MWh per trade; t=1.07",
      "interpretation": "The model-derived residual signal does not have conventional statistical support."
    },
    {
      "id": "strategy.fold_concentration",
      "label": "Residual strategy fold concentration",
      "value": "6/12 positive folds; best Nov 2025 +2,810.89 EUR; excluding it -1,414.23 EUR",
      "interpretation": "A large share of the aggregate result depends on one validation month."
    },
    {
      "id": "strategy.public_benchmark",
      "label": "Public SMARD ramp benchmark",
      "value": "7400 trades; +20.791 EUR/MWh per trade; 11.5x the residual signal",
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
      "value": "best of 5 tested points: 1,000 MW, +2.551 EUR/MWh per trade, unadjusted t=2.72",
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
