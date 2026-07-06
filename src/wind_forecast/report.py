"""Reporting and figure generation."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_forecast.features import TARGET
from wind_forecast.qa import markdown_table
from wind_forecast.strategy import PUBLIC_RAMP_STRATEGY, RESIDUAL_STRATEGY


def _fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def _newey_west_t_stat(loss_diff: pd.Series, max_lag: int = 24) -> float:
    """Return a simple Newey-West t-stat for mean loss differential."""

    values = loss_diff.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 3:
        return float("nan")

    mean = float(values.mean())
    centered = values - mean
    long_run_var = float(np.mean(centered**2))
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        long_run_var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance

    if long_run_var <= 0:
        return float("nan")
    return mean / math.sqrt(long_run_var / n)


def _fold_month(row: pd.Series) -> str:
    return pd.to_datetime(row["fold_start"]).strftime("%b %Y")


def write_figures(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    latest_fold = int(predictions["fold"].max())
    latest = predictions[predictions["fold"] == latest_fold].tail(14 * 24)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(latest["timestamp_utc"], latest[TARGET], label="Actual", linewidth=2.0)
    ax.plot(latest["timestamp_utc"], latest["smard_forecast"], label="SMARD forecast", linewidth=1.5)
    ax.plot(latest["timestamp_utc"], latest["xgboost_residual"], label="XGBoost", linewidth=1.5)
    ax.set_title("Latest Walk-Forward Fold: Actual vs Forecast")
    ax.set_ylabel("MW")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / "actual_vs_predicted_latest_fold.png", dpi=160)
    plt.close(fig)

    mae = fold_metrics[fold_metrics["model"].isin(["smard_forecast", "xgboost_residual"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, frame in mae.groupby("model"):
        label = "SMARD forecast" if model_name == "smard_forecast" else "XGBoost"
        ax.plot(frame["fold"], frame["mae"], marker="o", label=label)
    ax.set_title("Walk-Forward MAE By Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "walk_forward_mae.png", dpi=160)
    plt.close(fig)

    top = feature_importance.head(20).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_mean"], color="#4c78a8")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Mean importance across folds")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def write_case_study(
    dataset: pd.DataFrame,
    qa_summary: dict[str, object],
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    path: Path,
    strategy_metrics: pd.DataFrame | None = None,
    strategy_fold_metrics: pd.DataFrame | None = None,
    strategy_thresholds: pd.DataFrame | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    smard_row = metrics.loc[metrics["model"] == "smard_forecast"].iloc[0]
    xgb_row = metrics.loc[metrics["model"] == "xgboost_residual"].iloc[0]
    persistence_row = metrics.loc[metrics["model"] == "persistence_prev_week"].iloc[0]
    climatology_row = metrics.loc[metrics["model"] == "hour_month_climatology"].iloc[0]
    skill = float(xgb_row["skill_vs_smard_mae_pct"])
    xgb_fold_metrics = fold_metrics[fold_metrics["model"] == "xgboost_residual"].copy()
    fold_skill_mean = float(xgb_fold_metrics["skill_vs_smard_mae_pct"].mean())
    fold_wins = int((xgb_fold_metrics["skill_vs_smard_mae_pct"] > 0).sum())
    fold_count = int(len(xgb_fold_metrics))

    if abs(skill) < 2:
        skill_sentence = (
            f"XGBoost does not show a reliable edge over the SMARD forecast in this backtest: pooled MAE is only "
            f"{_fmt(skill, 2)}% better, while average fold skill is {_fmt(fold_skill_mean, 2)}%."
        )
    elif skill > 0:
        skill_sentence = (
            f"XGBoost improved pooled MAE versus SMARD by {_fmt(skill, 1)}%, but fold stability still matters more than the pooled headline."
        )
    else:
        skill_sentence = (
            f"XGBoost trailed the SMARD forecast by {_fmt(abs(skill), 1)}% MAE, which is useful negative evidence."
        )

    worst_skill_fold = xgb_fold_metrics.sort_values("skill_vs_smard_mae_pct", ascending=True).iloc[0]
    best_skill_fold = xgb_fold_metrics.sort_values("skill_vs_smard_mae_pct", ascending=False).iloc[0]
    first_fold_start = str(xgb_fold_metrics["fold_start"].iloc[0])
    last_fold_end = str(xgb_fold_metrics["fold_end"].iloc[-1])
    smard_abs_error = (predictions[TARGET] - predictions["smard_forecast"]).abs()
    xgb_abs_error = (predictions[TARGET] - predictions["xgboost_residual"]).abs()
    loss_diff_t = _newey_west_t_stat(smard_abs_error - xgb_abs_error, max_lag=24)

    metrics_for_doc = metrics.copy()
    for column in ["mae", "rmse", "bias", "skill_vs_smard_mae_pct"]:
        metrics_for_doc[column] = metrics_for_doc[column].map(lambda x: _fmt(x, 2))

    top_features = feature_importance.head(10).copy()
    top_features["importance_mean"] = top_features["importance_mean"].map(lambda x: _fmt(x, 4))
    top_features["importance_std"] = top_features["importance_std"].fillna(0).map(lambda x: _fmt(x, 4))

    strategy_lines: list[str] = []
    if strategy_metrics is not None and not strategy_metrics.empty:
        residual_rows = strategy_metrics[strategy_metrics["strategy"] == RESIDUAL_STRATEGY]
        public_rows = strategy_metrics[strategy_metrics["strategy"] == PUBLIC_RAMP_STRATEGY]
        strategy_row = residual_rows.iloc[0] if not residual_rows.empty else strategy_metrics.iloc[0]
        public_row = public_rows.iloc[0] if not public_rows.empty else None

        weak_strategy = (
            abs(float(strategy_row.get("hit_rate_z_stat", 0.0))) < 1.96
            and abs(float(strategy_row.get("net_pnl_t_stat", 0.0))) < 1.96
        )
        if weak_strategy:
            strategy_read = (
                "Read honestly, the residual strategy is a weak research signal rather than a tradable edge: "
                f"the 1.5 GW rule trades {_fmt(strategy_row['trade_rate'] * 100, 1)}% of hours, "
                f"has a {_fmt(strategy_row['hit_rate'] * 100, 1)}% gross hit rate, "
                f"earns {_fmt(strategy_row['avg_net_pnl_eur_mwh'], 3)} EUR/MWh after costs, "
                f"and has a Sharpe-like score of only {_fmt(strategy_row['sharpe_like_per_trade'], 3)}."
            )
        else:
            strategy_read = (
                "The residual strategy looks positive on the simple hourly statistics, but it still needs clustered and out-of-sample validation: "
                f"the 1.5 GW rule trades {_fmt(strategy_row['trade_rate'] * 100, 1)}% of hours, "
                f"has a {_fmt(strategy_row['hit_rate'] * 100, 1)}% gross hit rate, "
                f"earns {_fmt(strategy_row['avg_net_pnl_eur_mwh'], 3)} EUR/MWh after costs, "
                f"and has a Sharpe-like score of {_fmt(strategy_row['sharpe_like_per_trade'], 3)}."
            )

        benchmark_read = ""
        if public_row is not None:
            residual_total = float(strategy_row["total_net_pnl_eur_mwh"])
            public_total = float(public_row["total_net_pnl_eur_mwh"])
            delta = residual_total - public_total
            if delta <= 0:
                benchmark_read = (
                    f" The public SMARD forecast-ramp benchmark earns {_fmt(public_total, 2)} EUR/MWh versus "
                    f"{_fmt(residual_total, 2)} for the residual strategy, so the residual correction does not beat the public-only rule here."
                )
            else:
                benchmark_read = (
                    f" The residual strategy earns {_fmt(delta, 2)} EUR/MWh more than the public SMARD forecast-ramp benchmark, "
                    "but the strategy statistics are too weak to call that an edge."
                )
        significance_read = ""
        if {"hit_rate_z_stat", "net_pnl_t_stat"}.issubset(strategy_row.index):
            if weak_strategy:
                significance_read = (
                    f" Simple strategy-side significance checks are also weak: unclustered gross-hit-rate z-stat "
                    f"{_fmt(strategy_row['hit_rate_z_stat'], 2)} and per-trade net-P&L t-stat "
                    f"{_fmt(strategy_row['net_pnl_t_stat'], 2)}. Fold-level clustering would make this evidence weaker, not stronger."
                )
            else:
                significance_read = (
                    f" Simple hourly statistics are gross-hit-rate z-stat {_fmt(strategy_row['hit_rate_z_stat'], 2)} "
                    f"and per-trade net-P&L t-stat {_fmt(strategy_row['net_pnl_t_stat'], 2)}, before clustered adjustment."
                )
        fold_read = ""
        if strategy_fold_metrics is not None and not strategy_fold_metrics.empty:
            strategy_fold_frame = strategy_fold_metrics[strategy_fold_metrics["strategy"] == strategy_row["strategy"]]
            positive_folds = int((strategy_fold_frame["total_net_pnl_eur_mwh"] > 0).sum())
            strategy_fold_count = int(len(strategy_fold_frame))
            best_strategy_fold = strategy_fold_frame.sort_values("total_net_pnl_eur_mwh", ascending=False).iloc[0]
            worst_strategy_fold = strategy_fold_frame.sort_values("total_net_pnl_eur_mwh", ascending=True).iloc[0]
            total_pnl = float(strategy_row["total_net_pnl_eur_mwh"])
            best_pnl = float(best_strategy_fold["total_net_pnl_eur_mwh"])
            total_ex_best = total_pnl - best_pnl
            trades_ex_best = int(strategy_row["trades"] - best_strategy_fold["trades"])
            avg_ex_best = total_ex_best / trades_ex_best if trades_ex_best else float("nan")
            fold_read = (
                f" Fold-level P&L is uneven: {positive_folds} of {strategy_fold_count} folds are positive, "
                f"with the best fold in {_fold_month(best_strategy_fold)} and the worst in {_fold_month(worst_strategy_fold)}. "
                f"The full-sample P&L is {_fmt(total_pnl, 2)} EUR/MWh, but {_fold_month(best_strategy_fold)} alone contributes "
                f"{_fmt(best_pnl, 2)}; excluding that fold, the strategy loses {_fmt(abs(total_ex_best), 2)} over "
                f"{trades_ex_best} trades ({_fmt(avg_ex_best, 3)} EUR/MWh per trade)."
            )

        strategy_doc = strategy_metrics.copy()
        for column in [
            "trade_rate",
            "hit_rate",
            "avg_gross_pnl_eur_mwh",
            "avg_net_pnl_eur_mwh",
            "total_net_pnl_eur_mwh",
            "sharpe_like_per_trade",
            "hit_rate_z_stat",
            "net_pnl_t_stat",
        ]:
            if column in strategy_doc.columns:
                strategy_doc[column] = strategy_doc[column].map(lambda x: _fmt(x, 3) if pd.notna(x) else "")
        strategy_doc = strategy_doc.rename(columns={"hit_rate": "gross_hit_rate"})

        threshold_doc = pd.DataFrame()
        if strategy_thresholds is not None and not strategy_thresholds.empty:
            residual_thresholds = strategy_thresholds[strategy_thresholds["strategy"] == strategy_row["strategy"]]
            best_threshold = residual_thresholds.sort_values("total_net_pnl_eur_mwh", ascending=False).iloc[0]
            best_threshold_t = float(best_threshold.get("net_pnl_t_stat", 0.0))
            if abs(best_threshold_t) < 1.96:
                threshold_read = (
                    f"The best residual threshold in this five-point grid is {_fmt(best_threshold['threshold_mw'], 0)} MW, "
                    f"but that is a multiple-testing maximum with unadjusted net-P&L t-stat {_fmt(best_threshold_t, 2)}, "
                    "so it supports further research rather than deployment."
                )
            else:
                threshold_read = (
                    f"The best residual threshold in this five-point grid is {_fmt(best_threshold['threshold_mw'], 0)} MW, "
                    f"with unadjusted net-P&L t-stat {_fmt(best_threshold_t, 2)}; it would still need holdout validation after threshold selection."
                )
            threshold_doc = strategy_thresholds[
                [
                    "strategy",
                    "threshold_mw",
                    "trades",
                    "hit_rate",
                    "avg_net_pnl_eur_mwh",
                    "total_net_pnl_eur_mwh",
                    "sharpe_like_per_trade",
                    "net_pnl_t_stat",
                ]
            ].copy()
            for column in ["hit_rate", "avg_net_pnl_eur_mwh", "total_net_pnl_eur_mwh", "sharpe_like_per_trade", "net_pnl_t_stat"]:
                threshold_doc[column] = threshold_doc[column].map(lambda x: _fmt(x, 3) if pd.notna(x) else "")
            threshold_doc = threshold_doc.rename(columns={"hit_rate": "gross_hit_rate"})

        strategy_lines = [
            "## Strategy Backtest",
            "",
            "The strategy converts the wind residual into a paper price-surprise signal. If XGBoost forecasts wind at least 1.5 GW above the public SMARD forecast, the signal is short German day-ahead price; if it is at least 1.5 GW below, the signal is long. The benchmark applies the same rule to the public SMARD 24-hour forecast ramp alone. P&L is measured against a previous-day same-hour day-ahead price baseline with 0.5 EUR/MWh transaction cost.",
            "",
            "This is a research backtest, not an executable exchange P&L claim: the entry price is a transparent persistence proxy, not a historical traded forward quote. Economically, a production version would monetize a forecast-error edge through day-ahead to intraday or imbalance settlement, not through DA(t) minus DA(t-24). The goal here is narrower: test whether the wind residual contains directionally useful price information beyond a public forecast-ramp rule after costs.",
            "",
            markdown_table(
                strategy_doc[
                    [
                        "strategy",
                        "hours",
                        "trades",
                        "trade_rate",
                        "gross_hit_rate",
                        "avg_net_pnl_eur_mwh",
                        "total_net_pnl_eur_mwh",
                        "sharpe_like_per_trade",
                        "hit_rate_z_stat",
                        "net_pnl_t_stat",
                    ]
                ]
            ),
            "",
            f"{strategy_read}{benchmark_read}{significance_read}{fold_read}",
            "",
        ]
        if not threshold_doc.empty:
            strategy_lines.extend(
                [
                    "Threshold sensitivity is included to avoid pretending one hand-picked trigger tells the whole story.",
                    "",
                    markdown_table(threshold_doc),
                    "",
                    threshold_read,
                    "",
                ]
            )

    lines = [
        "# Germany Wind Generation Forecasting Case Study",
        "",
        "Jacob Mackey",
        "",
        "## Objective",
        "",
        "Build a clean, trading-relevant forecasting pipeline for German hourly wind generation. The model predicts actual onshore plus offshore wind generation and is evaluated against simple and market-relevant baselines using walk-forward validation.",
        "",
        "The intended use case is rolling 24-hour-ahead calibration of the public wind forecast. The information set is published forecasts plus observations and forecast errors that are at least 24 hours old relative to delivery. At true prompt or intraday horizons, recent metered actuals become available and a last-observation style benchmark would dominate this feature set.",
        "",
        "## Data",
        "",
        f"Sample: {qa_summary['start']} to {qa_summary['end']} UTC, {qa_summary['rows']} hourly rows.",
        "",
        "- SMARD / Bundesnetzagentur: actual wind generation and published wind generation forecasts for Germany.",
        "- SMARD / Bundesnetzagentur: German/Luxembourg day-ahead auction prices, series 4169, for the strategy research layer.",
        "- Open-Meteo Historical Forecast API: archived forecast-model weather covariates at four representative German wind locations.",
        "",
        f"QA passed: {qa_summary['passed']}. Checks cover timestamp uniqueness/continuity, local power-day hour counts, missing values, and plausible generation, weather, and day-ahead price ranges.",
        "",
        "## Features",
        "",
        "The feature set includes SMARD wind forecast levels and ramps, previous-day and previous-week actual wind lags, rolling means, lagged forecast errors, calendar seasonality, and weather forecast variables such as 100m wind speed/direction, gusts, temperature, and pressure.",
        "",
        "## Models And Validation",
        "",
        f"Baselines are previous-week same-hour persistence, train-fold hour/month climatology, and the SMARD published wind forecast. The improved model is an XGBoost residual calibrator: it predicts actual minus SMARD forecast, then adds the correction back to the SMARD baseline. Validation uses expanding walk-forward folds from {first_fold_start} to {last_fold_end}, not a random split.",
        "",
        markdown_table(metrics_for_doc),
        "",
        "## Interpretation",
        "",
        skill_sentence,
        "",
        f"A simple Newey-West lag-24 t-stat on the hourly absolute-error loss differential is {_fmt(loss_diff_t, 2)}. This is included only as a sanity check, but it supports the same conclusion: the observed pooled improvement is noise, not a statistically robust edge.",
        "",
        f"The previous-week persistence MAE was {_fmt(persistence_row['mae'])} MW and the hour/month climatology MAE was {_fmt(climatology_row['mae'])} MW. The serious benchmark is SMARD: MAE {_fmt(smard_row['mae'])} MW versus XGBoost {_fmt(xgb_row['mae'])} MW. XGBoost beat SMARD in {fold_wins} of {fold_count} folds, but the fold dispersion is large enough that I would not claim a production edge from this evidence alone.",
        "",
        f"Bias is also worth watching. SMARD bias was {_fmt(smard_row['bias'])} MW and XGBoost bias was {_fmt(xgb_row['bias'])} MW overall, but several folds show over-correction. The strongest relative fold was fold {int(best_skill_fold['fold'])} ({_fold_month(best_skill_fold)}), where XGBoost improved MAE versus SMARD by {_fmt(best_skill_fold['skill_vs_smard_mae_pct'], 1)}%. The weakest relative fold was fold {int(worst_skill_fold['fold'])} ({_fold_month(worst_skill_fold)}), where XGBoost lost by {_fmt(abs(worst_skill_fold['skill_vs_smard_mae_pct']), 1)}%. That October failure is the clearest stress case in the backtest and should be investigated around sharp ramps, storm regimes, curtailment/congestion, or weather-regime changes where lagged errors stop being stable.",
        "",
        "## Top Features",
        "",
        markdown_table(top_features[["feature", "importance_mean", "importance_std"]]),
        "",
        "## How This Would Be Used",
        "",
        "For trading or dispatch analysis, the useful signal is the rolling 24-hour-ahead calibrated deviation from the public wind forecast. A positive model residual says actual wind is expected above the published forecast, which is bearish for residual load and power prices all else equal. A negative residual says the opposite. The signal should be invalidated or down-weighted when fresh TSO/weather updates materially change the forecast, when observed wind errors diverge from lagged error patterns, or when grid constraints/curtailment dominate weather-driven production.",
        "",
        *strategy_lines,
        "## Limitations",
        "",
        "The validation now spans summer, autumn, winter, and spring folds, but it is still only one annual cycle. I would not treat the result as seasonally robust until it is repeated over multiple years and distinct weather regimes.",
        "",
        "The current feature set is valid for a rolling 24-hour-ahead information set. It is not a true prompt/intraday model; once recent metered actuals are available, a previous-hour persistence benchmark should be tested and would likely be hard to beat. It is also not a strict day-ahead auction signal: the German day-ahead auction clears around D-1 noon, so 24-hour lagged actuals would be unavailable for some later delivery hours. The weather covariates are archived near-delivery forecast values rather than D-1 noon model-run snapshots, so they should be treated as post-auction for strict day-ahead trading. For a strict D-1 noon forecast, all lagged features and weather rows should be recomputed relative to the issue timestamp.",
        "",
        "The XGBoost hyperparameters were kept fixed for the reported rerun, but they were chosen during prototyping on this same history. A production study should tune on a separate period or use nested time-series validation.",
        "",
        "Overall, this should be read as a validation-first research prototype rather than a production signal. The pipeline now closes the loop from forecast to backtest to trading-strategy support, but the trading result is threshold-sensitive, fold-concentrated, and benchmarked only with a paper price-surprise proxy. A deployable version would need a longer multi-year backtest, forecast-run/lead-time pinned weather inputs, explicit issue-time feature cuts, regional generation constraints, historical executable DA-to-intraday or imbalance price marks, realistic transaction costs, and hyperparameter tuning isolated from the evaluation window.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
