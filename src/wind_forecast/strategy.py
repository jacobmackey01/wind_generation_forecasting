"""Trading-signal backtest from wind forecast residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_forecast.features import SMARD_BASELINE


PRICE = "price_da_eur_mwh"
FORECAST_RAMP = "forecast_wind_total_ramp_24h"
RESIDUAL_STRATEGY = "xgboost_wind_residual_signal"
PUBLIC_RAMP_STRATEGY = "public_smard_forecast_ramp_signal"
TOTAL_NET_PNL = "total_net_pnl_eur_per_mw_clip"


def _metrics(frame: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    trades = frame[frame["position"] != 0].copy()
    out: dict[str, float | int | str] = {
        "strategy": label,
        "hours": int(len(frame)),
        "trades": int(len(trades)),
        "trade_rate": float(len(trades) / len(frame)) if len(frame) else 0.0,
    }
    if trades.empty:
        out.update(
            {
                "hit_rate": np.nan,
                "avg_gross_pnl_eur_mwh": 0.0,
                "avg_net_pnl_eur_mwh": 0.0,
                TOTAL_NET_PNL: 0.0,
                "sharpe_like_per_trade": np.nan,
                "hit_rate_z_stat": np.nan,
                "net_pnl_t_stat": np.nan,
            }
        )
        return out

    net = trades["net_pnl_eur_mwh"].to_numpy(dtype=float)
    hit_rate = float((trades["gross_pnl_eur_mwh"] > 0).mean())
    net_std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe_like = float(net.mean() / net_std) if net_std else np.nan
    out.update(
        {
            "hit_rate": hit_rate,
            "avg_gross_pnl_eur_mwh": float(trades["gross_pnl_eur_mwh"].mean()),
            "avg_net_pnl_eur_mwh": float(trades["net_pnl_eur_mwh"].mean()),
            TOTAL_NET_PNL: float(trades["net_pnl_eur_mwh"].sum()),
            "sharpe_like_per_trade": sharpe_like,
            "hit_rate_z_stat": float((hit_rate - 0.5) / np.sqrt(0.25 / len(trades))),
            "net_pnl_t_stat": float(sharpe_like * np.sqrt(len(trades))) if pd.notna(sharpe_like) else np.nan,
        }
    )
    return out


def _position_from_edge(wind_edge_mw: pd.Series, threshold_mw: float) -> pd.Series:
    # More wind than the public forecast is bearish for power prices, so short.
    position = pd.Series(0, index=wind_edge_mw.index, dtype=int)
    position[wind_edge_mw >= threshold_mw] = -1
    position[wind_edge_mw <= -threshold_mw] = 1
    return position


def build_trade_log(
    feature_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    threshold_mw: float = 1500.0,
    transaction_cost_eur_mwh: float = 0.5,
) -> pd.DataFrame:
    """Build a paper trade log for a wind-driven DA price surprise signal."""

    price_frame = feature_frame[["timestamp_utc", PRICE]].copy().sort_values("timestamp_utc")
    price_frame["price_lag_24h"] = price_frame[PRICE].shift(24)

    out = predictions.merge(price_frame, on="timestamp_utc", how="left")
    out["strategy"] = RESIDUAL_STRATEGY
    out["wind_edge_mw"] = out["xgboost_residual"] - out[SMARD_BASELINE]
    out["signal_mw"] = out["wind_edge_mw"]
    out["price_surprise_eur_mwh"] = out[PRICE] - out["price_lag_24h"]
    out["position"] = _position_from_edge(out["signal_mw"], threshold_mw)
    out["gross_pnl_eur_mwh"] = out["position"] * out["price_surprise_eur_mwh"]
    out["transaction_cost_eur_mwh"] = np.where(out["position"] != 0, transaction_cost_eur_mwh, 0.0)
    out["net_pnl_eur_mwh"] = out["gross_pnl_eur_mwh"] - out["transaction_cost_eur_mwh"]
    out["hit"] = np.where(out["position"] != 0, out["gross_pnl_eur_mwh"] > 0, np.nan)
    return out.dropna(subset=[PRICE, "price_lag_24h", "signal_mw"]).reset_index(drop=True)


def build_public_forecast_ramp_trade_log(
    feature_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    threshold_mw: float = 1500.0,
    transaction_cost_eur_mwh: float = 0.5,
) -> pd.DataFrame:
    """Build a benchmark trade log using only the public SMARD forecast ramp."""

    signal_frame = feature_frame[["timestamp_utc", PRICE, FORECAST_RAMP]].copy().sort_values("timestamp_utc")
    signal_frame["price_lag_24h"] = signal_frame[PRICE].shift(24)
    fold_frame = predictions[["timestamp_utc", "fold", "fold_start", "fold_end"]].copy()

    out = fold_frame.merge(signal_frame, on="timestamp_utc", how="left")
    out["strategy"] = PUBLIC_RAMP_STRATEGY
    out["signal_mw"] = out[FORECAST_RAMP]
    out["price_surprise_eur_mwh"] = out[PRICE] - out["price_lag_24h"]
    out["position"] = _position_from_edge(out["signal_mw"], threshold_mw)
    out["gross_pnl_eur_mwh"] = out["position"] * out["price_surprise_eur_mwh"]
    out["transaction_cost_eur_mwh"] = np.where(out["position"] != 0, transaction_cost_eur_mwh, 0.0)
    out["net_pnl_eur_mwh"] = out["gross_pnl_eur_mwh"] - out["transaction_cost_eur_mwh"]
    out["hit"] = np.where(out["position"] != 0, out["gross_pnl_eur_mwh"] > 0, np.nan)
    return out.dropna(subset=[PRICE, "price_lag_24h", "signal_mw"]).reset_index(drop=True)


def summarize_strategy(trade_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "strategy" not in trade_log.columns:
        trade_log = trade_log.copy()
        trade_log["strategy"] = RESIDUAL_STRATEGY

    overall_rows = []
    fold_rows = []
    for strategy, strategy_frame in trade_log.groupby("strategy", sort=False):
        overall_rows.append(_metrics(strategy_frame, str(strategy)))
        for fold, frame in strategy_frame.groupby("fold", sort=True):
            row = _metrics(frame, str(strategy))
            row["fold"] = int(fold)
            row["fold_start"] = str(frame["fold_start"].iloc[0])
            row["fold_end"] = str(frame["fold_end"].iloc[0])
            fold_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(fold_rows)


def threshold_sensitivity(
    feature_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    thresholds_mw: list[float] | None = None,
    transaction_cost_eur_mwh: float = 0.5,
) -> pd.DataFrame:
    thresholds_mw = thresholds_mw or [500.0, 1000.0, 1500.0, 2000.0, 3000.0]
    rows = []
    for threshold in thresholds_mw:
        residual_trades = build_trade_log(
            feature_frame,
            predictions,
            threshold_mw=threshold,
            transaction_cost_eur_mwh=transaction_cost_eur_mwh,
        )
        public_ramp_trades = build_public_forecast_ramp_trade_log(
            feature_frame,
            predictions,
            threshold_mw=threshold,
            transaction_cost_eur_mwh=transaction_cost_eur_mwh,
        )
        for trades in [residual_trades, public_ramp_trades]:
            strategy = str(trades["strategy"].iloc[0])
            row = _metrics(trades, strategy)
            row["threshold_mw"] = threshold
            rows.append(row)
    return pd.DataFrame(rows)


def backtest_wind_price_signal(
    feature_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    threshold_mw: float = 1500.0,
    transaction_cost_eur_mwh: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residual_trade_log = build_trade_log(
        feature_frame,
        predictions,
        threshold_mw=threshold_mw,
        transaction_cost_eur_mwh=transaction_cost_eur_mwh,
    )
    public_ramp_trade_log = build_public_forecast_ramp_trade_log(
        feature_frame,
        predictions,
        threshold_mw=threshold_mw,
        transaction_cost_eur_mwh=transaction_cost_eur_mwh,
    )
    trade_log = pd.concat([residual_trade_log, public_ramp_trade_log], ignore_index=True, sort=False)
    trade_log = trade_log.sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    metrics, fold_metrics = summarize_strategy(trade_log)
    sensitivity = threshold_sensitivity(
        feature_frame,
        predictions,
        transaction_cost_eur_mwh=transaction_cost_eur_mwh,
    )
    return trade_log, metrics, fold_metrics, sensitivity
