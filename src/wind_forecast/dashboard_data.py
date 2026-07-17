"""Data preparation helpers for the stakeholder dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wind_forecast.strategy import PUBLIC_RAMP_STRATEGY, RESIDUAL_STRATEGY, TOTAL_NET_PNL


MODEL_LABELS = {
    "smard_forecast": "Public SMARD forecast",
    "xgboost_residual": "XGBoost residual",
}

STRATEGY_LABELS = {
    RESIDUAL_STRATEGY: "XGBoost residual signal",
    PUBLIC_RAMP_STRATEGY: "Public SMARD ramp",
}


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def prepare_fold_mae(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the two serious forecast benchmarks in chart-ready form."""

    _require_columns(
        fold_metrics,
        {"model", "fold", "fold_start", "mae", "skill_vs_smard_mae_pct"},
        "fold_metrics",
    )
    out = fold_metrics.loc[fold_metrics["model"].isin(MODEL_LABELS)].copy()
    out["fold_start"] = pd.to_datetime(out["fold_start"], utc=True)
    out["month"] = out["fold_start"].dt.strftime("%b %Y")
    out["series"] = out["model"].map(MODEL_LABELS)
    return out.sort_values(["fold", "model"]).reset_index(drop=True)


def prepare_strategy_fold_paths(strategy_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return fold P&L and cumulative paths for both strategy definitions."""

    _require_columns(
        strategy_fold_metrics,
        {"strategy", "fold", "fold_start", "trades", TOTAL_NET_PNL},
        "strategy_fold_metrics",
    )
    out = strategy_fold_metrics.loc[strategy_fold_metrics["strategy"].isin(STRATEGY_LABELS)].copy()
    out["fold_start"] = pd.to_datetime(out["fold_start"], utc=True)
    out = out.sort_values(["strategy", "fold"]).reset_index(drop=True)
    out["month"] = out["fold_start"].dt.strftime("%b %Y")
    out["series"] = out["strategy"].map(STRATEGY_LABELS)
    out["cumulative_net_pnl_eur_per_mw_clip"] = out.groupby("strategy", sort=False)[TOTAL_NET_PNL].cumsum()
    out["cumulative_trades"] = out.groupby("strategy", sort=False)["trades"].cumsum()
    out["running_avg_net_pnl_eur_mwh"] = (
        out["cumulative_net_pnl_eur_per_mw_clip"] / out["cumulative_trades"].replace(0, pd.NA)
    )
    return out


def prepare_threshold_curves(strategy_thresholds: pd.DataFrame) -> pd.DataFrame:
    """Return threshold-grid results for both strategies."""

    _require_columns(
        strategy_thresholds,
        {"strategy", "threshold_mw", "trades", "avg_net_pnl_eur_mwh", "net_pnl_t_stat"},
        "strategy_threshold_sensitivity",
    )
    out = strategy_thresholds.loc[strategy_thresholds["strategy"].isin(STRATEGY_LABELS)].copy()
    out["series"] = out["strategy"].map(STRATEGY_LABELS)
    return out.sort_values(["strategy", "threshold_mw"]).reset_index(drop=True)


def residual_fold_concentration(strategy_fold_metrics: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize how much the residual result depends on its strongest fold."""

    _require_columns(
        strategy_fold_metrics,
        {"strategy", "fold_start", "trades", TOTAL_NET_PNL},
        "strategy_fold_metrics",
    )
    residual = strategy_fold_metrics.loc[strategy_fold_metrics["strategy"] == RESIDUAL_STRATEGY].copy()
    if residual.empty:
        raise ValueError("strategy_fold_metrics contains no residual strategy rows.")

    residual["fold_start"] = pd.to_datetime(residual["fold_start"], utc=True)
    best = residual.sort_values(TOTAL_NET_PNL, ascending=False).iloc[0]
    total = float(residual[TOTAL_NET_PNL].sum())
    best_pnl = float(best[TOTAL_NET_PNL])
    trades_ex_best = int(residual["trades"].sum() - best["trades"])
    return {
        "positive_folds": int((residual[TOTAL_NET_PNL] > 0).sum()),
        "fold_count": int(len(residual)),
        "best_month": best["fold_start"].strftime("%b %Y"),
        "best_fold_pnl": best_pnl,
        "total_pnl": total,
        "pnl_ex_best": total - best_pnl,
        "trades_ex_best": trades_ex_best,
    }


def load_llm_review_artifact(outputs_dir: Path) -> dict[str, object] | None:
    """Load the latest saved LLM review without making an API call."""

    path = Path(outputs_dir) / "llm" / "analyst_review.json"
    if not path.exists():
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "generated_at_utc",
        "model_returned",
        "reasoning_effort",
        "evidence_fingerprint",
        "evidence",
        "review",
        "boundary",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"analyst_review.json is missing fields: {sorted(missing)}")
    if not isinstance(artifact["evidence"], list) or not isinstance(artifact["review"], dict):
        raise ValueError("analyst_review.json has invalid evidence or review data.")
    from wind_forecast.llm_review import build_review_evidence, evidence_fingerprint

    current_fingerprint = evidence_fingerprint(build_review_evidence(Path(outputs_dir)))
    if artifact["evidence_fingerprint"] != current_fingerprint:
        raise ValueError("the saved AI review belongs to a different pipeline output snapshot")
    return artifact


def finding_evidence_labels(artifact: dict[str, object], finding: dict[str, object]) -> list[str]:
    """Return deterministic label/value strings for a grounded LLM finding."""

    evidence_items = artifact["evidence"]
    if not isinstance(evidence_items, list):
        raise ValueError("LLM review evidence must be a list.")
    evidence_map = {item["id"]: item for item in evidence_items if isinstance(item, dict) and "id" in item}
    evidence_ids = finding.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        raise ValueError("LLM finding evidence_ids must be a list.")
    unknown = set(evidence_ids).difference(evidence_map)
    if unknown:
        raise ValueError(f"LLM finding cites unknown evidence IDs: {sorted(unknown)}")
    return [f"{evidence_map[item]['label']}: {evidence_map[item]['value']}" for item in evidence_ids]
