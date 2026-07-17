"""Grounded OpenAI analyst review for deterministic pipeline outputs."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from wind_forecast.strategy import PUBLIC_RAMP_STRATEGY, RESIDUAL_STRATEGY, TOTAL_NET_PNL


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "low"
REQUIRED_OUTPUT_FILES = (
    "metrics.csv",
    "forecast_diagnostics.csv",
    "fold_metrics.csv",
    "qa_checks.csv",
    "strategy_metrics.csv",
    "strategy_fold_metrics.csv",
    "strategy_threshold_sensitivity.csv",
)

SYSTEM_PROMPT = """You are a European power-market research reviewer.
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
"""


class EvidenceBackedFinding(BaseModel):
    """One narrative finding tied to deterministic evidence IDs."""

    model_config = ConfigDict(extra="forbid")

    assessment: str
    evidence_ids: list[str]


class AnalystReview(BaseModel):
    """Structured output returned by the OpenAI Responses API."""

    model_config = ConfigDict(extra="forbid")

    headline: str
    overall_conclusion: str
    forecast_assessment: EvidenceBackedFinding
    strategy_assessment: EvidenceBackedFinding
    risk_flags: list[EvidenceBackedFinding]
    invalidation_conditions: list[str]
    production_next_steps: list[str]


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _single_row(frame: pd.DataFrame, column: str, value: str, name: str) -> pd.Series:
    rows = frame.loc[frame[column] == value]
    if len(rows) != 1:
        raise ValueError(f"{name} must contain exactly one {column}={value!r} row; found {len(rows)}.")
    return rows.iloc[0]


def _number(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; received {value!r}.")
    return result


def _fmt(value: object, digits: int = 2, signed: bool = False) -> str:
    number = _number(value, "evidence value")
    sign = "+" if signed else ""
    return f"{number:{sign},.{digits}f}"


def _evidence_item(evidence_id: str, label: str, value: str, interpretation: str) -> dict[str, str]:
    return {
        "id": evidence_id,
        "label": label,
        "value": value,
        "interpretation": interpretation,
    }


def _read_outputs(outputs_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [filename for filename in REQUIRED_OUTPUT_FILES if not (outputs_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing LLM review inputs in {outputs_dir}: {', '.join(missing)}")
    return {Path(filename).stem: pd.read_csv(outputs_dir / filename) for filename in REQUIRED_OUTPUT_FILES}


def build_review_evidence(outputs_dir: Path) -> list[dict[str, str]]:
    """Build a compact, auditable evidence package from deterministic CSV outputs."""

    frames = _read_outputs(Path(outputs_dir))
    metrics = frames["metrics"]
    diagnostics = frames["forecast_diagnostics"]
    fold_metrics = frames["fold_metrics"]
    qa_checks = frames["qa_checks"]
    strategy_metrics = frames["strategy_metrics"]
    strategy_folds = frames["strategy_fold_metrics"]
    thresholds = frames["strategy_threshold_sensitivity"]

    _require_columns(metrics, {"model", "mae", "bias", "skill_vs_smard_mae_pct"}, "metrics")
    _require_columns(
        diagnostics,
        {
            "newey_west_loss_diff_t_stat",
            "xgboost_fold_wins",
            "fold_count",
            "mean_fold_skill_pct",
            "best_fold",
            "best_fold_skill_pct",
            "worst_fold",
            "worst_fold_skill_pct",
        },
        "forecast_diagnostics",
    )
    _require_columns(
        fold_metrics,
        {"model", "fold", "fold_start", "fold_end", "mae", "skill_vs_smard_mae_pct"},
        "fold_metrics",
    )
    _require_columns(qa_checks, {"check", "value", "passed"}, "qa_checks")
    _require_columns(
        strategy_metrics,
        {"strategy", "trades", "hit_rate", "avg_net_pnl_eur_mwh", TOTAL_NET_PNL, "net_pnl_t_stat"},
        "strategy_metrics",
    )
    _require_columns(
        strategy_folds,
        {"strategy", "fold", "fold_start", "trades", TOTAL_NET_PNL},
        "strategy_fold_metrics",
    )
    _require_columns(
        thresholds,
        {"strategy", "threshold_mw", "trades", "avg_net_pnl_eur_mwh", "net_pnl_t_stat"},
        "strategy_threshold_sensitivity",
    )

    if len(diagnostics) != 1:
        raise ValueError(f"forecast_diagnostics must contain exactly one row; found {len(diagnostics)}.")

    smard = _single_row(metrics, "model", "smard_forecast", "metrics")
    xgb = _single_row(metrics, "model", "xgboost_residual", "metrics")
    diag = diagnostics.iloc[0]
    residual = _single_row(strategy_metrics, "strategy", RESIDUAL_STRATEGY, "strategy_metrics")
    public = _single_row(strategy_metrics, "strategy", PUBLIC_RAMP_STRATEGY, "strategy_metrics")

    xgb_folds = fold_metrics.loc[fold_metrics["model"] == "xgboost_residual"].copy()
    if xgb_folds.empty:
        raise ValueError("fold_metrics contains no xgboost_residual rows.")
    xgb_folds["fold_start"] = pd.to_datetime(xgb_folds["fold_start"], utc=True)
    xgb_folds["fold_end"] = pd.to_datetime(xgb_folds["fold_end"], utc=True)
    worst_forecast_fold = xgb_folds.sort_values("skill_vs_smard_mae_pct").iloc[0]

    residual_folds = strategy_folds.loc[strategy_folds["strategy"] == RESIDUAL_STRATEGY].copy()
    public_folds = strategy_folds.loc[strategy_folds["strategy"] == PUBLIC_RAMP_STRATEGY].copy()
    if residual_folds.empty or public_folds.empty:
        raise ValueError("strategy_fold_metrics must contain both residual and public-ramp rows.")
    residual_folds["fold_start"] = pd.to_datetime(residual_folds["fold_start"], utc=True)
    best_residual_fold = residual_folds.sort_values(TOTAL_NET_PNL, ascending=False).iloc[0]
    residual_total = _number(residual[TOTAL_NET_PNL], "residual total P&L")
    public_total = _number(public[TOTAL_NET_PNL], "public total P&L")
    if not math.isclose(residual_total, residual_folds[TOTAL_NET_PNL].astype(float).sum(), abs_tol=1e-6):
        raise ValueError("Residual strategy total does not reconcile to strategy_fold_metrics.")
    if not math.isclose(public_total, public_folds[TOTAL_NET_PNL].astype(float).sum(), abs_tol=1e-6):
        raise ValueError("Public-ramp total does not reconcile to strategy_fold_metrics.")

    residual_thresholds = thresholds.loc[thresholds["strategy"] == RESIDUAL_STRATEGY].copy()
    if residual_thresholds.empty:
        raise ValueError("strategy_threshold_sensitivity contains no residual strategy rows.")
    residual_thresholds["distance_to_reported"] = (
        residual_thresholds["avg_net_pnl_eur_mwh"].astype(float)
        - _number(residual["avg_net_pnl_eur_mwh"], "reported residual average P&L")
    ).abs()
    reported_threshold = residual_thresholds.sort_values(["distance_to_reported", "threshold_mw"]).iloc[0]
    if _number(reported_threshold["distance_to_reported"], "threshold match distance") > 1e-8:
        raise ValueError("Could not reconcile the reported residual strategy to the threshold grid.")
    best_threshold = residual_thresholds.sort_values("avg_net_pnl_eur_mwh", ascending=False).iloc[0]

    qa_passed = qa_checks["passed"].astype(str).str.lower().isin({"true", "1", "yes"})
    failed_checks = qa_checks.loc[~qa_passed, "check"].astype(str).tolist()
    row_count_rows = qa_checks.loc[qa_checks["check"] == "row_count_gt_90_days", "value"]
    row_count = row_count_rows.iloc[0] if len(row_count_rows) == 1 else "not recorded"

    residual_avg = _number(residual["avg_net_pnl_eur_mwh"], "residual average P&L")
    public_avg = _number(public["avg_net_pnl_eur_mwh"], "public average P&L")
    if residual_avg > 0:
        per_trade_comparison = f"{public_avg / residual_avg:,.1f}x the residual signal"
    else:
        per_trade_comparison = "not meaningful because residual average P&L is non-positive"

    best_fold_pnl = _number(best_residual_fold[TOTAL_NET_PNL], "best residual fold P&L")
    pnl_ex_best = residual_total - best_fold_pnl
    validation_start = xgb_folds["fold_start"].min().strftime("%Y-%m-%d")
    validation_end = xgb_folds["fold_end"].max().strftime("%Y-%m-%d")

    return [
        _evidence_item(
            "qa.checks",
            "Pipeline QA checks",
            f"{int(qa_passed.sum())}/{len(qa_checks)} passed; {row_count} hourly rows; failures: {failed_checks or 'none'}",
            "The review input passed the pipeline's deterministic data checks.",
        ),
        _evidence_item(
            "forecast.validation_design",
            "Forecast validation design",
            f"{int(diag['fold_count'])} expanding walk-forward folds, {validation_start} to {validation_end}; no random split",
            "The reported forecast result is out-of-sample in temporal order.",
        ),
        _evidence_item(
            "forecast.mae_comparison",
            "XGBoost versus public SMARD MAE",
            f"XGBoost {_fmt(xgb['mae'])} MW; SMARD {_fmt(smard['mae'])} MW; skill {_fmt(xgb['skill_vs_smard_mae_pct'], 2, signed=True)}%",
            "The pooled MAE difference is small relative to the public operational forecast.",
        ),
        _evidence_item(
            "forecast.significance",
            "Forecast loss-difference diagnostic",
            f"Newey-West t={_fmt(diag['newey_west_loss_diff_t_stat'], 2)}; mean fold skill {_fmt(diag['mean_fold_skill_pct'], 2, signed=True)}%",
            "The forecast result lacks conventional statistical support and is not consistently positive by fold.",
        ),
        _evidence_item(
            "forecast.fold_dispersion",
            "Forecast fold dispersion",
            f"XGBoost wins {int(diag['xgboost_fold_wins'])}/{int(diag['fold_count'])}; best {_fmt(diag['best_fold_skill_pct'], 1, signed=True)}%; worst {_fmt(diag['worst_fold_skill_pct'], 1, signed=True)}%",
            "Average performance hides material regime variation.",
        ),
        _evidence_item(
            "forecast.worst_regime",
            "Worst forecast fold",
            f"{worst_forecast_fold['fold_start'].strftime('%b %Y')}; skill {_fmt(worst_forecast_fold['skill_vs_smard_mae_pct'], 1, signed=True)}%",
            "This is the clearest observed forecast stress regime.",
        ),
        _evidence_item(
            "forecast.bias",
            "Forecast bias",
            f"XGBoost {_fmt(xgb['bias'], 1, signed=True)} MW; SMARD {_fmt(smard['bias'], 1, signed=True)} MW",
            "Bias should be monitored separately from absolute error.",
        ),
        _evidence_item(
            "strategy.residual_result",
            "Residual strategy result",
            f"{int(residual['trades'])} trades; {_fmt(residual['avg_net_pnl_eur_mwh'], 3, signed=True)} EUR/MWh per trade; t={_fmt(residual['net_pnl_t_stat'], 2)}",
            "The model-derived residual signal does not have conventional statistical support.",
        ),
        _evidence_item(
            "strategy.fold_concentration",
            "Residual strategy fold concentration",
            f"{int((residual_folds[TOTAL_NET_PNL].astype(float) > 0).sum())}/{len(residual_folds)} positive folds; best {best_residual_fold['fold_start'].strftime('%b %Y')} {_fmt(best_fold_pnl, 2, signed=True)} EUR; excluding it {_fmt(pnl_ex_best, 2, signed=True)} EUR",
            "A large share of the aggregate result depends on one validation month.",
        ),
        _evidence_item(
            "strategy.public_benchmark",
            "Public SMARD ramp benchmark",
            f"{int(public['trades'])} trades; {_fmt(public_avg, 3, signed=True)} EUR/MWh per trade; {per_trade_comparison}",
            "The proxy strongly rewards public forecast repricing, so aggregate proxy P&L is not proof of private edge.",
        ),
        _evidence_item(
            "strategy.price_proxy",
            "Price-settlement limitation",
            "DA(t) minus DA(t-24), not an executable day-ahead-to-intraday or imbalance spread",
            "Neither strategy's proxy P&L should be presented as an executable trading return.",
        ),
        _evidence_item(
            "strategy.reported_threshold",
            "Reported residual trigger",
            f"absolute signal >= {_fmt(reported_threshold['threshold_mw'], 0)} MW",
            "Hours below the trigger are no-trade observations.",
        ),
        _evidence_item(
            "strategy.threshold_robustness",
            "Residual threshold sensitivity",
            f"best of {len(residual_thresholds)} tested points: {_fmt(best_threshold['threshold_mw'], 0)} MW, {_fmt(best_threshold['avg_net_pnl_eur_mwh'], 3, signed=True)} EUR/MWh per trade, unadjusted t={_fmt(best_threshold['net_pnl_t_stat'], 2)}",
            "The best grid point is an in-sample multiple-choice maximum, not holdout evidence.",
        ),
        _evidence_item(
            "design.information_set",
            "Auction information-set limitation",
            "24-hour lagged actual/error features and archived weather are not pinned to strict D-1 noon availability",
            "The current signal is rolling 24-hour-ahead research, not a strict day-ahead auction implementation.",
        ),
        _evidence_item(
            "design.sample_scope",
            "Seasonal scope",
            "summer, autumn, winter, and spring folds within one annual cycle; no multi-year validation",
            "One pass through the seasons is insufficient evidence of regime robustness.",
        ),
        _evidence_item(
            "design.llm_boundary",
            "LLM role",
            "downstream summarization only; no influence on QA, features, XGBoost, signals, trades, or metrics",
            "The AI component reduces manual reporting work without contaminating the quantitative pipeline.",
        ),
    ]


def build_review_prompts(evidence: list[dict[str, str]]) -> tuple[str, str]:
    """Return the exact system and user prompts sent to the model."""

    user_prompt = (
        "Write the structured analyst review from the evidence package below. "
        "Use evidence IDs exactly as supplied. Keep risk flags, invalidation conditions, and next steps concise.\n\n"
        + json.dumps({"evidence": evidence}, indent=2, ensure_ascii=True)
    )
    return SYSTEM_PROMPT, user_prompt


def evidence_fingerprint(evidence: list[dict[str, str]]) -> str:
    """Return a stable fingerprint tying an LLM review to one output snapshot."""

    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ascii_clean(value: str) -> str:
    replacements = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
    for old, new in replacements.items():
        value = value.replace(old, new)
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def _clean_review(review: AnalystReview) -> AnalystReview:
    data = review.model_dump()

    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return _ascii_clean(value)
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    return AnalystReview.model_validate(clean(data))


def validate_review_grounding(review: AnalystReview, evidence: list[dict[str, str]]) -> None:
    """Reject responses that omit grounding or cite IDs outside the supplied evidence."""

    if not (2 <= len(review.risk_flags) <= 5):
        raise ValueError("The LLM review must contain between 2 and 5 risk flags.")
    if not (2 <= len(review.invalidation_conditions) <= 5):
        raise ValueError("The LLM review must contain between 2 and 5 invalidation conditions.")
    if not (2 <= len(review.production_next_steps) <= 5):
        raise ValueError("The LLM review must contain between 2 and 5 production next steps.")

    allowed_ids = {item["id"] for item in evidence}
    findings = [review.forecast_assessment, review.strategy_assessment, *review.risk_flags]
    for finding in findings:
        if not finding.evidence_ids:
            raise ValueError("Every LLM finding must cite at least one evidence ID.")
        unknown = set(finding.evidence_ids).difference(allowed_ids)
        if unknown:
            raise ValueError(f"LLM review cited unknown evidence IDs: {sorted(unknown)}")


def _model_dump(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return, union-attr]
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def render_review_markdown(artifact: dict[str, object]) -> str:
    """Render the structured review and deterministic evidence as Markdown."""

    review = AnalystReview.model_validate(artifact["review"])
    evidence = {item["id"]: item for item in artifact["evidence"]}  # type: ignore[index]

    def finding_lines(title: str, finding: EvidenceBackedFinding) -> list[str]:
        lines = [f"## {title}", "", finding.assessment, "", "Evidence:"]
        for evidence_id in finding.evidence_ids:
            item = evidence[evidence_id]
            lines.append(f"- `{evidence_id}`: {item['label']} - {item['value']}")
        lines.append("")
        return lines

    lines = [
        "# AI Analyst Review",
        "",
        f"Generated: {artifact['generated_at_utc']}",
        f"Model: {artifact['model_returned']}",
        f"Reasoning effort: {artifact['reasoning_effort']}",
        "",
        "> Downstream research summary only. The LLM did not alter forecasts, trades, QA checks, or metrics.",
        "",
        f"## {review.headline}",
        "",
        review.overall_conclusion,
        "",
        *finding_lines("Forecast Assessment", review.forecast_assessment),
        *finding_lines("Strategy Assessment", review.strategy_assessment),
        "## Risk Flags",
        "",
    ]
    for risk in review.risk_flags:
        labels = ", ".join(f"`{evidence_id}`" for evidence_id in risk.evidence_ids)
        lines.append(f"- {risk.assessment} ({labels})")
    lines.extend(["", "## Invalidation Conditions", ""])
    lines.extend(f"- {item}" for item in review.invalidation_conditions)
    lines.extend(["", "## Production Next Steps", ""])
    lines.extend(f"- {item}" for item in review.production_next_steps)
    lines.extend(
        [
            "",
            "## Audit Note",
            "",
            "Exact prompts, evidence, structured output, response metadata, and token usage are logged in `run_log.json`.",
            "This is not executable trading advice.",
            "",
        ]
    )
    return "\n".join(lines)


def run_llm_review(
    outputs_dir: Path,
    output_dir: Path | None = None,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    client: object | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Path]:
    """Call OpenAI, validate grounding, and persist auditable review artifacts."""

    outputs_dir = Path(outputs_dir)
    output_dir = Path(output_dir) if output_dir is not None else outputs_dir / "llm"
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = build_review_evidence(outputs_dir)
    system_prompt, user_prompt = build_review_prompts(evidence)
    prompt_path = output_dir / "prompt.md"
    prompt_path.write_text(
        f"# System Prompt\n\n{system_prompt}\n\n# User Prompt\n\n{user_prompt}\n",
        encoding="utf-8",
    )

    if client is None:
        from openai import OpenAI

        client = OpenAI()

    response = client.responses.parse(  # type: ignore[union-attr]
        model=model,
        reasoning={"effort": reasoning_effort},
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=AnalystReview,
        max_output_tokens=1800,
        store=False,
    )
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed analyst review. Check the response for a refusal or API error.")
    review = parsed if isinstance(parsed, AnalystReview) else AnalystReview.model_validate(parsed)
    review = _clean_review(review)
    validate_review_grounding(review, evidence)

    timestamp = (generated_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    response_model = str(getattr(response, "model", model))
    artifact: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": timestamp,
        "model_requested": model,
        "model_returned": response_model,
        "reasoning_effort": reasoning_effort,
        "boundary": "Downstream summarization only; no influence on quantitative outputs.",
        "evidence_fingerprint": evidence_fingerprint(evidence),
        "evidence": evidence,
        "review": review.model_dump(),
    }
    run_log = {
        **artifact,
        "response_id": getattr(response, "id", None),
        "request_id": getattr(response, "_request_id", None),
        "usage": _model_dump(getattr(response, "usage", None)),
        "prompts": {"system": system_prompt, "user": user_prompt},
    }

    artifact_path = output_dir / "analyst_review.json"
    markdown_path = output_dir / "analyst_review.md"
    log_path = output_dir / "run_log.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_review_markdown(artifact), encoding="utf-8")
    log_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "artifact": artifact_path,
        "markdown": markdown_path,
        "prompt": prompt_path,
        "log": log_path,
    }
