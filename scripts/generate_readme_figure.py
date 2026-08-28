"""Generate the source-backed fold-skill figure used by the public README."""

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


EXPECTED_MODELS = ("smard_forecast", "xgboost_residual")
# Stored fold skills are compared in percentage points; this is tighter than display precision.
SKILL_TOLERANCE_PCT = 1e-6
SVG_WIDTH = 1200
SVG_HEIGHT = 780


class EvidenceError(ValueError):
    """Raised when the committed evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class FoldEvidence:
    fold: int
    fold_start: datetime
    fold_end: datetime
    smard_mae: float
    xgboost_mae: float
    stored_skill_pct: float

    @property
    def skill_pct(self) -> float:
        return (self.smard_mae - self.xgboost_mae) / self.smard_mae * 100.0


@dataclass(frozen=True)
class FigureEvidence:
    folds: tuple[FoldEvidence, ...]
    smard_mae: float
    xgboost_mae: float
    pooled_skill_pct: float
    observations: int
    fold_wins: int
    nw_ci_lower_pct: float
    nw_ci_upper_pct: float
    bootstrap_ci_lower_pct: float
    bootstrap_ci_upper_pct: float
    best_fold: int
    best_fold_skill_pct: float
    worst_fold: int
    worst_fold_skill_pct: float
    mean_fold_skill_pct: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise EvidenceError(f"Required evidence file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EvidenceError(f"Evidence file has no header: {path}")
        return list(reader)


def _require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    actual = set(rows[0]) if rows else set()
    missing = sorted(required - actual)
    if missing:
        raise EvidenceError(f"{label} is missing columns: {missing}")


def _number(value: str | None, label: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} must be numeric; received {value!r}") from exc
    if not math.isfinite(parsed):
        raise EvidenceError(f"{label} must be finite; received {value!r}")
    return parsed


def _integer(value: str | None, label: str) -> int:
    parsed = _number(value, label)
    if not parsed.is_integer():
        raise EvidenceError(f"{label} must be an integer; received {value!r}")
    return int(parsed)


def _timestamp(value: str | None, label: str) -> datetime:
    if not value:
        raise EvidenceError(f"{label} is missing")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise EvidenceError(f"{label} must be an ISO-8601 timestamp; received {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceError(f"{label} must include a timezone offset; received {value!r}")
    return parsed


def _compare(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=SKILL_TOLERANCE_PCT):
        raise EvidenceError(
            f"{label} mismatch: stored {actual:.12f}, recomputed {expected:.12f}; "
            f"tolerance is {SKILL_TOLERANCE_PCT:g} percentage points"
        )


def load_evidence(repo_root: Path) -> FigureEvidence:
    """Load and validate the compact committed evidence used by the chart."""

    outputs = Path(repo_root) / "outputs"
    fold_rows = _read_csv(outputs / "fold_metrics.csv")
    metric_rows = _read_csv(outputs / "metrics.csv")
    diagnostic_rows = _read_csv(outputs / "forecast_diagnostics.csv")

    _require_columns(
        fold_rows,
        {"model", "fold", "fold_start", "fold_end", "mae", "skill_vs_smard_mae_pct"},
        "fold_metrics.csv",
    )
    _require_columns(metric_rows, {"model", "mae", "skill_vs_smard_mae_pct"}, "metrics.csv")
    _require_columns(
        diagnostic_rows,
        {
            "loss_diff_observations",
            "loss_diff_skill_pct",
            "newey_west_skill_ci_95_lower_pct",
            "newey_west_skill_ci_95_upper_pct",
            "day_block_bootstrap_skill_ci_95_lower_pct",
            "day_block_bootstrap_skill_ci_95_upper_pct",
            "xgboost_fold_wins",
            "fold_count",
            "mean_fold_skill_pct",
            "best_fold",
            "best_fold_skill_pct",
            "worst_fold",
            "worst_fold_skill_pct",
        },
        "forecast_diagnostics.csv",
    )
    if len(diagnostic_rows) != 1:
        raise EvidenceError(f"forecast_diagnostics.csv must contain exactly one row; found {len(diagnostic_rows)}")

    metric_by_model: dict[str, dict[str, str]] = {}
    for row in metric_rows:
        model = row["model"]
        if model in metric_by_model:
            raise EvidenceError(f"metrics.csv contains duplicate model row: {model}")
        metric_by_model[model] = row
    missing_metrics = sorted(set(EXPECTED_MODELS) - set(metric_by_model))
    if missing_metrics:
        raise EvidenceError(f"metrics.csv is missing expected model rows: {missing_metrics}")

    smard_mae = _number(metric_by_model["smard_forecast"].get("mae"), "SMARD pooled MAE")
    xgboost_mae = _number(metric_by_model["xgboost_residual"].get("mae"), "XGBoost pooled MAE")
    if smard_mae <= 0:
        raise EvidenceError("SMARD pooled MAE must be positive")
    pooled_skill_pct = (smard_mae - xgboost_mae) / smard_mae * 100.0
    _compare(
        _number(metric_by_model["xgboost_residual"].get("skill_vs_smard_mae_pct"), "stored pooled skill"),
        pooled_skill_pct,
        "Pooled skill",
    )

    key_counts: dict[tuple[int, str], int] = {}
    rows_by_fold: dict[int, dict[str, dict[str, str]]] = {}
    fold_order: list[int] = []
    for row in fold_rows:
        fold = _integer(row.get("fold"), "fold")
        model = row["model"]
        key = (fold, model)
        key_counts[key] = key_counts.get(key, 0) + 1
        if key_counts[key] > 1:
            raise EvidenceError(f"fold_metrics.csv contains duplicate fold/model combination: {key}")
        if fold not in rows_by_fold:
            rows_by_fold[fold] = {}
            fold_order.append(fold)
        rows_by_fold[fold][model] = row

    if not fold_order:
        raise EvidenceError("fold_metrics.csv contains no fold rows")
    if fold_order != sorted(fold_order):
        raise EvidenceError(f"fold_metrics.csv is not chronologically ordered by fold: {fold_order}")
    fold_ids = sorted(rows_by_fold)
    if fold_ids != list(range(fold_ids[0], fold_ids[-1] + 1)):
        raise EvidenceError(f"fold_metrics.csv has missing fold ids: {fold_ids}")

    folds: list[FoldEvidence] = []
    previous_start: datetime | None = None
    for fold in fold_ids:
        missing_models = sorted(set(EXPECTED_MODELS) - set(rows_by_fold[fold]))
        if missing_models:
            raise EvidenceError(f"fold {fold} is missing expected model rows: {missing_models}")
        smard_row = rows_by_fold[fold]["smard_forecast"]
        xgb_row = rows_by_fold[fold]["xgboost_residual"]
        fold_start = _timestamp(smard_row.get("fold_start"), f"fold {fold} start")
        fold_end = _timestamp(smard_row.get("fold_end"), f"fold {fold} end")
        if fold_end < fold_start:
            raise EvidenceError(f"fold {fold} ends before it starts")
        if previous_start is not None and fold_start <= previous_start:
            raise EvidenceError("fold start timestamps must increase chronologically")
        previous_start = fold_start
        smard_fold_mae = _number(smard_row.get("mae"), f"fold {fold} SMARD MAE")
        xgb_fold_mae = _number(xgb_row.get("mae"), f"fold {fold} XGBoost MAE")
        stored_skill = _number(xgb_row.get("skill_vs_smard_mae_pct"), f"fold {fold} stored skill")
        recomputed_skill = (smard_fold_mae - xgb_fold_mae) / smard_fold_mae * 100.0
        _compare(stored_skill, recomputed_skill, f"Fold {fold} skill")
        folds.append(
            FoldEvidence(
                fold=fold,
                fold_start=fold_start,
                fold_end=fold_end,
                smard_mae=smard_fold_mae,
                xgboost_mae=xgb_fold_mae,
                stored_skill_pct=stored_skill,
            )
        )

    diagnostic = diagnostic_rows[0]
    fold_wins = sum(fold.skill_pct > 0 for fold in folds)
    diagnostic_fold_count = _integer(diagnostic.get("fold_count"), "diagnostic fold count")
    if diagnostic_fold_count != len(folds):
        raise EvidenceError(
            f"diagnostic fold count {diagnostic_fold_count} does not agree with fold_metrics.csv count {len(folds)}"
        )
    diagnostic_wins = _integer(diagnostic.get("xgboost_fold_wins"), "diagnostic fold wins")
    if diagnostic_wins != fold_wins:
        raise EvidenceError(f"diagnostic fold wins {diagnostic_wins} does not agree with counted wins {fold_wins}")

    best = max(folds, key=lambda item: item.skill_pct)
    worst = min(folds, key=lambda item: item.skill_pct)
    diagnostic_best = _integer(diagnostic.get("best_fold"), "diagnostic best fold")
    diagnostic_worst = _integer(diagnostic.get("worst_fold"), "diagnostic worst fold")
    if diagnostic_best != best.fold or diagnostic_worst != worst.fold:
        raise EvidenceError(
            f"diagnostic best/worst folds {diagnostic_best}/{diagnostic_worst} do not agree with "
            f"recomputed {best.fold}/{worst.fold}"
        )
    _compare(_number(diagnostic.get("best_fold_skill_pct"), "diagnostic best skill"), best.skill_pct, "Best-fold skill")
    _compare(_number(diagnostic.get("worst_fold_skill_pct"), "diagnostic worst skill"), worst.skill_pct, "Worst-fold skill")
    _compare(_number(diagnostic.get("loss_diff_skill_pct"), "diagnostic pooled skill"), pooled_skill_pct, "Diagnostic pooled skill")

    nw_ci_lower = _number(diagnostic.get("newey_west_skill_ci_95_lower_pct"), "Newey-West lower skill CI")
    nw_ci_upper = _number(diagnostic.get("newey_west_skill_ci_95_upper_pct"), "Newey-West upper skill CI")
    bootstrap_ci_lower = _number(diagnostic.get("day_block_bootstrap_skill_ci_95_lower_pct"), "bootstrap lower skill CI")
    bootstrap_ci_upper = _number(diagnostic.get("day_block_bootstrap_skill_ci_95_upper_pct"), "bootstrap upper skill CI")
    if not nw_ci_lower <= 0 <= nw_ci_upper:
        raise EvidenceError("Newey-West skill interval must cross zero for this evidence figure")
    if not bootstrap_ci_lower <= 0 <= bootstrap_ci_upper:
        raise EvidenceError("day-block bootstrap skill interval must cross zero for this evidence figure")

    return FigureEvidence(
        folds=tuple(folds),
        smard_mae=smard_mae,
        xgboost_mae=xgboost_mae,
        pooled_skill_pct=pooled_skill_pct,
        observations=_integer(diagnostic.get("loss_diff_observations"), "out-of-sample observations"),
        fold_wins=fold_wins,
        nw_ci_lower_pct=nw_ci_lower,
        nw_ci_upper_pct=nw_ci_upper,
        bootstrap_ci_lower_pct=bootstrap_ci_lower,
        bootstrap_ci_upper_pct=bootstrap_ci_upper,
        best_fold=best.fold,
        best_fold_skill_pct=best.skill_pct,
        worst_fold=worst.fold,
        worst_fold_skill_pct=worst.skill_pct,
        mean_fold_skill_pct=_number(diagnostic.get("mean_fold_skill_pct"), "mean fold skill"),
    )


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}%"


def _fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _month_range(start: datetime, end: datetime) -> str:
    start_label = start.strftime("%b %Y")
    end_label = end.strftime("%b %Y")
    if start_label == end_label:
        return start_label
    if start.year == end.year:
        return f"{start.strftime('%b')}-{end.strftime('%b %Y')}"
    return f"{start.strftime('%b %Y')}-{end.strftime('%b %Y')}"


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    fill: str = "#24313a",
    weight: str = "400",
    anchor: str = "start",
    family: str = "system",
) -> str:
    font_family = (
        "ui-monospace, SFMono-Regular, Consolas, monospace"
        if family == "mono"
        else "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{font_family}" '
        f'font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}">{_esc(value)}</text>'
    )


def render_svg(evidence: FigureEvidence) -> str:
    """Render a deterministic, responsive SVG from validated evidence."""

    bg = "#fbfbf8"
    ink = "#24313a"
    muted = "#647078"
    rule = "#d9ddd9"
    grid = "#e8ebe7"
    positive = "#3d7288"
    negative = "#a4776a"
    plot_left = 220.0
    plot_width = 570.0
    plot_top = 158.0
    row_height = 32.0
    bar_height = 17.0
    plot_bottom = plot_top + len(evidence.folds) * row_height
    axis_y = plot_bottom + 7.0
    domain_min = -30.0
    domain_max = 30.0
    scale = plot_width / (domain_max - domain_min)
    zero_x = plot_left + (0.0 - domain_min) * scale

    def x_value(value: float) -> float:
        return plot_left + (value - domain_min) * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-labelledby="figure-title figure-desc">',
        '<title id="figure-title">Fold-level XGBoost MAE skill versus the SMARD wind forecast</title>',
        f'<desc id="figure-desc">12 chronological rolling-calibration folds. Seven folds improve and five worsen. Pooled skill is {_fmt_pct(evidence.pooled_skill_pct)}, while the Newey-West 95 percent skill interval is {_fmt_pct(evidence.nw_ci_lower_pct)} to {_fmt_pct(evidence.nw_ci_upper_pct)} and crosses zero.</desc>',
        f'<rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="{bg}"/>',
        _text(36, 43, "XGBoost residual calibration: fold-level MAE skill", size=25, fill=ink, weight="650"),
        _text(36, 70, "German wind | rolling-calibration reference through Jun 2026 | positive = lower MAE than SMARD", size=14, fill=muted),
        f'<line x1="36" y1="92" x2="1164" y2="92" stroke="{rule}" stroke-width="1"/>',
        _text(36, 126, "Chronological fold skill", size=16, fill=ink, weight="650"),
        _text(790, 126, f"{evidence.fold_wins} improved | {len(evidence.folds) - evidence.fold_wins} worse", size=13, fill=muted, anchor="end", family="mono"),
        f'<line x1="820" y1="112" x2="820" y2="640" stroke="{rule}" stroke-width="1"/>',
        _text(850, 126, "Pooled estimate and uncertainty", size=16, fill=ink, weight="650"),
    ]

    for tick in (-20, -10, 0, 10, 20):
        x = x_value(tick)
        stroke = ink if tick == 0 else grid
        width = 1.5 if tick == 0 else 1.0
        parts.append(f'<line x1="{x:.1f}" y1="{plot_top - 13:.1f}" x2="{x:.1f}" y2="{axis_y:.1f}" stroke="{stroke}" stroke-width="{width}"/>')
        parts.append(_text(x, axis_y + 22, f"{tick:+d}%" if tick else "0%", size=12, fill=muted, anchor="middle", family="mono"))
    parts.append(_text(zero_x, plot_top - 21, "zero reference", size=11, fill=muted, anchor="middle"))

    for index, fold in enumerate(evidence.folds):
        center_y = plot_top + index * row_height + row_height / 2
        endpoint = x_value(fold.skill_pct)
        bar_x = min(zero_x, endpoint)
        bar_width = abs(endpoint - zero_x)
        color = positive if fold.skill_pct > 0 else negative
        label = f"F{fold.fold:02d} | {_month_range(fold.fold_start, fold.fold_end)}"
        parts.append(_text(36, center_y + 5, label, size=12, fill=ink, family="mono"))
        parts.append(f'<rect x="{bar_x:.1f}" y="{center_y - bar_height / 2:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
        if fold.skill_pct >= 0:
            parts.append(_text(endpoint + 9, center_y + 5, _fmt_pct(fold.skill_pct, 1), size=12, fill=color, weight="650", family="mono"))
        else:
            parts.append(_text(endpoint - 9, center_y + 5, _fmt_pct(fold.skill_pct, 1), size=12, fill=color, weight="650", anchor="end", family="mono"))

    parts.extend(
        [
            _text(plot_left + plot_width / 2, axis_y + 50, "MAE skill versus SMARD forecast (%)", size=13, fill=muted, anchor="middle"),
            _text(850, 163, "95% skill interval", size=13, fill=muted),
        ]
    )

    interval_left = 850.0
    interval_width = 310.0
    interval_min = -4.0
    interval_max = 4.0

    def interval_x(value: float) -> float:
        return interval_left + (value - interval_min) / (interval_max - interval_min) * interval_width

    interval_y = 202.0
    interval_zero_x = interval_x(0.0)
    ci_left = interval_x(evidence.nw_ci_lower_pct)
    ci_right = interval_x(evidence.nw_ci_upper_pct)
    pooled_x = interval_x(evidence.pooled_skill_pct)
    parts.extend(
        [
            f'<line x1="{interval_left:.1f}" y1="{interval_y:.1f}" x2="{interval_left + interval_width:.1f}" y2="{interval_y:.1f}" stroke="{grid}" stroke-width="1"/>',
            f'<line x1="{interval_zero_x:.1f}" y1="{interval_y - 19:.1f}" x2="{interval_zero_x:.1f}" y2="{interval_y + 19:.1f}" stroke="{ink}" stroke-width="1.5"/>',
            f'<line x1="{ci_left:.1f}" y1="{interval_y:.1f}" x2="{ci_right:.1f}" y2="{interval_y:.1f}" stroke="{ink}" stroke-width="2"/>',
            f'<line x1="{ci_left:.1f}" y1="{interval_y - 8:.1f}" x2="{ci_left:.1f}" y2="{interval_y + 8:.1f}" stroke="{ink}" stroke-width="2"/>',
            f'<line x1="{ci_right:.1f}" y1="{interval_y - 8:.1f}" x2="{ci_right:.1f}" y2="{interval_y + 8:.1f}" stroke="{ink}" stroke-width="2"/>',
            f'<circle cx="{pooled_x:.1f}" cy="{interval_y:.1f}" r="6" fill="{bg}" stroke="{ink}" stroke-width="2"/>',
            _text(ci_left, interval_y + 32, _fmt_pct(evidence.nw_ci_lower_pct), size=12, fill=ink, anchor="middle", family="mono"),
            _text(ci_right, interval_y + 32, _fmt_pct(evidence.nw_ci_upper_pct), size=12, fill=ink, anchor="middle", family="mono"),
            _text(interval_zero_x, interval_y - 27, "0%", size=11, fill=muted, anchor="middle", family="mono"),
            _text(850, 282, _fmt_pct(evidence.pooled_skill_pct), size=29, fill=ink, weight="650", family="mono"),
            _text(850, 303, "pooled skill vs SMARD", size=13, fill=muted),
            f'<line x1="850" y1="322" x2="1160" y2="322" stroke="{rule}" stroke-width="1"/>',
            _text(850, 349, "XGBoost MAE", size=13, fill=muted),
            _text(1160, 349, f"{_fmt_num(evidence.xgboost_mae)} MW", size=14, fill=ink, weight="650", anchor="end", family="mono"),
            _text(850, 380, "SMARD forecast MAE", size=13, fill=muted),
            _text(1160, 380, f"{_fmt_num(evidence.smard_mae)} MW", size=14, fill=ink, weight="650", anchor="end", family="mono"),
            _text(850, 411, "Out-of-sample hours", size=13, fill=muted),
            _text(1160, 411, f"{evidence.observations:,}", size=14, fill=ink, weight="650", anchor="end", family="mono"),
            _text(850, 442, "Folds improved", size=13, fill=muted),
            _text(1160, 442, f"{evidence.fold_wins} / {len(evidence.folds)}", size=14, fill=ink, weight="650", anchor="end", family="mono"),
            f'<line x1="850" y1="464" x2="1160" y2="464" stroke="{rule}" stroke-width="1"/>',
            _text(850, 490, "Day-block bootstrap skill interval", size=12, fill=muted),
            _text(1160, 490, f"{_fmt_pct(evidence.bootstrap_ci_lower_pct)} to {_fmt_pct(evidence.bootstrap_ci_upper_pct)}", size=12, fill=ink, weight="650", anchor="end", family="mono"),
            _text(850, 522, "Mean fold skill", size=12, fill=muted),
            _text(1160, 522, _fmt_pct(evidence.mean_fold_skill_pct), size=12, fill=ink, weight="650", anchor="end", family="mono"),
            _text(850, 553, f"Best fold: F{evidence.best_fold:02d} {_fmt_pct(evidence.best_fold_skill_pct, 2)}", size=12, fill=muted, family="mono"),
            _text(850, 578, f"Worst fold: F{evidence.worst_fold:02d} {_fmt_pct(evidence.worst_fold_skill_pct, 2)}", size=12, fill=muted, family="mono"),
        ]
    )

    parts.extend(
        [
            f'<line x1="36" y1="650" x2="1164" y2="650" stroke="{rule}" stroke-width="1"/>',
            _text(36, 680, "Interpretation: Average error was marginally lower, but the interval crosses zero and results were inconsistent across folds.", size=14, fill=ink, weight="550"),
            _text(36, 708, f"Pooled skill uses aggregate MAEs, not the mean of fold percentages (mean fold skill {_fmt_pct(evidence.mean_fold_skill_pct)}).", size=12, fill=muted),
            _text(36, 739, "Historical rolling-calibration reference only. Inputs are not publication-vintage controlled to the frozen 11:00 D-1 auction gate; this is not prospective accuracy or trading evidence.", size=11, fill=muted),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def generate_figure(repo_root: Path, output_path: Path) -> FigureEvidence:
    evidence = load_evidence(Path(repo_root))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(evidence), encoding="utf-8", newline="")
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the README fold-skill evidence figure.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "assets" / "fold_skill_vs_smard.svg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        evidence = generate_figure(args.repo_root, args.output)
    except EvidenceError as exc:
        raise SystemExit(f"README figure evidence error: {exc}") from exc
    print(
        f"Generated {args.output} from {len(evidence.folds)} folds and "
        f"{evidence.observations:,} out-of-sample hours."
    )


if __name__ == "__main__":
    main()
