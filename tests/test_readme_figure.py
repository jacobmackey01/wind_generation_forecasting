from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_readme_figure import EvidenceError, generate_figure, load_evidence  # noqa: E402


def _copy_evidence_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    outputs = repo / "outputs"
    outputs.mkdir(parents=True)
    for filename in ("fold_metrics.csv", "metrics.csv", "forecast_diagnostics.csv"):
        shutil.copy2(ROOT / "outputs" / filename, outputs / filename)
    return repo


def _rewrite_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_readme_figure_is_deterministic_and_uses_current_evidence(tmp_path: Path) -> None:
    first_path = tmp_path / "first.svg"
    second_path = tmp_path / "second.svg"

    evidence = generate_figure(ROOT, first_path)
    generate_figure(ROOT, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert len(evidence.folds) == 12
    assert evidence.observations == 8_615
    assert evidence.fold_wins == 7
    assert sum(fold.skill_pct < 0 for fold in evidence.folds) == 5
    assert evidence.pooled_skill_pct == pytest.approx(0.3430735894656111)
    assert evidence.mean_fold_skill_pct == pytest.approx(-0.4875015597296727)
    assert evidence.pooled_skill_pct != pytest.approx(evidence.mean_fold_skill_pct)
    assert evidence.nw_ci_lower_pct < 0 < evidence.nw_ci_upper_pct
    assert evidence.bootstrap_ci_lower_pct < 0 < evidence.bootstrap_ci_upper_pct

    svg = first_path.read_text(encoding="utf-8")
    for required in (
        "12 chronological",
        "+0.34%",
        "-2.91%",
        "+3.60%",
        "-2.94%",
        "+3.62%",
        "1,384.67 MW",
        "1,389.44 MW",
        "8,615",
        "7 / 12",
        "+13.30%",
        "-26.44%",
        "Average error was marginally lower",
        "Historical rolling-calibration reference only",
        "not prospective accuracy or trading evidence",
    ):
        assert required in svg


def test_incomplete_fold_model_pair_is_rejected(tmp_path: Path) -> None:
    repo = _copy_evidence_repo(tmp_path)
    path = repo / "outputs" / "fold_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows = [row for row in rows if not (row["fold"] == "1" and row["model"] == "xgboost_residual")]
    _rewrite_csv(path, rows)

    with pytest.raises(EvidenceError, match="missing expected model rows"):
        load_evidence(repo)


def test_duplicate_fold_model_pair_is_rejected(tmp_path: Path) -> None:
    repo = _copy_evidence_repo(tmp_path)
    path = repo / "outputs" / "fold_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows.append(dict(rows[0]))
    _rewrite_csv(path, rows)

    with pytest.raises(EvidenceError, match="duplicate fold/model"):
        load_evidence(repo)


def test_stored_fold_skill_must_reconcile_to_mae(tmp_path: Path) -> None:
    repo = _copy_evidence_repo(tmp_path)
    path = repo / "outputs" / "fold_metrics.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        if row["fold"] == "1" and row["model"] == "xgboost_residual":
            row["skill_vs_smard_mae_pct"] = "0"
    _rewrite_csv(path, rows)

    with pytest.raises(EvidenceError, match="Fold 1 skill mismatch"):
        load_evidence(repo)


def test_readme_links_to_committed_figure_and_case_study() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    figure_path = ROOT / "docs" / "assets" / "fold_skill_vs_smard.svg"
    assert figure_path.is_file()
    assert "docs/assets/fold_skill_vs_smard.svg" in readme
    assert "](docs/wind_forecast_case_study.md)" in readme
    assert "8,615 out-of-sample hours" in readme
    assert "pooled +0.34%" in readme
    assert "crosses zero" in readme
    assert "historical rolling-calibration" in readme
