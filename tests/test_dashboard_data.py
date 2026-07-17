from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.dashboard_data import (  # noqa: E402
    finding_evidence_labels,
    load_llm_review_artifact,
    prepare_fold_mae,
    prepare_strategy_fold_paths,
    residual_fold_concentration,
)
from wind_forecast.llm_review import REQUIRED_OUTPUT_FILES, build_review_evidence, evidence_fingerprint  # noqa: E402
from wind_forecast.strategy import PUBLIC_RAMP_STRATEGY, RESIDUAL_STRATEGY, TOTAL_NET_PNL  # noqa: E402


def test_dashboard_paths_and_concentration_are_derived_from_fold_outputs() -> None:
    strategy_folds = pd.DataFrame(
        {
            "strategy": [RESIDUAL_STRATEGY, RESIDUAL_STRATEGY, PUBLIC_RAMP_STRATEGY, PUBLIC_RAMP_STRATEGY],
            "fold": [1, 2, 1, 2],
            "fold_start": ["2025-10-01", "2025-11-01", "2025-10-01", "2025-11-01"],
            "trades": [10, 5, 20, 20],
            TOTAL_NET_PNL: [-30.0, 80.0, 100.0, 120.0],
        }
    )

    paths = prepare_strategy_fold_paths(strategy_folds)
    residual = paths.loc[paths["strategy"] == RESIDUAL_STRATEGY]
    concentration = residual_fold_concentration(strategy_folds)

    assert residual["cumulative_net_pnl_eur_per_mw_clip"].tolist() == [-30.0, 50.0]
    assert concentration["positive_folds"] == 1
    assert concentration["best_month"] == "Nov 2025"
    assert concentration["pnl_ex_best"] == -30.0
    assert concentration["trades_ex_best"] == 10


def test_fold_mae_keeps_only_serious_benchmarks() -> None:
    fold_metrics = pd.DataFrame(
        {
            "model": ["smard_forecast", "xgboost_residual", "persistence_prev_week"],
            "fold": [1, 1, 1],
            "fold_start": ["2025-07-01"] * 3,
            "mae": [100.0, 99.0, 500.0],
            "skill_vs_smard_mae_pct": [0.0, 1.0, -400.0],
        }
    )

    out = prepare_fold_mae(fold_metrics)

    assert set(out["model"]) == {"smard_forecast", "xgboost_residual"}
    assert set(out["month"]) == {"Jul 2025"}


def test_llm_finding_uses_only_saved_evidence_values() -> None:
    artifact = {
        "evidence": [
            {"id": "forecast.skill", "label": "Forecast skill", "value": "+0.34%"},
        ]
    }

    labels = finding_evidence_labels(artifact, {"evidence_ids": ["forecast.skill"]})

    assert labels == ["Forecast skill: +0.34%"]
    with pytest.raises(ValueError, match="unknown evidence IDs"):
        finding_evidence_labels(artifact, {"evidence_ids": ["forecast.unknown"]})


def test_dashboard_rejects_review_from_a_different_output_snapshot(tmp_path: Path) -> None:
    for filename in REQUIRED_OUTPUT_FILES:
        shutil.copy2(ROOT / "outputs" / filename, tmp_path / filename)
    evidence = build_review_evidence(tmp_path)
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    artifact = {
        "generated_at_utc": "2026-07-16T12:00:00+00:00",
        "model_returned": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "evidence_fingerprint": evidence_fingerprint(evidence),
        "evidence": evidence,
        "review": {},
        "boundary": "Downstream summarization only.",
    }
    (llm_dir / "analyst_review.json").write_text(json.dumps(artifact), encoding="utf-8")
    assert load_llm_review_artifact(tmp_path) is not None

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    metrics.loc[metrics["model"] == "xgboost_residual", "mae"] += 1.0
    metrics.to_csv(tmp_path / "metrics.csv", index=False)

    with pytest.raises(ValueError, match="different pipeline output snapshot"):
        load_llm_review_artifact(tmp_path)
