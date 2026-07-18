from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.holdout import (  # noqa: E402
    HoldoutPolicyError,
    create_holdout_release_manifest,
    enforce_development_window,
    verify_holdout_release_manifest,
)


def _release_files(tmp_path: Path, *, promotion_status: str = "promoted") -> tuple[Path, Path, Path]:
    model_manifest = tmp_path / "models" / "model_manifest.json"
    predictions = tmp_path / "predictions" / "holdout_predictions.csv"
    release_manifest = tmp_path / "manifests" / "holdout_release_manifest.json"
    model_manifest.parent.mkdir()
    predictions.parent.mkdir()
    model_manifest.write_text(
        json.dumps(
            {
                "specification_id": "WG-D1-001",
                "promotion_status": promotion_status,
                "promoted_at_utc": "2026-06-30T08:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text("issue_date,y_pred\n2026-07-01,10000\n", encoding="utf-8")
    return model_manifest, predictions, release_manifest


def _freeze(
    tmp_path: Path,
    *,
    frozen_at_utc: datetime = datetime(2026, 9, 30, 20, 0, tzinfo=timezone.utc),
) -> Path:
    model_manifest, predictions, release_manifest = _release_files(tmp_path)
    create_holdout_release_manifest(
        project_root=tmp_path,
        model_manifest_path=model_manifest,
        prediction_artifact_path=predictions,
        output_path=release_manifest,
        git_commit="a" * 40,
        frozen_at_utc=frozen_at_utc,
    )
    return release_manifest


def test_development_pipeline_blocks_any_holdout_overlap() -> None:
    enforce_development_window("2025-01-01", "2026-06-30")
    enforce_development_window("2026-10-01", "2026-10-31")

    for start, end in [
        ("2025-01-01", "2026-07-01"),
        ("2026-07-01", "2026-09-30"),
        ("2026-09-30", "2026-10-31"),
    ]:
        with pytest.raises(HoldoutPolicyError, match="locked .* prospective holdout"):
            enforce_development_window(start, end)

    with pytest.raises(HoldoutPolicyError, match="precedes start"):
        enforce_development_window("2026-06-30", "2026-01-01")


def test_pipeline_cli_rejects_holdout_before_network_or_llm_work() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pipeline.py"),
            "--start",
            "2026-07-01",
            "--end",
            "2026-07-02",
            "--skip-llm-review",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode != 0
    assert "Holdout protection" in result.stderr
    assert "Fetching SMARD" not in result.stdout


def test_release_seal_verifies_only_after_embargo(tmp_path: Path) -> None:
    manifest = _freeze(tmp_path)

    with pytest.raises(HoldoutPolicyError, match="embargoed"):
        verify_holdout_release_manifest(
            manifest,
            project_root=tmp_path,
            scoring_started_at_utc="2026-09-30T12:00:00Z",
            now_utc=datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
        )

    payload = verify_holdout_release_manifest(
        manifest,
        project_root=tmp_path,
        scoring_started_at_utc="2026-10-01T08:00:00Z",
        now_utc=datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc),
    )
    assert payload["specification_id"] == "WG-D1-001"


def test_release_seal_detects_artifact_tampering(tmp_path: Path) -> None:
    manifest = _freeze(tmp_path)
    predictions = tmp_path / "predictions" / "holdout_predictions.csv"
    predictions.write_text("issue_date,y_pred\n2026-07-01,99999\n", encoding="utf-8")

    with pytest.raises(HoldoutPolicyError, match="predictions artifact"):
        verify_holdout_release_manifest(
            manifest,
            project_root=tmp_path,
            scoring_started_at_utc="2026-10-01T08:00:00Z",
            now_utc=datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc),
        )


def test_release_seal_detects_manifest_tampering(tmp_path: Path) -> None:
    manifest = _freeze(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["release"]["git_commit"] = "d" * 40
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HoldoutPolicyError, match="no longer matches"):
        verify_holdout_release_manifest(
            manifest,
            project_root=tmp_path,
            scoring_started_at_utc="2026-10-01T08:00:00Z",
            now_utc=datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc),
        )


def test_release_must_be_frozen_before_scoring(tmp_path: Path) -> None:
    manifest = _freeze(
        tmp_path,
        frozen_at_utc=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(HoldoutPolicyError, match="frozen before"):
        verify_holdout_release_manifest(
            manifest,
            project_root=tmp_path,
            scoring_started_at_utc="2026-10-01T08:00:00Z",
            now_utc=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        )


def test_unpromoted_model_cannot_be_sealed(tmp_path: Path) -> None:
    model_manifest, predictions, release_manifest = _release_files(tmp_path, promotion_status="candidate")

    with pytest.raises(HoldoutPolicyError, match="promotion_status='promoted'"):
        create_holdout_release_manifest(
            project_root=tmp_path,
            model_manifest_path=model_manifest,
            prediction_artifact_path=predictions,
            output_path=release_manifest,
            git_commit="b" * 40,
        )


def test_prediction_artifact_cannot_contain_targets(tmp_path: Path) -> None:
    model_manifest, predictions, release_manifest = _release_files(tmp_path)
    predictions.write_text("issue_date,y_pred,actual_wind_total_mw\n2026-07-01,10000,11000\n", encoding="utf-8")

    with pytest.raises(HoldoutPolicyError, match="contains target columns"):
        create_holdout_release_manifest(
            project_root=tmp_path,
            model_manifest_path=model_manifest,
            prediction_artifact_path=predictions,
            output_path=release_manifest,
            git_commit="c" * 40,
        )


def test_model_promoted_after_holdout_start_cannot_be_sealed(tmp_path: Path) -> None:
    model_manifest, predictions, release_manifest = _release_files(tmp_path)
    payload = json.loads(model_manifest.read_text(encoding="utf-8"))
    payload["promoted_at_utc"] = "2026-07-01T08:00:00Z"
    model_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HoldoutPolicyError, match="predate"):
        create_holdout_release_manifest(
            project_root=tmp_path,
            model_manifest_path=model_manifest,
            prediction_artifact_path=predictions,
            output_path=release_manifest,
            git_commit="e" * 40,
        )
