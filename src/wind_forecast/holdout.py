"""Prospective-holdout policy and hash-sealed release manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SPECIFICATION_ID = "WG-D1-001"
HOLDOUT_START_LOCAL = date(2026, 7, 1)
HOLDOUT_END_LOCAL = date(2026, 9, 30)
SCORE_NOT_BEFORE_LOCAL = date(2026, 10, 1)
MARKET_TIMEZONE = ZoneInfo("Europe/Berlin")
MANIFEST_SCHEMA_VERSION = 1
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_PREDICTION_COLUMNS = {
    "actual",
    "actual_wind_total_mw",
    "observed",
    "target",
    "y_true",
}


class HoldoutPolicyError(RuntimeError):
    """Raised when a command would violate the prospective-holdout policy."""


def _parse_local_date(value: str | date, label: str) -> date:
    if isinstance(value, datetime):
        raise HoldoutPolicyError(f"{label} must be a local calendar date, not a timestamp.")
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise HoldoutPolicyError(f"{label} must use YYYY-MM-DD; received {value!r}.") from exc


def _parse_utc_timestamp(value: str | datetime, label: str) -> datetime:
    if isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise HoldoutPolicyError(f"{label} must be an ISO-8601 timestamp.") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise HoldoutPolicyError(f"{label} must be an ISO-8601 timestamp.")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HoldoutPolicyError(f"{label} must include a timezone offset.")
    return parsed.astimezone(timezone.utc)


def range_overlaps_locked_holdout(start: str | date, end: str | date) -> bool:
    """Return whether an inclusive local-date range touches the locked holdout."""

    start_date = _parse_local_date(start, "start")
    end_date = _parse_local_date(end, "end")
    if end_date < start_date:
        raise HoldoutPolicyError(f"end {end_date} precedes start {start_date}.")
    return start_date <= HOLDOUT_END_LOCAL and end_date >= HOLDOUT_START_LOCAL


def enforce_development_window(start: str | date, end: str | date) -> None:
    """Block the research pipeline from reading or scoring locked holdout targets."""

    if range_overlaps_locked_holdout(start, end):
        raise HoldoutPolicyError(
            f"requested range {start} to {end} overlaps the locked {HOLDOUT_START_LOCAL} to "
            f"{HOLDOUT_END_LOCAL} prospective holdout. The research pipeline cannot ingest or score "
            "that period; use the frozen holdout release workflow after the scoring embargo."
        )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path, project_root: Path, label: str) -> tuple[Path, str]:
    root = project_root.resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise HoldoutPolicyError(f"{label} must be inside project root {root}.") from exc
    return resolved, relative.as_posix()


def _load_model_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutPolicyError(f"Could not read model manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HoldoutPolicyError("The model manifest must contain a JSON object.")
    if payload.get("specification_id") != SPECIFICATION_ID:
        raise HoldoutPolicyError(f"The model manifest must declare specification_id={SPECIFICATION_ID}.")
    if payload.get("promotion_status") != "promoted":
        raise HoldoutPolicyError("The model manifest must have promotion_status='promoted'.")
    promoted_at = _parse_utc_timestamp(payload.get("promoted_at_utc"), "model promoted_at_utc")
    if promoted_at.astimezone(MARKET_TIMEZONE).date() >= HOLDOUT_START_LOCAL:
        raise HoldoutPolicyError("The promoted model must predate the prospective holdout in market time.")
    return payload


def _validate_prediction_artifact(path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise HoldoutPolicyError("The prospective prediction artifact must be a CSV file.")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, StopIteration, csv.Error) as exc:
        raise HoldoutPolicyError(f"Could not read the prediction artifact header: {exc}") from exc
    normalized = {column.strip().lower() for column in header}
    leaked_targets = sorted(normalized.intersection(_FORBIDDEN_PREDICTION_COLUMNS))
    if leaked_targets:
        raise HoldoutPolicyError(
            f"The prospective prediction artifact contains target columns: {leaked_targets}."
        )


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _sidecar_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.name}.sha256")


def create_holdout_release_manifest(
    *,
    project_root: Path,
    model_manifest_path: Path,
    prediction_artifact_path: Path,
    output_path: Path,
    git_commit: str,
    frozen_at_utc: datetime | None = None,
) -> tuple[Path, Path]:
    """Seal promoted-model metadata and target-free predictions before scoring."""

    root = Path(project_root).resolve()
    model_path, model_relative = _project_relative(Path(model_manifest_path), root, "model manifest")
    prediction_path, prediction_relative = _project_relative(
        Path(prediction_artifact_path), root, "prediction artifact"
    )
    manifest_path, _ = _project_relative(Path(output_path), root, "release manifest")
    sidecar_path = _sidecar_path(manifest_path)

    if not model_path.is_file():
        raise HoldoutPolicyError(f"Model manifest does not exist: {model_path}")
    if not prediction_path.is_file():
        raise HoldoutPolicyError(f"Prediction artifact does not exist: {prediction_path}")
    if manifest_path.exists() or sidecar_path.exists():
        raise HoldoutPolicyError("Release manifest already exists; release seals are immutable and cannot be overwritten.")
    normalized_commit = git_commit.strip().lower()
    if not _COMMIT_PATTERN.fullmatch(normalized_commit):
        raise HoldoutPolicyError("git_commit must be a full 40-character hexadecimal commit hash.")

    _load_model_manifest(model_path)
    _validate_prediction_artifact(prediction_path)
    frozen_at = _parse_utc_timestamp(frozen_at_utc or datetime.now(timezone.utc), "frozen_at_utc")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "specification_id": SPECIFICATION_ID,
        "holdout": {
            "start_local": HOLDOUT_START_LOCAL.isoformat(),
            "end_local": HOLDOUT_END_LOCAL.isoformat(),
            "score_not_before_local": SCORE_NOT_BEFORE_LOCAL.isoformat(),
            "timezone": str(MARKET_TIMEZONE),
        },
        "release": {
            "frozen_at_utc": frozen_at.isoformat().replace("+00:00", "Z"),
            "git_commit": normalized_commit,
        },
        "artifacts": {
            "model_manifest": {"path": model_relative, "sha256": sha256_file(model_path)},
            "predictions": {"path": prediction_relative, "sha256": sha256_file(prediction_path)},
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_canonical_json(payload), encoding="utf-8", newline="")
    manifest_hash = sha256_file(manifest_path)
    sidecar_path.write_text(f"{manifest_hash}  {manifest_path.name}\n", encoding="ascii", newline="")
    return manifest_path, sidecar_path


def verify_holdout_release_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    scoring_started_at_utc: str | datetime,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Verify the release seal, embargo date, and artifact hashes before scoring."""

    root = Path(project_root).resolve()
    resolved_manifest, _ = _project_relative(Path(manifest_path), root, "release manifest")
    sidecar_path = _sidecar_path(resolved_manifest)
    if not resolved_manifest.is_file() or not sidecar_path.is_file():
        raise HoldoutPolicyError("Both the release manifest and its .sha256 sidecar are required.")

    sidecar_parts = sidecar_path.read_text(encoding="ascii").strip().split()
    if len(sidecar_parts) != 2 or sidecar_parts[1] != resolved_manifest.name:
        raise HoldoutPolicyError("The release-manifest SHA-256 sidecar is malformed.")
    if sidecar_parts[0] != sha256_file(resolved_manifest):
        raise HoldoutPolicyError("The release manifest no longer matches its SHA-256 seal.")

    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HoldoutPolicyError("The release manifest is not valid JSON.") from exc
    expected_policy = {
        "start_local": HOLDOUT_START_LOCAL.isoformat(),
        "end_local": HOLDOUT_END_LOCAL.isoformat(),
        "score_not_before_local": SCORE_NOT_BEFORE_LOCAL.isoformat(),
        "timezone": str(MARKET_TIMEZONE),
    }
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise HoldoutPolicyError("Unsupported release-manifest schema version.")
    if payload.get("specification_id") != SPECIFICATION_ID or payload.get("holdout") != expected_policy:
        raise HoldoutPolicyError("The release manifest does not match the frozen holdout policy.")

    scoring_started = _parse_utc_timestamp(scoring_started_at_utc, "scoring_started_at_utc")
    now = _parse_utc_timestamp(now_utc or datetime.now(timezone.utc), "now_utc")
    if scoring_started > now:
        raise HoldoutPolicyError("scoring_started_at_utc cannot be in the future.")
    if scoring_started.astimezone(MARKET_TIMEZONE).date() < SCORE_NOT_BEFORE_LOCAL:
        raise HoldoutPolicyError(f"Target scoring is embargoed until {SCORE_NOT_BEFORE_LOCAL} market time.")
    frozen_at = _parse_utc_timestamp(payload.get("release", {}).get("frozen_at_utc"), "frozen_at_utc")
    if frozen_at >= scoring_started:
        raise HoldoutPolicyError("The release manifest must be frozen before target scoring starts.")

    commit = str(payload.get("release", {}).get("git_commit", ""))
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise HoldoutPolicyError("The release manifest contains an invalid git commit hash.")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HoldoutPolicyError("The release manifest is missing artifact records.")
    for name in ("model_manifest", "predictions"):
        record = artifacts.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise HoldoutPolicyError(f"The release manifest is missing the {name} artifact record.")
        artifact_path, _ = _project_relative(root / record["path"], root, name)
        if not artifact_path.is_file() or sha256_file(artifact_path) != record.get("sha256"):
            raise HoldoutPolicyError(f"The {name} artifact does not match the frozen SHA-256 hash.")

    model_record = artifacts["model_manifest"]
    _load_model_manifest(root / model_record["path"])
    _validate_prediction_artifact(root / artifacts["predictions"]["path"])
    return payload
