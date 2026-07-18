from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.holdout import (  # noqa: E402
    HoldoutPolicyError,
    create_holdout_release_manifest,
    verify_holdout_release_manifest,
)


def _current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or verify the WG-D1-001 holdout release seal.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Hash-seal a promoted model manifest and predictions.")
    freeze.add_argument("--model-manifest", type=Path, required=True)
    freeze.add_argument("--predictions", type=Path, required=True)
    freeze.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manifests" / "holdout_release_manifest.json",
    )
    freeze.add_argument("--git-commit", help="Full code commit; defaults to the current Git HEAD.")

    verify = subparsers.add_parser("verify", help="Verify the seal immediately before target scoring.")
    verify.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "manifests" / "holdout_release_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "freeze":
            manifest, sidecar = create_holdout_release_manifest(
                project_root=ROOT,
                model_manifest_path=args.model_manifest,
                prediction_artifact_path=args.predictions,
                output_path=args.output,
                git_commit=args.git_commit or _current_commit(),
            )
            print(f"Frozen release manifest: {manifest}")
            print(f"SHA-256 sidecar: {sidecar}")
            print("Commit both files before any target scoring so Git records the external timestamp.")
        else:
            started_at = datetime.now(timezone.utc)
            verify_holdout_release_manifest(
                args.manifest,
                project_root=ROOT,
                scoring_started_at_utc=started_at,
                now_utc=started_at,
            )
            print(f"Holdout release verified for scoring at {started_at.isoformat()}.")
    except (HoldoutPolicyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Holdout protection: {exc}") from exc


if __name__ == "__main__":
    main()
