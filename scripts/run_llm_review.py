from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.llm_review import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    run_llm_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft a grounded AI analyst review from pipeline outputs.")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "llm")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default=DEFAULT_REASONING_EFFORT,
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(ROOT / ".env.local", override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Add it to .env.local before running the LLM review.")

    args = parse_args()
    paths = run_llm_review(
        outputs_dir=args.outputs_dir,
        output_dir=args.output_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    print("AI analyst review complete.")
    print(f"Markdown: {paths['markdown']}")
    print(f"Audit log: {paths['log']}")


if __name__ == "__main__":
    main()
