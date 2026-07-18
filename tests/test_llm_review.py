from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.llm_review import (  # noqa: E402
    AnalystReview,
    EvidenceBackedFinding,
    build_review_evidence,
    run_llm_review,
    validate_review_grounding,
)


def _review(evidence_id: str = "forecast.mae_comparison") -> AnalystReview:
    return AnalystReview(
        headline="No robust incremental edge",
        overall_conclusion="The evidence supports continued research rather than deployment.",
        forecast_assessment=EvidenceBackedFinding(
            assessment="The residual model does not reliably improve the public forecast.",
            evidence_ids=[evidence_id, "forecast.significance"],
        ),
        strategy_assessment=EvidenceBackedFinding(
            assessment="The paper strategy result is not statistically reliable or executable.",
            evidence_ids=["strategy.residual_result", "strategy.price_proxy"],
        ),
        risk_flags=[
            EvidenceBackedFinding(
                assessment="Performance varies materially by validation fold.",
                evidence_ids=["forecast.fold_dispersion"],
            ),
            EvidenceBackedFinding(
                assessment="The information set is not pinned to auction gate closure.",
                evidence_ids=["design.information_set"],
            ),
        ],
        invalidation_conditions=[
            "Invalidate any day-ahead claim until features are cut at D-1 noon.",
            "Invalidate deployment if the signal fails on executable settlement marks.",
        ],
        production_next_steps=[
            "Rebuild weather and lag inputs at explicit issue timestamps.",
            "Backtest against executable intraday or imbalance prices over multiple years.",
        ],
    )


class FakeResponses:
    def __init__(self, review: AnalystReview) -> None:
        self.review = review
        self.kwargs: dict[str, object] = {}

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 700, "output_tokens": 250})
        return SimpleNamespace(
            id="resp_test",
            _request_id="req_test",
            model="gpt-5.6-luna-2026-07-01",
            output_parsed=self.review,
            usage=usage,
        )


class FakeClient:
    def __init__(self, review: AnalystReview) -> None:
        self.responses = FakeResponses(review)


def test_evidence_is_reconciled_and_contains_pipeline_boundaries() -> None:
    evidence = build_review_evidence(ROOT / "outputs")
    evidence_map = {item["id"]: item for item in evidence}

    assert "forecast.mae_comparison" in evidence_map
    assert "strategy.fold_concentration" in evidence_map
    assert "strategy.price_proxy" in evidence_map
    assert "design.llm_boundary" in evidence_map
    assert "no influence" in evidence_map["design.llm_boundary"]["value"]
    assert "95% CI" in evidence_map["forecast.significance"]["value"]


def test_unknown_evidence_id_is_rejected() -> None:
    evidence = build_review_evidence(ROOT / "outputs")

    with pytest.raises(ValueError, match="unknown evidence IDs"):
        validate_review_grounding(_review("forecast.not_a_real_metric"), evidence)


def test_llm_review_logs_prompt_response_and_dashboard_artifact(tmp_path: Path) -> None:
    client = FakeClient(_review())
    generated_at = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

    paths = run_llm_review(
        ROOT / "outputs",
        tmp_path,
        client=client,
        generated_at=generated_at,
    )

    assert all(path.exists() for path in paths.values())
    artifact = json.loads(paths["artifact"].read_text(encoding="utf-8"))
    run_log = json.loads(paths["log"].read_text(encoding="utf-8"))
    assert artifact["review"]["headline"] == "No robust incremental edge"
    assert artifact["boundary"].startswith("Downstream summarization only")
    assert len(artifact["evidence_fingerprint"]) == 64
    assert run_log["response_id"] == "resp_test"
    assert run_log["request_id"] == "req_test"
    assert run_log["prompts"]["system"]
    assert run_log["prompts"]["user"]
    assert run_log["usage"]["input_tokens"] == 700
    assert client.responses.kwargs["store"] is False
    assert client.responses.kwargs["reasoning"] == {"effort": "low"}
    assert client.responses.kwargs["text_format"] is AnalystReview
