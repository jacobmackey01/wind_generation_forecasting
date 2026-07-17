from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.llm_review import (  # noqa: E402
    AnalystReview,
    build_review_evidence,
    evidence_fingerprint,
    validate_review_grounding,
)


def test_static_ai_review_matches_original_site_snapshot() -> None:
    site_data = ROOT / "site" / "data"
    artifact = json.loads((site_data / "llm" / "analyst_review.json").read_text(encoding="utf-8"))
    evidence = build_review_evidence(site_data)
    review = AnalystReview.model_validate(artifact["review"])

    assert artifact["evidence_fingerprint"] == evidence_fingerprint(evidence)
    assert artifact["evidence"] == evidence
    validate_review_grounding(review, evidence)


def test_static_site_renders_saved_review_without_browser_api_credentials() -> None:
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "site" / "app.js").read_text(encoding="utf-8")
    stylesheet = (ROOT / "site" / "styles.css").read_text(encoding="utf-8")

    assert 'id="ai-heading"' in html
    assert 'id="ai-forecast-evidence"' in html
    assert 'id="ai-next-steps"' in html
    assert 'loadJson("llm/analyst_review.json")' in javascript
    assert "textContent" in javascript
    assert "innerHTML" not in javascript
    assert ".evidence-grid > *" in stylesheet
    assert "max-width: 100% !important" in stylesheet

    published_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "site").rglob("*")
        if path.is_file()
    )
    assert "OPENAI_API_KEY" not in published_text
    assert "sk-proj-" not in published_text
