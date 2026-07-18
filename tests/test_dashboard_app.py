from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_renders_from_tracked_outputs() -> None:
    strategy_metrics = pd.read_csv(ROOT / "outputs" / "strategy_metrics.csv").set_index("strategy")
    diagnostics = pd.read_csv(ROOT / "outputs" / "forecast_diagnostics.csv").iloc[0]
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "German Wind Forecast Validation"
    assert len(app.metric) == 7
    assert "No reliable incremental edge" in app.info[0].value
    assert "not evidence of a free executable edge" in app.info[0].value
    assert any(
        f"{int(diagnostics['fold_count'])} expanding walk-forward folds" in caption.value
        for caption in app.caption
    )
    assert any("95% skill CI" in caption.value for caption in app.caption)
    assert app.metric[1].value == (
        f"{strategy_metrics.loc['xgboost_wind_residual_signal', 'avg_net_pnl_eur_mwh']:+.3f}"
    )
    assert app.metric[2].value == (
        f"{strategy_metrics.loc['public_smard_forecast_ramp_signal', 'avg_net_pnl_eur_mwh']:+.3f}"
    )
    residual_total = strategy_metrics.loc[
        "xgboost_wind_residual_signal", "total_net_pnl_eur_per_mw_clip"
    ]
    public_total = strategy_metrics.loc[
        "public_smard_forecast_ramp_signal", "total_net_pnl_eur_per_mw_clip"
    ]
    assert any(
        f"{public_total:,.2f} EUR" in caption.value and f"{residual_total:,.2f} EUR" in caption.value
        for caption in app.caption
    )
    assert any(subheader.value == "AI analyst review" for subheader in app.subheader)
    artifact_path = ROOT / "outputs" / "llm" / "analyst_review.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        headline_rendered = any(artifact["review"]["headline"] in markdown.value for markdown in app.markdown)
        stale_warning_rendered = any(
            "different pipeline output snapshot" in warning.value for warning in app.warning
        )
        assert headline_rendered or stale_warning_rendered
