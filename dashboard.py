from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wind_forecast.dashboard_data import (  # noqa: E402
    MODEL_LABELS,
    STRATEGY_LABELS,
    finding_evidence_labels,
    load_llm_review_artifact,
    prepare_fold_mae,
    prepare_strategy_fold_paths,
    prepare_threshold_curves,
    residual_fold_concentration,
)
from wind_forecast.strategy import PUBLIC_RAMP_STRATEGY, RESIDUAL_STRATEGY, TOTAL_NET_PNL  # noqa: E402


OUTPUTS = ROOT / "outputs"
SMARD_COLOR = "#4E79A7"
XGB_COLOR = "#C45A3D"
PUBLIC_COLOR = "#2C7A5B"
POSITIVE_COLOR = "#2C7A5B"
NEGATIVE_COLOR = "#B34747"
NOVEMBER_COLOR = "#B7791F"
GRID_COLOR = "rgba(120, 128, 126, 0.22)"


st.set_page_config(
    page_title="German Wind Forecast Validation",
    page_icon=":material/air:",
    layout="wide",
)

st.markdown(
    """
    <style>
    h1 {
        font-size: 2.25rem !important;
        line-height: 1.18 !important;
        letter-spacing: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_outputs(outputs_dir: str) -> dict[str, pd.DataFrame]:
    base = Path(outputs_dir)
    names = {
        "metrics": "metrics.csv",
        "diagnostics": "forecast_diagnostics.csv",
        "fold_metrics": "fold_metrics.csv",
        "strategy_metrics": "strategy_metrics.csv",
        "strategy_folds": "strategy_fold_metrics.csv",
        "thresholds": "strategy_threshold_sensitivity.csv",
    }
    missing = [filename for filename in names.values() if not (base / filename).exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return {key: pd.read_csv(base / filename) for key, filename in names.items()}


def chart_layout(fig: go.Figure, *, height: int = 360, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        hovermode=hovermode,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


try:
    data = load_outputs(str(OUTPUTS))
except FileNotFoundError as exc:
    st.error(f"Dashboard outputs are missing: {exc}")
    st.code("python scripts/run_pipeline.py --start 2025-01-01 --end 2026-06-30")
    st.stop()

metrics = data["metrics"]
diagnostics = data["diagnostics"].iloc[0]
strategy_metrics = data["strategy_metrics"]
fold_mae = prepare_fold_mae(data["fold_metrics"])
strategy_paths = prepare_strategy_fold_paths(data["strategy_folds"])
thresholds = prepare_threshold_curves(data["thresholds"])
concentration = residual_fold_concentration(data["strategy_folds"])

xgb_metric = metrics.loc[metrics["model"] == "xgboost_residual"].iloc[0]
residual_metric = strategy_metrics.loc[strategy_metrics["strategy"] == RESIDUAL_STRATEGY].iloc[0]
public_metric = strategy_metrics.loc[strategy_metrics["strategy"] == PUBLIC_RAMP_STRATEGY].iloc[0]
per_trade_ratio = float(public_metric["avg_net_pnl_eur_mwh"] / residual_metric["avg_net_pnl_eur_mwh"])
has_forecast_interval = {
    "newey_west_skill_ci_95_lower_pct",
    "newey_west_skill_ci_95_upper_pct",
}.issubset(diagnostics.index)
if has_forecast_interval:
    forecast_interval_caption = (
        f"95% skill CI [{float(diagnostics['newey_west_skill_ci_95_lower_pct']):+.2f}%, "
        f"{float(diagnostics['newey_west_skill_ci_95_upper_pct']):+.2f}%] | "
        f"NW t = {float(diagnostics['newey_west_loss_diff_t_stat']):.2f}"
    )
else:
    forecast_interval_caption = (
        f"NW t = {float(diagnostics['newey_west_loss_diff_t_stat']):.2f} | not significant"
    )

st.title("German Wind Forecast Validation")
st.caption(
    f"XGBoost residual calibration | {int(diagnostics['fold_count'])} expanding walk-forward folds | "
    "paper wind-to-price strategy"
)

st.info(
    "No reliable incremental edge: forecast skill is within noise, residual strategy P&L is statistically weak "
    "and fold-concentrated. The large public-ramp result diagnoses proxy structure; it is not evidence of a free "
    "executable edge."
)

kpi_columns = st.columns(4)
with kpi_columns[0]:
    with st.container(border=True):
        st.metric("Forecast MAE skill vs SMARD", f"{float(xgb_metric['skill_vs_smard_mae_pct']):+.2f}%")
        st.caption(forecast_interval_caption)
with kpi_columns[1]:
    with st.container(border=True):
        st.metric("Residual net P&L / trade", f"{float(residual_metric['avg_net_pnl_eur_mwh']):+.3f}")
        st.caption(f"EUR/MWh | t = {float(residual_metric['net_pnl_t_stat']):.2f} | statistically weak")
with kpi_columns[2]:
    with st.container(border=True):
        st.metric("Public-ramp proxy / trade", f"{float(public_metric['avg_net_pnl_eur_mwh']):+.3f}")
        st.caption(f"EUR/MWh | {per_trade_ratio:.1f}x residual on this proxy")
with kpi_columns[3]:
    with st.container(border=True):
        st.metric("Positive residual folds", f"{int(concentration['positive_folds'])} / {int(concentration['fold_count'])}")
        st.caption(f"Ex-{concentration['best_month']}: {float(concentration['pnl_ex_best']):+,.2f} EUR")

st.subheader("Forecast validation")
st.markdown("#### Fold-by-fold MAE against the public forecast")

mae_fig = go.Figure()
for model, color in [("smard_forecast", SMARD_COLOR), ("xgboost_residual", XGB_COLOR)]:
    frame = fold_mae.loc[fold_mae["model"] == model]
    mae_fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["mae"],
            mode="lines+markers",
            name=MODEL_LABELS[model],
            line=dict(color=color, width=2),
            marker=dict(size=8, symbol="circle" if model == "smard_forecast" else "diamond"),
            customdata=frame[["skill_vs_smard_mae_pct"]],
            hovertemplate="%{x}<br>MAE %{y:,.0f} MW<br>Skill vs SMARD %{customdata[0]:+.1f}%<extra></extra>",
        )
    )

worst_fold = fold_mae.loc[fold_mae["model"] == "xgboost_residual"].sort_values(
    "skill_vs_smard_mae_pct"
).iloc[0]
mae_fig.add_annotation(
    x=worst_fold["month"],
    y=worst_fold["mae"],
    text=f"{worst_fold['month']} stress: {float(worst_fold['skill_vs_smard_mae_pct']):.1f}% skill",
    showarrow=True,
    arrowhead=2,
    ax=44,
    ay=-48,
)
chart_layout(mae_fig, height=390)
mae_fig.update_yaxes(title="MAE (MW)")
st.plotly_chart(mae_fig, width="stretch", config={"displayModeBar": False})
st.caption(
    f"XGBoost beat SMARD in {int(diagnostics['xgboost_fold_wins'])} of {int(diagnostics['fold_count'])} folds. "
    f"Pooled skill is {float(xgb_metric['skill_vs_smard_mae_pct']):.2f}% and the Newey-West t-stat is "
    f"{float(diagnostics['newey_west_loss_diff_t_stat']):.2f}. "
    + (
        f"The approximate 95% skill interval is "
        f"[{float(diagnostics['newey_west_skill_ci_95_lower_pct']):+.2f}%, "
        f"{float(diagnostics['newey_west_skill_ci_95_upper_pct']):+.2f}%]."
        if has_forecast_interval
        else ""
    )
)

st.subheader("Strategy evidence")
st.markdown("#### Cumulative proxy P&L on one shared scale")

pnl_fig = go.Figure()
for strategy, color, symbol in [
    (RESIDUAL_STRATEGY, XGB_COLOR, "diamond"),
    (PUBLIC_RAMP_STRATEGY, PUBLIC_COLOR, "circle"),
]:
    frame = strategy_paths.loc[strategy_paths["strategy"] == strategy]
    pnl_fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["cumulative_net_pnl_eur_per_mw_clip"],
            mode="lines+markers",
            name=STRATEGY_LABELS[strategy],
            line=dict(color=color, width=2),
            marker=dict(size=8, symbol=symbol),
            customdata=frame[[TOTAL_NET_PNL, "trades"]],
            hovertemplate=(
                "%{x}<br>Cumulative %{y:,.2f} EUR per 1 MW clip"
                "<br>Fold P&L %{customdata[0]:+,.2f} EUR"
                "<br>Trades %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
chart_layout(pnl_fig, height=400)
pnl_fig.update_yaxes(title="EUR per 1 MW hourly clip")
st.plotly_chart(pnl_fig, width="stretch", config={"displayModeBar": False})
st.caption(
    f"The common scale is intentional: the public rule reaches "
    f"{float(public_metric[TOTAL_NET_PNL]):,.2f} EUR while the residual signal finishes at "
    f"{float(residual_metric[TOTAL_NET_PNL]):,.2f} EUR. Because both are marked against DA(t-24), the public "
    "total shows that this proxy rewards public-information repricing; it does not establish an unarbitraged "
    "trading strategy."
)

left, right = st.columns([3, 2])
with left:
    st.markdown("#### Residual P&L by validation fold")
    residual_folds = strategy_paths.loc[strategy_paths["strategy"] == RESIDUAL_STRATEGY].copy()
    colors = [
        NOVEMBER_COLOR
        if month == concentration["best_month"]
        else POSITIVE_COLOR
        if pnl > 0
        else NEGATIVE_COLOR
        for month, pnl in zip(residual_folds["month"], residual_folds[TOTAL_NET_PNL])
    ]
    fold_pnl_fig = go.Figure(
        go.Bar(
            x=residual_folds["month"],
            y=residual_folds[TOTAL_NET_PNL],
            marker_color=colors,
            customdata=residual_folds[["trades", "hit_rate", "net_pnl_t_stat"]],
            hovertemplate=(
                "%{x}<br>P&L %{y:+,.2f} EUR per 1 MW clip"
                "<br>Trades %{customdata[0]:,.0f}"
                "<br>Hit rate %{customdata[1]:.1%}"
                "<br>t-stat %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    chart_layout(fold_pnl_fig, height=340, hovermode="closest")
    fold_pnl_fig.update_yaxes(title="EUR per 1 MW hourly clip", zeroline=True, zerolinecolor=GRID_COLOR)
    fold_pnl_fig.add_annotation(
        x=concentration["best_month"],
        y=concentration["best_fold_pnl"],
        text=f"{concentration['best_month']}: {float(concentration['best_fold_pnl']):+,.0f} EUR",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-42,
    )
    st.plotly_chart(fold_pnl_fig, width="stretch", config={"displayModeBar": False})

with right:
    st.markdown("#### What survives the best fold?")
    with st.container(border=True):
        st.metric("Full-sample residual P&L", f"{float(concentration['total_pnl']):+,.2f} EUR")
        st.caption("1 MW hourly clips")
    with st.container(border=True):
        st.metric(
            f"Excluding {concentration['best_month']}",
            f"{float(concentration['pnl_ex_best']):+,.2f} EUR",
        )
        st.caption(f"{int(concentration['trades_ex_best'])} trades")
    with st.container(border=True):
        st.metric("Residual strategy significance", f"t = {float(residual_metric['net_pnl_t_stat']):.2f}")
        st.caption("No statistical support")

st.subheader("Threshold robustness")
st.markdown("#### Net P&L per trade across the pre-defined grid")

threshold_fig = go.Figure()
for strategy, color, symbol in [
    (RESIDUAL_STRATEGY, XGB_COLOR, "diamond"),
    (PUBLIC_RAMP_STRATEGY, PUBLIC_COLOR, "circle"),
]:
    frame = thresholds.loc[thresholds["strategy"] == strategy]
    threshold_fig.add_trace(
        go.Scatter(
            x=frame["threshold_mw"],
            y=frame["avg_net_pnl_eur_mwh"],
            mode="lines+markers",
            name=STRATEGY_LABELS[strategy],
            line=dict(color=color, width=2),
            marker=dict(size=9, symbol=symbol),
            customdata=frame[["trades", "net_pnl_t_stat"]],
            hovertemplate=(
                "Threshold %{x:,.0f} MW<br>Net P&L / trade %{y:+.3f} EUR/MWh"
                "<br>Trades %{customdata[0]:,.0f}<br>t-stat %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
chart_layout(threshold_fig, height=390)
threshold_fig.update_xaxes(title="Absolute wind-signal threshold (MW)", tickformat=",.0f")
threshold_fig.update_yaxes(title="Average net P&L (EUR/MWh per trade)", zeroline=True, zerolinecolor=GRID_COLOR)
threshold_fig.add_vline(x=1500, line_width=1, line_dash="dot", line_color=GRID_COLOR)
residual_thresholds = thresholds.loc[thresholds["strategy"] == RESIDUAL_STRATEGY].sort_values("threshold_mw")
public_thresholds = thresholds.loc[thresholds["strategy"] == PUBLIC_RAMP_STRATEGY].sort_values("threshold_mw")
best_residual_threshold = residual_thresholds.sort_values("avg_net_pnl_eur_mwh", ascending=False).iloc[0]
threshold_fig.add_annotation(
    x=best_residual_threshold["threshold_mw"],
    y=best_residual_threshold["avg_net_pnl_eur_mwh"],
    text=f"Best residual grid point; t={float(best_residual_threshold['net_pnl_t_stat']):.2f}",
    showarrow=True,
    arrowhead=2,
    ax=70,
    ay=-38,
)
st.plotly_chart(threshold_fig, width="stretch", config={"displayModeBar": False})
negative_residual_thresholds = residual_thresholds.loc[residual_thresholds["avg_net_pnl_eur_mwh"] < 0]
if negative_residual_thresholds.empty:
    residual_sign_summary = "remains non-negative across the tested grid"
else:
    first_negative_threshold = float(negative_residual_thresholds["threshold_mw"].min())
    residual_sign_summary = f"turns negative from {first_negative_threshold:,.0f} MW"
public_sign_summary = (
    "remains positive at every tested threshold"
    if (public_thresholds["avg_net_pnl_eur_mwh"] > 0).all()
    else "is not positive at every tested threshold"
)
st.caption(
    f"The residual rule peaks at {float(best_residual_threshold['threshold_mw']):,.0f} MW and "
    f"{residual_sign_summary}. Its best point is an unadjusted {len(residual_thresholds)}-choice maximum, not "
    f"holdout evidence; the public rule {public_sign_summary}."
)

st.warning(
    "Research proxy only: P&L uses DA(t) minus DA(t-24), not executable auction-to-intraday or imbalance marks. "
    "The public-ramp result is therefore a proxy diagnostic, not an executable edge. The weather and lag "
    "information set is also not pinned to strict D-1 noon gate closure."
)

st.subheader("AI analyst review")
try:
    ai_artifact = load_llm_review_artifact(OUTPUTS)
except (OSError, ValueError) as exc:
    ai_artifact = None
    st.warning(f"The saved AI review could not be loaded: {exc}")

if ai_artifact is None:
    st.info("No saved AI review is available for this output snapshot.")
else:
    ai_review = ai_artifact["review"]
    st.caption(
        f"Generated {ai_artifact['generated_at_utc']} | {ai_artifact['model_returned']} | "
        f"reasoning effort: {ai_artifact['reasoning_effort']}"
    )
    st.markdown(f"#### {ai_review['headline']}")
    st.write(ai_review["overall_conclusion"])

    ai_left, ai_right = st.columns(2)
    for column, title, key in [
        (ai_left, "Forecast assessment", "forecast_assessment"),
        (ai_right, "Strategy assessment", "strategy_assessment"),
    ]:
        finding = ai_review[key]
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(finding["assessment"])
                for evidence_label in finding_evidence_labels(ai_artifact, finding):
                    st.caption(evidence_label)

    st.markdown("#### Risks and next checks")
    risk_column, invalidation_column, next_steps_column = st.columns(3)
    with risk_column:
        st.markdown("**Risk flags**")
        for risk in ai_review["risk_flags"]:
            st.markdown(f"- {risk['assessment']}")
            st.caption(" | ".join(finding_evidence_labels(ai_artifact, risk)))
    with invalidation_column:
        st.markdown("**Invalidation conditions**")
        for condition in ai_review["invalidation_conditions"]:
            st.markdown(f"- {condition}")
    with next_steps_column:
        st.markdown("**Production next steps**")
        for next_step in ai_review["production_next_steps"]:
            st.markdown(f"- {next_step}")

    st.caption(
        "Downstream summary only: exact prompts and outputs are logged, deterministic evidence values are "
        "attached after generation, and the LLM cannot alter forecasts, trades, QA checks, or metrics."
    )
