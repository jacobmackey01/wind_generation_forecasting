const COLORS = {
  ink: "#17211e",
  muted: "#65716d",
  grid: "rgba(101, 113, 109, 0.18)",
  blue: "#426f8e",
  rust: "#bd5b3d",
  green: "#2c765a",
  amber: "#a87319",
  red: "#b14b51",
};

const RESIDUAL = "xgboost_wind_residual_signal";
const PUBLIC_RAMP = "public_smard_forecast_ramp_signal";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"' && quoted && next === '"') {
      field += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(field);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  const [headers, ...values] = rows;
  return values.map((cells) => Object.fromEntries(headers.map((header, index) => {
    const value = cells[index] ?? "";
    const numeric = Number(value);
    return [header, value !== "" && Number.isFinite(numeric) ? numeric : value];
  })));
}

async function loadCsv(filename) {
  const response = await fetch(`/data/${filename}`);
  if (!response.ok) throw new Error(`Could not load ${filename}`);
  return parseCsv(await response.text());
}

async function loadJson(filename) {
  const response = await fetch(`/data/${filename}`);
  if (!response.ok) throw new Error(`Could not load ${filename}`);
  return response.json();
}

function monthLabel(timestamp) {
  return new Intl.DateTimeFormat("en-GB", { month: "short", year: "numeric", timeZone: "UTC" })
    .format(new Date(timestamp));
}

function signed(value, digits = 2) {
  return `${value >= 0 ? "+" : ""}${Number(value).toLocaleString("en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

function baseOptions(yTitle) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { intersect: false, mode: "index" },
    plugins: {
      legend: {
        align: "start",
        labels: { color: COLORS.ink, boxWidth: 14, boxHeight: 3, usePointStyle: true, pointStyle: "line" },
      },
      tooltip: {
        backgroundColor: COLORS.ink,
        padding: 11,
        titleFont: { weight: "600" },
        bodySpacing: 5,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: COLORS.muted, maxRotation: 0, autoSkipPadding: 18 },
        border: { color: COLORS.grid },
      },
      y: {
        title: { display: true, text: yTitle, color: COLORS.muted },
        grid: { color: COLORS.grid },
        ticks: { color: COLORS.muted },
        border: { display: false },
      },
    },
  };
}

function setMetric(id, value, contextId, context) {
  document.getElementById(id).textContent = value;
  document.getElementById(contextId).textContent = context;
}

function appendListItems(elementId, items) {
  const list = document.getElementById(elementId);
  list.replaceChildren();
  items.forEach((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    list.appendChild(item);
  });
}

function buildAiReview(artifact) {
  const review = artifact.review;
  const evidence = new Map(artifact.evidence.map((item) => [item.id, item]));
  const generated = new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Europe/London",
  }).format(new Date(artifact.generated_at_utc));

  document.getElementById("ai-meta").textContent = `${artifact.model_returned} | ${artifact.reasoning_effort} reasoning | generated ${generated}`;
  document.getElementById("ai-headline").textContent = review.headline;
  document.getElementById("ai-conclusion").textContent = review.overall_conclusion;

  function populateFinding(prefix, finding) {
    document.getElementById(`ai-${prefix}-assessment`).textContent = finding.assessment;
    appendListItems(`ai-${prefix}-evidence`, finding.evidence_ids.map((id) => {
      const item = evidence.get(id);
      if (!item) throw new Error(`Unknown evidence ID: ${id}`);
      return `${item.label}: ${item.value}`;
    }));
  }

  populateFinding("forecast", review.forecast_assessment);
  populateFinding("strategy", review.strategy_assessment);
  appendListItems("ai-risks", review.risk_flags.map((risk) => risk.assessment));
  appendListItems("ai-invalidations", review.invalidation_conditions);
  appendListItems("ai-next-steps", review.production_next_steps);
}

function buildDashboard({ metrics, diagnostics, foldMetrics, strategyMetrics, strategyFolds, thresholds }) {
  Chart.defaults.font.family = 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  Chart.defaults.color = COLORS.muted;

  const xgbMetric = metrics.find((row) => row.model === "xgboost_residual");
  const diag = diagnostics[0];
  const residualMetric = strategyMetrics.find((row) => row.strategy === RESIDUAL);
  const publicMetric = strategyMetrics.find((row) => row.strategy === PUBLIC_RAMP);
  const residualFolds = strategyFolds.filter((row) => row.strategy === RESIDUAL).sort((a, b) => a.fold - b.fold);
  const publicFolds = strategyFolds.filter((row) => row.strategy === PUBLIC_RAMP).sort((a, b) => a.fold - b.fold);
  const bestFold = residualFolds.reduce((best, row) => row.total_net_pnl_eur_per_mw_clip > best.total_net_pnl_eur_per_mw_clip ? row : best);
  const totalResidual = residualFolds.reduce((sum, row) => sum + row.total_net_pnl_eur_per_mw_clip, 0);
  const pnlExBest = totalResidual - bestFold.total_net_pnl_eur_per_mw_clip;
  const tradesExBest = residualFolds.reduce((sum, row) => sum + row.trades, 0) - bestFold.trades;
  const positiveFolds = residualFolds.filter((row) => row.total_net_pnl_eur_per_mw_clip > 0).length;
  const bestMonth = monthLabel(bestFold.fold_start);
  const perTradeRatio = publicMetric.avg_net_pnl_eur_mwh / residualMetric.avg_net_pnl_eur_mwh;

  setMetric("forecast-skill", `${signed(xgbMetric.skill_vs_smard_mae_pct, 2)}%`, "forecast-context", `NW t = ${diag.newey_west_loss_diff_t_stat.toFixed(2)} | not significant`);
  setMetric("residual-pnl", signed(residualMetric.avg_net_pnl_eur_mwh, 3), "residual-context", `EUR/MWh | t = ${residualMetric.net_pnl_t_stat.toFixed(2)} | statistically weak`);
  setMetric("public-pnl", signed(publicMetric.avg_net_pnl_eur_mwh, 3), "public-context", `EUR/MWh | ${perTradeRatio.toFixed(1)}x residual on this proxy`);
  setMetric("positive-folds", `${positiveFolds} / ${residualFolds.length}`, "fold-context", `Ex-${bestMonth}: ${signed(pnlExBest, 2)} EUR`);

  document.getElementById("forecast-summary").textContent = `XGBoost beat SMARD in ${diag.xgboost_fold_wins} of ${diag.fold_count} folds; pooled skill ${signed(xgbMetric.skill_vs_smard_mae_pct, 2)}%, Newey-West t = ${diag.newey_west_loss_diff_t_stat.toFixed(2)}.`;
  document.getElementById("total-residual-pnl").textContent = `${signed(totalResidual, 2)} EUR`;
  document.getElementById("exclude-label").textContent = `Excluding ${bestMonth}`;
  document.getElementById("pnl-ex-best").textContent = `${signed(pnlExBest, 2)} EUR`;
  document.getElementById("trades-ex-best").textContent = `${tradesExBest.toLocaleString("en-GB")} trades`;
  document.getElementById("residual-t-stat").textContent = `t = ${residualMetric.net_pnl_t_stat.toFixed(2)}`;

  const seriousFoldRows = foldMetrics.filter((row) => row.model === "smard_forecast" || row.model === "xgboost_residual");
  const folds = [...new Set(seriousFoldRows.map((row) => row.fold))].sort((a, b) => a - b);
  const labels = folds.map((fold) => monthLabel(seriousFoldRows.find((row) => row.fold === fold).fold_start));

  new Chart(document.getElementById("mae-chart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Public SMARD forecast",
          data: folds.map((fold) => seriousFoldRows.find((row) => row.fold === fold && row.model === "smard_forecast").mae),
          borderColor: COLORS.blue,
          backgroundColor: COLORS.blue,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          tension: 0.12,
        },
        {
          label: "XGBoost residual",
          data: folds.map((fold) => seriousFoldRows.find((row) => row.fold === fold && row.model === "xgboost_residual").mae),
          borderColor: COLORS.rust,
          backgroundColor: COLORS.rust,
          borderWidth: 2,
          pointRadius: 5,
          pointStyle: "rectRot",
          pointHoverRadius: 7,
          tension: 0.12,
        },
      ],
    },
    options: {
      ...baseOptions("MAE (MW)"),
      plugins: {
        ...baseOptions("MAE (MW)").plugins,
        tooltip: {
          ...baseOptions("MAE (MW)").plugins.tooltip,
          callbacks: {
            label: (context) => `${context.dataset.label}: ${Math.round(context.parsed.y).toLocaleString("en-GB")} MW`,
            afterBody: (items) => {
              const fold = folds[items[0].dataIndex];
              const row = seriousFoldRows.find((entry) => entry.fold === fold && entry.model === "xgboost_residual");
              return `XGBoost skill: ${signed(row.skill_vs_smard_mae_pct, 1)}%`;
            },
          },
        },
      },
    },
  });

  function cumulative(rows) {
    let running = 0;
    return rows.map((row) => {
      running += row.total_net_pnl_eur_per_mw_clip;
      return running;
    });
  }

  new Chart(document.getElementById("pnl-chart"), {
    type: "line",
    data: {
      labels: residualFolds.map((row) => monthLabel(row.fold_start)),
      datasets: [
        {
          label: "XGBoost residual signal",
          data: cumulative(residualFolds),
          borderColor: COLORS.rust,
          backgroundColor: COLORS.rust,
          borderWidth: 2,
          pointRadius: 4,
          pointStyle: "rectRot",
          tension: 0.1,
        },
        {
          label: "Public SMARD ramp",
          data: cumulative(publicFolds),
          borderColor: COLORS.green,
          backgroundColor: COLORS.green,
          borderWidth: 2,
          pointRadius: 4,
          tension: 0.1,
        },
      ],
    },
    options: {
      ...baseOptions("EUR per 1 MW hourly clip"),
      plugins: {
        ...baseOptions("EUR per 1 MW hourly clip").plugins,
        tooltip: {
          ...baseOptions("EUR per 1 MW hourly clip").plugins.tooltip,
          callbacks: { label: (context) => `${context.dataset.label}: ${signed(context.parsed.y, 2)} EUR` },
        },
      },
    },
  });

  new Chart(document.getElementById("fold-pnl-chart"), {
    type: "bar",
    data: {
      labels: residualFolds.map((row) => monthLabel(row.fold_start)),
      datasets: [{
        label: "Residual fold P&L",
        data: residualFolds.map((row) => row.total_net_pnl_eur_per_mw_clip),
        backgroundColor: residualFolds.map((row) => {
          if (row.fold === bestFold.fold) return COLORS.amber;
          return row.total_net_pnl_eur_per_mw_clip > 0 ? COLORS.green : COLORS.red;
        }),
        borderWidth: 0,
        borderRadius: 2,
      }],
    },
    options: {
      ...baseOptions("EUR per 1 MW hourly clip"),
      interaction: { intersect: true, mode: "nearest" },
      plugins: {
        ...baseOptions("EUR per 1 MW hourly clip").plugins,
        legend: { display: false },
        tooltip: {
          ...baseOptions("EUR per 1 MW hourly clip").plugins.tooltip,
          callbacks: {
            label: (context) => `P&L: ${signed(context.parsed.y, 2)} EUR`,
            afterBody: (items) => {
              const row = residualFolds[items[0].dataIndex];
              return [`Trades: ${row.trades}`, `Gross hit rate: ${(row.hit_rate * 100).toFixed(1)}%`, `t-stat: ${row.net_pnl_t_stat.toFixed(2)}`];
            },
          },
        },
      },
    },
  });

  const residualThresholds = thresholds.filter((row) => row.strategy === RESIDUAL).sort((a, b) => a.threshold_mw - b.threshold_mw);
  const publicThresholds = thresholds.filter((row) => row.strategy === PUBLIC_RAMP).sort((a, b) => a.threshold_mw - b.threshold_mw);
  const thresholdLabels = residualThresholds.map((row) => `${row.threshold_mw.toLocaleString("en-GB")} MW`);

  new Chart(document.getElementById("threshold-chart"), {
    type: "line",
    data: {
      labels: thresholdLabels,
      datasets: [
        {
          label: "XGBoost residual signal",
          data: residualThresholds.map((row) => row.avg_net_pnl_eur_mwh),
          borderColor: COLORS.rust,
          backgroundColor: COLORS.rust,
          borderWidth: 2,
          pointRadius: residualThresholds.map((row) => row.threshold_mw === 1500 ? 7 : 4),
          pointStyle: "rectRot",
          tension: 0.08,
        },
        {
          label: "Public SMARD ramp",
          data: publicThresholds.map((row) => row.avg_net_pnl_eur_mwh),
          borderColor: COLORS.green,
          backgroundColor: COLORS.green,
          borderWidth: 2,
          pointRadius: publicThresholds.map((row) => row.threshold_mw === 1500 ? 7 : 4),
          tension: 0.08,
        },
      ],
    },
    options: {
      ...baseOptions("Average net P&L (EUR/MWh per trade)"),
      plugins: {
        ...baseOptions("Average net P&L (EUR/MWh per trade)").plugins,
        tooltip: {
          ...baseOptions("Average net P&L (EUR/MWh per trade)").plugins.tooltip,
          callbacks: {
            label: (context) => `${context.dataset.label}: ${signed(context.parsed.y, 3)} EUR/MWh`,
            afterBody: (items) => {
              const index = items[0].dataIndex;
              const rows = items.map((item) => item.datasetIndex === 0 ? residualThresholds[index] : publicThresholds[index]);
              return rows.map((row) => `${row.strategy === RESIDUAL ? "Residual" : "Public"}: ${row.trades.toLocaleString("en-GB")} trades, t=${row.net_pnl_t_stat.toFixed(2)}`);
            },
          },
        },
      },
    },
  });
}

Promise.all([
  loadCsv("metrics.csv"),
  loadCsv("forecast_diagnostics.csv"),
  loadCsv("fold_metrics.csv"),
  loadCsv("strategy_metrics.csv"),
  loadCsv("strategy_fold_metrics.csv"),
  loadCsv("strategy_threshold_sensitivity.csv"),
])
  .then(([metrics, diagnostics, foldMetrics, strategyMetrics, strategyFolds, thresholds]) => {
    buildDashboard({ metrics, diagnostics, foldMetrics, strategyMetrics, strategyFolds, thresholds });
  })
  .catch((error) => {
    console.error(error);
    document.getElementById("load-error").hidden = false;
  });

loadJson("llm/analyst_review.json")
  .then(buildAiReview)
  .catch((error) => {
    console.error(error);
    document.getElementById("ai-load-error").hidden = false;
  });
