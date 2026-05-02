import { getDownloadUrl } from "./api.js";

const stored = sessionStorage.getItem("mlResults");

if (!stored) {
  window.location.href = "index.html";
}

const { results, modelId } = JSON.parse(stored);

const taskBadge = document.getElementById("taskBadge");
const bestModelName = document.getElementById("bestModelName");
const bestMetrics = document.getElementById("bestMetrics");
const allModelsGrid = document.getElementById("allModelsGrid");
const dataReport = document.getElementById("dataReport");
const dataReportWrap = document.getElementById("dataReportWrap");
const clusterSizes = document.getElementById("clusterSizes");
const clusterWrap = document.getElementById("clusterWrap");
const downloadBtn = document.getElementById("downloadBtn");

renderPage(results, modelId);

function renderPage(data, mId) {
  const task = data.task;

  taskBadge.textContent = task.toUpperCase();
  taskBadge.dataset.task = task;

  bestModelName.textContent = data.best_model;
  bestMetrics.innerHTML = formatMetrics(data.metrics, task);
  allModelsGrid.innerHTML = renderAllModels(data.all_models, data.best_model, task);

  if (data.data_report) {
    dataReportWrap.classList.remove("hidden");
    dataReport.innerHTML = renderDataReport(data.data_report);
  }

  if (data.cluster_sizes) {
    clusterWrap.classList.remove("hidden");
    clusterSizes.innerHTML = renderClusterSizes(data.cluster_sizes);
  }

  downloadBtn.href = getDownloadUrl(mId);
}

function formatMetrics(metrics, task) {
  if (!metrics || typeof metrics !== "object") return "<p>No metrics available.</p>";

  const entries = Object.entries(metrics);

  return entries
    .map(([key, val]) => {
      const label = formatMetricLabel(key);
      if (label === "Confusion Matrix") return "";
      const display = formatMetricValue(key, val, task);
      const barWidth = getBarWidth(key, val, task);

      return `
        <div class="metric-item">
          <div class=" flex justify-between items-baseline max-w-full nin-w-0 gap-2">
            <span class="metric-label">${label}</span>
            <span class="metric-value">${display}</span>
          </div>
          ${barWidth !== null
          ? `<div class="metric-bar-track">
                 <div class="metric-bar-fill" style="width:${barWidth}%"></div>
               </div>`
          : ""}
        </div>
      `;
    })
    .join("");
}

function formatMetricLabel(key) {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatMetricValue(key, val, task) {
  if (typeof val !== "number") return String(val);

  const errorMetrics = ["mae", "mse", "rmse", "mean_absolute_error",
    "mean_squared_error", "root_mean_squared_error"];
  if (errorMetrics.some((m) => key.toLowerCase().includes(m))) {
    return val.toFixed(4);
  }

  return `${(val * 100).toFixed(1)}%`;
}

function getBarWidth(key, val, task) {
  if (typeof val !== "number") return null;
  const errorMetrics = ["mae", "mse", "rmse"];
  if (errorMetrics.some((m) => key.toLowerCase().includes(m))) return null;
  return Math.min(Math.max(val * 100, 0), 100);
}

function renderAllModels(allModels, bestName, task) {
  if (!allModels || typeof allModels !== "object") return "";

  return Object.entries(allModels)
    .map(([modelName, metrics]) => {
      const isBest = modelName === bestName;

      const metricsHtml =
        typeof metrics === "number"
          ? `<div class="metric-item">
               <div class="flex justify-between items-baseline max-w-full min-w-0 gap-2">
                 <span class="metric-label">Silhouette Score</span>
                 <span class="metric-value">${metrics.toFixed(4)}</span>
               </div>
             </div>`
          : formatMetrics(metrics, task);

      return `
        <div class="model-card">
          <div class="model-card-header">
            <span class="model-name">${modelName}</span>
            ${isBest ? `<span class="best-badge">✦ BEST</span>` : ""}
          </div>
          <div class="model-metrics">
            ${metricsHtml}
          </div>
        </div>
      `;
    })
    .join("");
}

function renderDataReport(report) {
  if (!report || typeof report !== "object") return "";

  return Object.entries(report)
    .map(([key, val]) => {
      const label = formatMetricLabel(key);
      const display =
        typeof val === "object" ? JSON.stringify(val, null, 2) : String(val);

      return `
        <div class="report-item">
          <span class="report-label">${label}</span>
          <span class="report-value">${display}</span>
        </div>
      `;
    })
    .join("");
}

function renderClusterSizes(sizes) {
  if (!sizes || typeof sizes !== "object") return "";

  const total = Object.values(sizes).reduce((a, b) => a + b, 0);

  return Object.entries(sizes)
    .map(([cluster, count]) => {
      const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
      return `
        <div class="cluster-item">
          <div class="cluster-header">
            <span class="cluster-label">Cluster ${cluster}</span>
            <span class="cluster-count">${count} samples · ${pct}%</span>
          </div>
          <div class="metric-bar-track">
            <div class="metric-bar-fill" style="width:${pct}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}