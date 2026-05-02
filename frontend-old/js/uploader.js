import { uploadFile, previewDataset } from "./api.js";

const fileInput = document.querySelector("#fileInput");
const fileNameLabel = document.querySelector("#fileName");
const emptyPreview = document.querySelector("#emptyPreview");
const tableWrapper = document.querySelector("#tableWrapper");

export let currentDatasetId = null;
export let currentColumns = [];

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  fileNameLabel.textContent = file.name;

  showPreviewState("loading");

  try {
    const uploadRes = await uploadFile(file);
    currentDatasetId = uploadRes.dataset_id;

    if (!currentDatasetId) {
      throw new Error("Missing dataset_id from upload response");
    }

    const previewRes = await previewDataset(currentDatasetId);

    if (!previewRes?.columns || !previewRes?.preview) {
      throw new Error("Invalid preview response");
    }

    currentColumns = previewRes.columns;

    renderTable(previewRes.columns, previewRes.preview);
    showPreviewState("table");

    document.dispatchEvent(new CustomEvent("datasetReady", {
      detail: { datasetId: currentDatasetId, columns: currentColumns }
    }));
  } catch (err) {
    console.error("UPLOAD/PREVIEW ERROR:", err);
    showPreviewState("error", err.message);
  }
});

function renderTable(columns, rows) {
  const thead = tableWrapper.querySelector("thead tr");
  thead.innerHTML = columns
    .map((col) => `<th>${col}</th>`)
    .join("");

  const tbody = tableWrapper.querySelector("tbody");
  tbody.innerHTML = rows
    .map(
      (row) =>
        `<tr>${columns.map((col) => `<td>${row[col] ?? ""}</td>`).join("")}</tr>`
    )
    .join("");
}

function showPreviewState(state, message = "") {
  switch (state) {
    case "loading":
      emptyPreview.classList.remove("hidden");
      tableWrapper.classList.add("hidden");
      emptyPreview.innerHTML = `
        <div class="preview-loading">
          <div class="spinner"></div>
          <span>UPLOADING & LOADING PREVIEW…</span>
        </div>`;
      break;

    case "table":
      emptyPreview.classList.add("hidden");
      tableWrapper.classList.remove("hidden");
      break;

    case "error":
      emptyPreview.classList.remove("hidden");
      tableWrapper.classList.add("hidden");
      emptyPreview.innerHTML = `
        <div class="preview-error">
          <i class="fa-solid fa-triangle-exclamation"></i>
          <span>${message || "Something went wrong."}</span>
        </div>`;
      break;
  }
}