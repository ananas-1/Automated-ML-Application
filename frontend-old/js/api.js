const API_BASE = "http://localhost:8000";

async function apiFetch(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, options);

    const text = await res.text();

    if (!res.ok) {
      console.error("apiFetch HTTP ERROR:", res.status, text);
      throw new Error(text || `Request failed (${res.status})`);
    }

    try {
      const parsed = JSON.parse(text);
      return parsed;
    } catch (e) {
      console.error("INVALID JSON RESPONSE:", text);
      throw new Error("Backend did not return valid JSON");
    }
  } catch (err) {
    console.error("apiFetch EXCEPTION:", endpoint, err);
    throw err;
  }
}


// UPLOAD
export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);

  return apiFetch("/upload", {
    method: "POST",
    body: form,
  });
}

// PREVIEW
export async function previewDataset(datasetId) {
  return apiFetch(`/preview/${datasetId}`);
}

// CONFIGURE
export async function configureTask({ datasetId, task, targetColumn = null }) {
  return apiFetch("/configure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: datasetId,
      task,
      target_column: targetColumn,
    }),
  });
}

// TRAIN
export async function trainModel(datasetId) {
  return apiFetch(`/train/${datasetId}`, { method: "POST" });
}

// SAVE
export async function saveModel(datasetId) {
  return apiFetch(`/save/${datasetId}`, { method: "POST" });
}

// DOWNLOAD
export function getDownloadUrl(modelId) {
  return `${API_BASE}/download/${modelId}`;
}