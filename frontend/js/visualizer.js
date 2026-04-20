/**
 * Visualizer - Display results and metrics
 */

function displayResults(results) {
  const resultsSection = document.getElementById("results-section");
  const metricsDisplay = document.getElementById("metrics-display");

  resultsSection.classList.remove("hidden");

  // Clear previous metrics
  metricsDisplay.innerHTML = "";

  // Display each metric
  if (results.metrics) {
    Object.entries(results.metrics).forEach(([key, value]) => {
      // Skip arrays (like confusion matrix)
      if (typeof value === "object") return;

      const card = document.createElement("div");
      card.className = "metric-card";
      card.innerHTML = `
                <h3>${key.replace(/_/g, " ")}</h3>
                <div class="value">${typeof value === "number" ? value.toFixed(4) : value}</div>
            `;
      metricsDisplay.appendChild(card);
    });
  }

  // Store model ID for download
  window.modelId = results.model_id;
}

function displayConfusionMatrix(confusionMatrix) {
  // This would display confusion matrix as a heatmap or table
  console.log("Confusion Matrix:", confusionMatrix);
}

function displayTrainingProgress() {
  document.getElementById("training-progress").classList.remove("hidden");
  const progressBar = document.getElementById("progress-bar");
  progressBar.value = 50;
}

function hideTrainingProgress() {
  document.getElementById("training-progress").classList.add("hidden");
}
