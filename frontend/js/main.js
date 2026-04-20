/**
 * Main Application Logic
 */

const trainButton = document.getElementById("train-button");
const downloadButton = document.getElementById("download-button");

trainButton.addEventListener("click", startTraining);
downloadButton.addEventListener("click", downloadModel);

async function startTraining() {
  if (!uploadedFileId) {
    alert("Please upload a file first");
    return;
  }

  if (!selectedTask) {
    alert("Please select an ML task");
    return;
  }

  if (
    (selectedTask === "classification" || selectedTask === "regression") &&
    !window.selectedTargetColumn
  ) {
    alert("Please select a target column");
    return;
  }

  try {
    displayTrainingProgress();

    const response = await apiClient.startTraining(
      uploadedFileId,
      selectedTask,
      window.selectedTargetColumn || null,
    );

    // Poll for results
    pollResults(response.task_id);
  } catch (error) {
    hideTrainingProgress();
    alert("Error starting training: " + error.message);
  }
}

async function pollResults(taskId) {
  const maxAttempts = 30; // 30 attempts with 2 second interval = 60 seconds max
  let attempts = 0;

  const pollInterval = setInterval(async () => {
    attempts++;

    try {
      const results = await apiClient.getResults(taskId);

      if (results.status === "completed") {
        clearInterval(pollInterval);
        hideTrainingProgress();
        displayResults(results);
      } else if (attempts >= maxAttempts) {
        clearInterval(pollInterval);
        hideTrainingProgress();
        alert("Training took too long");
      }
    } catch (error) {
      if (attempts >= maxAttempts) {
        clearInterval(pollInterval);
        hideTrainingProgress();
        alert("Error polling results: " + error.message);
      }
    }
  }, 2000);
}

function downloadModel() {
  if (!window.modelId) {
    alert("No model available for download");
    return;
  }

  apiClient.downloadModel(window.modelId);
}
