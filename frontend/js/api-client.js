/**
 * API Client - Handle HTTP requests to the backend
 */

const API_BASE_URL = "http://127.0.0.1:8000/api";

class APIClient {
  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");
      return await response.json();
    } catch (error) {
      console.error("Upload error:", error);
      throw error;
    }
  }

  async startTraining(fileId, taskType, targetColumn = null) {
    try {
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_id: fileId,
          task_type: taskType,
          target_column: targetColumn,
        }),
      });

      if (!response.ok) throw new Error("Training start failed");
      return await response.json();
    } catch (error) {
      console.error("Training error:", error);
      throw error;
    }
  }

  async getResults(taskId) {
    try {
      const response = await fetch(`${API_BASE_URL}/results/${taskId}`);
      if (!response.ok) throw new Error("Failed to get results");
      return await response.json();
    } catch (error) {
      console.error("Results error:", error);
      throw error;
    }
  }

  async downloadModel(modelId) {
    try {
      window.location.href = `${API_BASE_URL}/download/${modelId}`;
    } catch (error) {
      console.error("Download error:", error);
      throw error;
    }
  }
}

const apiClient = new APIClient();
