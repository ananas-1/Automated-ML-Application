/**
 * File Uploader - Handle file upload functionality
 */

let uploadedFileId = null;

const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const filePreview = document.getElementById("file-preview");
const fileName = document.getElementById("file-name");

// Drag and drop handling
uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.style.background = "#f0f0f0";
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.style.background = "white";
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.style.background = "white";

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFileUpload(files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileUpload(e.target.files[0]);
  }
});

async function handleFileUpload(file) {
  try {
    // Validate file type
    if (!file.name.endsWith(".csv") && !file.name.endsWith(".xlsx")) {
      alert("Please upload a CSV or XLSX file");
      return;
    }

    // Upload file
    const result = await apiClient.uploadFile(file);
    uploadedFileId = result.file_id;

    // Display file preview
    fileName.textContent = `File: ${result.file_name} (${result.rows_count} rows)`;
    filePreview.classList.remove("hidden");

    // Show next sections
    displayDataPreview(result);
    document.getElementById("task-section").classList.remove("hidden");
  } catch (error) {
    alert("Error uploading file: " + error.message);
  }
}

function displayDataPreview(fileData) {
  const previewSection = document.getElementById("preview-section");
  previewSection.classList.remove("hidden");

  // This would display actual data preview once backend implements it
  const previewDiv = document.getElementById("data-preview");
  previewDiv.innerHTML = `
        <table>
            <thead>
                <tr>
                    ${fileData.columns.map((col) => `<th>${col}</th>`).join("")}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colspan="${fileData.columns.length}">Data preview will be displayed here</td>
                </tr>
            </tbody>
        </table>
    `;
}
