/**
 * Task Selector - Handle ML task selection
 */

let selectedTask = null;

const taskRadios = document.querySelectorAll('input[name="task"]');
const targetSection = document.getElementById("target-section");
const targetColumn = document.getElementById("target-column");

taskRadios.forEach((radio) => {
  radio.addEventListener("change", (e) => {
    selectedTask = e.target.value;

    // Show target selection for supervised learning
    if (selectedTask === "classification" || selectedTask === "regression") {
      targetSection.classList.remove("hidden");
      // Populate target columns (this would come from uploaded file)
      populateTargetColumns();
    } else {
      targetSection.classList.add("hidden");
    }

    // Show training section
    document.getElementById("training-section").classList.remove("hidden");
  });
});

function populateTargetColumns() {
  // This would populate with actual columns from uploaded file
  // For now, it's a placeholder
  targetColumn.innerHTML = "<option>Select a column...</option>";
}

targetColumn.addEventListener("change", (e) => {
  // Store selected target column
  window.selectedTargetColumn = e.target.value;
});
