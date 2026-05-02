const dropdownTriggers = document.querySelectorAll(".dropdown-trigger");

dropdownTriggers.forEach((trigger) => {
  trigger.addEventListener("click", () => {
    const dropdownId = trigger.dataset.dropdown;
    const dropdown = document.getElementById(dropdownId);
    const arrow = trigger.querySelector(".dropdown-arrow");

    dropdown.classList.toggle("hidden");
    arrow.classList.toggle("rotate");
  });
});

window.addEventListener("click", (e) => {
  if (!e.target.closest(".dropdown-container")) {
    document
      .querySelectorAll(".dropdown-menu")
      .forEach((m) => m.classList.add("hidden"));
    document
      .querySelectorAll(".dropdown-arrow")
      .forEach((a) => a.classList.remove("rotate"));
  }
});

const taskItems = document.querySelectorAll("#taskDropdown .dropdown-item");
const selectedTask = document.getElementById("selectedTask");
const targetContainer = document.getElementById("targetContainer");

export let currentTask = null;

taskItems.forEach((item) => {
  item.addEventListener("click", () => {
    currentTask = item.dataset.value;
    selectedTask.textContent = item.textContent.trim();

    document.getElementById("taskDropdown").classList.add("hidden");
    document
      .querySelector('[data-dropdown="taskDropdown"] .dropdown-arrow')
      ?.classList.remove("rotate");

    if (currentTask === "clustering") {
      targetContainer.classList.add("hidden");
    } else {
      targetContainer.classList.remove("hidden");
    }
  });
});

const selectedTarget = document.getElementById("selectedTarget");
export let currentTarget = null;

export function populateTargetDropdown(columns) {
  const targetDropdown = document.getElementById("targetDropdown");
  targetDropdown.innerHTML = columns
    .map(
      (col) =>
        `<div class="dropdown-item" data-col="${col}">${col}</div>`
    )
    .join("");

  targetDropdown.querySelectorAll(".dropdown-item").forEach((item) => {
    item.addEventListener("click", () => {
      currentTarget = item.dataset.col;
      selectedTarget.textContent = item.textContent.trim();
      targetDropdown.classList.add("hidden");
      document
        .querySelector('[data-dropdown="targetDropdown"] .dropdown-arrow')
        ?.classList.remove("rotate");
    });
  });

  selectedTarget.textContent = "TARGET VARIABLE";
  currentTarget = null;
}

document.addEventListener("datasetReady", (e) => {
  populateTargetDropdown(e.detail.columns);
});