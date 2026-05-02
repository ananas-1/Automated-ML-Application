import { configureTask, trainModel, saveModel } from "./api.js";
import { currentDatasetId } from "./uploader.js";
import { currentTask, currentTarget } from "./dropdown.js";

const form      = document.getElementById("mlForm");
const submitBtn = form.querySelector(".btn-primary");

submitBtn.addEventListener("click", async (e) => {
  e.preventDefault();

  if (!currentDatasetId) {
    showFormError("Please upload a file first.");
    return;
  }

  if (!currentTask) {
    showFormError("Please select a task.");
    return;
  }

  if (currentTask !== "clustering" && !currentTarget) {
    showFormError("Please select a target variable.");
    return;
  }

  clearFormError();
  setSubmitState("loading");

  try {
    /* 1. Configure */
    await configureTask({
      datasetId: currentDatasetId,
      task:       currentTask,
      targetColumn: currentTask === "clustering" ? null : currentTarget,
    });

    /* 2. Train */
    const results = await trainModel(currentDatasetId);

    /* 3. Save model */
    const saveRes = await saveModel(currentDatasetId);

    /* 4. Pass results to output page via sessionStorage */
    sessionStorage.setItem(
      "mlResults",
      JSON.stringify({
        results,
        modelId: saveRes.model_id,
        datasetId: currentDatasetId,
      })
    );

    /* 5. Navigate */
    window.location.href = "output.html";

  } catch (err) {
    setSubmitState("idle");
    showFormError(err.message || "Something went wrong. Please try again.");
  }
});

/* ── Submit button states ── */
function setSubmitState(state) {
  if (state === "loading") {
    submitBtn.disabled = true;
    submitBtn.innerHTML = `
      <span class="btn-spinner"></span>
      TRAINING…
    `;
  } else {
    submitBtn.disabled = false;
    submitBtn.textContent = "SUBMIT";
  }
}

/* ── Inline form error ── */
function showFormError(msg) {
  let el = document.getElementById("formError");
  if (!el) {
    el = document.createElement("p");
    el.id = "formError";
    el.className = "form-error";
    submitBtn.insertAdjacentElement("beforebegin", el);
  }
  el.textContent = msg;
}

function clearFormError() {
  document.getElementById("formError")?.remove();
}