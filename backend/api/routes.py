from backend.utils.file_handler import get_dataset_path, load_dataset
from backend.services.pipeline import run_pipeline, MODELS_STORE
from fastapi import APIRouter, UploadFile, File, HTTPException, Path
from fastapi.responses import FileResponse
import os, uuid, shutil, joblib
from pydantic import BaseModel
from typing import Optional


router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIG_STORE = {}

class ConfigRequest(BaseModel):
    dataset_id: str
    task: str
    target_column: Optional[str] = None


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    dataset_id = str(uuid.uuid4())
    file_ext = file.filename.split(".")[-1].lower()

    if file_ext not in ["csv", "xlsx"]:
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are allowed")

    file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.{file_ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully",
        "dataset_id": dataset_id,
        "file_path": file_path,
        "filename": file.filename
    }


@router.get("/preview/{dataset_id}")
def preview_dataset(dataset_id: str = Path(...)):
    file_path = get_dataset_path(dataset_id)

    if not file_path:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = load_dataset(file_path)

    return {
        "columns": df.columns.tolist(),
        "preview": df.head(10).to_dict(orient="records")
    }


@router.post("/configure")
def configure_task(config: ConfigRequest):
    dataset_id = config.dataset_id

    if config.task not in ["classification", "regression", "clustering"]:
        raise HTTPException(status_code=400, detail="Invalid task type")

    if config.task in ["classification", "regression"] and not config.target_column:
        raise HTTPException(status_code=400, detail="Target column is required")

    CONFIG_STORE[dataset_id] = {
        "task": config.task,
        "target_column": config.target_column
    }

    file_path = get_dataset_path(dataset_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = load_dataset(file_path)

    if config.target_column and config.target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column does not exist")

    return {
        "message": "Configuration saved successfully",
        "config": CONFIG_STORE[dataset_id]
    }


@router.post("/train/{dataset_id}")
def train_model(dataset_id: str):
    if dataset_id not in CONFIG_STORE:
        raise HTTPException(status_code=400, detail="Dataset not configured")

    config = CONFIG_STORE[dataset_id]

    result = run_pipeline(dataset_id, config)

    return result


@router.post("/save/{dataset_id}")
def save_model(dataset_id: str):
    if dataset_id not in MODELS_STORE:
        raise HTTPException(status_code=404, detail="No trained model found")

    data = MODELS_STORE[dataset_id]
    model_id = str(uuid.uuid4())

    file_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")

    joblib.dump({
        "model": data["model"],
        "pipeline": data["pipeline"],
        "task": data["task"],
        "metrics": data["metrics"]
    }, file_path)

    return {
        "message": "Model saved successfully",
        "model_id": model_id
    }


@router.get("/download/{model_id}")
def download_model(model_id: str):
    file_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(
        path=file_path,
        filename="model.joblib",
        media_type="application/octet-stream"
    )
