"""
API Routes for the ML Application
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from backend.api.schemas import FileUploadResponse, TrainingRequest, TrainingResponse, ResultsResponse
import os

router = APIRouter()

# Temporary storage for uploaded files (will be replaced with proper database)
uploaded_files = {}
training_results = {}


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or XLSX file
    """
    try:
        # Validate file type
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported")
        
        # Save file
        file_id = file.filename
        file_path = f"uploaded_files/{file_id}"
        os.makedirs("uploaded_files", exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # TODO: Parse file and extract columns
        columns = ["column1", "column2", "column3"]  # Placeholder
        rows_count = 100  # Placeholder
        
        return FileUploadResponse(
            message="File uploaded successfully",
            file_name=file.filename,
            file_id=file_id,
            columns=columns,
            rows_count=rows_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """
    Start model training process
    """
    try:
        task_id = f"task_{request.file_id}_{request.task_type}"
        
        # TODO: Implement training logic
        
        return TrainingResponse(
            task_id=task_id,
            status="processing",
            message="Training started"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{task_id}", response_model=ResultsResponse)
async def get_results(task_id: str):
    """
    Get training results for a specific task
    """
    try:
        if task_id not in training_results:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return training_results[task_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{model_id}")
async def download_model(model_id: str):
    """
    Download trained model
    """
    try:
        model_path = f"backend/models/{model_id}.pkl"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        return FileResponse(
            path=model_path,
            filename=f"{model_id}.pkl"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
