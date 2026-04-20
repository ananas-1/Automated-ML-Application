"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel
from typing import List, Dict, Optional


class FileUploadResponse(BaseModel):
    """Response after file upload"""
    message: str
    file_name: str
    file_id: str
    columns: List[str]
    rows_count: int


class TrainingRequest(BaseModel):
    """Request to start model training"""
    file_id: str
    task_type: str  # "classification", "regression", "clustering"
    target_column: Optional[str] = None  # Required for supervised learning


class TrainingResponse(BaseModel):
    """Response after training completes"""
    task_id: str
    status: str
    message: str


class MetricsResponse(BaseModel):
    """Response with model metrics"""
    task_type: str
    metrics: Dict[str, float]
    best_model: str


class ResultsResponse(BaseModel):
    """Complete results response"""
    task_id: str
    status: str
    task_type: str
    best_model: str
    metrics: Dict[str, float]
    model_id: str


class ModelDownloadResponse(BaseModel):
    """Response for model download"""
    message: str
    file_path: str
