"""
Configuration settings for the ML Application Backend
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BACKEND_DIR / "models"
LOGS_DIR = BACKEND_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI settings
API_TITLE = "Automated ML Application"
API_VERSION = "1.0.0"
API_DESCRIPTION = "An end-to-end Automated Machine Learning Application"

# Upload settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}

# Model training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
