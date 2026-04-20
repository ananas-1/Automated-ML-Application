"""
Data Loader - Load CSV and XLSX files
"""

import pandas as pd
from pathlib import Path


def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file"""
    return pd.read_csv(file_path)


def load_xlsx(file_path: str) -> pd.DataFrame:
    """Load XLSX file"""
    return pd.read_excel(file_path)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV or XLSX file
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with loaded data
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        return load_csv(file_path)
    elif file_ext == '.xlsx':
        return load_xlsx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
