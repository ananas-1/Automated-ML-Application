"""
Pipeline - Main preprocessing pipeline orchestrator
"""

import pandas as pd
from preprocessing.data_loader import load_data
from preprocessing.data_cleaner import handle_missing_values
from preprocessing.encoder import encode_categorical
from preprocessing.scaler import scale_features
from preprocessing.balancer import balance_data


def run_preprocessing_pipeline(file_path: str, target_column: str = None, 
                               task_type: str = 'classification') -> tuple:
    """
    Execute the complete preprocessing pipeline
    
    Args:
        file_path: Path to the data file
        target_column: Target column name (for supervised learning)
        task_type: Type of ML task - 'classification', 'regression', or 'clustering'
        
    Returns:
        Tuple of (X, y) for supervised learning or (X,) for clustering
    """
    # Load data
    df = load_data(file_path)
    
    # Handle missing values
    df = handle_missing_values(df, strategy='mean')
    
    # Encode categorical variables
    df = encode_categorical(df, method='label')
    
    # Scale features
    df = scale_features(df, method='standard')
    
    # Separate features and target if supervised learning
    if task_type in ['classification', 'regression'] and target_column:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Balance data if classification and imbalanced
        if task_type == 'classification':
            X, y = balance_data(X, y, method='oversample')
        
        return X, y
    else:
        # For clustering, return only features
        return df,


def get_pipeline_info() -> dict:
    """Get information about the preprocessing pipeline"""
    return {
        "steps": [
            "Load data",
            "Handle missing values (mean strategy)",
            "Encode categorical variables (label encoding)",
            "Scale features (standard scaling)",
            "Balance data (if classification)"
        ]
    }
