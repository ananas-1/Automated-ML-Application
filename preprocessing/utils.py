"""
Utilities for preprocessing module
"""

import pandas as pd


def get_data_info(df: pd.DataFrame) -> dict:
    """Get information about the dataset"""
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }


def display_data_summary(df: pd.DataFrame) -> None:
    """Display summary statistics"""
    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
