"""
Data Cleaner - Handle missing values
"""

import pandas as pd
from sklearn.impute import SimpleImputer


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    
    Args:
        df: Input DataFrame
        strategy: Strategy for imputation - 'mean', 'median', 'most_frequent', or 'constant'
        
    Returns:
        DataFrame with missing values handled
    """
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = df.copy()
    
    # Apply imputer to numerical columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df_imputed
