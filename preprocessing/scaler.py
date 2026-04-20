"""
Scaler - Scale/Normalize numerical features
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_features(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Scale/Normalize numerical features
    
    Args:
        df: Input DataFrame
        method: Scaling method - 'standard' or 'minmax'
        
    Returns:
        DataFrame with scaled features
    """
    df_scaled = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_scaled
