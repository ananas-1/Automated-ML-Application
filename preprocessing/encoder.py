"""
Encoder - Encode categorical variables
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_categorical(df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
    """
    Encode categorical variables
    
    Args:
        df: Input DataFrame
        method: Encoding method - 'label' or 'onehot'
        
    Returns:
        DataFrame with encoded categorical variables
    """
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if method == 'label':
        encoder = LabelEncoder()
        for col in categorical_cols:
            df_encoded[col] = encoder.fit_transform(df[col].astype(str))
    
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    return df_encoded
