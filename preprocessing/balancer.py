"""
Balancer - Handle imbalanced data using resampling techniques
"""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def balance_data(X: pd.DataFrame, y: pd.Series, method: str = 'oversample') -> tuple:
    """
    Handle imbalanced data
    
    Args:
        X: Features DataFrame
        y: Target Series
        method: Balancing method - 'oversample' or 'undersample'
        
    Returns:
        Tuple of balanced (X, y)
    """
    if method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    X_balanced, y_balanced = sampler.fit_resample(X, y)
    
    return X_balanced, y_balanced
