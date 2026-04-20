"""
Regression Evaluator - Calculate metrics
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate regression model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with all metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'mse': float(mean_squared_error(y_test, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'r2_score': float(r2_score(y_test, y_pred))
    }
    
    return metrics
