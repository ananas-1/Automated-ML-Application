"""
Regression Trainer
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from regression.models import get_regression_models
from backend.config import TEST_SIZE, RANDOM_STATE


def train_regression_models(X, y):
    """
    Train all regression models and select the best
    
    Args:
        X: Features
        y: Target
        
    Returns:
        Tuple of (best_model, best_model_name, X_test, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    models = get_regression_models()
    best_model = None
    best_model_name = None
    best_score = float('-inf')
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = model_name
    
    return best_model, best_model_name, X_test, y_test
