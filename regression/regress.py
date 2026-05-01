"""
regression.py
AutoML Regression Training + Evaluation
"""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def run_regression(X_train, X_test, y_train, y_test):

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = {}
    best_model = None
    best_score = -np.inf

    for name, model in models.items():

        # train
        model.fit(X_train, y_train)

        # predict
        y_pred = model.predict(X_test)

        # metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "MAE": float(mae),
            "MSE": float(mse),
            "R2": float(r2)
        }

        # select best model using R2
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

    return {
        "best_model": best_model,
        "best_model_name": best_name,
        "results": results
    }