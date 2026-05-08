from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def find_best_k_for_knn(X_train, y_train, k_values: list = None, cv: int = 5, scoring: str = "f1_weighted") -> Dict[str, Any]:
  
    if k_values is None:
        k_values = [3, 5, 7, 9, 11]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {"k_values": k_values, "cv_scores": {}}

    for k in k_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring=scoring)
        mean_score = scores.mean()
        results["cv_scores"][k] = {"mean": mean_score, "std": scores.std()}

    best_k = max(results["cv_scores"], key=lambda k: results["cv_scores"][k]["mean"])
    results["best_k"] = best_k
    results["best_score"] = results["cv_scores"][best_k]["mean"]

    return results


def _validate_preprocessed_input(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise TypeError("data must be the dictionary returned by preprocessing.run_preprocessing")

    required_keys = ["X_train", "X_test", "y_train", "y_test"]
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"Preprocessing output is missing required keys: {missing}")

    if data["X_train"] is None or data["y_train"] is None:
        raise ValueError("X_train and y_train cannot be None")

    if data["X_test"] is None or data["y_test"] is None:
        raise ValueError("X_test and y_test cannot be None")

    if len(data["X_train"]) == 0 or len(data["y_train"]) == 0:
        raise ValueError("X_train and y_train must not be empty")

    if len(data["X_test"]) == 0 or len(data["y_test"]) == 0:
        raise ValueError("X_test and y_test must not be empty")


def _get_models(random_state: int, best_k: int = 5) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "KNN": KNeighborsClassifier(n_neighbors=best_k),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
    }

    return models


def evaluate_classification_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def run_classification(
    data: Dict[str, Any],
    random_state: int = 42,
) -> Dict[str, Any]:

    _validate_preprocessed_input(data=data)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    k_search_results = find_best_k_for_knn(X_train, y_train, k_values=[3, 5, 7, 9, 11], cv=5)
    best_k = k_search_results["best_k"]
    print(f"\nOptimal K for KNN found: {best_k} (CV F1-Score: {k_search_results['best_score']:.4f})")

    models = _get_models(random_state=random_state, best_k=best_k)

    all_results: Dict[str, Dict[str, Any]] = {}
    best_model_name = ""
    best_model = None
    best_f1 = -1.0

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_classification_model(model=model, X_test=X_test, y_test=y_test)
        all_results[model_name] = metrics

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_model = model

    return {
        "split": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "test_ratio": float(len(X_test) / (len(X_train) + len(X_test))),
        },
        "used_xgboost": False,
        "results_by_model": all_results,
        "best_model_name": best_model_name,
        "best_model_metrics": all_results[best_model_name],
        "best_model_object": best_model,
    }


def print_classification_summary(output: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("GENERIC CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Split: train={output['split']['train_size']} | test={output['split']['test_size']}")
    print(f"Best model: {output['best_model_name']}")

    print("\n" + "-" * 40)
    print("MODEL COMPARISON")
    print("-" * 40)

    for model_name, metrics in output["results_by_model"].items():
        suffix = " (best)" if model_name == output["best_model_name"] else ""
        print(f"\n{model_name}{suffix}")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")
