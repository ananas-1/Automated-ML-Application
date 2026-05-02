from backend.utils.file_handler import get_dataset_path, load_dataset
from preprocessing.preprocessing import run_preprocessing
from classification.classification import run_classification
from regression.regression import run_regression
from clustering.clustering import run_clustering


MODELS_STORE = {}


def run_pipeline(dataset_id: str, config: dict):
    file_path = get_dataset_path(dataset_id)
    if not file_path:
        raise ValueError("Dataset not found")

    df = load_dataset(file_path)

    task = config["task"]
    target = config.get("target_column")

    if task in ["classification", "regression"]:
        preprocessing_result = (run_preprocessing(
            df=df,
            task=task,
            target_col=target
        ))

        if task == "classification":
            model_result = run_classification(preprocessing_result)

            MODELS_STORE[dataset_id] = {
                "task": task,
                "model": model_result["best_model_object"],
                "pipeline": preprocessing_result["pipeline"],
                "metrics": model_result["best_model_metrics"],
            }

            return {
                "task": task,
                "best_model": model_result["best_model_name"],
                "metrics": model_result["best_model_metrics"],
                "all_models": model_result["results_by_model"],
                "data_report": preprocessing_result["report"]
            }

        else:
            model_result = run_regression(
                preprocessing_result["X_train"],
                preprocessing_result["X_test"],
                preprocessing_result["y_train"],
                preprocessing_result["y_test"]
            )

            MODELS_STORE[dataset_id] = {
                "task": task,
                "model": model_result["best_model"],
                "pipeline": preprocessing_result["pipeline"],
                "metrics": model_result["results"][model_result["best_model_name"]],
            }

            return {
                "task": task,
                "best_model": model_result["best_model_name"],
                "metrics": model_result["results"][model_result["best_model_name"]],
                "all_models": model_result["results"],
                "data_report": preprocessing_result["report"]
            }

    elif task == "clustering":
        model_result = run_clustering(df)

        MODELS_STORE[dataset_id] = {
            "task": task,
            "model": model_result["best_model"],
            "pipeline": model_result["pipeline"],
            "metrics": {"silhouette_score": model_result["best_score"]}
        }

        return {
            "task": task,
            "best_model": model_result["best_name"],
            "metrics": {
                "silhouette_score": model_result["best_score"]
            },
            "all_models": {
                "KMeans": model_result["kmeans_score"],
                "DBSCAN": model_result["dbscan_score"]
            },
            "cluster_sizes": model_result["cluster_sizes"],
            "cluster_labels": model_result["cluster_labels"],
            "n_clusters": model_result["n_clusters"]
        }

    else:
        raise ValueError("Invalid task")
