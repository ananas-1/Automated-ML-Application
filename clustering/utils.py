"""
Clustering Utilities
"""


def print_clustering_results(model_name: str, metrics: dict) -> None:
    """Print clustering results"""
    print(f"\n{'='*50}")
    print(f"Clustering Results: {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
