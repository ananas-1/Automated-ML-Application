"""
Classification Utilities
"""


def print_classification_results(model_name: str, metrics: dict) -> None:
    """Print classification results"""
    print(f"\n{'='*50}")
    print(f"Classification Results: {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric.upper()}: {value:.4f}")
