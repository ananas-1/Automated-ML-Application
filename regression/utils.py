"""
Regression Utilities
"""


def print_regression_results(model_name: str, metrics: dict) -> None:
    """Print regression results"""
    print(f"\n{'='*50}")
    print(f"Regression Results: {model_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
