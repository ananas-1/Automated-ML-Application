import pandas as pd

from preprocessing.preprocessing import run_preprocessing
from regression.regression import run_regression


# 1 load dataset
df = pd.read_csv("regression/Mall_Customers (1).csv")


# 2 run preprocessing
data = run_preprocessing(
    df,
    task="regression",
    target_col="Spending Score (1-100)"   
)


# 3 run regression
result = run_regression(
    data["X_train"],
    data["X_test"],
    data["y_train"],
    data["y_test"]
)


# 4 print results
print("Best Model:", result["best_model_name"])
print()

for model, metrics in result["results"].items():
    print(model)
    print(metrics)
    print()