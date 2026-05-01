import os
import pandas as pd


UPLOAD_DIR = "uploads"


def get_dataset_path(dataset_id: str):
    possible_files = [
        os.path.join(UPLOAD_DIR, f"{dataset_id}.csv"),
        os.path.join(UPLOAD_DIR, f"{dataset_id}.xlsx")
    ]

    for path in possible_files:
        if os.path.exists(path):
            return path

    return None


def load_dataset(file_path: str):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)
