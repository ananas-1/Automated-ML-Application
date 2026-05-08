import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


def load_dataset(file_obj, filename):
    fname = filename.lower()

    if fname.endswith(".csv"):
        df = pd.read_csv(file_obj)
    elif fname.endswith(".xlsx"):
        df = pd.read_excel(file_obj, engine="openpyxl")
    elif fname.endswith(".xls"):
        df = pd.read_excel(file_obj, engine="xlrd")
    else:
        raise ValueError("Please upload a .csv or .xlsx file only.")

    if df.empty:
        raise ValueError("The file you uploaded is empty!")

    return df


def run_preprocessing(df, task, target_col=None, test_size=0.2, random_state=42):
    task = task.strip().lower()

    if task not in ("classification", "regression", "clustering"):
        raise ValueError("Task must be 'classification', 'regression', or 'clustering'.")

    df = df.copy()
    report = {"task": task}

    if task == "classification" or task == "regression":
        if target_col is None or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

        missing_target = df[target_col].isnull().sum()
        if missing_target > 0:
            df = df.dropna(subset=[target_col])
            report["dropped_missing_target_rows"] = int(missing_target)

        if df.empty:
            raise ValueError("All rows were dropped because the target column was entirely empty.")

        y_raw = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        y_raw = None
        X = df.copy()

    report["original_shape"] = list(X.shape)

    missing_ratios = X.isnull().mean()
    cols_to_drop = missing_ratios[missing_ratios > 0.8].index.tolist()
    X.drop(columns=cols_to_drop, inplace=True)
    report["dropped_high_missing"] = cols_to_drop

    constant_cols = []
    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    X.drop(columns=constant_cols, inplace=True)
    report["dropped_constant"] = constant_cols

    high_card = []
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].nunique(dropna=True) > 50:
            high_card.append(col)
    X.drop(columns=high_card, inplace=True)
    report["dropped_high_cardinality"] = high_card

    if X.shape[1] == 0:
        raise ValueError("No usable columns left after cleaning. Check your dataset.")

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    report["numeric_columns"] = num_cols
    report["categorical_columns"] = cat_cols

    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("No numeric or categorical columns found after preprocessing.")

    label_encoder = None

    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        report["target_classes"] = label_encoder.classes_.tolist()

    elif task == "regression":
        try:
            y = pd.to_numeric(y_raw, errors="raise").values.astype(float)
        except Exception:
            raise ValueError(
                f"Target column '{target_col}' contains non-numeric values. "
                "Regression requires a numeric target."
            )
    else:
        y = None

    if task in ("classification", "regression"):
        if task == "classification":
            try:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y
                )
            except ValueError:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state
                )
                report["stratify_warning"] = "Stratified split failed, used random split instead."
        else:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )
    else:
        X_train_raw = X
        X_test_raw = pd.DataFrame()
        y_train = None
        y_test = None

    report["train_samples"] = len(X_train_raw)
    report["test_samples"] = len(X_test_raw)

    transformers = []

    if len(num_cols) > 0:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if len(cat_cols) > 0:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    report["imputation"] = "median for numeric / most_frequent for categorical"
    report["encoding"] = "OneHotEncoder"
    report["scaling"] = "StandardScaler"

    use_smote = False
    smote_k = 5
    report["resampling"] = "none"

    if task == "classification":
        class_counts = np.bincount(y_train)
        min_class_size = class_counts.min()
        ratio = min_class_size / class_counts.max()
        report["imbalance_ratio"] = round(float(ratio), 4)

        if ratio < 0.4 and min_class_size >= 2:
            smote_k = max(1, min(5, min_class_size - 1))
            use_smote = True
        elif ratio < 0.4 and min_class_size < 2:
            report["resampling"] = "SMOTE skipped - minority class has too few samples"

    if use_smote:
        smote = SMOTE(k_neighbors=smote_k, random_state=random_state)
        final_pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", smote)
        ])
        report["resampling"] = f"SMOTE applied (k_neighbors={smote_k})"
    else:
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor)
        ])

    if use_smote:
        X_train, y_train = final_pipeline.fit_resample(X_train_raw, y_train)
    else:
        X_train = final_pipeline.fit_transform(X_train_raw)

    if len(X_test_raw) > 0:
        X_test = final_pipeline.named_steps["preprocessor"].transform(X_test_raw)
    else:
        X_test = np.array([])

    feature_names = get_feature_names(final_pipeline["preprocessor"], num_cols, cat_cols)

    report["feature_names_after_encoding"] = feature_names
    report["final_feature_count"] = len(feature_names)
    report["final_train_shape"] = list(X_train.shape)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": final_pipeline,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "report": report
    }


def get_feature_names(column_transformer, num_cols, cat_cols):
    names = []

    for name, pipe, cols in column_transformer.transformers_:
        if name == "remainder":
            continue

        last_step = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe

        try:
            if hasattr(last_step, "get_feature_names_out"):
                encoded_names = last_step.get_feature_names_out(cols)
                names.extend(encoded_names.tolist())
            else:
                names.extend(list(cols))
        except Exception:
            names.extend([f"{name}_feature_{i}" for i in range(len(cols))])

    return names
