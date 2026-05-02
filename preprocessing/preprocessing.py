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


# ── File Loading ────────────────────────────────────────────────────────────


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


# ── Main Preprocessing ──────────────────────────────────────────────────────

def run_preprocessing(df, task, target_col=None, test_size=0.2, random_state=42):
    """
    Runs the full preprocessing pipeline on the given dataframe.

    Steps:
        1. Separate features and target
        2. Drop rows where target is missing (supervised only)
        3. Drop columns with too many missing values (> 80%)
        4. Drop constant columns (only 1 unique value)
        5. Drop high-cardinality categorical columns (> 50 unique)
        6. Fix mixed-type columns (force object cols to string)
        7. Encode the target column
        8. Train/test split (with safe fallback if stratify fails)
        9. Build imputer + scaler + encoder transformers
        10. Check for class imbalance and apply SMOTE if needed
        11. Fit and transform train set, transform test set
        12. Extract feature names
    """

    task = task.strip().lower()

    if task not in ("classification", "regression", "clustering"):
        raise ValueError("Task must be 'classification', 'regression', or 'clustering'.")

    df = df.copy()
    report = {}
    report["task"] = task

    # Step 1: split features from target
    if task == "classification" or task == "regression":
        if target_col is None or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

        # Step 2: drop rows where the target itself is missing
        # (can't train on rows with no label)
        missing_target = df[target_col].isnull().sum()
        if missing_target > 0:
            df = df.dropna(subset=[target_col])
            report["dropped_missing_target_rows"] = int(missing_target)

        if df.empty:
            raise ValueError("All rows were dropped because the target column was entirely empty.")

        y_raw = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        # clustering has no target
        y_raw = None
        X = df.copy()

    report["original_shape"] = list(X.shape)

    # Step 3: remove columns where more than 80% of values are missing
    missing_ratios = X.isnull().mean()
    cols_to_drop = missing_ratios[missing_ratios > 0.8].index.tolist()
    X.drop(columns=cols_to_drop, inplace=True)
    report["dropped_high_missing"] = cols_to_drop

    # Step 4: remove columns that have only 1 unique value (no info at all)
    constant_cols = []
    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    X.drop(columns=constant_cols, inplace=True)
    report["dropped_constant"] = constant_cols

    # Step 5: remove categorical columns with way too many unique values
    # (e.g. Name, ID, Ticket - these would create thousands of dummy columns)
    high_card = []
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].nunique(dropna=True) > 50:
            high_card.append(col)
    X.drop(columns=high_card, inplace=True)
    report["dropped_high_cardinality"] = high_card

    if X.shape[1] == 0:
        raise ValueError("No usable columns left after cleaning. Check your dataset.")

    # Step 6: fix mixed-type columns
    # some columns have a mix of strings and numbers which breaks the pipeline
    # forcing them to string makes sure they get treated as categorical
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str)

    # Step 7: identify numeric vs categorical columns
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    report["numeric_columns"] = num_cols
    report["categorical_columns"] = cat_cols

    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("No numeric or categorical columns found after preprocessing.")

    # Step 8: encode the target column
    label_encoder = None

    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        report["target_classes"] = label_encoder.classes_.tolist()

    elif task == "regression":
        # make sure the target is actually numeric
        try:
            y = pd.to_numeric(y_raw, errors="raise").values.astype(float)
        except Exception:
            raise ValueError(
                f"Target column '{target_col}' contains non-numeric values. "
                "Regression requires a numeric target."
            )
    else:
        y = None

    # Step 9: train/test split
    # for classification we try stratified split first
    # if it fails (e.g. some class has only 1 sample), we fall back to normal split
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
                # stratify failed - probably a class with very few samples
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

    # Step 10: build the transformers
    #   numeric  → fill missing with median  → StandardScaler
    #   categorical → fill missing with most frequent → OneHotEncoder
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

    # Step 11: check if we need SMOTE
    # only applies to classification, and only if classes are imbalanced
    # SMOTE needs at least 2 samples in the minority class to work
    use_smote = False
    smote_k = 5
    report["resampling"] = "none"

    if task == "classification":
        class_counts = np.bincount(y_train)
        min_class_size = class_counts.min()
        ratio = min_class_size / class_counts.max()
        report["imbalance_ratio"] = round(float(ratio), 4)

        if ratio < 0.4 and min_class_size >= 2:
            # k_neighbors must be less than the number of minority samples
            smote_k = max(1, min(5, min_class_size - 1))
            use_smote = True
        elif ratio < 0.4 and min_class_size < 2:
            report["resampling"] = "SMOTE skipped - minority class has too few samples"

    # Step 12: build the final pipeline (with or without SMOTE)
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

    # Step 13: fit on train, transform both sets
    if use_smote:
        X_train, y_train = final_pipeline.fit_resample(X_train_raw, y_train)
    else:
        X_train = final_pipeline.fit_transform(X_train_raw)

    if len(X_test_raw) > 0:
        X_test = final_pipeline.named_steps["preprocessor"].transform(X_test_raw)
    else:
        X_test = np.array([])

    # Step 14: get feature names after encoding
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


# ── Helper : correct feature names after OneHotEncoder ───

def get_feature_names(column_transformer, num_cols, cat_cols):
    """
    Gets the feature names after ColumnTransformer runs.
    Needed because OneHotEncoder expands one column into many.
    Falls back to index-based names if something goes wrong.
    """
    names = []

    for name, pipe, cols in column_transformer.transformers_:
        if name == "remainder":
            continue

        # get the last step in the sub-pipeline
        last_step = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe

        try:
            if hasattr(last_step, "get_feature_names_out"):
                # OneHotEncoder returns names like "col_value"
                encoded_names = last_step.get_feature_names_out(cols)
                names.extend(encoded_names.tolist())
            else:
                # StandardScaler and similar keep original names
                names.extend(list(cols))
        except Exception:
            # fallback: use generic names so the pipeline doesn't crash
            names.extend([f"{name}_feature_{i}" for i in range(len(cols))])

    return names
