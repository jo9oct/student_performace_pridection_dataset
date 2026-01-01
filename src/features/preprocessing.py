# src/features/preprocessing.py

import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# --------------------------------------------------
# 1. Column Removal
# --------------------------------------------------
def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns not required for modeling.
    """
    return df.drop(columns=["Student_ID", "Final_Exam_Score"], errors="ignore")


# --------------------------------------------------
# 2. Split Features and Target
# --------------------------------------------------
def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Separate features (X) and target (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# --------------------------------------------------
# 3. Build Preprocessing Pipeline
# --------------------------------------------------
def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create preprocessing pipeline for numerical and categorical features.
    """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # Step 1: fill missing values
        ("scaler", StandardScaler())                    # Step 2: normalize/scale
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessing_pipeline


# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# --------------------------------------------------
# 5. Apply Preprocessing
# --------------------------------------------------
def apply_preprocessing(pipeline, X_train, X_test):
    """
    Fit preprocessing pipeline and transform data.
    """
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    return X_train_processed, X_test_processed


# --------------------------------------------------
# 6. Save Outputs
# --------------------------------------------------
def save_outputs(
    X_train_processed,
    X_test_processed,
    y_train,
    y_test,
    pipeline
):
    """
    Save processed data and preprocessing pipeline.
    """

    pd.DataFrame(X_train_processed).to_csv(
        "../data/processed/X_train.csv", index=False
    )
    pd.DataFrame(X_test_processed).to_csv(
        "../data/processed/X_test.csv", index=False
    )
    y_train.to_csv("../data/processed/y_train.csv", index=False)
    y_test.to_csv("../data/processed/y_test.csv", index=False)

    with open("../models/preprocessing.pkl", "wb") as f:
        pickle.dump(pipeline, f)


# --------------------------------------------------
# 7. Master Orchestration Function
# --------------------------------------------------
def run_full_preprocessing(df: pd.DataFrame, target_col: str):
    """
    Run full preprocessing workflow step by step.
    """

    df = drop_unnecessary_columns(df)

    X, y = split_features_target(df, target_col)

    pipeline = build_preprocessing_pipeline(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_processed, X_test_processed = apply_preprocessing(
        pipeline, X_train, X_test
    )

    save_outputs(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        pipeline
    )

    return X_train, X_test, y_train, y_test
