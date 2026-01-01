# models/model_training.py

import json
import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier , ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load Preprocessing Pipeline
# --------------------------------------------------
def load_preprocessing_pipeline(path="../models/preprocessing.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------
# 2. Define Candidate Models
# --------------------------------------------------
def get_candidate_models():
    """
    Returns a dictionary of candidate models.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ),
        # "SVM": SVC(kernel="rbf", probability=True)
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ),
    }


# --------------------------------------------------
# 3. Cross-Validation Evaluation
# --------------------------------------------------
def evaluate_models(models, X_train, y_train, X_test, y_test, cv=5):
    """
    Evaluate models using cross-validation and test set metrics.
    Assumes 'Pass' is the positive class.
    """

    results = {}

    for name, model in models.items():

        # ---------------------------
        # Cross-validation (Accuracy)
        # ---------------------------
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        # ---------------------------
        # Train model
        # ---------------------------
        model.fit(X_train, y_train)

        # ---------------------------
        # Predictions
        # ---------------------------
        y_pred = model.predict(X_test)

        # Probabilities (if supported)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(
                (y_test == "Pass").astype(int), y_prob
            )
        else:
            roc_auc = None

        # ---------------------------
        # Metrics
        # ---------------------------
        results[name] = {
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std(),

            "test_accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="Pass"),
            "recall": recall_score(y_test, y_pred, pos_label="Pass"),

            # F2-score → recall focused
            "f2_score": fbeta_score(
                y_test, y_pred, beta=2, pos_label="Pass"
            ),

            "roc_auc": roc_auc,

            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=["Fail", "Pass"]
            ).tolist()
        }

    return results



# --------------------------------------------------
# 4. Select Best Model
# --------------------------------------------------
def select_best_model(results: dict, metric: str = "f2_score"):
    """
    Select the best model based on a specified metric.
    Default metric: f2_score (recall-focused).
    """

    best_model_name = max(
        results,
        key=lambda k: results[k][metric]
    )

    return best_model_name


# --------------------------------------------------
# 5. Train Final Model
# --------------------------------------------------
def train_final_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------
# 6. Save Model and Metadata
# --------------------------------------------------
def save_model_and_metadata(model, metadata: dict):
    with open("../models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("../models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
