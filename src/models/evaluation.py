# models/evaluation.py

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


# --------------------------------------------------
# 1. Load Trained Model
# --------------------------------------------------
def load_trained_model(path="../models/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------
# 2. Compute Classification Metrics
# --------------------------------------------------
def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics for string labels.
    Assumes 'Pass' is the positive class.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="Pass"),
        "recall": recall_score(y_true, y_pred, pos_label="Pass"),
        "f1_score": f1_score(y_true, y_pred, pos_label="Pass"),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(
            (y_true == "Pass").astype(int), y_prob
        )

    return metrics

# --------------------------------------------------
# 3. Confusion Matrix
# --------------------------------------------------
def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


# --------------------------------------------------
# 4. ROC Curve
# --------------------------------------------------
def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve for binary classification.
    Assumes 'Pass' is the positive class.
    """

    # Convert string labels to binary
    y_true_binary = (y_true == "Pass").astype(int)

    fpr, tpr, _ = roc_curve(y_true_binary, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()



# --------------------------------------------------
# 5. Precision-Recall Curve
# --------------------------------------------------
def plot_precision_recall_curve(y_true, y_prob):
    """
    Plot Precision-Recall curve.
    Assumes 'Pass' is the positive class.
    """

    # Convert string labels to binary
    y_true_binary = (y_true == "Pass").astype(int)

    precision, recall, _ = precision_recall_curve(
        y_true_binary, y_prob
    )

    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()



# --------------------------------------------------
# 6. Error Analysis
# --------------------------------------------------
def get_misclassifications(X_test, y_true, y_pred):
    """
    Return misclassified samples for inspection.
    """
    misclassified_idx = np.where(y_true != y_pred)[0]
    return X_test.iloc[misclassified_idx]


# --------------------------------------------------
# 7. Feature Importance (Tree-Based Models)
# --------------------------------------------------
def get_feature_importance(model):
    """
    Extract feature importance if supported.
    """
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return None


# --------------------------------------------------
# 8. Save Evaluation Results
# --------------------------------------------------
def save_evaluation_results(results: dict):
    with open("../models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
