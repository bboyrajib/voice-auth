"""
src/evaluation/metrics.py
Evaluation functions: EER, standard classification metrics, and plotting.
EER (Equal Error Rate) is the primary metric in anti-spoofing literature.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    EER is the point where False Acceptance Rate == False Rejection Rate.
    Lower EER = better model.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr  # False Negative Rate = False Rejection Rate

    # Find threshold where FPR ≈ FNR
    abs_diff = np.abs(fpr - fnr)
    eer_idx = np.argmin(abs_diff)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return float(eer)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_score: np.ndarray) -> dict:
    """
    Compute the full set of evaluation metrics.
    y_true:  ground truth labels (0 or 1)
    y_pred:  predicted labels
    y_score: predicted probability for class 1 (synthetic)
    """
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_score)),
        "eer":       compute_eer(y_true, y_score),
    }


def print_metrics(metrics: dict, model_name: str = ""):
    header = f"── {model_name} ──" if model_name else "── Metrics ──"
    print(header)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  EER      : {metrics['eer']:.4f}  ← primary metric")
