"""Metrics and evaluation plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def plot_roc(y_true, y_prob, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    _save(out_path)


def plot_pr(y_true, y_prob, out_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f"PR (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend()
    _save(out_path)


def plot_confusion(y_true, y_prob, out_path: Path, threshold: float = 0.5) -> None:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["No churn", "Churn"])
    plt.yticks([0, 1], ["No churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    _save(out_path)


def _save(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
