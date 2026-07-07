from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score
)
import numpy as np

def compute_metrics(y_true, y_pred):
    # --- All-class metrics ---
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # --- Imbalance-aware metrics (binarised: New Track == 0 vs Existing Class != 0) ---
    binary_true = (y_true != 0).astype(int)
    binary_pred = (y_pred != 0).astype(int)

    bal_acc = balanced_accuracy_score(binary_true, binary_pred)

    per_class_precision, per_class_recall, per_class_f1, _ = \
        precision_recall_fscore_support(binary_true, binary_pred, average=None, zero_division=0)

    class_weights = 1.0 / (np.bincount(binary_true) + 1e-6)
    class_weights /= class_weights.sum()

    return {
        # Multi-class
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Imbalance-aware binary
        "balanced_accuracy": bal_acc,
        "weighted_precision": np.sum(per_class_precision * class_weights),
        "weighted_recall": np.sum(per_class_recall * class_weights),
        "weighted_f1": np.sum(per_class_f1 * class_weights),
    }