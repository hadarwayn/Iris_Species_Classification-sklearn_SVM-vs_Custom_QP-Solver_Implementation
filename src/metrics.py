"""
Metrics calculation for SVM comparison.

WHY: Comparing sklearn and QP solver requires multiple metrics beyond accuracy:
weight similarity, support vector overlap, and timing comparisons.
"""
import numpy as np
from typing import Any


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    WHY: Simple accuracy shows overall performance, though it doesn't
    reveal class-specific issues like imbalanced predictions.
    """
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """
    Build confusion matrix.

    WHY: Confusion matrix shows which classes are confused with each other.
    For Iris, we expect Versicolor/Virginica to be confused more than Setosa.

    Returns:
        Matrix where [i,j] = count of true class i predicted as class j
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict[str, Any]:
    """
    Calculate precision, recall, F1 per class.

    WHY: Per-class metrics reveal if classifier struggles with specific classes.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    report: dict[str, Any] = {}

    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp  # Column sum minus true positive
        fn = np.sum(cm[c, :]) - tp  # Row sum minus true positive

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[f"class_{c}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    report["accuracy"] = calculate_accuracy(y_true, y_pred)
    return report


def cosine_similarity(w1: np.ndarray, w2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two weight vectors.

    WHY: Cosine similarity measures if two vectors point in the same direction,
    ignoring magnitude. If sklearn and QP have similar directions (>0.95),
    they've found essentially the same separating hyperplane.
    """
    dot_product = np.dot(w1, w2)
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def support_vector_overlap(indices1: np.ndarray, indices2: np.ndarray) -> float:
    """
    Calculate overlap percentage between two sets of support vector indices.

    WHY: If both methods identify similar support vectors, they've found
    similar decision boundaries. High overlap validates our QP implementation.
    """
    set1 = set(indices1)
    set2 = set(indices2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return float(intersection / union)


def format_comparison_summary(
    sklearn_accuracy: float,
    qp_accuracy: float,
    sklearn_time: float,
    qp_time: float,
    weight_similarity: float | None = None,
    sv_overlap: float | None = None,
) -> str:
    """Format a comparison summary for console output."""
    lines = [
        "=" * 50,
        "         SVM Comparison Summary",
        "=" * 50,
        f"Accuracy:     sklearn={sklearn_accuracy:.4f}  QP={qp_accuracy:.4f}",
        f"Train Time:   sklearn={sklearn_time:.4f}s  QP={qp_time:.4f}s",
    ]
    if weight_similarity is not None:
        lines.append(f"Weight Cosine Similarity: {weight_similarity:.4f}")
    if sv_overlap is not None:
        lines.append(f"Support Vector Overlap:   {sv_overlap:.4f}")
    lines.append("=" * 50)
    return "\n".join(lines)
