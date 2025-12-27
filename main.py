"""
Main entry point for Iris SVM Comparison project.

WHY: This orchestrates the full comparison pipeline: loading data, training
both SVM implementations, computing metrics, and generating visualizations.
"""
import argparse
import json
import numpy as np
from pathlib import Path

from src.data_loader import load_iris_data, split_data, normalize_features
from src.sklearn_svm import SklearnSVM
from src.multiclass import OneVsRestQPSVM
from src.metrics import (
    calculate_accuracy, confusion_matrix, classification_report,
    cosine_similarity, format_comparison_summary
)
from src.visualizer import (
    plot_decision_boundary, plot_confusion_matrix,
    plot_accuracy_comparison, plot_time_comparison, plot_feature_importance
)
from src.utils.paths import get_graphs_dir, get_run_dir
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Iris SVM Comparison")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument("--kernel", type=str, default="linear", help="Kernel type for sklearn")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--run", type=int, default=1, help="Run number for saving results")
    return parser.parse_args()


def run_comparison(C: float, kernel: str, test_size: float, run_number: int) -> dict:
    """Run full comparison between sklearn and QP solver."""
    logger.info(f"Starting Run {run_number}: C={C}, kernel={kernel}, test_size={test_size}")

    # Load and prepare data
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)

    # Train sklearn SVM
    logger.info("Training sklearn SVM...")
    sklearn_svm = SklearnSVM(kernel=kernel, C=C)
    sklearn_svm.fit(X_train_norm, y_train)
    sklearn_pred = sklearn_svm.predict(X_test_norm)
    sklearn_acc = calculate_accuracy(y_test, sklearn_pred)
    sklearn_cm = confusion_matrix(y_test, sklearn_pred)

    # Train QP solver SVM (OvR multi-class)
    logger.info("Training QP Solver SVM...")
    qp_svm = OneVsRestQPSVM(n_classes=3)
    qp_svm.fit(X_train_norm, y_train)
    qp_pred = qp_svm.predict(X_test_norm)
    qp_acc = calculate_accuracy(y_test, qp_pred)
    qp_cm = confusion_matrix(y_test, qp_pred)

    # Calculate weight similarity (for linear kernel only)
    weight_sim = None
    if kernel == "linear":
        try:
            sklearn_w, _ = sklearn_svm.get_weights()
            # Get first classifier's weights from QP (Setosa vs rest)
            qp_w, _ = qp_svm.classifiers[0].get_weights()
            weight_sim = abs(cosine_similarity(sklearn_w, qp_w))
        except Exception as e:
            logger.warning(f"Could not compute weight similarity: {e}")

    # Prepare output directories
    graphs_dir = get_graphs_dir()
    run_dir = get_run_dir(run_number)
    run_graphs = run_dir / "graphs"
    run_graphs.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_decision_boundary(X_train_norm, y_train, sklearn_svm.predict,
                           f"sklearn SVM Decision Boundary (C={C})",
                           run_graphs / "decision_boundary_sklearn.png",
                           sklearn_svm.get_support_indices())

    plot_decision_boundary(X_train_norm, y_train, qp_svm.predict,
                           f"QP Solver SVM Decision Boundary (C={C})",
                           run_graphs / "decision_boundary_qp.png")

    plot_confusion_matrix(sklearn_cm, "sklearn SVM Confusion Matrix",
                          run_graphs / "confusion_matrix_sklearn.png")
    plot_confusion_matrix(qp_cm, "QP Solver SVM Confusion Matrix",
                          run_graphs / "confusion_matrix_qp.png")
    plot_accuracy_comparison(sklearn_acc, qp_acc, run_graphs / "comparison_accuracy.png")
    plot_time_comparison(sklearn_svm.training_time, qp_svm.training_time,
                         run_graphs / "comparison_time.png")

    # Feature importance from first QP classifier
    if qp_svm.classifiers:
        w, _ = qp_svm.classifiers[0].get_weights()
        plot_feature_importance(w, run_graphs / "feature_importance.png")

    # Save metrics
    results = {
        "run_number": run_number,
        "parameters": {"C": C, "kernel": kernel, "test_size": test_size},
        "sklearn": {
            "accuracy": round(sklearn_acc, 4),
            "training_time": round(sklearn_svm.training_time, 4),
            "n_support_vectors": sklearn_svm.get_n_support_vectors(),
            "classification_report": classification_report(y_test, sklearn_pred),
        },
        "qp_solver": {
            "accuracy": round(qp_acc, 4),
            "training_time": round(qp_svm.training_time, 4),
            "n_support_vectors": qp_svm.get_total_n_support_vectors(),
            "classification_report": classification_report(y_test, qp_pred),
        },
        "comparison": {"weight_cosine_similarity": round(weight_sim, 4) if weight_sim else None},
    }

    # Save JSON results
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save summary text
    summary = format_comparison_summary(sklearn_acc, qp_acc, sklearn_svm.training_time,
                                        qp_svm.training_time, weight_sim)
    with open(run_dir / "summary.txt", "w") as f:
        f.write(summary)

    print(summary)
    logger.info(f"Run {run_number} complete. Results saved to {run_dir}")
    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_comparison(args.C, args.kernel, args.test_size, args.run)


if __name__ == "__main__":
    main()
