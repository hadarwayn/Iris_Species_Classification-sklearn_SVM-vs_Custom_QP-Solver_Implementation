"""
Multi-class wrapper using One-vs-Rest strategy.

WHY: Standard SVM is binary. For 3 Iris classes, we train 3 binary classifiers,
each detecting one class vs all others, then combine predictions.
"""
import numpy as np

from src.qp_svm import QPSVM
from src.data_loader import prepare_binary_labels
from src.utils.logger import logger


class OneVsRestQPSVM:
    """
    Multi-class SVM using One-vs-Rest (OvR) strategy with QP solver.

    WHY: OvR is simpler than One-vs-One and works well when classes are
    reasonably balanced. Each classifier learns to separate one class.

    Strategy:
        Classifier 0: Setosa (+1) vs Versicolor+Virginica (-1)
        Classifier 1: Versicolor (+1) vs Setosa+Virginica (-1)
        Classifier 2: Virginica (+1) vs Setosa+Versicolor (-1)
    """

    def __init__(self, n_classes: int = 3) -> None:
        """Initialize OvR classifier with specified number of classes."""
        self.n_classes = n_classes
        self.classifiers: list[QPSVM] = []
        self.training_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsRestQPSVM":
        """
        Train one binary classifier per class.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Multi-class labels (0, 1, 2)
        """
        logger.info(f"Training OvR with {self.n_classes} classifiers...")
        self.classifiers = []
        total_time = 0.0

        for class_idx in range(self.n_classes):
            # Create binary labels: +1 for this class, -1 for others
            y_binary = prepare_binary_labels(y, class_idx)
            classifier = QPSVM()
            classifier.fit(X, y_binary)
            self.classifiers.append(classifier)
            total_time += classifier.training_time
            logger.info(f"  Classifier {class_idx}: {len(classifier.support_indices)} SVs")

        self.training_time = total_time
        logger.info(f"OvR training complete in {self.training_time:.4f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels by choosing highest confidence classifier.

        WHY: Each classifier outputs a confidence score. We pick the class
        whose "I am this class" classifier is most confident.
        """
        # Get confidence scores from all classifiers
        confidences = self.decision_function(X)
        # Return class with highest confidence
        return np.argmax(confidences, axis=1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get confidence scores from all classifiers.

        Returns:
            Array of shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        confidences = np.zeros((n_samples, self.n_classes))

        for class_idx, clf in enumerate(self.classifiers):
            confidences[:, class_idx] = clf.decision_function(X)

        return confidences

    def get_all_support_vectors(self) -> list[np.ndarray]:
        """Get support vectors from all classifiers."""
        return [clf.support_vectors for clf in self.classifiers]

    def get_all_weights(self) -> list[tuple[np.ndarray, float]]:
        """Get (w, b) from all classifiers."""
        return [clf.get_weights() for clf in self.classifiers]

    def get_total_n_support_vectors(self) -> int:
        """Get total number of support vectors across all classifiers."""
        return sum(len(clf.support_indices) for clf in self.classifiers)
