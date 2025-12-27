"""
scikit-learn SVM wrapper for Iris classification.

WHY: Wrapping sklearn's SVC gives us a consistent interface and allows us
to extract internal data (support vectors, weights) for comparison with QP solver.
"""
import time
import numpy as np
from sklearn.svm import SVC

from src.utils.logger import logger


class SklearnSVM:
    """
    Wrapper around sklearn.svm.SVC for easy comparison with QP solver.

    WHY: sklearn uses highly optimized LIBSVM internally. This serves as
    our reference "correct" implementation to validate our QP solver.
    """

    def __init__(self, kernel: str = "linear", C: float = 1.0):
        """
        Initialize the SVM classifier.

        Args:
            kernel: 'linear' for comparison with QP, 'rbf' for best accuracy
            C: Regularization parameter (smaller = more regularization)
        """
        self.kernel = kernel
        self.C = C
        self.model: SVC | None = None
        self.training_time: float = 0.0
        logger.info(f"SklearnSVM initialized: kernel={kernel}, C={C}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnSVM":
        """
        Train the SVM on the provided data.

        WHY: We measure training time to compare efficiency with QP solver.
        """
        start = time.perf_counter()
        self.model = SVC(kernel=self.kernel, C=self.C, random_state=42)
        self.model.fit(X, y)
        self.training_time = time.perf_counter() - start
        logger.info(f"sklearn SVM trained in {self.training_time:.4f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get confidence scores for predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.decision_function(X)

    def get_support_vectors(self) -> np.ndarray:
        """Get the support vectors from the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.support_vectors_

    def get_support_indices(self) -> np.ndarray:
        """Get indices of support vectors in training data."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.support_

    def get_weights(self) -> tuple[np.ndarray, float]:
        """
        Get weight vector and bias for linear kernel.

        WHY: For linear SVM, we can extract w and b directly to compare
        with our QP solver's solution.

        Returns:
            (w, b): Weight vector and bias term
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        if self.kernel != "linear":
            raise ValueError("Weights only available for linear kernel")
        w = self.model.coef_.flatten()
        b = self.model.intercept_[0]
        return w, b

    def get_training_time(self) -> float:
        """Get the training time in seconds."""
        return self.training_time

    def get_n_support_vectors(self) -> int:
        """Get total number of support vectors."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return len(self.model.support_)
