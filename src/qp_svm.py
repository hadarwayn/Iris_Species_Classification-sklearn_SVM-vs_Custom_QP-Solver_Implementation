"""
Custom QP Solver SVM implementation using cvxopt.

WHY: Building SVM from scratch using Quadratic Programming demonstrates the
mathematical foundations: Lagrange multipliers, KKT conditions, support vectors.

Dual formulation: min (1/2) * alpha.T @ H @ alpha - sum(alpha)
Where H[i,j] = y_i * y_j * (x_i.T @ x_j)
"""
import time
import numpy as np
from cvxopt import matrix, solvers

from src.utils.logger import logger

solvers.options["show_progress"] = False


def build_gram_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Build the Gram matrix H for QP. H[i,j] = y_i * y_j * (x_i.T @ x_j).
    WHY: Encodes pairwise similarities weighted by labels.
    """
    n = X.shape[0]
    gram = np.outer(y, y) * (X @ X.T)
    gram += 1e-8 * np.eye(n)  # Numerical stability
    return gram


def build_qp_matrices(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Build cvxopt QP matrices: min (1/2)x'Px + q'x, s.t. Gx<=h, Ax=b.
    WHY: Maps SVM dual to cvxopt's standard QP format.
    """
    n = X.shape[0]
    P = matrix(build_gram_matrix(X, y))
    q = matrix(-np.ones(n))
    G = matrix(-np.eye(n))  # alpha >= 0
    h = matrix(np.zeros(n))
    A = matrix(y.reshape(1, -1).astype(np.float64))  # sum(alpha*y) = 0
    b = matrix(np.zeros(1))
    return P, q, G, h, A, b


def solve_qp(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve QP and return Lagrange multipliers. Points with alpha > 0 are SVs."""
    P, q, G, h, A, b = build_qp_matrices(X, y)
    solution = solvers.qp(P, q, G, h, A, b)
    return np.array(solution["x"]).flatten()


def extract_support_vectors(alphas: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """Find indices of support vectors (alpha > threshold)."""
    return np.where(alphas > threshold)[0]


def compute_weights(alphas: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute w = sum(alpha_i * y_i * x_i). Normal vector to hyperplane."""
    return np.sum((alphas * y).reshape(-1, 1) * X, axis=0)


def compute_bias(w: np.ndarray, X_sv: np.ndarray, y_sv: np.ndarray) -> float:
    """Compute bias from KKT: b = y_s - w.T @ x_s, averaged over SVs."""
    return float(np.mean(y_sv - X_sv @ w))


class QPSVM:
    """Complete QP-based SVM classifier for binary classification."""

    def __init__(self) -> None:
        """Initialize QP SVM with empty model parameters."""
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.alphas: np.ndarray | None = None
        self.support_indices: np.ndarray | None = None
        self.support_vectors: np.ndarray | None = None
        self.training_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QPSVM":
        """Train QP SVM on binary labels {-1, +1}."""
        start = time.perf_counter()
        logger.info("Training QP SVM...")
        self.alphas = solve_qp(X, y)
        self.support_indices = extract_support_vectors(self.alphas)
        self.support_vectors = X[self.support_indices]
        self.w = compute_weights(self.alphas, X, y)
        self.b = compute_bias(self.w, self.support_vectors, y[self.support_indices])
        self.training_time = time.perf_counter() - start
        logger.info(f"QP SVM trained: {len(self.support_indices)} SVs, {self.training_time:.4f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels {-1, +1}."""
        return np.sign(self.decision_function(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision values: w.T @ x + b."""
        if self.w is None:
            raise ValueError("Model not trained")
        return X @ self.w + self.b

    def get_weights(self) -> tuple[np.ndarray, float]:
        """Return (w, b) for comparison with sklearn."""
        if self.w is None:
            raise ValueError("Model not trained")
        return self.w, self.b
