"""
Visualization module for SVM comparison.

WHY: Visual comparison makes abstract concepts concrete. Decision boundaries,
support vectors, and accuracy charts help understand how algorithms differ.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path

from src.utils.paths import get_graphs_dir


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn,
    title: str,
    save_path: Path,
    support_indices: np.ndarray | None = None,
) -> None:
    """
    Plot 2D decision boundary using PCA projection.

    WHY: Iris has 4 features, but we can only visualize 2D. PCA finds the
    2 dimensions with most variance, giving the best 2D view of the data.
    """
    # Reduce to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Create mesh grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict on mesh (transform back to original space)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_original = pca.inverse_transform(mesh_points)
    Z = predict_fn(mesh_original).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="RdYlBu", edgecolors="black", s=50)

    # Highlight support vectors
    if support_indices is not None and len(support_indices) > 0:
        plt.scatter(X_2d[support_indices, 0], X_2d[support_indices, 1],
                    facecolors="none", edgecolors="green", s=150, linewidths=2, label="Support Vectors")
        plt.legend()

    plt.colorbar(scatter, label="Class")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path) -> None:
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(8, 6))
    labels = ["Setosa", "Versicolor", "Virginica"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_comparison(sklearn_acc: float, qp_acc: float, save_path: Path) -> None:
    """Bar chart comparing accuracy of both methods."""
    plt.figure(figsize=(8, 6))
    methods = ["sklearn SVM", "QP Solver SVM"]
    accuracies = [sklearn_acc, qp_acc]
    colors = ["#3498db", "#e74c3c"]
    bars = plt.bar(methods, accuracies, color=colors)
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison: sklearn vs QP Solver")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{acc:.2%}", ha="center", fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_comparison(sklearn_time: float, qp_time: float, save_path: Path) -> None:
    """Bar chart comparing training time."""
    plt.figure(figsize=(8, 6))
    methods = ["sklearn SVM", "QP Solver SVM"]
    times = [sklearn_time, qp_time]
    colors = ["#3498db", "#e74c3c"]
    bars = plt.bar(methods, times, color=colors)
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time Comparison")
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{t:.4f}s", ha="center", fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(w: np.ndarray, save_path: Path) -> None:
    """Bar chart of feature importance from weight vector."""
    plt.figure(figsize=(10, 6))
    features = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    importance = np.abs(w)
    colors = plt.cm.viridis(importance / importance.max())
    bars = plt.bar(features, importance, color=colors)
    plt.ylabel("|Weight|")
    plt.title("Feature Importance (Weight Magnitudes)")
    for bar, imp in zip(bars, importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{imp:.3f}", ha="center", fontsize=10)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
