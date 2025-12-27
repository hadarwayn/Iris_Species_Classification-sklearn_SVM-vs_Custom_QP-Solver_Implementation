"""
Data loader for Iris dataset.

WHY: Centralizing data loading ensures consistent preprocessing and makes
it easy to modify data handling in one place.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.paths import get_iris_csv_path
from src.utils.logger import logger


# Species encoding mapping
SPECIES_MAP = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


def load_iris_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Iris dataset from CSV.

    WHY: Loading from local CSV gives us control over the data source and
    ensures reproducibility without network dependencies.

    Returns:
        X: Feature matrix of shape (150, 4)
        y: Labels array of shape (150,) with values 0, 1, 2
    """
    csv_path = get_iris_csv_path()
    logger.info(f"Loading Iris data from {csv_path}")

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df["Species"].map(SPECIES_MAP).values.astype(np.int32)

    logger.info(f"Loaded {len(y)} samples with {X.shape[1]} features")
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    WHY: Fixed random_state ensures reproducible splits for comparing
    different algorithms on exactly the same data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Split: {len(y_train)} train, {len(y_test)} test samples")
    return X_train, X_test, y_train, y_test


def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize features using StandardScaler.

    WHY: SVM is sensitive to feature scales. Normalizing ensures all features
    contribute equally to the decision boundary.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Features normalized with StandardScaler")
    return X_train_scaled, X_test_scaled


def prepare_binary_labels(y: np.ndarray, positive_class: int) -> np.ndarray:
    """
    Convert multi-class labels to binary {-1, +1} for SVM.

    WHY: Standard SVM formulation uses +1/-1 labels. This function enables
    One-vs-Rest strategy by treating one class as positive, rest as negative.

    Args:
        y: Multi-class labels (0, 1, 2)
        positive_class: Which class should be +1

    Returns:
        Binary labels: +1 for positive_class, -1 for others
    """
    return np.where(y == positive_class, 1, -1).astype(np.float64)
