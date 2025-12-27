"""
Path utilities for the Iris SVM project.

WHY: Using pathlib ensures cross-platform compatibility and relative paths
prevent hardcoded paths that break when the project is moved.
"""
from pathlib import Path


def get_project_root() -> Path:
    """
    Get the project root directory.

    WHY: We detect the root by looking for known project files, making the
    code work regardless of where it's executed from.
    """
    current = Path(__file__).resolve()
    # Navigate up from src/utils/paths.py to project root
    return current.parent.parent.parent


def get_data_dir() -> Path:
    """Get the Data directory containing datasets."""
    return get_project_root() / "Data"


def get_iris_csv_path() -> Path:
    """Get the path to Iris.csv dataset."""
    return get_data_dir() / "Iris-Species" / "Iris.csv"


def get_results_dir() -> Path:
    """Get the results directory for outputs."""
    path = get_project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_graphs_dir() -> Path:
    """Get the graphs directory for visualizations."""
    path = get_results_dir() / "graphs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_dir(run_number: int) -> Path:
    """Get the directory for a specific test run."""
    path = get_results_dir() / "examples" / f"run_{run_number}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir() -> Path:
    """Get the logs directory."""
    path = get_project_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
