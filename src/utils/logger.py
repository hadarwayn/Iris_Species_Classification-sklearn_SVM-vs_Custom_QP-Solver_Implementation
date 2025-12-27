"""
Ring buffer logger for the Iris SVM project.

WHY: Ring buffer prevents log files from growing infinitely by automatically
removing old entries when the maximum is reached.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from src.utils.paths import get_logs_dir, get_project_root

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class RingBufferLogger:
    """
    Logger that maintains a fixed-size buffer of log entries.

    WHY: Production systems need bounded log sizes to prevent disk exhaustion.
    """

    def __init__(self, max_lines: int = 1000, log_file: str = "app.log") -> None:
        """Initialize ring buffer logger with max size and output file."""
        self.max_lines = max_lines
        self.log_path = get_logs_dir() / log_file
        self.entries: list[str] = []
        self._load_config()
        self._load_existing()

    def _load_config(self) -> None:
        """Load configuration from log_config.json if it exists."""
        config_path = get_project_root() / "logs" / "config" / "log_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                self.max_lines = config.get("max_lines", self.max_lines)

    def _load_existing(self) -> None:
        """Load existing log entries if file exists."""
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                self.entries = f.read().strip().split("\n")
                self.entries = [e for e in self.entries if e]  # Remove empty

    def _write(self) -> None:
        """Write entries to file, maintaining ring buffer size."""
        # Keep only the last max_lines entries
        if len(self.entries) > self.max_lines:
            self.entries = self.entries[-self.max_lines:]
        with open(self.log_path, "w") as f:
            f.write("\n".join(self.entries) + "\n")

    def log(self, level: LogLevel, message: str) -> None:
        """Add a log entry with timestamp and level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} - {level} - {message}"
        self.entries.append(entry)
        self._write()

    def info(self, message: str) -> None:
        """Log an INFO level message."""
        self.log("INFO", message)

    def warning(self, message: str) -> None:
        """Log a WARNING level message."""
        self.log("WARNING", message)

    def error(self, message: str) -> None:
        """Log an ERROR level message."""
        self.log("ERROR", message)

    def debug(self, message: str) -> None:
        """Log a DEBUG level message."""
        self.log("DEBUG", message)


# Global logger instance
logger = RingBufferLogger()
