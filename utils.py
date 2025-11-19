"""
Simple color-coded logging utility with timing support.
"""

import time
from contextlib import contextmanager
from typing import Optional


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Colors
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    # Bright colors
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


class Logger:
    """Simple logger with color coding and timing."""

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self.start_time = None

    def _format(self, message: str, color: str, prefix: str = "") -> str:
        """Format a log message with color."""
        timestamp = time.strftime("%H:%M:%S")
        colored_prefix = f"{color}{Colors.BOLD}{prefix}{Colors.RESET}" if prefix else ""
        colored_msg = f"{color}{message}{Colors.RESET}"
        return f"[{timestamp}] {colored_prefix} {colored_msg}"

    def info(self, message: str):
        """Log an info message (blue)."""
        print(self._format(message, Colors.BLUE, "INFO"))

    def success(self, message: str):
        """Log a success message (green)."""
        print(self._format(message, Colors.BRIGHT_GREEN, "✓"))

    def warning(self, message: str):
        """Log a warning message (yellow)."""
        print(self._format(message, Colors.YELLOW, "⚠"))

    def error(self, message: str):
        """Log an error message (red)."""
        print(self._format(message, Colors.RED, "✗"))

    def step(self, message: str):
        """Log a pipeline step (cyan)."""
        print(self._format(message, Colors.CYAN, "→"))

    def stage(self, message: str):
        """Log a pipeline stage (magenta)."""
        print(self._format(message, Colors.MAGENTA, "═"))

    @contextmanager
    def timer(self, step_name: str):
        """Context manager to time a step."""
        self.step(f"Starting: {step_name}")
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.success(f"Completed: {step_name} ({elapsed:.2f}s)")

    def time(self, step_name: str, elapsed: float):
        """Log a completed step with timing."""
        self.success(f"{step_name} ({elapsed:.2f}s)")


# Global logger instance
logger = Logger("Flow OCR")
