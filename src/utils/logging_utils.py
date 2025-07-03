"""
Logging utilities for the Clipper application.
"""

import logging
import sys
from pathlib import Path
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clipper")


def configure_logging(log_level="INFO", log_file=None):
    """Configure logging for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Logger instance
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Return the clipper logger
    clipper_logger = logging.getLogger("clipper")
    return clipper_logger


def setup_logger():
    """Configure the logger for the application"""
    return logger


class StreamlitHandler(logging.Handler):
    """Custom logging handler that captures logs for display in Streamlit"""

    def emit(self, record):
        log_entry = self.format(record)
        if "log_messages" not in st.session_state:
            st.session_state.log_messages = []

        st.session_state.log_messages.append(f"{record.levelname}: {log_entry}")
        # Keep only last 100 messages
        if len(st.session_state.log_messages) > 100:
            st.session_state.log_messages.pop(0)


def add_streamlit_handler(logger):
    """Add a Streamlit handler to the logger"""
    handler = StreamlitHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def display_logs(max_logs=20):
    """Display logs in the Streamlit UI"""
    if "log_messages" in st.session_state:
        for msg in reversed(st.session_state.log_messages[-max_logs:]):
            st.text(msg)
