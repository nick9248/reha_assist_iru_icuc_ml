"""
Utility functions for project setup.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


def create_project_structure():
    """
    Create structured project folders for logs, plots, and models

    Returns:
    --------
    log_dir : str
        Path to the log directory for this run
    plot_dir : str
        Path to the plot directory for this run
    model_dir : str
        Path to the model directory for this run
    """
    # Load environment variables
    load_dotenv()

    # Get base folders from environment variables
    log_base = os.environ.get('LOG_FOLDER')
    plot_base = os.environ.get('PLOT_FOLDER')
    model_base = os.environ.get('MODEL_FOLDER', 'models')  # Default to 'models' if not set

    # Validate environment variables
    if not log_base:
        raise ValueError("LOG_FOLDER environment variable is not set.")
    if not plot_base:
        raise ValueError("PLOT_FOLDER environment variable is not set.")

    # Create timestamp for unique folder names
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create structured folders
    stage_name = "04_model_tuning"
    log_dir = os.path.join(log_base, stage_name, timestamp)
    plot_dir = os.path.join(plot_base, stage_name, timestamp)
    model_dir = os.path.join(model_base, stage_name, timestamp)

    # Create directories
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    return log_dir, plot_dir, model_dir


def setup_logging(log_folder, logger_name='model_tuning'):
    """
    Setup logging configuration

    Parameters:
    -----------
    log_folder : str
        Path to the folder where logs should be saved
    logger_name : str
        Name for the logger

    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    # Configure logging
    log_file = os.path.join(log_folder, f"{logger_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger