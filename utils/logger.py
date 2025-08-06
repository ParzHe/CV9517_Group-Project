# utils/logger.py
# This script provides a utility function to create a logger with rich formatting for better readability.

import os
from rich.logging import RichHandler
import logging

def make_logger(name, log_path, file_mode='a', show_level_name=False):
    """Create a logger with a specified name and log file path.
    Args:
        name (str): The name of the logger.
        log_path (str): The path to the log file.
    Returns:
        logging.Logger: Configured logger instance.
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    logger.propagate = False
    
    for handler in logger.handlers:
        logger.removeHandler(handler)
        
    rich_handler = RichHandler(show_time=False, rich_tracebacks=True, markup=True)
    
    # Create a file handler to log to a file
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding='utf-8')

    if show_level_name:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
    file_handler.setFormatter(formatter)
    
    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    
    return logger