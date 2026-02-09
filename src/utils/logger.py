"""Logging utilities"""

import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    json_format: bool = True
) -> logging.Logger:
    """
    Setup logger with optional JSON formatting
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        json_format: Whether to use JSON formatting
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_format:
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            timestamp=True
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return logging.getLogger(name)
