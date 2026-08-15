"""
Structured Logging Module for Healthcare Provider Fraud Intelligence System.
"""

import logging
import sys
from pathlib import Path
from app.config.settings import settings

def setup_logger(name: str = "FraudIntelligence") -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    console_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File Handler
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
