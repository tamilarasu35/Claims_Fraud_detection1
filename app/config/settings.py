"""
Centralized Configuration Settings for Healthcare Provider Fraud Intelligence System.
Reads from environment variables and .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file
dotenv_path = BASE_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Healthcare Provider Fraud Intelligence System")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    MODEL_DIR: Path = BASE_DIR / os.getenv("MODEL_DIR", "models")
    DATABASE_PATH: Path = BASE_DIR / os.getenv("DATABASE_PATH", "data/fraud_intelligence.db")
    
    # LLM Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")
    
    # Auth & Security
    PASSWORD_SALT: str = os.getenv("PASSWORD_SALT", "dev-salt-12345")
    SESSION_EXPIRY_HOURS: int = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
    
    # Ensure directories exist
    def __init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
