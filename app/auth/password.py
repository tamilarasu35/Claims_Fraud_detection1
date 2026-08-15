"""
Password Security and Hashing Utilities.
"""

import hashlib
from app.config.settings import settings

def hash_password(password: str) -> str:
    """Hash password securely using SHA256 and configured salt."""
    salted = f"{settings.PASSWORD_SALT}:{password}"
    return hashlib.sha256(salted.encode('utf-8')).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against stored hash."""
    return hash_password(password) == hashed_password
