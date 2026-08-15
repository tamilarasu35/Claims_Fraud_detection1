"""
Authentication Service and Bootstrap Admin Manager.
"""

from typing import Optional, Dict, Any
from app.database.repositories import UserRepository, AuditRepository
from app.auth.password import hash_password, verify_password
from app.utils.logger import logger

class AuthService:
    @staticmethod
    def bootstrap_default_users():
        """Ensure default admin, investigator, manager, and user accounts exist for dev/demo."""
        default_accounts = [
            ("admin", "Admin123!", "System Administrator", "admin@healthcare.gov", "ADMIN"),
            ("investigator", "Investigator123!", "Lead Investigator", "investigator@healthcare.gov", "INVESTIGATOR"),
            ("manager", "Manager123!", "Fraud Analytics Manager", "manager@healthcare.gov", "MANAGER"),
            ("user", "User123!", "Standard Analyst", "user@healthcare.gov", "USER"),
        ]
        
        for username, plain_pass, name, email, role in default_accounts:
            existing = UserRepository.get_by_username(username)
            if not existing:
                pass_hash = hash_password(plain_pass)
                UserRepository.create_user(username, pass_hash, name, email, role)
                logger.info(f"Bootstrapped default user account: {username} ({role})")

    @staticmethod
    def authenticate(username: str, password_plain: str) -> Optional[Dict[str, Any]]:
        user = UserRepository.get_by_username(username)
        if not user:
            return None
            
        if not user.get("is_active", True):
            logger.warning(f"Authentication attempt for inactive user: {username}")
            return None
            
        if verify_password(password_plain, user["password_hash"]):
            AuditRepository.log_action(
                username=username,
                role=user["role_name"],
                action="LOGIN_SUCCESS",
                details="User logged in successfully"
            )
            return user
            
        AuditRepository.log_action(
            username=username,
            role=user.get("role_name", "UNKNOWN"),
            action="LOGIN_FAILED",
            details="Invalid password attempt"
        )
        return None
