"""
Audit Logging Utilities for Security and Privacy Compliance.
"""

from typing import Optional
from app.database.repositories import AuditRepository
from app.utils.logger import logger

def audit_log(username: str, role: str, action: str, target_resource: Optional[str] = None, details: Optional[str] = None):
    """Log audit trail events safely without exposing PII."""
    logger.info(f"AUDIT | User: {username} ({role}) | Action: {action} | Resource: {target_resource} | Details: {details}")
    AuditRepository.log_action(
        username=username,
        role=role,
        action=action,
        target_resource=target_resource,
        details=details
    )
