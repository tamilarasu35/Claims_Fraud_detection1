"""
Role-Based Access Control (RBAC) System.
Permissions per role: USER, INVESTIGATOR, MANAGER, ADMIN.
"""

from typing import Set

ROLE_PERMISSIONS = {
    "USER": {
        "upload_dataset",
        "submit_provider_input",
        "initiate_analysis",
        "view_basic_results"
    },
    "INVESTIGATOR": {
        "upload_dataset",
        "submit_provider_input",
        "initiate_analysis",
        "view_basic_results",
        "view_suspicious_providers",
        "view_provider_drilldown",
        "inspect_evidence",
        "inspect_explanations",
        "inspect_negotiation",
        "record_investigation"
    },
    "MANAGER": {
        "upload_dataset",
        "submit_provider_input",
        "initiate_analysis",
        "view_basic_results",
        "view_executive_dashboard",
        "view_risk_trends",
        "view_high_risk_providers",
        "review_investigations"
    },
    "ADMIN": {
        "upload_dataset",
        "submit_provider_input",
        "initiate_analysis",
        "view_basic_results",
        "view_suspicious_providers",
        "view_provider_drilldown",
        "inspect_evidence",
        "inspect_explanations",
        "inspect_negotiation",
        "record_investigation",
        "view_executive_dashboard",
        "view_risk_trends",
        "view_high_risk_providers",
        "review_investigations",
        "manage_users",
        "manage_roles",
        "manage_datasets",
        "manage_models",
        "view_audit_logs",
        "system_config"
    }
}

class AuthorizationService:
    @staticmethod
    def has_permission(role_name: str, permission: str) -> bool:
        """Check if a role possesses a specific permission."""
        role_name = role_name.upper()
        permissions: Set[str] = ROLE_PERMISSIONS.get(role_name, set())
        return permission in permissions

    @staticmethod
    def require_permission(user: dict, permission: str) -> bool:
        """Enforce authorization check for a user dict."""
        if not user or "role_name" not in user:
            return False
        return AuthorizationService.has_permission(user["role_name"], permission)
