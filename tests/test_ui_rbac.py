"""
Unit tests for Phase 7 Streamlit UI Dashboards & RBAC Authorization Checks.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.database import init_db
from app.auth.authorization import AuthorizationService
from app.database.repositories import UserRepository, AuditRepository

def test_rbac_dashboard_permissions():
    user_role = {"username": "standard_user", "role_name": "USER"}
    inv_role = {"username": "investigator_user", "role_name": "INVESTIGATOR"}
    mgr_role = {"username": "manager_user", "role_name": "MANAGER"}
    admin_role = {"username": "admin_user", "role_name": "ADMIN"}
    
    # USER permission checks
    assert AuthorizationService.require_permission(user_role, "upload_dataset") is True
    assert AuthorizationService.require_permission(user_role, "view_suspicious_providers") is False
    assert AuthorizationService.require_permission(user_role, "view_executive_dashboard") is False
    assert AuthorizationService.require_permission(user_role, "manage_users") is False
    
    # INVESTIGATOR permission checks
    assert AuthorizationService.require_permission(inv_role, "view_suspicious_providers") is True
    assert AuthorizationService.require_permission(inv_role, "inspect_explanations") is True
    assert AuthorizationService.require_permission(inv_role, "record_investigation") is True
    assert AuthorizationService.require_permission(inv_role, "manage_users") is False
    
    # MANAGER permission checks
    assert AuthorizationService.require_permission(mgr_role, "view_executive_dashboard") is True
    assert AuthorizationService.require_permission(mgr_role, "view_risk_trends") is True
    assert AuthorizationService.require_permission(mgr_role, "manage_users") is False
    
    # ADMIN permission checks
    assert AuthorizationService.require_permission(admin_role, "manage_users") is True
    assert AuthorizationService.require_permission(admin_role, "view_audit_logs") is True
    assert AuthorizationService.require_permission(admin_role, "record_investigation") is True

import uuid

def test_admin_user_creation():
    init_db()
    uname = f"test_inv_{uuid.uuid4().hex[:8]}"
    success = UserRepository.create_user(uname, "hash123", "New Investigator", "inv@hc.gov", "INVESTIGATOR")
    assert success is True
    
    user_db = UserRepository.get_by_username(uname)
    assert user_db["role_name"] == "INVESTIGATOR"
    assert user_db["full_name"] == "New Investigator"

