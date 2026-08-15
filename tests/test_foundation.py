"""
Unit tests for Phase 1 Foundation: Database, Config, Logging, Auth & RBAC.
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import settings
from app.database.database import init_db, get_db_connection

from app.database.repositories import UserRepository, AuditRepository
from app.auth.password import hash_password, verify_password
from app.auth.authentication import AuthService
from app.auth.authorization import AuthorizationService

def test_settings_loaded():
    assert settings.APP_NAME is not None
    assert settings.DATA_DIR.exists()
    assert settings.MODEL_DIR.exists()

def test_database_initialization():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    expected_tables = {"roles", "users", "datasets", "analysis_runs", "providers", "fraud_results", "investigations", "model_versions", "audit_logs"}
    for table in expected_tables:
        assert table in tables, f"Table {table} missing from SQLite schema."

def test_password_hashing():
    raw_pass = "SecurePass123!"
    pass_hash = hash_password(raw_pass)
    assert pass_hash != raw_pass
    assert verify_password(raw_pass, pass_hash) is True
    assert verify_password("WrongPass", pass_hash) is False

def test_bootstrap_accounts_and_auth():
    init_db()
    AuthService.bootstrap_default_users()
    
    admin_user = AuthService.authenticate("admin", "Admin123!")
    assert admin_user is not None
    assert admin_user["role_name"] == "ADMIN"
    
    inv_user = AuthService.authenticate("investigator", "Investigator123!")
    assert inv_user is not None
    assert inv_user["role_name"] == "INVESTIGATOR"
    
    invalid = AuthService.authenticate("admin", "WrongPass")
    assert invalid is None

def test_rbac_permissions():
    # User cannot manage users
    user_dict = {"username": "user", "role_name": "USER"}
    assert AuthorizationService.require_permission(user_dict, "upload_dataset") is True
    assert AuthorizationService.require_permission(user_dict, "manage_users") is False
    
    # Investigator can view provider drilldown but cannot manage users
    inv_dict = {"username": "investigator", "role_name": "INVESTIGATOR"}
    assert AuthorizationService.require_permission(inv_dict, "view_provider_drilldown") is True
    assert AuthorizationService.require_permission(inv_dict, "manage_users") is False
    
    # Admin has manage_users
    admin_dict = {"username": "admin", "role_name": "ADMIN"}
    assert AuthorizationService.require_permission(admin_dict, "manage_users") is True

def test_audit_logging():
    init_db()
    AuditRepository.log_action("testuser", "ADMIN", "TEST_ACTION", target_resource="SYS", details="Testing audit log")
    logs = AuditRepository.get_recent_logs(limit=10)
    user_logs = [log for log in logs if log["username"] == "testuser"]
    assert len(user_logs) > 0
    assert user_logs[0]["action"] == "TEST_ACTION"

