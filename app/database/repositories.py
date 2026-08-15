"""
Database Repository Layer for CRUD operations.
"""

import json
import pandas as pd
from typing import Optional, List, Dict, Any

from app.database.database import get_db_connection
from app.database.models import User, ProviderResult, AuditLogEntry
from app.utils.logger import logger

class UserRepository:
    @staticmethod
    def get_by_username(username: str) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return dict(row) if row else None

    @staticmethod
    def create_user(username: str, password_hash: str, full_name: str, email: str, role_name: str = "USER") -> bool:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email, role_name)
            VALUES (?, ?, ?, ?, ?)
            """, (username, password_hash, full_name, email, role_name))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            conn.rollback()
            return False

    @staticmethod
    def list_all_users() -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, email, role_name, is_active, created_at, last_login FROM users")
        return [dict(row) for row in cursor.fetchall()]

class AuditRepository:
    @staticmethod
    def log_action(username: str, role: str, action: str, target_resource: Optional[str] = None, details: Optional[str] = None):
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO audit_logs (username, role, action, target_resource, details)
            VALUES (?, ?, ?, ?, ?)
            """, (username, role, action, target_resource, details))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to record audit log: {e}")

    @staticmethod
    def get_recent_logs(limit: int = 100) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

class ResultsRepository:
    @staticmethod
    def upsert_providers_from_features(features_df: pd.DataFrame):
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            for _, row in features_df.iterrows():
                cursor.execute("""
                INSERT OR REPLACE INTO providers (
                    provider_id, total_claims, total_reimbursement,
                    beneficiary_count, inpatient_claims, outpatient_claims
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(row['Provider']),
                    int(row.get('TotalClaims', 0)),
                    float(row.get('TotalReimbursement', 0.0)),
                    int(row.get('UniqueBeneficiaries', 0)),
                    int(row.get('InpatientClaims', 0)),
                    int(row.get('OutpatientClaims', 0))
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error upserting providers into SQLite DB: {e}")
            conn.rollback()

    @staticmethod
    def save_provider_result(result: ProviderResult) -> bool:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO fraud_results (
                run_uuid, provider_id, classification, fraud_probability,
                risk_score, risk_level, recommendation, top_features_json,
                negotiation_argument, negotiation_challenge, arbitrator_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.run_uuid, result.provider_id, result.classification,
                result.fraud_probability, result.risk_score, result.risk_level,
                result.recommendation, json.dumps(result.top_features),
                result.negotiation_argument, result.negotiation_challenge,
                result.arbitrator_reasoning
            ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving provider result for {result.provider_id}: {e}")
            conn.rollback()
            return False


    @staticmethod
    def get_results_by_run(run_uuid: str) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM fraud_results WHERE run_uuid = ?", (run_uuid,))
        return [dict(row) for row in cursor.fetchall()]
