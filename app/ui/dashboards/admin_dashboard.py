"""
Admin Dashboard Module.
System Administration panel for User & Role Management, Model Version Control,
Audit Log inspection, and System Configuration.
"""

import streamlit as st
import pandas as pd
from app.database.repositories import UserRepository, AuditRepository
from app.database.database import get_db_connection
from app.auth.password import hash_password
from app.audit.audit_logger import audit_log

def render_admin_dashboard(user: dict):
    st.title("⚙️ System Administration & Governance")
    st.write("Manage application users, role permissions, model version artifacts, and inspect security audit logs.")
    
    tabs = st.tabs(["👥 User & Role Management", "📜 Security Audit Trail Logs", "🧠 Model Version Artifacts", "⚙️ System Status"])
    
    # ---------------- TAB 1: USER MANAGEMENT ----------------
    with tabs[0]:
        st.subheader("Existing User Accounts")
        users_list = UserRepository.list_all_users()
        st.dataframe(pd.DataFrame(users_list), use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("➕ Create New User Account")
        with st.form("create_user_form"):
            c1, c2 = st.columns(2)
            with c1:
                new_uname = st.text_input("Username")
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email")
            with c2:
                new_pass = st.text_input("Password", type="password")
                new_role = st.selectbox("Assign Role", ["USER", "INVESTIGATOR", "MANAGER", "ADMIN"])
                
            create_btn = st.form_submit_button("Create User")
            if create_btn:
                if new_uname and new_pass:
                    p_hash = hash_password(new_pass)
                    success = UserRepository.create_user(new_uname, p_hash, new_name, new_email, new_role)
                    if success:
                        audit_log(user["username"], user["role_name"], "USER_CREATED", target_resource=new_uname, details=f"Role: {new_role}")
                        st.success(f"User '{new_uname}' created successfully with role {new_role}!")
                        st.rerun()
                    else:
                        st.error("Failed to create user. Username may already exist.")
                else:
                    st.warning("Please provide username and password.")

    # ---------------- TAB 2: AUDIT LOGS ----------------
    with tabs[1]:
        st.subheader("Security & Operational Audit Logs")
        logs = AuditRepository.get_recent_logs(limit=200)
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)

    # ---------------- TAB 3: MODEL VERSIONS ----------------
    with tabs[2]:
        st.subheader("Model Artifacts & Performance Specifications")
        if "orchestrator" in st.session_state and st.session_state.get("pipeline_run"):
            xgb_meta = st.session_state["orchestrator"].fraud_analysis_agent.xgb_model.metrics
            st.json({
                "active_model_version": "v1.0.0",
                "xgb_classifier_metrics": xgb_meta,
                "ebm_risk_explainer": "ExplainableBoostingClassifier (Glass-Box Additive Log-Odds)",
                "scale_pos_weight": xgb_meta.get("scale_pos_weight", 9.69),
                "optimal_decision_threshold": xgb_meta.get("optimal_threshold", 0.62)
            })
        else:
            st.info("Run the multi-agent pipeline to view active model version metrics.")

    # ---------------- TAB 4: SYSTEM STATUS ----------------
    with tabs[3]:
        st.subheader("System Architecture Status")
        st.json({
            "database_engine": "SQLite3 (WAL Mode, Foreign Key Enforcement)",
            "authentication_mode": "Salted SHA256 Password Hashing",
            "rbac_roles": ["USER", "INVESTIGATOR", "MANAGER", "ADMIN"],
            "perception_agent": "Active",
            "fraud_analysis_agent": "Active (XGBoost + EBM)",
            "negotiation_agent": "Active (Adversarial Review)",
            "arbitrator": "Active (Final Decision Resolution)"
        })
