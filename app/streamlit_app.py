"""
Healthcare Provider Fraud Intelligence System - Premium Streamlit Portal.
Features Centered Minimalist Login Screen & Multi-Format Fraud Intelligence Hub.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config.settings import settings
from app.database.database import init_db
from app.auth.authentication import AuthService
from app.audit.audit_logger import audit_log
from app.utils.logger import logger
from app.agents.orchestrator import FraudIntelligenceOrchestrator

# 1. Page Configuration
st.set_page_config(
    page_title="Healthcare Fraud Intelligence Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed" if "authenticated" not in st.session_state or not st.session_state["authenticated"] else "expanded"
)

# 2. Styling (Dark Mode & Centered Card Aesthetics)
st.markdown("""
<style>
    /* Dark Slate Body Background */
    .stApp {
        background-color: #090d16;
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Centered Login Card Styling */
    .login-card {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 40px 36px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
        text-align: center;
        margin-top: 50px;
    }
    .login-card h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .login-card p {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 28px;
    }

    /* Top Navigation Header for Logged-In Users */
    .top-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }
    
    /* Role Badges */
    .badge-user { background: #3b82f6; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .badge-investigator { background: #f59e0b; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .badge-manager { background: #10b981; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .badge-admin { background: #ef4444; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    
    /* Risk Pill Badges */
    .risk-critical { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .risk-high { background-color: #f97316; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .risk-medium { background-color: #eab308; color: black; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .risk-low { background-color: #22c55e; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# 3. System Boot Cache
@st.cache_resource
def initialize_system():
    init_db()
    AuthService.bootstrap_default_users()
    logger.info("Database and default accounts initialized.")

initialize_system()

# Ensure orchestrator is initialized in session state
if "orchestrator" not in st.session_state:
    orchestrator = FraudIntelligenceOrchestrator()
    orchestrator.fraud_analysis_agent.try_load_pretrained("v1.0.0")
    st.session_state["orchestrator"] = orchestrator

# Session State Keys
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# ==============================================================================
# SCREEN A: PURE CENTERED LOGIN SCREEN (WHEN UNAUTHENTICATED)
# ==============================================================================
if not st.session_state["authenticated"]:
    # Hide sidebar completely when not logged in
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2.2, 1])
    
    with c2:
        st.markdown("""
        <div class="login-card">
            <h1>🛡️ Healthcare Fraud Intelligence</h1>
            <p>Multi-Agent Medicare Claims Risk & Explainable Decision Portal</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔐 Sign In to Platform")
        
        persona = st.selectbox(
            "Select Persona Role to Test:",
            [
                "👤 User / Data Analyst (user / User123!)",
                "🔍 Fraud Investigator (investigator / Investigator123!)",
                "📊 Executive Manager (manager / Manager123!)",
                "⚙️ System Admin (admin / Admin123!)"
            ]
        )
        
        if "User" in persona:
            def_user, def_pass = "user", "User123!"
        elif "Investigator" in persona:
            def_user, def_pass = "investigator", "Investigator123!"
        elif "Manager" in persona:
            def_user, def_pass = "manager", "Manager123!"
        else:
            def_user, def_pass = "admin", "Admin123!"

        with st.form("centered_login_form"):
            username = st.text_input("Username", value=def_user)
            password = st.text_input("Password", type="password", value=def_pass)
            login_btn = st.form_submit_button("🔑 Sign In to Platform", type="primary", use_container_width=True)
            
            if login_btn:
                user = AuthService.authenticate(username, password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    audit_log(user["username"], user["role_name"], "LOGIN_SUCCESS", details="User signed in")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

# ==============================================================================
# SCREEN B: AUTHENTICATED DASHBOARD PORTAL
# ==============================================================================
else:
    user = st.session_state["user"]
    role_name = user['role_name']
    badge_class = f"badge-{role_name.lower()}"
    
    # Sidebar Navigation & Controls
    st.sidebar.markdown(f"### 🛡️ Portal Navigation")
    st.sidebar.markdown(f"**Logged in:** `{user['full_name']}`")
    st.sidebar.markdown(f"**Role:** <span class='{badge_class}'>{role_name}</span>", unsafe_allow_html=True)
    st.sidebar.divider()
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        audit_log(user["username"], role_name, "LOGOUT", details="User signed out")
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()
        
    st.sidebar.divider()

    from app.ui.dashboards.user_dashboard import render_user_dashboard
    from app.ui.dashboards.investigator_dashboard import render_investigator_dashboard
    from app.ui.dashboards.manager_dashboard import render_manager_dashboard
    from app.ui.dashboards.admin_dashboard import render_admin_dashboard

    # Strict RBAC Navigation Locking
    if role_name == "USER":
        st.sidebar.markdown("**Active Role Workspace:**")
        st.sidebar.info("📥 Multi-Format Ingestion & Fraud Assessment")
        st.caption("🔒 Role Limited: Analyst Mode")
        render_user_dashboard(user)

    elif role_name == "INVESTIGATOR":
        selected_page = st.sidebar.radio(
            "Select Workspace Module:",
            ["🔍 Provider Investigation Workspace", "📥 Multi-Format Ingestion & Assessment"]
        )
        st.caption("🔒 Role Limited: Investigator Mode")
        if selected_page == "🔍 Provider Investigation Workspace":
            render_investigator_dashboard(user)
        else:
            render_user_dashboard(user)

    elif role_name == "MANAGER":
        selected_page = st.sidebar.radio(
            "Select Workspace Module:",
            ["📊 Executive Manager Command Center", "📥 Multi-Format Ingestion & Assessment"]
        )
        st.caption("🔒 Role Limited: Executive Manager Mode")
        if selected_page == "📊 Executive Manager Command Center":
            render_manager_dashboard(user)
        else:
            render_user_dashboard(user)

    elif role_name == "ADMIN":
        selected_page = st.sidebar.radio(
            "Select Workspace Module:",
            [
                "📥 Multi-Format Ingestion & Assessment",
                "🔍 Provider Investigation Workspace",
                "📊 Executive Manager Command Center",
                "⚙️ System Admin & Security Governance"
            ]
        )
        st.caption("🔓 Unlimited Access: Administrator Mode")
        
        if selected_page == "📥 Multi-Format Ingestion & Assessment":
            render_user_dashboard(user)
        elif selected_page == "🔍 Provider Investigation Workspace":
            render_investigator_dashboard(user)
        elif selected_page == "📊 Executive Manager Command Center":
            render_manager_dashboard(user)
        elif selected_page == "⚙️ System Admin & Security Governance":
            render_admin_dashboard(user)
