"""
Healthcare Provider Fraud Intelligence System - Premium Streamlit Portal.
Featuring High-Contrast Visual Aesthetics & Pure Centered Login Screen.
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

# 2. High-Contrast CSS Override for 100% Visibility
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #0b0f19 !important;
        color: #ffffff !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Ensure ALL Headings, Paragraphs, Labels, and Text are Bright White / Cyan */
    h1, h2, h3, h4, h5, h6, label, p, span, div, caption, .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Input Fields, Selectboxes, Textareas High Contrast */
    input, select, textarea, div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    /* Fix Streamlit Selectbox Option Dropdowns */
    ul[data-baseweb="menu"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }
    li[data-baseweb="option"] {
        color: #ffffff !important;
    }

    /* Centered Login Card Styling */
    .login-card {
        background: #1e293b;
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 36px 32px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.7);
        text-align: center;
        margin-top: 30px;
    }
    .login-card h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8 !important;
        margin-bottom: 8px;
    }
    .login-card p {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        margin-bottom: 24px;
    }

    /* Metric Cards High Contrast */
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 2.0rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 1.0rem !important;
        font-weight: 600 !important;
    }
    
    /* Role Badges */
    .badge-user { background: #3b82f6; color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
    .badge-investigator { background: #f59e0b; color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
    .badge-manager { background: #10b981; color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
    .badge-admin { background: #ef4444; color: #ffffff !important; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
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
        st.sidebar.info("📥 File Fraud Detection Hub")
        st.caption("🔒 Role Limited: Analyst Mode")
        render_user_dashboard(user)

    elif role_name == "INVESTIGATOR":
        selected_page = st.sidebar.radio(
            "Select Workspace Module:",
            ["🔍 Provider Investigation Workspace", "📥 File Fraud Detection Hub"]
        )
        st.caption("🔒 Role Limited: Investigator Mode")
        if selected_page == "🔍 Provider Investigation Workspace":
            render_investigator_dashboard(user)
        else:
            render_user_dashboard(user)

    elif role_name == "MANAGER":
        selected_page = st.sidebar.radio(
            "Select Workspace Module:",
            ["📊 Executive Manager Command Center", "📥 File Fraud Detection Hub"]
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
                "📥 File Fraud Detection Hub",
                "🔍 Provider Investigation Workspace",
                "📊 Executive Manager Command Center",
                "⚙️ System Admin & Security Governance"
            ]
        )
        st.caption("🔓 Unlimited Access: Administrator Mode")
        
        if selected_page == "📥 File Fraud Detection Hub":
            render_user_dashboard(user)
        elif selected_page == "🔍 Provider Investigation Workspace":
            render_investigator_dashboard(user)
        elif selected_page == "📊 Executive Manager Command Center":
            render_manager_dashboard(user)
        elif selected_page == "⚙️ System Admin & Security Governance":
            render_admin_dashboard(user)
