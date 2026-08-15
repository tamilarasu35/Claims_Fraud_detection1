"""
Healthcare Provider Fraud Intelligence System - Premium Streamlit Portal.
Featuring No Sidebar, Sleek Top Navigation Header, and Centered Minimalist Login Screen.
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

# 1. Page Configuration (Sidebar Permanently Collapsed/Hidden)
st.set_page_config(
    page_title="Healthcare Fraud Intelligence Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS Override: Completely Hide Sidebar Globally & High-Contrast Styling
st.markdown("""
<style>
    /* REMOVE SIDEBAR COMPLETELY */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Make Streamlit Top Header match dark background */
    header[data-testid="stHeader"], .stAppHeader, header, div[data-testid="stHeader"] {
        background-color: #0b0f19 !important;
        color: #ffffff !important;
    }
    
    /* Main Content Width & Background */
    .stApp {
        background-color: #0b0f19 !important;
        color: #ffffff !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    
    /* Headings & Text Color High Contrast */
    h1, h2, h3, h4, h5, h6, label, p, span, div, caption, .stMarkdown {
        color: #ffffff !important;
    }

    /* Input Fields & Selectboxes */
    input, select, textarea, div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    ul[data-baseweb="menu"], li[data-baseweb="option"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }

    /* Centered Login Card Styling */
    .login-card {
        background: #1e293b;
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 40px 36px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.7);
        text-align: center;
        margin-top: 40px;
    }
    .login-card h1 {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38bdf8 !important;
        margin-bottom: 8px;
    }
    .login-card p {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        margin-bottom: 28px;
    }

    /* Top Navigation Header Container */
    .top-nav-bar {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 24px;
    }

    /* Metric Cards */
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
# SCREEN B: AUTHENTICATED DASHBOARD PORTAL (NO SIDEBAR - TOP HEADER NAV)
# ==============================================================================
else:
    user = st.session_state["user"]
    role_name = user['role_name']
    badge_class = f"badge-{role_name.lower()}"
    
    # Top Navigation Header Bar
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        st.markdown(f"### 🛡️ Healthcare Fraud Intelligence Portal &nbsp; <span class='{badge_class}'>{role_name}</span>", unsafe_allow_html=True)
        st.caption(f"Logged in as: **{user['full_name']}** (`{user['username']}`)")
    with t_col2:
        if st.button("🚪 Sign Out / Switch Role", use_container_width=True):
            audit_log(user["username"], role_name, "LOGOUT", details="User signed out")
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.rerun()
            
    st.divider()

    from app.ui.dashboards.user_dashboard import render_user_dashboard
    from app.ui.dashboards.investigator_dashboard import render_investigator_dashboard
    from app.ui.dashboards.manager_dashboard import render_manager_dashboard
    from app.ui.dashboards.admin_dashboard import render_admin_dashboard

    # Strict Horizontal Tab Navigation based on Role
    if role_name == "USER":
        st.caption("🔒 Role Workspace: Analyst Mode")
        render_user_dashboard(user)

    elif role_name == "INVESTIGATOR":
        st.caption("🔒 Role Workspace: Investigator Mode")
        inv_tabs = st.tabs(["🔍 Provider Investigation Workspace", "📥 File Fraud Detection Hub"])
        with inv_tabs[0]:
            render_investigator_dashboard(user)
        with inv_tabs[1]:
            render_user_dashboard(user)

    elif role_name == "MANAGER":
        st.caption("🔒 Role Workspace: Executive Manager Mode")
        mgr_tabs = st.tabs(["📊 Executive Manager Command Center", "📥 File Fraud Detection Hub"])
        with mgr_tabs[0]:
            render_manager_dashboard(user)
        with mgr_tabs[1]:
            render_user_dashboard(user)

    elif role_name == "ADMIN":
        st.caption("🔓 Role Workspace: Administrator Mode (Unlimited Access)")
        admin_tabs = st.tabs([
            "📥 File Fraud Detection Hub",
            "🔍 Provider Investigation Workspace",
            "📊 Executive Manager Command Center",
            "⚙️ System Admin & Security Governance"
        ])
        with admin_tabs[0]:
            render_user_dashboard(user)
        with admin_tabs[1]:
            render_investigator_dashboard(user)
        with admin_tabs[2]:
            render_manager_dashboard(user)
        with admin_tabs[3]:
            render_admin_dashboard(user)
