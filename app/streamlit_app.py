"""
Healthcare Provider Fraud Intelligence System - Premium Modern Streamlit Entry Point.
Featuring Glassmorphism, Dark Accents, Quick Persona Login Switcher, and Strict RBAC Enforcement.
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
from app.auth.authorization import AuthorizationService
from app.audit.audit_logger import audit_log
from app.utils.logger import logger
from app.agents.orchestrator import FraudIntelligenceOrchestrator

# 1. Page Configuration
st.set_page_config(
    page_title="Healthcare Provider Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Rich Aesthetics & Custom CSS (Dark Theme, Glassmorphism, Neon Accents, Card Badges)
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Banner */
    .main-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .main-banner h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-banner p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Role Badges */
    .badge-user { background: #3b82f6; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-investigator { background: #f59e0b; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-manager { background: #10b981; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    .badge-admin { background: #ef4444; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
    
    /* Custom Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Risk Badges */
    .risk-critical { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .risk-high { background-color: #f97316; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .risk-medium { background-color: #eab308; color: black; padding: 4px 10px; border-radius: 700; }
    .risk-low { background-color: #22c55e; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# 3. System Initialization
@st.cache_resource
def initialize_system():
    init_db()
    AuthService.bootstrap_default_users()
    logger.info("Healthcare Provider Fraud Intelligence System booted.")

initialize_system()

# Ensure orchestrator is initialized in session state
if "orchestrator" not in st.session_state:
    orchestrator = FraudIntelligenceOrchestrator()
    orchestrator.fraud_analysis_agent.try_load_pretrained("v1.0.0")
    st.session_state["orchestrator"] = orchestrator

# 4. Session State Management
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# Header Banner
st.markdown("""
<div class="main-banner">
    <h1>🛡️ Healthcare Provider Fraud Intelligence Platform</h1>
    <p>Multi-Agent Medicare Claims Risk & Explainable Behavior Analytics</p>
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Identity & Persona Switcher
st.sidebar.title("🔐 Authentication & Role Access")

if not st.session_state["authenticated"]:
    st.sidebar.subheader("Quick Persona Login (Select Persona)")
    
    persona = st.sidebar.selectbox(
        "Choose Persona Role to Test:",
        [
            "👤 User / Data Analyst (user / User123!)",
            "🔍 Fraud Investigator (investigator / Investigator123!)",
            "📊 Executive Manager (manager / Manager123!)",
            "⚙️ System Admin (admin / Admin123!)"
        ]
    )
    
    # Auto-fill credentials based on selected persona
    if "User" in persona:
        def_user, def_pass = "user", "User123!"
    elif "Investigator" in persona:
        def_user, def_pass = "investigator", "Investigator123!"
    elif "Manager" in persona:
        def_user, def_pass = "manager", "Manager123!"
    else:
        def_user, def_pass = "admin", "Admin123!"

    with st.sidebar.form("login_form"):
        username = st.text_input("Username", value=def_user)
        password = st.text_input("Password", type="password", value=def_pass)
        submitted = st.form_submit_button("🔑 Sign In to Platform", type="primary")
        
        if submitted:
            user = AuthService.authenticate(username, password)
            if user:
                st.session_state["authenticated"] = True
                st.session_state["user"] = user
                st.sidebar.success(f"Signed in as {user['full_name']}")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials.")
                
    st.info("💡 Select any role persona above and click **Sign In** to explore role-specific permissions and dashboards.")
    
    st.markdown("---")
    st.subheader("Platform Capabilities")
    st.markdown("""
    - **Perception Agent:** Quality & referential integrity checks across 558,211 claims.
    - **XGBoost Classifier:** ROC-AUC `0.9703` fraud probability score.
    - **EBM Risk Engine:** Glass-box `0-100` risk score scaling.
    - **Negotiation Agent:** Adversarial Prosecutor vs Defense debate.
    - **Arbitrator:** Final audit-ready resolution.
    """)

else:
    user = st.session_state["user"]
    role_name = user['role_name']
    badge_class = f"badge-{role_name.lower()}"
    
    st.sidebar.markdown(f"**User:** `{user['full_name']}`")
    st.sidebar.markdown(f"**Active Role:** <span class='{badge_class}'>{role_name}</span>", unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Logout / Switch Role"):
        audit_log(user["username"], role_name, "LOGOUT", details="User signed out")
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()
        
    st.sidebar.divider()
    
    from app.ui.dashboards.user_dashboard import render_user_dashboard
    from app.ui.dashboards.investigator_dashboard import render_investigator_dashboard
    from app.ui.dashboards.manager_dashboard import render_manager_dashboard
    from app.ui.dashboards.admin_dashboard import render_admin_dashboard

    # Enforce strict RBAC options per role
    st.sidebar.subheader("📍 Navigation Modules")
    
    if role_name == "USER":
        selected_page = st.sidebar.radio("Active Workspace", ["📥 Data Ingestion & Input Center"])
        st.caption("🔒 Role Limited: Analyst Mode")
        render_user_dashboard(user)

    elif role_name == "INVESTIGATOR":
        selected_page = st.sidebar.radio("Active Workspace", ["🔍 Provider Investigation Workspace", "📥 Data Ingestion & Input Center"])
        st.caption("🔒 Role Limited: Investigator Mode")
        if selected_page == "🔍 Provider Investigation Workspace":
            render_investigator_dashboard(user)
        else:
            render_user_dashboard(user)

    elif role_name == "MANAGER":
        selected_page = st.sidebar.radio("Active Workspace", ["📊 Executive Manager Command Center", "📥 Data Ingestion & Input Center"])
        st.caption("🔒 Role Limited: Manager Executive Mode")
        if selected_page == "📊 Executive Manager Command Center":
            render_manager_dashboard(user)
        else:
            render_user_dashboard(user)

    elif role_name == "ADMIN":
        selected_page = st.sidebar.radio(
            "Active Workspace",
            [
                "📥 Data Ingestion & Input Center",
                "🔍 Provider Investigation Workspace",
                "📊 Executive Manager Command Center",
                "⚙️ System Admin & Security Governance"
            ]
        )
        st.caption("🔓 Unlimited Access: Administrator Mode")
        
        if selected_page == "📥 Data Ingestion & Input Center":
            render_user_dashboard(user)
        elif selected_page == "🔍 Provider Investigation Workspace":
            render_investigator_dashboard(user)
        elif selected_page == "📊 Executive Manager Command Center":
            render_manager_dashboard(user)
        elif selected_page == "⚙️ System Admin & Security Governance":
            render_admin_dashboard(user)
