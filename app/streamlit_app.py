"""
Healthcare Provider Fraud Intelligence System - Streamlit Entry Point.
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

# 1. Page Configuration
st.set_page_config(
    page_title="Healthcare Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Vanilla CSS with rich aesthetics, gradients, and dark accents)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    .main-header p {
        color: #e0e6ed;
        margin-top: 8px;
        font-size: 1.1rem;
    }
    .badge-role {
        background-color: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .card-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# 2. System Initialization
@st.cache_resource
def initialize_system():
    init_db()
    AuthService.bootstrap_default_users()
    logger.info("Healthcare Provider Fraud Intelligence System booted.")

initialize_system()

# 3. Session State Management
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>🛡️ Healthcare Provider Fraud Intelligence System</h1>
    <p>Explainable Multi-Agent Medicare Claims Risk & Behavior Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & Auth
st.sidebar.title("Navigation & Identity")

if not st.session_state["authenticated"]:
    st.sidebar.subheader("🔐 Login")
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")
        
        if submitted:
            user = AuthService.authenticate(username, password)
            if user:
                st.session_state["authenticated"] = True
                st.session_state["user"] = user
                st.sidebar.success(f"Welcome, {user['full_name']}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials.")
                
    st.info("👋 Please sign in using one of the bootstrap demo accounts (e.g. `admin` / `Admin123!`, `investigator` / `Investigator123!`, `manager` / `Manager123!`, `user` / `User123!`).")
    
else:
    user = st.session_state["user"]
    st.sidebar.markdown(f"**Logged in as:** {user['full_name']}")
    st.sidebar.markdown(f"**Role:** <span class='badge-role'>{user['role_name']}</span>", unsafe_allow_html=True)
    
    if st.sidebar.button("Logout"):
        audit_log(user["username"], user["role_name"], "LOGOUT", details="User signed out")
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()
        
    st.sidebar.divider()
    
    # Available Views based on RBAC
    options = []
    if AuthorizationService.require_permission(user, "upload_dataset"):
        options.append("📥 Data Ingestion & Perception")
    if AuthorizationService.require_permission(user, "view_suspicious_providers"):
        options.append("🔍 Provider Investigation Drill-Down")
    if AuthorizationService.require_permission(user, "view_executive_dashboard"):
        options.append("📊 Manager Executive Dashboard")
    if AuthorizationService.require_permission(user, "manage_users"):
        options.append("⚙️ Admin System Management")
        
    selected_page = st.sidebar.radio("Module Selection", options if options else ["Overview"])
    
    st.subheader(f"Active Module: {selected_page}")
    st.write("System foundation initialized cleanly. Phase 1 active.")
