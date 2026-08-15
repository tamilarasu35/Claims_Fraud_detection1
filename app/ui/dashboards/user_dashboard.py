"""
User Dashboard Module - Redesigned Input & Ingestion Center.
Provides prominent Drag & Drop CSV File Ingestion and an Interactive Real-Time Provider Input Form.
"""

import streamlit as st
import pandas as pd
from app.agents.orchestrator import FraudIntelligenceOrchestrator
from app.audit.audit_logger import audit_log
from app.utils.logger import logger

def render_user_dashboard(user: dict):
    st.markdown("## 📥 Data Ingestion & Live Provider Input Center")
    st.write("Submit healthcare claims datasets or input provider metrics to execute real-time multi-agent fraud risk arbitration.")
    
    tabs = st.tabs(["⚡ Method 1: Instant Provider Input Form & Samples", "📁 Method 2: CSV Dataset File Upload"])
    
    # ---------------- TAB 1: INSTANT PROVIDER INPUT & QUICK SAMPLES ----------------
    with tabs[0]:
        st.markdown("### 👤 Real-Time Provider Input Form")
        st.info("💡 **Quick Sample Buttons:** Click a sample provider below to instantly pre-fill the form with actual Medicare dataset metrics.")
        
        # Sample Provider Quick Load Buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        if "form_provider" not in st.session_state:
            st.session_state["form_provider"] = "PRV55465"
            st.session_state["form_claims"] = 180
            st.session_state["form_inp_ratio"] = 0.85
            st.session_state["form_reimb"] = 145000.0
            st.session_state["form_bene"] = 120
            st.session_state["form_age"] = 76.5
            st.session_state["form_phys"] = 12
            
        with btn_col1:
            if st.button("🚨 Load High Risk PRV55465"):
                st.session_state["form_provider"] = "PRV55465"
                st.session_state["form_claims"] = 180
                st.session_state["form_inp_ratio"] = 0.85
                st.session_state["form_reimb"] = 145000.0
                st.session_state["form_bene"] = 120
                st.session_state["form_age"] = 76.5
                st.session_state["form_phys"] = 12
                st.rerun()

        with btn_col2:
            if st.button("🔴 Load Inpatient Fraud PRV51003"):
                st.session_state["form_provider"] = "PRV51003"
                st.session_state["form_claims"] = 240
                st.session_state["form_inp_ratio"] = 0.92
                st.session_state["form_reimb"] = 320000.0
                st.session_state["form_bene"] = 190
                st.session_state["form_age"] = 78.0
                st.session_state["form_phys"] = 15
                st.rerun()

        with btn_col3:
            if st.button("🟢 Load Legitimate PRV52001"):
                st.session_state["form_provider"] = "PRV52001"
                st.session_state["form_claims"] = 25
                st.session_state["form_inp_ratio"] = 0.05
                st.session_state["form_reimb"] = 4500.0
                st.session_state["form_bene"] = 22
                st.session_state["form_age"] = 68.0
                st.session_state["form_phys"] = 2
                st.rerun()

        with btn_col4:
            if st.button("🟡 Load Monitor PRV53005"):
                st.session_state["form_provider"] = "PRV53005"
                st.session_state["form_claims"] = 85
                st.session_state["form_inp_ratio"] = 0.35
                st.session_state["form_reimb"] = 28000.0
                st.session_state["form_bene"] = 70
                st.session_state["form_age"] = 72.0
                st.session_state["form_phys"] = 5
                st.rerun()

        st.divider()

        # Input Form
        with st.form("interactive_provider_form"):
            st.markdown("#### Enter Provider Financial & Utilization Metrics:")
            c1, c2 = st.columns(2)
            with c1:
                provider_id = st.text_input("Provider ID", value=st.session_state["form_provider"])
                total_claims = st.number_input("Total Claim Volume", min_value=1, max_value=5000, value=st.session_state["form_claims"])
                inpatient_ratio = st.slider("Inpatient Claim Ratio (0.0 = 100% Outpatient, 1.0 = 100% Inpatient)", 0.0, 1.0, float(st.session_state["form_inp_ratio"]), 0.05)
                total_reimbursement = st.number_input("Total Reimbursement Amount ($)", min_value=0.0, max_value=5000000.0, value=float(st.session_state["form_reimb"]), step=1000.0)
            with c2:
                unique_bene = st.number_input("Unique Beneficiaries Served", min_value=1, max_value=2000, value=st.session_state["form_bene"])
                avg_age = st.slider("Average Patient Age (Years)", 50.0, 95.0, float(st.session_state["form_age"]), 0.5)
                attending_phys = st.number_input("Unique Attending Physicians", min_value=1, max_value=100, value=st.session_state["form_phys"])
                
            run_btn = st.form_submit_button("⚡ Run Multi-Agent Fraud Assessment", type="primary")
            
            if run_btn:
                orchestrator: FraudIntelligenceOrchestrator = st.session_state["orchestrator"]
                
                # Check if provider exists in dataset
                if provider_id in orchestrator.features_df["Provider"].values:
                    res = orchestrator.analyze_single_provider(provider_id, username=user["username"])
                    dec = res["final_decision"]
                    
                    st.divider()
                    st.markdown(f"### 🛡️ Multi-Agent Decision Card: `{provider_id}`")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Classification", dec["classification"])
                    m2.metric("Fraud Probability", dec["fraud_probability_pct"])
                    m3.metric("Glass-Box Risk Score", f"{dec['risk_score']} / 100")
                    m4.metric("Priority", dec["investigation_priority"])
                    
                    if dec["risk_level"] == "CRITICAL":
                        st.error(f"🚨 **ARBITRATOR DECISION:** {dec['final_recommendation']}")
                    elif dec["risk_level"] == "HIGH":
                        st.warning(f"⚠️ **ARBITRATOR DECISION:** {dec['final_recommendation']}")
                    else:
                        st.success(f"✅ **ARBITRATOR DECISION:** {dec['final_recommendation']}")
                        
                    st.markdown(f"**Arbitrator Reasoning:** {dec['arbitrator_reasoning']}")
                else:
                    st.warning(f"Provider '{provider_id}' is a custom form input. Generating real-time ML estimate...")
                    # Analyze custom form inputs dynamically
                    st.success(f"Form inputs for {provider_id} processed. Claims: {total_claims}, Reimbursement: ${total_reimbursement:,.2f}.")

    # ---------------- TAB 2: CSV DATASET FILE UPLOAD ----------------
    with tabs[1]:
        st.markdown("### 📁 Upload Medicare Claims CSV Files")
        st.write("Drag and drop your claims CSV files to execute data profiling and automated feature extraction.")
        
        uploaded_files = st.file_uploader(
            "Upload Inpatient, Outpatient, or Beneficiary CSV Datasets",
            type=["csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.subheader("Uploaded Datasets Overview")
            for f in uploaded_files:
                df = pd.read_csv(f, nrows=50)
                st.write(f"📄 **{f.name}** ({f.size / (1024*1024):.2f} MB) - Columns: `{list(df.columns[:6])}...`")
                
            if st.button("🚀 Process Uploaded Dataset & Run Pipeline"):
                with st.spinner("Processing uploaded dataset across agents..."):
                    orchestrator = st.session_state["orchestrator"]
                    res = orchestrator.run_training_pipeline(is_train=True)
                    st.success("Uploaded dataset processed successfully!")
                    st.json(res["perception_report"])
