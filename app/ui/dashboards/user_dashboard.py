"""
User Dashboard Module.
Handles claims dataset uploads, single provider manually-entered input forms,
initiating multi-agent fraud analysis, and displaying live Perception Agent findings.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from app.agents.orchestrator import FraudIntelligenceOrchestrator
from app.audit.audit_logger import audit_log
from app.utils.logger import logger

def render_user_dashboard(user: dict):
    st.title("📥 Data Ingestion & Provider Fraud Analysis")
    st.write("Upload Medicare claims datasets or analyze individual healthcare providers through the multi-agent pipeline.")
    
    tabs = st.tabs(["📁 Ingest Dataset & Run Agent Pipeline", "👤 Individual Provider Analysis Form"])
    
    # ---------------- TAB 1: DATASET INGESTION ----------------
    with tabs[0]:
        st.subheader("Ingest Healthcare Dataset")
        st.info("The system automatically ingests Inpatient, Outpatient, and Beneficiary claims tables from the `data/` repository.")
        
        if st.button("🚀 Trigger Full Multi-Agent Pipeline", type="primary"):
            with st.status("Executing Multi-Agent Fraud Intelligence Pipeline...", expanded=True) as status:
                orchestrator = FraudIntelligenceOrchestrator()
                st.write("🔍 **Perception Agent:** Ingesting & validating claims dataset...")
                
                output = orchestrator.run_training_pipeline(is_train=True)
                st.write("⚡ **Fraud Analysis Agent:** Generating provider features & training XGBoost + EBM models...")
                st.write("⚖️ **Negotiation Agent & Arbitrator:** System ready for provider risk decisions.")
                
                status.update(label="✅ Multi-Agent Pipeline Execution Complete!", state="complete", expanded=False)
                
            st.session_state["orchestrator"] = orchestrator
            st.session_state["pipeline_run"] = True
            audit_log(user["username"], user["role_name"], "PIPELINE_RUN_COMPLETED", details=f"Run UUID: {output['run_uuid']}")
            
            # Display Perception Findings KPI Summary
            perc = output["perception_report"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Providers", f"{perc['total_providers']:,}")
            col2.metric("Total Claims Processed", f"{perc['total_claims']:,}")
            col3.metric("Total Beneficiaries", f"{perc['total_beneficiaries']:,}")
            col4.metric("Provider Coverage", f"{perc['referential_integrity']['provider_coverage_pct']}%")
            
            st.success("Dataset successfully ingested and ML models trained. Proceed to Investigation or Manager Dashboards!")

    # ---------------- TAB 2: SINGLE PROVIDER INPUT FORM ----------------
    with tabs[1]:
        st.subheader("Single Provider Analysis Form")
        st.write("Enter provider behavioral metrics to perform real-time multi-agent fraud risk arbitration.")
        
        with st.form("single_provider_form"):
            col1, col2 = st.columns(2)
            with col1:
                provider_id_input = st.text_input("Provider ID", value="PRV55465")
                total_claims = st.number_input("Total Claim Volume", min_value=1, value=150)
                inpatient_claims = st.number_input("Inpatient Claim Volume", min_value=0, value=45)
                total_reimbursement = st.number_input("Total Reimbursement Amount ($)", min_value=0.0, value=85000.0)
            with col2:
                unique_bene = st.number_input("Unique Beneficiaries Served", min_value=1, value=110)
                avg_age = st.number_input("Average Patient Age", min_value=18, max_value=110, value=74)
                unique_attending = st.number_input("Unique Attending Physicians", min_value=1, value=8)
                unique_diags = st.number_input("Unique Diagnosis Codes Count", min_value=1, value=25)
                
            submit_form = st.form_submit_button("🔍 Run Multi-Agent Fraud Assessment")
            
            if submit_form:
                if "orchestrator" not in st.session_state or not st.session_state.get("pipeline_run"):
                    st.warning("Please trigger the main pipeline first in Tab 1 to initialize trained models.")
                else:
                    orchestrator: FraudIntelligenceOrchestrator = st.session_state["orchestrator"]
                    
                    if provider_id_input in orchestrator.features_df["Provider"].values:
                        res = orchestrator.analyze_single_provider(provider_id_input, username=user["username"])
                        dec = res["final_decision"]
                        
                        st.divider()
                        st.subheader(f"Results for {provider_id_input}")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Classification", dec["classification"])
                        m2.metric("Fraud Probability", dec["fraud_probability_pct"])
                        m3.metric("Risk Score", f"{dec['risk_score']} / 100")
                        m4.metric("Risk Level", dec["risk_level"])
                        
                        st.info(f"**Arbitrator Recommendation:** {dec['final_recommendation']}")
                        st.markdown(f"**Reasoning:** {dec['arbitrator_reasoning']}")
                    else:
                        st.error(f"Provider ID '{provider_id_input}' not found in current dataset.")
