"""
Investigator Dashboard Module.
Dedicated fraud investigation view with suspicious provider ranking, risk filters,
detailed provider drill-down, glass-box explanations, agent reasoning, and case recording.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from app.agents.orchestrator import FraudIntelligenceOrchestrator
from app.database.database import get_db_connection
from app.audit.audit_logger import audit_log
from app.utils.logger import logger

def render_investigator_dashboard(user: dict):
    st.title("🔍 Investigator Provider Fraud Case Management")
    st.write("Drill down into suspicious healthcare providers, inspect ML explanations, multi-agent adversarial challenges, and record investigation case decisions.")
    
    if "orchestrator" not in st.session_state or not st.session_state.get("pipeline_run"):
        st.warning("⚠️ No active analysis pipeline found. Please navigate to 'Data Ingestion' and run the multi-agent pipeline first.")
        return
        
    orchestrator: FraudIntelligenceOrchestrator = st.session_state["orchestrator"]
    features_df = orchestrator.features_df
    
    # ---------------- FILTERS & PROVIDER SEARCH ----------------
    st.subheader("Filter Suspicious Providers")
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Risk Level Filter", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH"])
    with col2:
        class_filter = st.selectbox("Classification Filter", ["All", "Potentially Fraudulent", "Likely Legitimate"])
    with col3:
        min_score = st.slider("Minimum Risk Score (0-100)", 0, 100, 50)
        
    # Get all batch provider results
    batch_results = orchestrator.fraud_analysis_agent.analyze_all_providers(features_df)
    
    # Filter dataset
    filtered_df = batch_results[batch_results["RiskScore"] >= min_score]
    if risk_filter:
        filtered_df = filtered_df[filtered_df["RiskLevel"].isin(risk_filter)]
    if class_filter != "All":
        filtered_df = filtered_df[filtered_df["Classification"] == class_filter]
        
    st.dataframe(
        filtered_df.sort_values(by="RiskScore", ascending=False),
        column_config={
            "FraudProbability": st.column_config.NumberColumn("Fraud Probability", format="%.4f"),
            "RiskScore": st.column_config.ProgressColumn("Risk Score", format="%d/100", min_value=0, max_value=100),
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # ---------------- PROVIDER DRILL-DOWN VIEW ----------------
    st.subheader("🔬 Deep Provider Investigation Drill-Down")
    selected_provider = st.selectbox("Select Provider ID to Investigate", options=filtered_df["Provider"].tolist() if not filtered_df.empty else ["No providers matching filters"])
    
    if selected_provider and selected_provider != "No providers matching filters":
        res = orchestrator.analyze_single_provider(selected_provider, username=user["username"])
        dec = res["final_decision"]
        evidence = res["evidence_package"]
        neg = res["negotiation"]
        
        # Provider Overview Banner
        st.markdown(f"### Case File: Provider `{selected_provider}`")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Classification", dec["classification"])
        c2.metric("Fraud Probability", dec["fraud_probability_pct"])
        c3.metric("Risk Score", f"{dec['risk_score']} / 100")
        c4.metric("Priority", dec["investigation_priority"])
        
        st.info(f"**Final Arbitrator Recommendation:** `{dec['final_recommendation']}`")
        
        # Tabs for ML Explanation & Multi-Agent Evidence Chain
        exp_tabs = st.tabs(["📊 ML & Glass-Box EBM Explanations", "⚔️ Negotiation (Argument vs Defense)", "🏛️ Arbitrator Decision Reasoning", "📝 Case Investigation Notes"])
        
        with exp_tabs[0]:
            st.subheader("Top Contributing Risk Factors (Additive EBM Log-Odds Effects)")
            top_contribs = dec.get("top_features", [])
            contrib_df = pd.DataFrame(top_contribs)
            if not contrib_df.empty:
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                
            st.subheader("Peer Group Behavioral Deviations (Z-Scores)")
            peer_z = dec.get("peer_deviations", {})
            st.json(peer_z)

        with exp_tabs[1]:
            st.subheader("Negotiation Agent Adversarial Review")
            st.markdown("#### 🚨 Prosecutorial Fraud Argument")
            st.warning(neg["fraud_argument"])
            
            st.markdown("#### 🛡️ Defense Counter-Argument & Legitimate Explanations")
            st.success(neg["counter_argument"])

        with exp_tabs[2]:
            st.subheader("Arbitrator Resolution & Reasoning")
            st.markdown(f"**Arbitrator Decision:** `{dec['classification']}`")
            st.write(dec["arbitrator_reasoning"])
            st.caption(f"Analysis Timestamp: {dec['analysis_timestamp']} | Model Version: {dec['model_version']}")

        with exp_tabs[3]:
            st.subheader("Record Investigator Decision & Case Notes")
            with st.form("investigation_notes_form"):
                status = st.selectbox("Investigation Status", ["NEW", "UNDER_INVESTIGATION", "CASE_PAUSED", "CLOSED_CONFIRMED_FRAUD", "CLOSED_LEGITIMATE"])
                notes = st.text_area("Investigator Case Notes", value=f"Inspected provider {selected_provider}. Probability: {dec['fraud_probability_pct']}.")
                submit_notes = st.form_submit_button("💾 Save Case Investigation Record")
                
                if submit_notes:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                    INSERT INTO investigations (provider_id, investigator_username, status, notes)
                    VALUES (?, ?, ?, ?)
                    """, (selected_provider, user["username"], status, notes))
                    conn.commit()
                    
                    audit_log(user["username"], user["role_name"], "INVESTIGATION_RECORDED", target_resource=selected_provider, details=f"Status: {status}")
                    st.success(f"Investigation case notes saved cleanly for Provider {selected_provider}!")
