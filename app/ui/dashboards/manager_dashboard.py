"""
Manager Executive Dashboard Module.
High-level executive dashboard featuring risk distribution metrics, high-risk provider rankings,
financial impact analysis, and analytical visualizations for management decision support.
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.agents.orchestrator import FraudIntelligenceOrchestrator

def render_manager_dashboard(user: dict):
    st.title("📊 Executive Fraud Intelligence & Risk Dashboard")
    st.write("Executive overview of healthcare provider fraud risks, risk level distributions, and financial reimbursement patterns.")
    
    if "orchestrator" not in st.session_state or not st.session_state.get("pipeline_run"):
        st.warning("⚠️ No active analysis pipeline found. Please navigate to 'Data Ingestion' and run the multi-agent pipeline first.")
        return
        
    orchestrator: FraudIntelligenceOrchestrator = st.session_state["orchestrator"]
    features_df = orchestrator.features_df
    
    # Get batch predictions
    results_df = orchestrator.fraud_analysis_agent.analyze_all_providers(features_df)
    full_df = features_df.merge(results_df[["Provider", "FraudProbability", "RiskScore", "RiskLevel", "Classification", "Recommendation"]], on="Provider")
    
    # ---------------- EXECUTIVE KPI CARDS ----------------
    total_provs = len(full_df)
    fraud_provs = len(full_df[full_df["Classification"] == "Potentially Fraudulent"])
    critical_provs = len(full_df[full_df["RiskLevel"] == "CRITICAL"])
    high_provs = len(full_df[full_df["RiskLevel"] == "HIGH"])
    total_reimb = full_df["TotalReimbursement"].sum()
    fraud_reimb = full_df[full_df["Classification"] == "Potentially Fraudulent"]["TotalReimbursement"].sum()
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Analyzed Providers", f"{total_provs:,}")
    k2.metric("Flagged Fraudulent", f"{fraud_provs:,}", delta=f"{fraud_provs/total_provs*100:.1f}%")
    k3.metric("Critical Risk", f"{critical_provs:,}")
    k4.metric("High Risk", f"{high_provs:,}")
    k5.metric("Fraud Reimbursement Risk", f"${fraud_reimb:,.2f}")
    
    st.divider()
    
    # ---------------- CHARTS & VISUALIZATIONS ----------------
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Risk Level Breakdown")
        risk_counts = full_df["RiskLevel"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0)
        st.bar_chart(risk_counts, color="#d97706")
        
    with col_chart2:
        st.subheader("Risk Score Distribution (0 - 100)")
        st.bar_chart(np.histogram(full_df["RiskScore"], bins=10, range=(0,100))[0], color="#2563eb")

    st.divider()
    
    # ---------------- TOP 10 HIGHEST RISK PROVIDERS TABLE ----------------
    st.subheader("🚨 Top 10 Highest Fraud Risk Providers")
    top_10 = full_df.sort_values(by="RiskScore", ascending=False).head(10)
    
    display_cols = [
        "Provider", "RiskScore", "FraudProbability", "RiskLevel",
        "Classification", "TotalClaims", "TotalReimbursement", "InpatientRatio", "UniqueBeneficiaries"
    ]
    st.dataframe(
        top_10[display_cols],
        column_config={
            "TotalReimbursement": st.column_config.NumberColumn("Total Reimbursement", format="$%.2f"),
            "FraudProbability": st.column_config.NumberColumn("Fraud Prob", format="%.4f"),
            "InpatientRatio": st.column_config.NumberColumn("Inpatient Ratio", format="%.2f")
        },
        use_container_width=True,
        hide_index=True
    )
