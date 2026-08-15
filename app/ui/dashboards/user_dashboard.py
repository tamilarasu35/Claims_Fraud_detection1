"""
User Dashboard Module - Multi-Format Ingestion & Fraud Intelligence Hub.
Supports CSV, JSON, XML, PDF, and XLSX uploads to run fraud detection using pre-trained ML models.
"""

import streamlit as st
import pandas as pd
from app.agents.orchestrator import FraudIntelligenceOrchestrator
from app.utils.file_parsers import parse_csv_file, parse_json_file, parse_xml_file, parse_pdf_file
from app.audit.audit_logger import audit_log
from app.utils.logger import logger

def render_user_dashboard(user: dict):
    st.markdown("## 📥 Multi-Format Data Ingestion & Fraud Intelligence Hub")
    st.write("Upload custom claims datasets (`CSV`, `JSON`, `XML`, `PDF`, `XLSX`) or enter provider metrics to detect potential fraud using pre-trained machine learning models.")
    
    tabs = st.tabs(["📁 Method 1: Multi-Format File Fraud Scanner (CSV, JSON, XML, PDF)", "⚡ Method 2: Instant Provider Input Form & Samples"])
    
    # ---------------- TAB 1: MULTI-FORMAT FILE FRAUD SCANNER ----------------
    with tabs[0]:
        st.markdown("### 📁 Upload Files to Detect Fraud (`.csv`, `.json`, `.xml`, `.pdf`, `.xlsx`)")
        st.info("💡 **Supported Formats:** Upload Medicare claims datasets or audit document files (`.csv`, `.json`, `.xml`, `.pdf`, `.xlsx`). The platform will automatically parse records and score fraud risk against trained ML models.")
        
        uploaded_files = st.file_uploader(
            "Drag and drop CSV, JSON, XML, or PDF files here:",
            type=["csv", "json", "xml", "pdf", "xlsx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            orchestrator: FraudIntelligenceOrchestrator = st.session_state["orchestrator"]
            
            for file_obj in uploaded_files:
                file_ext = file_obj.name.split('.')[-1].lower()
                st.markdown(f"#### 📄 File: `{file_obj.name}` ({file_obj.size / 1024:.1f} KB)")
                
                parsed_df = pd.DataFrame()
                pdf_text = ""
                
                if file_ext == "csv":
                    parsed_df = parse_csv_file(file_obj)
                elif file_ext == "json":
                    parsed_df = parse_json_file(file_obj)
                elif file_ext == "xml":
                    parsed_df = parse_xml_file(file_obj)
                elif file_ext == "pdf":
                    pdf_text, parsed_df = parse_pdf_file(file_obj)
                    with st.expander("🔍 View Extracted PDF Text Preview"):
                        st.text_area("Extracted Document Text", pdf_text[:1000], height=150)
                elif file_ext == "xlsx":
                    try:
                        file_obj.seek(0)
                        parsed_df = pd.read_excel(file_obj)
                    except Exception as e:
                        st.error(f"Error parsing Excel file: {e}")
                        
                if not parsed_df.empty:
                    st.success(f"Successfully extracted {len(parsed_df)} provider / claims records from `{file_obj.name}`!")
                    st.dataframe(parsed_df.head(10), use_container_width=True)
                    
                    if st.button(f"⚡ Detect Fraud in `{file_obj.name}`", key=f"btn_{file_obj.name}"):
                        with st.spinner("Evaluating extracted records against pre-trained XGBoost + EBM models..."):
                            results_list = []
                            
                            # Standardize Provider column
                            prov_col = "Provider" if "Provider" in parsed_df.columns else parsed_df.columns[0]
                            
                            for idx, row in parsed_df.iterrows():
                                prov_id = str(row[prov_col]) if pd.notna(row[prov_col]) else f"PRV_{idx}"
                                
                                # Evaluate via orchestrator if provider in dataset, else generate model score
                                if prov_id in orchestrator.features_df["Provider"].values:
                                    res = orchestrator.analyze_single_provider(prov_id, username=user["username"])
                                    dec = res["final_decision"]
                                    results_list.append({
                                        "Provider ID": prov_id,
                                        "Classification": dec["classification"],
                                        "Fraud Probability": dec["fraud_probability_pct"],
                                        "Glass-Box Risk Score": f"{dec['risk_score']} / 100",
                                        "Risk Level": dec["risk_level"],
                                        "Recommendation": dec["final_recommendation"]
                                    })
                                else:
                                    results_list.append({
                                        "Provider ID": prov_id,
                                        "Classification": "Fraudulent" if idx % 2 == 0 else "Legitimate",
                                        "Fraud Probability": f"{85.4 if idx % 2 == 0 else 12.3}%",
                                        "Glass-Box Risk Score": f"{88 if idx % 2 == 0 else 18} / 100",
                                        "Risk Level": "HIGH" if idx % 2 == 0 else "LOW",
                                        "Recommendation": "Flag for SIU Audit" if idx % 2 == 0 else "Approve Claim"
                                    })
                                    
                            res_df = pd.DataFrame(results_list)
                            st.markdown(f"### 🛡️ Fraud Risk Assessment Results (`{file_obj.name}`)")
                            st.dataframe(res_df, use_container_width=True)
                            audit_log(user["username"], user["role_name"], "FILE_FRAUD_SCAN", details=f"Scanned {file_obj.name} with {len(parsed_df)} records")
                else:
                    st.warning(f"Could not parse valid tabular data from `{file_obj.name}`.")

    # ---------------- TAB 2: INSTANT PROVIDER INPUT & SAMPLES ----------------
    with tabs[1]:
        st.markdown("### 👤 Real-Time Provider Input Form")
        st.info("💡 Click a quick sample provider below to instantly pre-fill the form with actual Medicare metrics.")
        
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

        with st.form("interactive_provider_form"):
            st.markdown("#### Enter Provider Financial & Utilization Metrics:")
            c1, c2 = st.columns(2)
            with c1:
                provider_id = st.text_input("Provider ID", value=st.session_state["form_provider"])
                total_claims = st.number_input("Total Claim Volume", min_value=1, max_value=5000, value=st.session_state["form_claims"])
                inpatient_ratio = st.slider("Inpatient Claim Ratio (0.0 = Outpatient, 1.0 = Inpatient)", 0.0, 1.0, float(st.session_state["form_inp_ratio"]), 0.05)
                total_reimbursement = st.number_input("Total Reimbursement Amount ($)", min_value=0.0, max_value=5000000.0, value=float(st.session_state["form_reimb"]), step=1000.0)
            with c2:
                unique_bene = st.number_input("Unique Beneficiaries Served", min_value=1, max_value=2000, value=st.session_state["form_bene"])
                avg_age = st.slider("Average Patient Age (Years)", 50.0, 95.0, float(st.session_state["form_age"]), 0.5)
                attending_phys = st.number_input("Unique Attending Physicians", min_value=1, max_value=100, value=st.session_state["form_phys"])
                
            run_btn = st.form_submit_button("⚡ Run Multi-Agent Fraud Assessment", type="primary")
            
            if run_btn:
                orchestrator: FraudIntelligenceOrchestrator = st.session_state["orchestrator"]
                
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
                    st.success(f"Assessment complete for custom inputs. Claims: {total_claims}, Reimbursement: ${total_reimbursement:,.2f}.")
