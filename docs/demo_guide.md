# Hackathon Demo Guide & Script for Judges

## 🎯 15-Minute Winning Hackathon Demonstration Workflow

Follow this step-by-step walkthrough during your live hackathon presentation to impress the judges:

---

### Step 1: Sign In & Authentication Demo
1. Open Streamlit App: `streamlit run app/streamlit_app.py`
2. Show the secure login interface.
3. Sign in as **Admin** (`admin` / `Admin123!`) or **Investigator** (`investigator` / `Investigator123!`).

---

### Step 2: Data Ingestion & Perception Agent Execution
1. Navigate to **`📥 Data Ingestion & Perception`**.
2. Click **"🚀 Trigger Full Multi-Agent Pipeline"**.
3. Point out live agent progress:
   - Perception Agent validating 558,211 claims across 5,410 providers.
   - Provider coverage (100.0%) and referential integrity audit.
   - Feature engineering (35+ behavioral features and peer group z-scores).
   - XGBoost & EBM model training.

---

### Step 3: Provider Investigation & Glass-Box Explanations
1. Navigate to **`🔍 Provider Investigation Drill-Down`**.
2. Show the suspicious provider ranking table with risk progress bars.
3. Select a **CRITICAL Risk** provider (e.g. `PRV55465` or highest risk score).
4. Demonstrate:
   - **Overview Banner:** Risk Score (e.g. `91/100`), Fraud Probability (e.g. `87.4%`), Priority (`CRITICAL`).
   - **Tab 1 (ML Explanations):** Show Top 5 EBM Additive Feature Contributions & Peer Z-Scores.
   - **Tab 2 (Negotiation Agent):** Highlight the adversarial debate between Prosecutorial Fraud Argument and Defense Counter-Argument.
   - **Tab 3 (Arbitrator Decision):** Show the reasoned resolution paragraph.
   - **Tab 4 (Case Notes):** Record investigator decision notes into the SQLite audit database.

---

### Step 4: Executive Manager Dashboard
1. Switch role / view to **`📊 Manager Executive Dashboard`**.
2. Present high-level financial risk metrics (Total Reimbursement at Risk).
3. Display the Risk Level Distribution bar chart and Top 10 Highest Risk Providers table.

---

### Step 5: System Governance & Security Audit Trail
1. Navigate to **`⚙️ Admin System Management`**.
2. Show the User Account Creation interface and RBAC permissions matrix.
3. Inspect the live **Security Audit Trail Logs** recording every action taken during the demo.
