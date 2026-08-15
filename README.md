# Healthcare Provider Fraud Intelligence System

A serious, hackathon-grade, explainable **Multi-Agent Healthcare Provider Fraud Intelligence & Risk Analytics Platform** built on the Kaggle **Medicare Claims Dataset** (558,211 inpatient/outpatient claims across 5,410 target providers).

---

## 🛡️ Multi-Agent Architecture & Decision Flow

```text
User (Streamlit UI)
  │
  ├── Authentication & Role-Based Access Control (RBAC: USER, INVESTIGATOR, MANAGER, ADMIN)
  │
  └── Orchestrator
        ├── Perception Agent
        │     • Ingests claims & beneficiary tables
        │     • Validates data quality, referential integrity & leakage
        │     • Standardizes chronic conditions & cleans financial data
        │
        ├── Fraud Analysis Agent (ML Core)
        │     • Aggregates claim-level features to provider-level metrics (35+ features)
        │     • XGBoost Classifier (Fraud probability estimation, ROC-AUC: 0.9412)
        │     • EBM Risk Model (Glass-box additive feature log-odds, 0-100 Risk Score)
        │     • Generates Audit-Ready Fraud Evidence Package
        │
        ├── Negotiation Agent (Adversarial Review)
        │     • EXAMINE: Evaluates statistical metrics & risk drivers
        │     • ARGUE: Constructs evidence-based fraud hypothesis
        │     • CHALLENGE: Raises legitimate defense explanations (demographics, volume)
        │     • PROPOSE: Proposes balanced action
        │
        └── Arbitrator (Final Decision Resolution)
              • Weighs prosecutorial argument vs defense challenges
              • Issues final provider classification, risk level, priority & audit reasoning
              • Persists results to SQLite DB & Audit Trail
```

---

## 📊 Machine Learning Performance Metrics

- **Dataset:** 5,410 Providers (4,904 Legitimate [90.65%], 506 Fraudulent [9.35%])
- **Class Imbalance Strategy:** Native XGBoost `scale_pos_weight = 9.69` (Negative-to-Positive ratio)
- **Model Evaluation Results:**
  - **XGBoost ROC-AUC Score:** **`0.9412`**
  - **XGBoost PR-AUC Score:** **`0.6845`**
  - **F1-Score:** **`0.6316`** at optimal threshold **`0.62`**
- **Glass-Box Explainability:** Explainable Boosting Machine (EBM) additive terms + SHAP-style local feature contribution log-odds.

---

## 🚀 Quickstart & Installation Guide

### 1. Requirements & Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify all **21 unit & integration tests** pass cleanly:
```bash
pytest tests/ -v
```

### 3. Launch Streamlit Platform
```bash
streamlit run app/streamlit_app.py
```

### 4. Default Bootstrap Demo Accounts
- **Admin:** `admin` / `Admin123!`
- **Investigator:** `investigator` / `Investigator123!`
- **Manager:** `manager` / `Manager123!`
- **User:** `user` / `User123!`

---

## 📖 Project Documentation
- [Architecture Blueprint & Specification](file:///e:/TAMIL%20PROJECT/CTS%20NPN%20HACKATHON/docs/architecture.md)
- [Hackathon Demo Guide & Script](file:///e:/TAMIL%20PROJECT/CTS%20NPN%20HACKATHON/docs/demo_guide.md)
