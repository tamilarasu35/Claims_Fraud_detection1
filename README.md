# Healthcare Provider Fraud Intelligence System

A serious, hackathon-grade, explainable Multi-Agent Healthcare Provider Fraud Detection & Intelligence Platform built on the Kaggle Medicare Claims Dataset.

---

## 🛡️ Key Features & Multi-Agent Architecture

```
User (Streamlit Application)
  │
  ├── Authentication & Role-Based Access Control (RBAC: USER, INVESTIGATOR, MANAGER, ADMIN)
  │
  └── Orchestrator
        ├── Perception Agent (Ingestion, Data Profiling, Quality Checks, Provider-level Aggregation)
        ├── Fraud Analysis Agent (XGBoost Fraud Probability + EBM Risk Score 0–100 + Evidence Package)
        ├── Negotiation Agent (Examine ──► Argue ──► Challenge ──► Propose)
        └── Arbitrator (Final Decision Engine & Audit Summary)
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and adjust settings if required:
```bash
cp .env.example .env
```

### 3. Launch Streamlit Application
```bash
streamlit run app/streamlit_app.py
```

### 4. Default Bootstrap Accounts for Demo
- **Admin:** `admin` / `Admin123!`
- **Investigator:** `investigator` / `Investigator123!`
- **Manager:** `manager` / `Manager123!`
- **User:** `user` / `User123!`

---

## 🧪 Running Tests
```bash
pytest tests/
```
