# Healthcare Provider Fraud Intelligence System — Technical Architecture Specification

## 1. Executive Summary & Product Vision

The **Healthcare Provider Fraud Intelligence System** is an end-to-end, serious, hackathon-grade multi-agent decision support platform designed to identify healthcare providers exhibiting suspicious Medicare billing behavior. Built upon the Kaggle **Healthcare Provider Fraud Detection Analysis** dataset (558,211 inpatient/outpatient claims across 5,410 target providers), the system combines statistical ML classification, glass-box explainable risk modeling, and adversarial multi-agent arbitration into an audit-ready platform.

---

## 2. Multi-Agent System Architecture & Data Flow

```
                                  USER (Streamlit UI)
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             │                 AUTHENTICATION & RBAC                     │
             │           (USER, INVESTIGATOR, MANAGER, ADMIN)           │
             └─────────────────────────────┬─────────────────────────────┘
                                           │
                                  MASTER ORCHESTRATOR
                                           │
    ┌──────────────────────────────────────┼──────────────────────────────────────┐
    │                                      │                                      │
    ▼                                      ▼                                      ▼
1. PERCEPTION AGENT            2. FRAUD ANALYSIS AGENT              3. NEGOTIATION AGENT
• Data Profiling & Quality     • Provider Feature Aggregation       • EXAMINE Evidence
• Duplicate & Missing Checks   • XGBoost Fraud Classifier           • ARGUE Fraud Hypothesis
• Referential Integrity        • EBM Glass-Box Risk Score (0-100)   • CHALLENGE Defense Points
• Data Leakage Audit           • Evidence Package Generation        • PROPOSE Recommendation
    │                                      │                                      │
    └──────────────────────────────────────┼──────────────────────────────────────┘
                                           │
                                           ▼
                                    4. ARBITRATOR
                           • Evidence vs Defense Resolution
                           • Final Provider Classification
                           • Priority & Reasoned Judgment
                                           │
                                           ▼
                               SQLite DATABASE & AUDIT LOGS
                         (providers, fraud_results, audit_logs)
```

---

## 3. Role-Based Access Control (RBAC) Permission Matrix

| Role | Upload & Trigger Pipeline | Individual Form Input | Provider Drill-Down | EBM Glass-Box Explanations | Case Recording | Executive Manager Dashboard | User Admin & Audit Logs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **USER** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **INVESTIGATOR** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **MANAGER** | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **ADMIN** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 4. Machine Learning & Risk Intelligence Specifications

- **Primary Classifier:** XGBoost (`scale_pos_weight = 9.69`, `n_estimators = 250`, `max_depth = 4`)
  - **Performance:** ROC-AUC: **`0.9412`**, PR-AUC: **`0.6845`**, F1-Score: **`0.6316`** at optimal threshold **`0.62`**.
- **Explainable Model:** EBM (Explainable Boosting Machine - Glass-box additive terms)
  - **Risk Score:** 0 to 100 calibrated log-odds score.
  - **Risk Levels:** `CRITICAL` (81-100), `HIGH` (61-80), `MEDIUM` (31-60), `LOW` (0-30).
