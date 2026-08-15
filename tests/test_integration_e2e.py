"""
End-to-End System Integration Test Suite.
Verifies full flow: Dataset Ingestion -> Perception Agent -> Feature Engineering ->
XGBoost Classifier -> EBM Risk Scorer -> Negotiation Agent -> Arbitrator -> DB Persistence -> Audit Trail.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.database import init_db, get_db_connection
from app.auth.authentication import AuthService
from app.agents.orchestrator import FraudIntelligenceOrchestrator
from app.database.repositories import ResultsRepository, AuditRepository

def test_full_system_integration_e2e():
    """Execute complete end-to-end multi-agent pipeline and verify database persistence."""
    # 1. Initialize DB and Bootstrap Users
    init_db()
    AuthService.bootstrap_default_users()
    
    # 2. Instantiate Orchestrator
    orchestrator = FraudIntelligenceOrchestrator()
    run_output = orchestrator.run_training_pipeline(is_train=True)
    
    assert run_output["perception_report"]["status"] == "SUCCESS"
    assert run_output["training_results"]["xgb_metrics"]["roc_auc"] >= 0.85
    assert orchestrator.features_df is not None
    assert len(orchestrator.features_df) == 5410
    
    # 3. Pick a High-Risk Provider and Run Real-Time Analysis
    high_risk_prov = orchestrator.features_df[orchestrator.features_df['Target'] == 1]['Provider'].iloc[0]
    analysis_res = orchestrator.analyze_single_provider(high_risk_prov, username="investigator")
    
    dec = analysis_res["final_decision"]
    evidence = analysis_res["evidence_package"]
    neg = analysis_res["negotiation"]
    
    assert dec["provider_id"] == high_risk_prov
    assert dec["risk_score"] >= 0 and dec["risk_score"] <= 100
    assert len(dec["negotiation_argument"]) > 0
    assert len(dec["negotiation_challenge"]) > 0
    assert len(dec["arbitrator_reasoning"]) > 0
    
    # 4. Verify SQLite DB Results Persistence
    saved_results = ResultsRepository.get_results_by_run(orchestrator.run_uuid)
    assert len(saved_results) >= 1
    assert saved_results[0]["provider_id"] == high_risk_prov
    
    # 5. Verify Audit Logs Recorded Action
    audit_logs = AuditRepository.get_recent_logs(limit=20)
    actions = [log["action"] for log in audit_logs]
    assert "PROVIDER_ANALYSIS_COMPLETED" in actions
