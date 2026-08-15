"""
Unit tests for Phase 6 Negotiation Agent, Arbitrator & Orchestrator Workflow.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.perception_agent import PerceptionAgent
from app.ml.feature_engineering import FeatureEngineer
from app.agents.fraud_analysis_agent import FraudAnalysisAgent
from app.agents.negotiation_agent import NegotiationAgent
from app.agents.arbitrator import Arbitrator
from app.agents.orchestrator import FraudIntelligenceOrchestrator
from app.database.database import init_db

def test_negotiation_agent_workflow():
    sample_evidence = {
        "provider_id": "PRV-TEST1",
        "classification": "Potentially Fraudulent",
        "fraud_probability": 0.874,
        "fraud_probability_pct": "87.4%",
        "risk_score": 91,
        "risk_level": "CRITICAL",
        "recommendation": "HIGH-PRIORITY INVESTIGATION",
        "top_contributions": [
            {"feature": "TotalReimbursement_PeerZScore", "value": 3.4, "score_effect": 1.8},
            {"feature": "InpatientRatio", "value": 0.85, "score_effect": 1.2}
        ],
        "behavioral_summary": {
            "TotalClaims": 500,
            "TotalReimbursement": 150000.0,
            "InpatientRatio": 0.85,
            "UniqueBeneficiaries": 600,
            "AvgPatientAge": 76.5
        },
        "peer_deviations": {"TotalReimbursement_PeerZScore": 3.4}
    }
    
    neg = NegotiationAgent()
    neg_res = neg.process_evidence(sample_evidence)
    
    assert neg_res["provider_id"] == "PRV-TEST1"
    assert len(neg_res["fraud_argument"]) > 20
    assert len(neg_res["counter_argument"]) > 20
    assert neg_res["proposed_recommendation"] == "HIGH-PRIORITY INVESTIGATION"

def test_arbitrator_resolution():
    sample_perception = {"status": "SUCCESS", "total_claims": 5000}
    sample_evidence = {
        "provider_id": "PRV-TEST1",
        "fraud_probability": 0.874,
        "risk_score": 91,
        "risk_level": "CRITICAL",
        "top_contributions": [{"feature": "TotalReimbursement", "value": 150000, "score_effect": 1.8}],
        "peer_deviations": {}
    }
    sample_neg = {
        "fraud_argument": "High statistical risk.",
        "counter_argument": "High patient volume."
    }
    
    arb = Arbitrator()
    final_dec = arb.resolve_decision(sample_perception, sample_evidence, sample_neg)
    
    assert final_dec["provider_id"] == "PRV-TEST1"
    assert final_dec["classification"] == "Potentially Fraudulent"
    assert final_dec["investigation_priority"] == "CRITICAL"
    assert len(final_dec["arbitrator_reasoning"]) > 30

def test_orchestrator_end_to_end():
    init_db()
    orchestrator = FraudIntelligenceOrchestrator()
    train_output = orchestrator.run_training_pipeline(is_train=True)
    
    assert train_output["perception_report"]["status"] == "SUCCESS"
    assert len(train_output["timeline"]) >= 6
    
    # Run analysis for first provider
    provider_id = orchestrator.features_df['Provider'].iloc[0]
    analysis = orchestrator.analyze_single_provider(provider_id, username="admin")
    
    assert analysis["final_decision"]["provider_id"] == provider_id
    assert analysis["final_decision"]["classification"] in ["Potentially Fraudulent", "Likely Legitimate"]
    assert len(analysis["final_decision"]["arbitrator_reasoning"]) > 10
