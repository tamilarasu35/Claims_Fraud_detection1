"""
Unit tests for Phase 5 EBM Risk Scoring, Explainability & Evidence Package.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.perception_agent import PerceptionAgent
from app.ml.feature_engineering import FeatureEngineer
from app.agents.fraud_analysis_agent import FraudAnalysisAgent
from app.ml.ebm_model import EBMExplainerModel

def test_ebm_risk_scoring_and_explainability():
    agent = PerceptionAgent()
    agent.process(is_train=True)
    features_df = FeatureEngineer.generate_provider_features(agent.cleaned_data)
    feature_names = FeatureEngineer.get_feature_names(features_df)
    
    X = features_df[feature_names]
    y = features_df['Target']
    
    ebm = EBMExplainerModel()
    ebm.train(X, y)
    
    risk_scores, decision_scores = ebm.predict_risk_score(X)
    assert len(risk_scores) == len(X)
    assert (risk_scores >= 0).all() and (risk_scores <= 100).all()
    
    # Check risk level assignment
    assert EBMExplainerModel.get_risk_level(95) == "CRITICAL"
    assert EBMExplainerModel.get_risk_level(75) == "HIGH"
    assert EBMExplainerModel.get_risk_level(45) == "MEDIUM"
    assert EBMExplainerModel.get_risk_level(15) == "LOW"
    
    # Local explanation
    local_exp = ebm.get_local_explanation(X.iloc[0])
    assert len(local_exp) == len(feature_names)
    assert "feature" in local_exp[0]
    assert "score_effect" in local_exp[0]

def test_fraud_analysis_agent_evidence_package():
    agent = PerceptionAgent()
    agent.process(is_train=True)
    features_df = FeatureEngineer.generate_provider_features(agent.cleaned_data)
    
    fa_agent = FraudAnalysisAgent()
    train_res = fa_agent.train_pipeline(features_df)
    assert train_res["num_features"] >= 30
    
    # Analyze single provider
    provider_id = features_df['Provider'].iloc[0]
    evidence = fa_agent.analyze_provider(provider_id, features_df)
    
    assert evidence["provider_id"] == provider_id
    assert evidence["classification"] in ["Potentially Fraudulent", "Likely Legitimate"]
    assert 0.0 <= evidence["fraud_probability"] <= 1.0
    assert 0 <= evidence["risk_score"] <= 100
    assert evidence["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert len(evidence["top_contributions"]) == 5
    assert len(evidence["evidence_summary"]) > 20
