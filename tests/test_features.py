"""
Unit tests for Phase 3 Feature Engineering & Peer Metrics.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.perception_agent import PerceptionAgent
from app.ml.feature_engineering import FeatureEngineer

def test_feature_engineering_pipeline():
    agent = PerceptionAgent()
    agent.process(is_train=True)
    
    features_df = FeatureEngineer.generate_provider_features(agent.cleaned_data)
    feature_names = FeatureEngineer.get_feature_names(features_df)
    
    assert len(features_df) == 5410
    assert 'Target' in features_df.columns
    assert 'Provider' in features_df.columns
    assert 'TotalReimbursement' in features_df.columns
    assert 'TotalReimbursement_PeerZScore' in features_df.columns
    
    # Verify no NaN or Inf in feature matrix
    X = features_df[feature_names]
    assert X.isnull().sum().sum() == 0
    assert np.isinf(X.values).sum() == 0
    assert len(feature_names) >= 30

def test_leakage_prevention():
    agent = PerceptionAgent()
    agent.process(is_train=True)
    
    features_df = FeatureEngineer.generate_provider_features(agent.cleaned_data)
    feature_names = FeatureEngineer.get_feature_names(features_df)
    
    # Ensure target labels and raw IDs are NOT in feature matrix X
    assert 'Target' not in feature_names
    assert 'PotentialFraud' not in feature_names
    assert 'Provider' not in feature_names
