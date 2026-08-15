"""
Unit tests for Phase 4 XGBoost Fraud Classification & Model Persistence.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.perception_agent import PerceptionAgent
from app.ml.feature_engineering import FeatureEngineer
from app.ml.xgboost_model import XGBoostFraudClassifier

def test_xgboost_training_and_evaluation(tmp_path):
    agent = PerceptionAgent()
    agent.process(is_train=True)
    
    features_df = FeatureEngineer.generate_provider_features(agent.cleaned_data)
    feature_names = FeatureEngineer.get_feature_names(features_df)
    
    X = features_df[feature_names]
    y = features_df['Target']
    
    classifier = XGBoostFraudClassifier(model_dir=tmp_path)
    metrics = classifier.train(X, y, test_size=0.2, random_state=42)
    
    # Assert performance threshold standards
    assert metrics['roc_auc'] >= 0.85, f"ROC-AUC score {metrics['roc_auc']} below standard threshold 0.85"
    assert metrics['pr_auc'] > 0.40
    assert metrics['f1_score'] > 0.40
    assert 0.1 <= metrics['optimal_threshold'] <= 0.9
    
    # Probabilities sanity check
    probs = classifier.predict_proba(X)
    assert len(probs) == len(X)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_xgboost_model_persistence_and_reload(tmp_path):
    agent = PerceptionAgent()
    agent.process(is_train=True)
    
    features_df = FeatureEngineer.generate_provider_features(agent.cleaned_data)
    feature_names = FeatureEngineer.get_feature_names(features_df)
    
    X = features_df[feature_names]
    y = features_df['Target']
    
    classifier = XGBoostFraudClassifier(model_dir=tmp_path)
    classifier.train(X, y, test_size=0.2, random_state=42)
    
    # Save Model
    version_tag = "v1.0.0-test"
    classifier.save_model(version_tag=version_tag)
    
    # Reload Model
    reloaded_clf = XGBoostFraudClassifier(model_dir=tmp_path)
    reloaded_clf.load_model(version_tag=version_tag)
    
    reloaded_probs = reloaded_clf.predict_proba(X)
    original_probs = classifier.predict_proba(X)
    
    np.testing.assert_array_almost_equal(original_probs, reloaded_probs)
    assert reloaded_clf.feature_names == classifier.feature_names
