"""
Train and Save Production ML Model Artifacts (XGBoost + EBM) to models/v1.0.0/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.perception_agent import PerceptionAgent
from app.ml.feature_engineering import FeatureEngineer
from app.agents.fraud_analysis_agent import FraudAnalysisAgent
from app.database.database import init_db
from app.database.repositories import ResultsRepository
from app.utils.logger import logger

def main():
    print("==================================================")
    print("TRAINING & SAVING PRODUCTION ML MODELS (v1.0.0)")
    print("==================================================")
    
    init_db()
    
    # 1. Perception Agent
    print("\n[1/4] Running Perception Agent on Medicare Claims...")
    perception = PerceptionAgent()
    perception.process(is_train=True)
    
    # 2. Feature Engineering
    print("\n[2/4] Extracting Provider Behavioral Features & Peer Z-Scores...")
    features_df = FeatureEngineer.generate_provider_features(perception.cleaned_data)
    ResultsRepository.upsert_providers_from_features(features_df)
    
    # 3. Fraud Analysis Agent Training
    print("\n[3/4] Training XGBoost Classifier and EBM Glass-Box Risk Model...")
    fa_agent = FraudAnalysisAgent()
    metrics = fa_agent.train_pipeline(features_df)
    
    print("\n--- MODEL TRAINING RESULTS ---")
    print(f"XGBoost ROC-AUC: {metrics['xgb_metrics']['roc_auc']:.4f}")
    print(f"XGBoost PR-AUC:  {metrics['xgb_metrics']['pr_auc']:.4f}")
    print(f"XGBoost F1-Score:{metrics['xgb_metrics']['f1_score']:.4f} (Optimal Threshold: {metrics['xgb_metrics']['optimal_threshold']})")
    
    # 4. Save Model Artifacts
    print("\n[4/4] Persisting Trained Model Artifacts to models/v1.0.0/...")
    fa_agent.xgb_model.save_model("v1.0.0")
    fa_agent.ebm_model.save_model("v1.0.0")
    
    print("\nProduction ML models successfully trained and saved to models/v1.0.0/!")

if __name__ == "__main__":
    main()

