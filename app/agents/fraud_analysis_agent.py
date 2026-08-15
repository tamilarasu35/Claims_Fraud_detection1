"""
Fraud Analysis Agent (Agent 2) Module.
Combines XGBoost Probability Estimation, EBM Glass-Box Risk Scoring, SHAP-style Explainability,
and Evidence Package generation into a unified ML intelligence layer.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.ml.xgboost_model import XGBoostFraudClassifier
from app.ml.ebm_model import EBMExplainerModel
from app.ml.feature_engineering import FeatureEngineer
from app.utils.logger import logger

class FraudAnalysisAgent:
    """Fraud Analysis Agent - Core Machine Learning & Risk Intelligence Layer."""
    
    def __init__(self):
        self.xgb_model = XGBoostFraudClassifier()
        self.ebm_model = EBMExplainerModel()
        self.feature_names: List[str] = []
        self.is_trained: bool = False


    def try_load_pretrained(self, version_tag: str = "v1.0.0") -> bool:
        """Attempt to auto-load pre-trained XGBoost and EBM models from disk if available."""
        try:
            self.xgb_model.load_model(version_tag)
            self.ebm_model.load_model(version_tag)
            self.feature_names = self.xgb_model.feature_names
            self.is_trained = True
            logger.info(f"FraudAnalysisAgent: Pre-trained production models ({version_tag}) auto-loaded successfully!")
            return True
        except Exception as e:
            logger.info(f"FraudAnalysisAgent: No pre-trained model found for {version_tag} ({e}).")
            return False


    def train_pipeline(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost and EBM models on extracted provider behavioral features."""
        logger.info("FraudAnalysisAgent: Training ML models (XGBoost + EBM)...")
        
        self.feature_names = FeatureEngineer.get_feature_names(features_df)
        X = features_df[self.feature_names]
        y = features_df['Target']
        
        # Train XGBoost
        xgb_metrics = self.xgb_model.train(X, y)
        
        # Train EBM
        ebm_metrics = self.ebm_model.train(X, y)
        
        self.is_trained = True
        logger.info("FraudAnalysisAgent: ML pipeline training complete.")
        
        return {
            "xgb_metrics": xgb_metrics,
            "ebm_metrics": ebm_metrics,
            "total_providers_trained": len(X),
            "num_features": len(self.feature_names)
        }

    def analyze_provider(self, provider_id: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive, audit-ready Fraud Evidence Package for a specific provider.
        """
        if not self.is_trained:
            raise ValueError("FraudAnalysisAgent Error: Models must be trained before inference.")
            
        prov_row = features_df[features_df['Provider'] == provider_id]
        if prov_row.empty:
            raise ValueError(f"Provider {provider_id} not found in feature dataset.")
            
        single_row = prov_row.iloc[0]
        X_single = single_row[self.feature_names]
        X_single_df = pd.DataFrame([X_single])
        
        # 1. XGBoost Prediction & Probability
        prob = float(self.xgb_model.predict_proba(X_single_df)[0])
        classification = "Potentially Fraudulent" if prob >= self.xgb_model.optimal_threshold else "Likely Legitimate"
        
        # 2. EBM Glass-Box Risk Score & Level
        risk_scores, _ = self.ebm_model.predict_risk_score(X_single_df)
        risk_score = int(risk_scores[0])
        risk_level = EBMExplainerModel.get_risk_level(risk_score)
        recommendation = EBMExplainerModel.get_recommendation(risk_level)
        
        # 3. Top Feature Contributions (EBM Additive Effects)
        contributions = self.ebm_model.get_local_explanation(X_single)
        top_contributions = contributions[:5]
        
        # 4. Extract Key Behavioral Metrics & Peer Z-Scores
        zscore_cols = [c for c in self.feature_names if c.endswith('_PeerZScore')]
        peer_deviations = {
            col: float(single_row[col]) for col in zscore_cols
        }
        
        behavioral_summary = {
            "TotalClaims": int(single_row.get("TotalClaims", 0)),
            "TotalReimbursement": float(single_row.get("TotalReimbursement", 0.0)),
            "InpatientRatio": float(single_row.get("InpatientRatio", 0.0)),
            "UniqueBeneficiaries": int(single_row.get("UniqueBeneficiaries", 0)),
            "AvgReimbursementPerClaim": float(single_row.get("AvgReimbursementPerClaim", 0.0)),
            "ReimbursementPerBeneficiary": float(single_row.get("ReimbursementPerBeneficiary", 0.0))
        }

        # 5. Model Uncertainty & Confidence
        # Higher confidence when probability is far from optimal decision threshold
        dist_from_threshold = abs(prob - self.xgb_model.optimal_threshold)
        if dist_from_threshold >= 0.25:
            confidence = "HIGH"
        elif dist_from_threshold >= 0.10:
            confidence = "MODERATE"
        else:
            confidence = "LOW (NEAR DECISION BOUNDARY)"

        # Construct Evidence Summary text
        top_feat_desc = ", ".join([f"{c['feature']} ({c['value']})" for c in top_contributions[:3]])
        evidence_summary = (
            f"Provider {provider_id} exhibits a model-estimated fraud probability of {prob*100:.1f}% "
            f"and an EBM risk score of {risk_score}/100 ({risk_level} Risk). Primary risk factors include: {top_feat_desc}. "
            f"Recommended Action: {recommendation}."
        )

        # 6. Assemble Structured Evidence Package
        evidence_package = {
            "provider_id": provider_id,
            "classification": classification,
            "fraud_probability": round(prob, 4),
            "fraud_probability_pct": f"{prob*100:.1f}%",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "top_contributions": top_contributions,
            "peer_deviations": peer_deviations,
            "behavioral_summary": behavioral_summary,
            "model_confidence": confidence,
            "evidence_summary": evidence_summary,
            "limitations": "Probability estimate based on Medicare claims data. Requires clinical & financial audit confirmation."
        }
        
        return evidence_package

    def analyze_all_providers(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch inference for all providers returning structured summary table.
        """
        if not self.is_trained:
            raise ValueError("FraudAnalysisAgent Error: Models must be trained before inference.")
            
        X = features_df[self.feature_names]
        probs = self.xgb_model.predict_proba(X)
        risk_scores, _ = self.ebm_model.predict_risk_score(X)
        
        results_df = features_df[['Provider']].copy()
        results_df['FraudProbability'] = np.round(probs, 4)
        results_df['RiskScore'] = risk_scores
        results_df['RiskLevel'] = results_df['RiskScore'].apply(EBMExplainerModel.get_risk_level)
        results_df['Classification'] = (results_df['FraudProbability'] >= self.xgb_model.optimal_threshold).map(
            {True: "Potentially Fraudulent", False: "Likely Legitimate"}
        )
        results_df['Recommendation'] = results_df['RiskLevel'].apply(EBMExplainerModel.get_recommendation)
        
        return results_df
