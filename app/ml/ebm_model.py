"""
Explainable Boosting Machine (EBM) & Risk Scoring Module.
Provides glass-box interpretable risk modeling, additive feature effect scoring,
risk level assignment (0-100 score, LOW/MEDIUM/HIGH/CRITICAL), and transparent contribution analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from app.config.settings import settings
from app.utils.logger import logger

class EBMExplainerModel:
    """Glass-Box Explainable Boosting Machine and 0-100 Risk Scorer."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or settings.MODEL_DIR
        self.model: Optional[ExplainableBoostingClassifier] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}

    def train(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Dict[str, Any]:
        """Train EBM glass-box classifier on provider features."""
        logger.info(f"EBM: Training Explainable Boosting Machine on {len(X):,} providers...")
        self.feature_names = list(X.columns)
        
        self.model = ExplainableBoostingClassifier(
            feature_names=self.feature_names,
            max_bins=128,
            max_interaction_bins=32,
            interactions=0,  # Main additive effects for maximum direct explainability
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        logger.info("EBM training complete. Glass-box additive feature terms extracted.")
        return {"status": "TRAINED", "num_features": len(self.feature_names)}

    def predict_risk_score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute risk score (0 to 100) and raw EBM log-odds risk score.
        Risk Score 0-100 is computed via calibrated sigmoid scaling of glass-box log-odds.
        """
        if self.model is None:
            raise ValueError("EBM Model is not trained.")
            
        X_ordered = X[self.feature_names]
        # Raw prediction log-odds (margin)
        decision_scores = self.model.decision_function(X_ordered)
        
        # Scale decision scores to 0-100 range
        # Sigmoid transform: 1 / (1 + exp(-margin)) -> 0.0 to 1.0 -> scale to 0-100
        probabilities = 1.0 / (1.0 + np.exp(-decision_scores))
        risk_scores = np.round(probabilities * 100).astype(int)
        risk_scores = np.clip(risk_scores, 0, 100)
        
        return risk_scores, decision_scores

    @staticmethod
    def get_risk_level(risk_score: int) -> str:
        """Assign standardized risk level category based on 0-100 risk score."""
        if risk_score >= 81:
            return "CRITICAL"
        elif risk_score >= 61:
            return "HIGH"
        elif risk_score >= 31:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def get_recommendation(risk_level: str) -> str:
        """Assign recommended action based on risk level."""
        if risk_level == "CRITICAL":
            return "HIGH-PRIORITY INVESTIGATION"
        elif risk_level == "HIGH":
            return "PRIORITY INVESTIGATION CANDIDATE"
        elif risk_level == "MEDIUM":
            return "MONITOR PROVIDER BEHAVIOR"
        else:
            return "LOW CONCERN / LIKELY LEGITIMATE"

    def get_local_explanation(self, X_single: pd.Series) -> List[Dict[str, Any]]:
        """
        Extract top feature-level additive contributions (log-odds effects) for a single provider.
        """
        if self.model is None:
            raise ValueError("EBM Model is not trained.")
            
        single_df = pd.DataFrame([X_single])[self.feature_names]
        
        # Local explanation object from interpret library
        ebm_local = self.model.explain_local(single_df)
        data = ebm_local.data(0)
        
        feature_names = data['names']
        feature_scores = data['scores']
        feature_values = data['values']
        
        contributions = []
        for name, score, val in zip(feature_names, feature_scores, feature_values):
            contributions.append({
                "feature": name,
                "value": round(float(val), 2) if isinstance(val, (int, float, np.number)) else str(val),
                "score_effect": round(float(score), 4)  # positive = increases fraud risk, negative = decreases
            })
            
        # Sort by absolute impact on risk score
        contributions = sorted(contributions, key=lambda x: abs(x["score_effect"]), reverse=True)
        return contributions

    def save_model(self, version_tag: str = "v1.0.0") -> Path:
        """Save EBM model metadata and configurations."""
        version_dir = self.model_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save EBM as pickle/joblib or json metadata
        import joblib
        joblib.dump(self.model, version_dir / "ebm_model.joblib")
        logger.info(f"EBM model persisted cleanly at {version_dir / 'ebm_model.joblib'}")
        return version_dir / "ebm_model.joblib"

    def load_model(self, version_tag: str = "v1.0.0") -> None:
        """Load EBM model from persistence storage."""
        version_dir = self.model_dir / version_tag
        model_path = version_dir / "ebm_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"EBM Model missing at {model_path}")
            
        import joblib
        self.model = joblib.load(model_path)
        self.feature_names = list(self.model.feature_names)
        logger.info(f"EBM model version {version_tag} loaded successfully.")
