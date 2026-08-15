"""
XGBoost Provider Fraud Classification Model Module.
Handles stratified train/test splitting, class imbalance handling via scale_pos_weight,
hyperparameter configuration, probability estimation, evaluation metrics, and reproducible model persistence.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

from app.config.settings import settings
from app.utils.logger import logger
from app.database.database import get_db_connection

class XGBoostFraudClassifier:
    """XGBoost Classifier for Provider Fraud Assessment."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or settings.MODEL_DIR
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.version_tag: str = "v1.0.0"
        self.optimal_threshold: float = 0.5

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train XGBoost classifier with class imbalance weighting and stratified test evaluation.
        """
        logger.info(f"XGBoost: Starting training on {len(X):,} providers with {X.shape[1]} features...")
        
        self.feature_names = list(X.columns)
        
        # 1. Stratified Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 2. Handle Class Imbalance via scale_pos_weight
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight = round(num_neg / max(1, num_pos), 4)
        logger.info(f"Class Distribution - Negative (Legit): {num_neg}, Positive (Fraud): {num_pos} | ScalePosWeight: {scale_pos_weight}")
        
        # 3. XGBoost Hyperparameters
        self.model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
        
        # 4. Model Fitting
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 5. Model Evaluation
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        # Find best threshold for F1-Score optimization
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1, best_thresh = 0.0, 0.5
        for t in thresholds:
            f1 = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
                
        self.optimal_threshold = round(best_thresh, 3)
        y_pred = (y_prob >= self.optimal_threshold).astype(int)
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1_opt = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Feature Importance (Gain)
        importance_scores = self.model.feature_importances_
        feature_importance_dict = dict(sorted(
            zip(self.feature_names, [round(float(s), 4) for s in importance_scores]),
            key=lambda x: x[1], reverse=True
        ))
        
        self.metrics = {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1_opt), 4),
            "optimal_threshold": self.optimal_threshold,
            "confusion_matrix": cm,
            "scale_pos_weight": scale_pos_weight,
            "top_10_features": dict(list(feature_importance_dict.items())[:10])
        }
        
        logger.info(f"XGBoost Evaluation: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1_opt:.4f} at threshold={self.optimal_threshold}")
        return self.metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability scores (0.0 to 1.0) for providers."""
        if self.model is None:
            raise ValueError("XGBoost Model is not trained or loaded.")
            
        # Reorder columns to match training feature order
        X_ordered = X[self.feature_names]
        return self.model.predict_proba(X_ordered)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary classification (1: Potentially Fraudulent, 0: Legitimate)."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def save_model(self, version_tag: str = "v1.0.0") -> Path:
        """Save model weights, feature names, and evaluation metadata."""
        if self.model is None:
            raise ValueError("Cannot save an untrained XGBoost model.")
            
        self.version_tag = version_tag
        version_dir = self.model_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = version_dir / "xgb_model.json"
        features_path = version_dir / "features.json"
        metadata_path = version_dir / "metadata.json"
        
        self.model.save_model(str(model_path))
        
        with open(features_path, "w") as f:
            json.dump({"feature_names": self.feature_names}, f, indent=2)
            
        with open(metadata_path, "w") as f:
            json.dump({
                "version_tag": version_tag,
                "metrics": self.metrics,
                "optimal_threshold": self.optimal_threshold
            }, f, indent=2)
            
        logger.info(f"XGBoost model artifacts persisted cleanly at {version_dir}")
        return version_dir

    def load_model(self, version_tag: str = "v1.0.0") -> None:
        """Load XGBoost model, feature order, and metadata from version directory."""
        version_dir = self.model_dir / version_tag
        model_path = version_dir / "xgb_model.json"
        features_path = version_dir / "features.json"
        metadata_path = version_dir / "metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing at {model_path}")
            
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        
        with open(features_path, "r") as f:
            self.feature_names = json.load(f)["feature_names"]
            
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            self.metrics = meta.get("metrics", {})
            self.optimal_threshold = meta.get("optimal_threshold", 0.5)
            
        self.version_tag = version_tag
        logger.info(f"XGBoost model version {version_tag} loaded successfully.")
