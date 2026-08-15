"""
Multi-Agent Orchestrator System Module.
Orchestrates end-to-end execution of Perception Agent -> Fraud Analysis Agent -> Negotiation Agent -> Arbitrator.
Manages workflow state, database logging, audit trail tracking, and execution timelines.
"""

import uuid
from typing import Dict, Any, List, Optional
import pandas as pd

from app.agents.perception_agent import PerceptionAgent
from app.agents.fraud_analysis_agent import FraudAnalysisAgent
from app.agents.negotiation_agent import NegotiationAgent
from app.agents.arbitrator import Arbitrator
from app.ml.feature_engineering import FeatureEngineer
from app.database.repositories import ResultsRepository, AuditRepository
from app.database.models import ProviderResult
from app.utils.logger import logger

class FraudIntelligenceOrchestrator:
    """Master Multi-Agent System Orchestrator."""
    
    def __init__(self):
        self.perception_agent = PerceptionAgent()
        self.fraud_analysis_agent = FraudAnalysisAgent()
        self.negotiation_agent = NegotiationAgent()
        self.arbitrator = Arbitrator()
        
        self.run_uuid: str = str(uuid.uuid4())
        self.features_df: Optional[pd.DataFrame] = None
        self.perception_report: Dict[str, Any] = {}
        self.execution_timeline: List[Dict[str, str]] = []

    def log_step(self, agent_name: str, status: str, details: str):
        """Log agent state transition to execution timeline."""
        entry = {"agent": agent_name, "status": status, "details": details}
        self.execution_timeline.append(entry)
        logger.info(f"ORCHESTRATOR TIMELINE | [{agent_name}] {status}: {details}")

    def run_training_pipeline(self, is_train: bool = True) -> Dict[str, Any]:
        """Execute full training multi-agent pipeline from raw dataset."""
        logger.info(f"Orchestrator: Initiating full pipeline run (UUID: {self.run_uuid})...")
        
        # Step 1: Perception Agent
        self.log_step("Perception Agent", "RUNNING", "Ingesting, validating, and preprocessing healthcare claims...")
        self.perception_report = self.perception_agent.process(is_train=is_train)
        self.log_step("Perception Agent", "COMPLETED", f"Ingested {self.perception_report['total_claims']:,} claims across {self.perception_report['total_providers']:,} providers.")
        
        # Step 2: Feature Engineering
        self.log_step("Feature Engineering", "RUNNING", "Aggregating claims to provider-level behavioral features & peer z-scores...")
        self.features_df = FeatureEngineer.generate_provider_features(self.perception_agent.cleaned_data)
        ResultsRepository.upsert_providers_from_features(self.features_df)
        self.log_step("Feature Engineering", "COMPLETED", f"Generated {self.features_df.shape[1]-2} behavioral features.")

        
        # Step 3: Fraud Analysis Agent Training
        self.log_step("Fraud Analysis Agent", "RUNNING", "Training XGBoost classifier and EBM glass-box risk explainer...")
        train_res = self.fraud_analysis_agent.train_pipeline(self.features_df)
        self.log_step("Fraud Analysis Agent", "COMPLETED", f"Models trained. XGBoost ROC-AUC: {train_res['xgb_metrics']['roc_auc']:.4f}.")
        
        return {
            "run_uuid": self.run_uuid,
            "perception_report": self.perception_report,
            "training_results": train_res,
            "timeline": self.execution_timeline
        }

    def analyze_single_provider(self, provider_id: str, username: str = "system") -> Dict[str, Any]:
        """
        Execute full multi-agent analysis chain for a specific provider:
        Perception -> ML Evidence -> Negotiation -> Arbitrator -> DB Persistence.
        """
        if self.features_df is None or not self.fraud_analysis_agent.is_trained:
            self.run_training_pipeline(is_train=True)
            
        # Step 1: Fraud Analysis Agent Evidence Package
        evidence = self.fraud_analysis_agent.analyze_provider(provider_id, self.features_df)
        
        # Step 2: Negotiation Agent Adversarial Review
        negotiation = self.negotiation_agent.process_evidence(evidence)
        
        # Step 3: Arbitrator Resolution
        final_decision = self.arbitrator.resolve_decision(
            perception_report=self.perception_report,
            evidence_package=evidence,
            negotiation_output=negotiation
        )
        
        # Step 4: Persist to SQLite Database
        db_result = ProviderResult(
            provider_id=provider_id,
            classification=final_decision["classification"],
            fraud_probability=final_decision["fraud_probability"],
            risk_score=final_decision["risk_score"],
            risk_level=final_decision["risk_level"],
            recommendation=final_decision["final_recommendation"],
            top_features=final_decision["top_features"],
            negotiation_argument=final_decision["negotiation_argument"],
            negotiation_challenge=final_decision["negotiation_challenge"],
            arbitrator_reasoning=final_decision["arbitrator_reasoning"],
            run_uuid=self.run_uuid
        )
        ResultsRepository.save_provider_result(db_result)
        
        AuditRepository.log_action(
            username=username,
            role="SYSTEM",
            action="PROVIDER_ANALYSIS_COMPLETED",
            target_resource=provider_id,
            details=f"Decision: {final_decision['classification']}, Score: {final_decision['risk_score']}/100"
        )
        
        return {
            "run_uuid": self.run_uuid,
            "evidence_package": evidence,
            "negotiation": negotiation,
            "final_decision": final_decision,
            "timeline": self.execution_timeline
        }
