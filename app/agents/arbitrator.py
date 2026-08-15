"""
Arbitrator (Agent 4) Module.
Final decision layer that evaluates ML evidence, data quality, adversarial arguments,
and counter-arguments to issue audit-ready provider decisions.
"""

from typing import Dict, Any
from datetime import datetime
from app.utils.logger import logger

class Arbitrator:
    """Final Decision Engine & Resolution Arbitrator."""
    
    def resolve_decision(
        self,
        perception_report: Dict[str, Any],
        evidence_package: Dict[str, Any],
        negotiation_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate full multi-agent evidence chain and emit final arbitrated provider risk decision.
        """
        provider_id = evidence_package["provider_id"]
        prob = evidence_package["fraud_probability"]
        risk_score = evidence_package["risk_score"]
        risk_level = evidence_package["risk_level"]
        
        logger.info(f"Arbitrator: Resolving final risk decision for Provider {provider_id}...")
        
        # Arbitrator Conflict Resolution Logic
        fraud_arg = negotiation_output["fraud_argument"]
        counter_arg = negotiation_output["counter_argument"]
        
        if risk_score >= 81:
            classification = "Potentially Fraudulent"
            recommendation = "HIGH-PRIORITY INVESTIGATION"
            priority = "CRITICAL"
            reasoning = (
                f"Statistical ML evidence strongly supports potential fraud risk ({prob*100:.1f}% probability, {risk_score}/100 score). "
                f"While defense notes potential volume factors, severe peer z-score deviations and abnormal utilization justify immediate investigation."
            )
        elif risk_score >= 61:
            classification = "Potentially Fraudulent"
            recommendation = "PRIORITY INVESTIGATION CANDIDATE"
            priority = "HIGH"
            reasoning = (
                f"Elevated fraud probability ({prob*100:.1f}%) and risk score ({risk_score}/100). "
                f"Substantial behavioral anomalies warrant formal investigator review despite legitimate patient demographic considerations."
            )
        elif risk_score >= 31:
            classification = "Likely Legitimate"
            recommendation = "MONITOR PROVIDER BEHAVIOR"
            priority = "MEDIUM"
            reasoning = (
                f"Moderate risk score ({risk_score}/100) with low-to-moderate fraud probability ({prob*100:.1f}%). "
                f"Counter-arguments indicate behavioral patterns are consistent with regional provider variance. Recommend routine monitoring."
            )
        else:
            classification = "Likely Legitimate"
            recommendation = "LOW CONCERN / LIKELY LEGITIMATE"
            priority = "LOW"
            reasoning = (
                f"Low fraud probability ({prob*100:.1f}%) and low risk score ({risk_score}/100). "
                f"Provider behavior aligns with standard peer medical practices."
            )

        final_decision = {
            "provider_id": provider_id,
            "classification": classification,
            "fraud_probability": prob,
            "fraud_probability_pct": f"{prob*100:.1f}%",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "investigation_priority": priority,
            "final_recommendation": recommendation,
            "arbitrator_reasoning": reasoning,
            "negotiation_argument": fraud_arg,
            "negotiation_challenge": counter_arg,
            "top_features": evidence_package.get("top_contributions", []),
            "peer_deviations": evidence_package.get("peer_deviations", {}),
            "model_version": "v1.0.0",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Arbitrator Resolution Complete: Provider {provider_id} -> {classification} ({priority} Priority)")
        return final_decision
