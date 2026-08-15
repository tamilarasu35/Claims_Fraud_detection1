"""
Negotiation Agent (Agent 3) Module.
Implements adversarial evidence examination, fraud hypothesis argumentation,
legitimate alternative explanation challenges, and balanced recommendation proposals.
"""

from typing import Dict, Any, List
from app.utils.logger import logger

class NegotiationAgent:
    """Adversarial Negotiation Agent that argues both fraud hypothesis and legitimate defense."""
    
    def process_evidence(self, evidence_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute EXAMINE -> ARGUE -> CHALLENGE -> PROPOSE workflow.
        """
        provider_id = evidence_package["provider_id"]
        prob_pct = evidence_package["fraud_probability_pct"]
        risk_score = evidence_package["risk_score"]
        risk_level = evidence_package["risk_level"]
        top_contribs = evidence_package.get("top_contributions", [])
        behavior = evidence_package.get("behavioral_summary", {})
        peer_z = evidence_package.get("peer_deviations", {})
        
        logger.info(f"NegotiationAgent: Examining evidence for Provider {provider_id} (Score: {risk_score}/100, Prob: {prob_pct})")
        
        # 1. EXAMINE
        # Analyze positive and negative contributing factors
        risk_drivers = [c for c in top_contribs if c["score_effect"] > 0]
        mitigating_factors = [c for c in top_contribs if c["score_effect"] < 0]
        
        # 2. ARGUE (Fraud Hypothesis)
        argue_points = []
        if risk_score >= 60 or evidence_package["classification"] == "Potentially Fraudulent":
            argue_points.append(f"Model probability stands at {prob_pct} with an elevated EBM risk score of {risk_score}/100 ({risk_level}).")
            if risk_drivers:
                top_driver_str = ", ".join([f"{d['feature']} = {d['value']}" for d in risk_drivers[:3]])
                argue_points.append(f"Primary risk drivers detected: {top_driver_str}.")
            if peer_z.get("TotalReimbursement_PeerZScore", 0) > 2.0:
                argue_points.append(f"Total reimbursement is {peer_z['TotalReimbursement_PeerZScore']:.1f} standard deviations above the peer group mean.")
            if behavior.get("InpatientRatio", 0) > 0.5:
                argue_points.append(f"Abnormally high inpatient utilization ratio ({behavior['InpatientRatio']*100:.1f}%).")
        else:
            argue_points.append(f"Model probability ({prob_pct}) and risk score ({risk_score}/100) indicate low statistical fraud risk.")
            
        fraud_argument = " ".join(argue_points)

        # 3. CHALLENGE (Skeptical Defense / Legitimate Explanations)
        challenge_points = []
        if behavior.get("UniqueBeneficiaries", 0) > 500:
            challenge_points.append(f"Provider serves a large beneficiary volume ({behavior['UniqueBeneficiaries']:,} patients), which naturally elevates aggregate financial billing.")
        if behavior.get("AvgPatientAge", 0) > 75:
            challenge_points.append("Patient demographic skews elderly with complex chronic conditions, justifying higher care intensity and reimbursement.")
        if mitigating_factors:
            mit_str = ", ".join([f"{m['feature']} ({m['value']})" for m in mitigating_factors[:2]])
            challenge_points.append(f"Mitigating behavioral factors observed: {mit_str}.")
        if not challenge_points:
            challenge_points.append("Patterns show significant variance from peer averages that cannot be fully explained by patient volume alone.")
            
        counter_argument = " ".join(challenge_points)

        # 4. PROPOSE
        if risk_score >= 81:
            proposed_action = "HIGH-PRIORITY INVESTIGATION"
        elif risk_score >= 61:
            proposed_action = "PRIORITY INVESTIGATION CANDIDATE"
        elif risk_score >= 31:
            proposed_action = "MONITOR PROVIDER BEHAVIOR"
        else:
            proposed_action = "LOW CONCERN / LIKELY LEGITIMATE"

        return {
            "provider_id": provider_id,
            "examine_summary": f"Reviewed {len(top_contribs)} features, peer z-scores, and {behavior.get('TotalClaims', 0)} total claims.",
            "fraud_argument": fraud_argument,
            "counter_argument": counter_argument,
            "proposed_recommendation": proposed_action,
            "risk_drivers_count": len(risk_drivers),
            "mitigating_factors_count": len(mitigating_factors)
        }
