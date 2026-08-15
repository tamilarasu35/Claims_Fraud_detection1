"""
Data models and schemas for Healthcare Provider Fraud Intelligence System.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

@dataclass
class User:
    id: Optional[int]
    username: str
    password_hash: str
    full_name: str
    email: Optional[str]
    role_name: str = "USER"
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

@dataclass
class Role:
    id: Optional[int]
    name: str
    description: str

@dataclass
class ProviderResult:
    provider_id: str
    classification: str  # "Potentially Fraudulent" or "Likely Legitimate"
    fraud_probability: float  # 0.0 - 1.0 (e.g. 0.874)
    risk_score: int  # 0 - 100
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommendation: str  # HIGH-PRIORITY INVESTIGATION, MONITOR, LOW CONCERN
    top_features: Dict[str, Any] = field(default_factory=dict)
    negotiation_argument: str = ""
    negotiation_challenge: str = ""
    arbitrator_reasoning: str = ""
    run_uuid: str = ""
    created_at: Optional[datetime] = None

@dataclass
class AuditLogEntry:
    id: Optional[int]
    username: str
    role: str
    action: str
    target_resource: Optional[str]
    details: Optional[str]
    timestamp: Optional[datetime] = None
    ip_address: str = "127.0.0.1"
