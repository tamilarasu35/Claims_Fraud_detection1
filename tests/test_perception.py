"""
Unit tests for Phase 2 Perception Agent & Healthcare Preprocessor.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.perception_agent import PerceptionAgent
from app.ml.preprocessing import HealthcarePreprocessor

def test_healthcare_preprocessor_beneficiary():
    sample_bene = pd.DataFrame([{
        "BeneID": "BENE1",
        "DOB": "1943-01-01",
        "DOD": None,
        "Gender": 1,
        "Race": 1,
        "RenalDiseaseIndicator": "Y",
        "ChronicCond_Alzheimer": 1,
        "ChronicCond_Heartfailure": 2
    }])
    
    cleaned = HealthcarePreprocessor.clean_beneficiary_data(sample_bene)
    assert cleaned["Age"].iloc[0] > 60
    assert cleaned["IsDeceased"].iloc[0] == 0
    assert cleaned["RenalDiseaseIndicator"].iloc[0] == 1
    assert cleaned["ChronicCond_Alzheimer"].iloc[0] == 1
    assert cleaned["ChronicCond_Heartfailure"].iloc[0] == 0

def test_healthcare_preprocessor_claims():
    sample_claims = pd.DataFrame([{
        "ClaimID": "CLM1",
        "ClaimStartDt": "2009-01-01",
        "ClaimEndDt": "2009-01-05",
        "AdmissionDt": "2009-01-01",
        "DischargeDt": "2009-01-05",
        "InscClaimAmtReimbursed": 1000.0,
        "DeductibleAmtPaid": 100.0,
        "ClmDiagnosisCode_1": "4019",
        "ClmDiagnosisCode_2": None,
        "ClmProcedureCode_1": "9904",
        "AttendingPhysician": "PHY1",
        "OperatingPhysician": None
    }])
    
    cleaned = HealthcarePreprocessor.clean_claims_data(sample_claims, is_inpatient=True)
    assert cleaned["ClaimDurationDays"].iloc[0] == 5
    assert cleaned["LengthOfStayDays"].iloc[0] == 5
    assert cleaned["TotalClaimAmt"].iloc[0] == 1100.0
    assert cleaned["NumDiagnoses"].iloc[0] == 1
    assert cleaned["NumProcedures"].iloc[0] == 1

def test_perception_agent_actual_dataset():
    agent = PerceptionAgent()
    report = agent.process(is_train=True)
    
    assert report["status"] == "SUCCESS"
    assert report["total_providers"] == 5410
    assert report["total_inpatient_claims"] == 40474
    assert report["total_outpatient_claims"] == 517737
    assert report["total_beneficiaries"] == 138556
    assert report["referential_integrity"]["provider_coverage_pct"] == 100.0
    assert report["leakage_analysis"]["leakage_detected"] is False
    assert len(agent.cleaned_data) == 4
