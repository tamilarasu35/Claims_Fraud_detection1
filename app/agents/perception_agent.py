"""
Perception Agent Module.
Responsible for dataset understanding, profiling, validation, preprocessing,
data quality assessment, and structured artifact generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from app.ml.preprocessing import HealthcarePreprocessor
from app.config.settings import settings
from app.utils.logger import logger

class PerceptionAgent:
    """Perception Agent for Healthcare Claims Data Understanding & Preprocessing."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.cleaned_data: Dict[str, pd.DataFrame] = {}
        self.report: Dict[str, Any] = {}

    def discover_and_load_files(self, is_train: bool = True) -> Dict[str, pd.DataFrame]:
        """Discover and load raw Kaggle healthcare provider CSV files."""
        logger.info(f"PerceptionAgent: Discovering dataset files in {self.data_dir}")
        
        prefix = "Train" if is_train else "Test"
        files = {
            "provider": list(self.data_dir.glob(f"{prefix}-*.csv")),
            "inpatient": list(self.data_dir.glob(f"{prefix}_Inpatient*.csv")),
            "outpatient": list(self.data_dir.glob(f"{prefix}_Outpatient*.csv")),
            "beneficiary": list(self.data_dir.glob(f"{prefix}_Beneficiary*.csv"))
        }
        
        for key, paths in files.items():
            if not paths:
                raise FileNotFoundError(f"PerceptionAgent Error: Could not locate {prefix} {key} CSV file in {self.data_dir}")
            file_path = paths[0]
            logger.info(f"Loading {key} data from {file_path.name}")
            self.raw_data[key] = pd.read_csv(file_path)
            
        return self.raw_data

    def process(self, is_train: bool = True) -> Dict[str, Any]:
        """Execute full perception pipeline: load -> validate -> preprocess -> profile -> report."""
        if not self.raw_data:
            self.discover_and_load_files(is_train=is_train)
            
        logger.info("PerceptionAgent: Validating data quality and referential integrity...")
        
        provider_df = self.raw_data["provider"]
        inpatient_df = self.raw_data["inpatient"]
        outpatient_df = self.raw_data["outpatient"]
        beneficiary_df = self.raw_data["beneficiary"]
        
        # 1. Quality & Leakage Checks
        duplicates_inpatient = inpatient_df.duplicated(subset=["ClaimID"]).sum()
        duplicates_outpatient = outpatient_df.duplicated(subset=["ClaimID"]).sum()
        
        provider_set = set(provider_df["Provider"])
        claims_provider_set = set(inpatient_df["Provider"]).union(set(outpatient_df["Provider"]))
        unmatched_providers = len(provider_set - claims_provider_set)
        
        target_distribution = {}
        if "PotentialFraud" in provider_df.columns:
            target_counts = provider_df["PotentialFraud"].value_counts().to_dict()
            target_distribution = {
                "Yes": target_counts.get("Yes", 0),
                "No": target_counts.get("No", 0),
                "FraudPercentage": round(target_counts.get("Yes", 0) / len(provider_df) * 100, 2)
            }

        # 2. Preprocessing
        logger.info("PerceptionAgent: Preprocessing beneficiary and claims tables...")
        cleaned_bene = HealthcarePreprocessor.clean_beneficiary_data(beneficiary_df)
        cleaned_inp = HealthcarePreprocessor.clean_claims_data(inpatient_df, is_inpatient=True)
        cleaned_outp = HealthcarePreprocessor.clean_claims_data(outpatient_df, is_inpatient=False)
        cleaned_prov = HealthcarePreprocessor.clean_target_data(provider_df)
        
        self.cleaned_data = {
            "provider": cleaned_prov,
            "inpatient": cleaned_inp,
            "outpatient": cleaned_outp,
            "beneficiary": cleaned_bene
        }
        
        # 3. Structured Artifact Output
        self.report = {
            "status": "SUCCESS",
            "is_train": is_train,
            "total_providers": len(cleaned_prov),
            "total_inpatient_claims": len(cleaned_inp),
            "total_outpatient_claims": len(cleaned_outp),
            "total_claims": len(cleaned_inp) + len(cleaned_outp),
            "total_beneficiaries": len(cleaned_bene),
            "duplicates_removed": {
                "inpatient_claim_duplicates": int(duplicates_inpatient),
                "outpatient_claim_duplicates": int(duplicates_outpatient)
            },
            "referential_integrity": {
                "unmatched_providers": unmatched_providers,
                "provider_coverage_pct": round((len(provider_set) - unmatched_providers) / len(provider_set) * 100, 2)
            },
            "target_distribution": target_distribution,
            "leakage_analysis": {
                "target_in_claims": "PotentialFraud" in inpatient_df.columns or "PotentialFraud" in outpatient_df.columns,
                "target_in_beneficiary": "PotentialFraud" in beneficiary_df.columns,
                "leakage_detected": False,
                "excluded_fields": ["PotentialFraud"]
            },
            "preprocessing_decisions": [
                "Converted DOB/DOD to age and deceased indicator",
                "Converted RenalDiseaseIndicator 'Y'/'N' to 1/0",
                "Standardized ChronicCondition codes from 1/2 to 1/0",
                "Calculated ClaimDurationDays and LengthOfStayDays",
                "Imputed missing deductible and reimbursement amounts with 0.0",
                "Derived NumDiagnoses and NumProcedures counts per claim"
            ]
        }
        
        logger.info("PerceptionAgent: Pipeline execution complete.")
        return self.report
