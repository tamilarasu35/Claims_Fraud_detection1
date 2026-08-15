"""
Healthcare Claims Data Preprocessing and Cleaning Module.
Handles date parsing, missing value imputation, chronic condition standardization,
and data quality validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from app.utils.logger import logger

class HealthcarePreprocessor:
    """Preprocesses raw Medicare inpatient, outpatient, beneficiary, and target datasets."""
    
    @staticmethod
    def clean_beneficiary_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format beneficiary table."""
        df = df.copy()
        
        # Convert date fields
        df['DOB'] = pd.to_datetime(df['DOB'], format='%Y-%m-%d', errors='coerce')
        df['DOD'] = pd.to_datetime(df['DOD'], format='%Y-%m-%d', errors='coerce')
        
        # Calculate Age (or Age at death if deceased, else relative reference date 2009-12-31)
        ref_date = pd.to_datetime('2009-12-31')
        end_date = df['DOD'].fillna(ref_date)
        df['Age'] = ((end_date - df['DOB']).dt.days / 365.25).round(1)
        df['IsDeceased'] = df['DOD'].notnull().astype(int)
        
        # Convert RenalDiseaseIndicator ('Y' -> 1, '0'/'N' -> 0)
        df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].astype(str).str.upper().apply(
            lambda x: 1 if x in ['Y', '1'] else 0
        )
        
        # Clean Chronic Conditions (Kaggle dataset uses 1=Yes, 2=No -> convert to 1=Yes, 0=No)
        chronic_cols = [c for c in df.columns if c.startswith('ChronicCond_')]
        for col in chronic_cols:
            df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)
            
        return df

    @staticmethod
    def clean_claims_data(df: pd.DataFrame, is_inpatient: bool = False) -> pd.DataFrame:
        """Clean and format inpatient or outpatient claims table."""
        df = df.copy()
        
        # Date conversions
        df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'], format='%Y-%m-%d', errors='coerce')
        df['ClaimEndDt'] = pd.to_datetime(df['ClaimEndDt'], format='%Y-%m-%d', errors='coerce')
        df['ClaimDurationDays'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days + 1
        df['ClaimDurationDays'] = df['ClaimDurationDays'].clip(lower=1)
        
        if is_inpatient and 'AdmissionDt' in df.columns:
            df['AdmissionDt'] = pd.to_datetime(df['AdmissionDt'], format='%Y-%m-%d', errors='coerce')
            df['DischargeDt'] = pd.to_datetime(df['DischargeDt'], format='%Y-%m-%d', errors='coerce')
            df['LengthOfStayDays'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days + 1
            df['LengthOfStayDays'] = df['LengthOfStayDays'].fillna(0).clip(lower=0)
        else:
            df['LengthOfStayDays'] = 0
            
        # Financial fields numeric conversion & missing handling
        df['InscClaimAmtReimbursed'] = pd.to_numeric(df['InscClaimAmtReimbursed'], errors='coerce').fillna(0.0)
        df['DeductibleAmtPaid'] = pd.to_numeric(df['DeductibleAmtPaid'], errors='coerce').fillna(0.0)
        df['TotalClaimAmt'] = df['InscClaimAmtReimbursed'] + df['DeductibleAmtPaid']
        
        # Count non-null diagnosis and procedure codes
        diag_cols = [c for c in df.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols = [c for c in df.columns if c.startswith('ClmProcedureCode_')]
        
        df['NumDiagnoses'] = df[diag_cols].notnull().sum(axis=1)
        df['NumProcedures'] = df[proc_cols].notnull().sum(axis=1)
        
        # Attending physician flag
        df['HasAttendingPhysician'] = df['AttendingPhysician'].notnull().astype(int)
        df['HasOperatingPhysician'] = df['OperatingPhysician'].notnull().astype(int) if 'OperatingPhysician' in df.columns else 0
        
        return df

    @staticmethod
    def clean_target_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean target provider dataset."""
        df = df.copy()
        if 'PotentialFraud' in df.columns:
            df['Target'] = df['PotentialFraud'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
        return df
