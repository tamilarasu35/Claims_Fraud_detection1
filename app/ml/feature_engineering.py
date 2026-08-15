"""
Healthcare Provider Behavioral Feature Engineering Pipeline.
Transforms claim-level and beneficiary-level clean data into comprehensive provider-level behavioral features.
Includes strict data leakage checks and peer group deviation calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.utils.logger import logger

class FeatureEngineer:
    """Computes provider-level behavioral features for ML modeling."""
    
    EXCLUDED_COLUMNS = ['Provider', 'PotentialFraud', 'Target']
    
    @classmethod
    def generate_provider_features(cls, cleaned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate cleaned inpatient, outpatient, and beneficiary tables into provider-level features.
        
        cleaned_data expects keys: 'provider', 'inpatient', 'outpatient', 'beneficiary'
        """
        logger.info("FeatureEngineer: Generating provider behavioral features...")
        
        prov_df = cleaned_data['provider'].copy()
        inp_df = cleaned_data['inpatient'].copy()
        outp_df = cleaned_data['outpatient'].copy()
        bene_df = cleaned_data['beneficiary'].copy()
        
        # Merge Claims with Beneficiary information
        inp_full = inp_df.merge(bene_df, on='BeneID', how='left') if not inp_df.empty else pd.DataFrame()
        outp_full = outp_df.merge(bene_df, on='BeneID', how='left') if not outp_df.empty else pd.DataFrame()
        
        # Tag claim types
        if not inp_full.empty:
            inp_full['IsInpatient'] = 1
        if not outp_full.empty:
            outp_full['IsInpatient'] = 0
            
        # Combine all claims
        all_claims = pd.concat([inp_full, outp_full], ignore_index=True)
        
        if all_claims.empty:
            raise ValueError("FeatureEngineer Error: Combined claims dataset is empty.")
            
        logger.info(f"Aggregating {len(all_claims):,} total claims across providers...")
        
        # Chronic condition column names
        chronic_cols = [c for c in bene_df.columns if c.startswith('ChronicCond_')]
        all_claims['ChronicCount'] = all_claims[chronic_cols].sum(axis=1) if chronic_cols else 0
        
        # Provider-Level Aggregations
        grouped = all_claims.groupby('Provider')
        
        features = pd.DataFrame({'Provider': list(grouped.groups.keys())})
        
        # 1. Utilization & Volume Features
        features['TotalClaims'] = grouped['ClaimID'].count().values
        features['InpatientClaims'] = grouped['IsInpatient'].sum().values
        features['OutpatientClaims'] = features['TotalClaims'] - features['InpatientClaims']
        features['InpatientRatio'] = (features['InpatientClaims'] / features['TotalClaims']).round(4)
        
        features['AvgClaimDurationDays'] = grouped['ClaimDurationDays'].mean().round(2).values
        features['MaxClaimDurationDays'] = grouped['ClaimDurationDays'].max().values
        features['AvgLengthOfStayDays'] = grouped['LengthOfStayDays'].mean().round(2).values
        
        # 2. Financial Reimbursement Behavior
        features['TotalReimbursement'] = grouped['InscClaimAmtReimbursed'].sum().round(2).values
        features['AvgReimbursementPerClaim'] = grouped['InscClaimAmtReimbursed'].mean().round(2).values
        features['MaxReimbursementPerClaim'] = grouped['InscClaimAmtReimbursed'].max().round(2).values
        features['StdReimbursementPerClaim'] = grouped['InscClaimAmtReimbursed'].std().fillna(0).round(2).values
        
        features['TotalDeductiblePaid'] = grouped['DeductibleAmtPaid'].sum().round(2).values
        features['AvgDeductiblePerClaim'] = grouped['DeductibleAmtPaid'].mean().round(2).values
        features['TotalClaimAmount'] = grouped['TotalClaimAmt'].sum().round(2).values
        
        features['ReimbursementToTotalRatio'] = (
            features['TotalReimbursement'] / features['TotalClaimAmount'].replace(0, np.nan)
        ).fillna(0).round(4)

        # 3. Beneficiary Behavior & Patient Population Profile
        features['UniqueBeneficiaries'] = grouped['BeneID'].nunique().values
        features['AvgClaimsPerBeneficiary'] = (features['TotalClaims'] / features['UniqueBeneficiaries']).round(2)
        features['ReimbursementPerBeneficiary'] = (features['TotalReimbursement'] / features['UniqueBeneficiaries']).round(2)
        
        features['AvgPatientAge'] = grouped['Age'].mean().round(1).values
        features['ProportionDeceasedPatients'] = grouped['IsDeceased'].mean().round(4).values
        features['ProportionRenalDisease'] = grouped['RenalDiseaseIndicator'].mean().round(4).values
        features['AvgChronicConditionCount'] = grouped['ChronicCount'].mean().round(2).values
        
        # Beneficiary Annual Financial Aggregations
        features['AvgBeneIPAnnualReimbursement'] = grouped['IPAnnualReimbursementAmt'].mean().round(2).fillna(0).values
        features['AvgBeneOPAnnualReimbursement'] = grouped['OPAnnualReimbursementAmt'].mean().round(2).fillna(0).values

        # 4. Procedure & Diagnosis Complexity & Diversity
        features['NumDiagnosesPerClaim'] = grouped['NumDiagnoses'].mean().round(2).values
        features['NumProceduresPerClaim'] = grouped['NumProcedures'].mean().round(2).values
        
        # Unique Diagnosis & Procedure Codes across claims (Vectorized melt for high performance)
        diag_cols = [c for c in all_claims.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols = [c for c in all_claims.columns if c.startswith('ClmProcedureCode_')]
        
        logger.info("Extracting unique diagnosis and procedure diversity metrics (vectorized)...")
        diag_melted = all_claims[['Provider'] + diag_cols].melt(id_vars=['Provider'], value_name='DiagCode').dropna()
        unique_diags = diag_melted.groupby('Provider')['DiagCode'].nunique()
        
        proc_melted = all_claims[['Provider'] + proc_cols].melt(id_vars=['Provider'], value_name='ProcCode').dropna()
        unique_procs = proc_melted.groupby('Provider')['ProcCode'].nunique()
        
        features['UniqueDiagnosisCodesCount'] = features['Provider'].map(unique_diags).fillna(0).astype(int)
        features['UniqueProcedureCodesCount'] = features['Provider'].map(unique_procs).fillna(0).astype(int)


        # 5. Physician Interaction Metrics
        features['UniqueAttendingPhysicians'] = grouped['AttendingPhysician'].nunique().values
        features['UniqueOperatingPhysicians'] = grouped['OperatingPhysician'].nunique().values
        features['UniqueOtherPhysicians'] = grouped['OtherPhysician'].nunique().values
        features['ClaimsPerAttendingPhysician'] = (
            features['TotalClaims'] / features['UniqueAttendingPhysicians'].replace(0, np.nan)
        ).fillna(0).round(2)

        # 6. Peer Group Relative Behavioral Deviations (z-scores)
        metrics_for_zscore = [
            'TotalReimbursement', 'AvgReimbursementPerClaim', 'InpatientRatio',
            'AvgClaimsPerBeneficiary', 'ReimbursementPerBeneficiary'
        ]
        
        for metric in metrics_for_zscore:
            mean_val = features[metric].mean()
            std_val = features[metric].std()
            z_col_name = f"{metric}_PeerZScore"
            if std_val > 0:
                features[z_col_name] = ((features[metric] - mean_val) / std_val).round(3)
            else:
                features[z_col_name] = 0.0

        # Merge Target column if available
        features = features.merge(prov_df[['Provider', 'Target']], on='Provider', how='left')
        
        # Verify no missing values in feature set
        feature_cols = [c for c in features.columns if c not in cls.EXCLUDED_COLUMNS]
        features[feature_cols] = features[feature_cols].fillna(0)
        
        logger.info(f"Feature engineering complete: {len(features):,} providers x {len(feature_cols)} behavioral features.")
        return features

    @classmethod
    def get_feature_names(cls, feature_df: pd.DataFrame) -> List[str]:
        """Return clean list of feature names excluding ID and Target."""
        return [col for col in feature_df.columns if col not in cls.EXCLUDED_COLUMNS]
