"""
Phase 0 Dataset Profiling Script.
Inspects all uploaded Kaggle Healthcare Provider Fraud CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path("data")

files = {
    "train_provider": DATA_DIR / "Train-1542865627584.csv",
    "train_inpatient": DATA_DIR / "Train_Inpatientdata-1542865627584.csv",
    "train_outpatient": DATA_DIR / "Train_Outpatientdata-1542865627584.csv",
    "train_beneficiary": DATA_DIR / "Train_Beneficiarydata-1542865627584.csv",
    "test_provider": DATA_DIR / "Test-1542969243754.csv",
    "test_inpatient": DATA_DIR / "Test_Inpatientdata-1542969243754.csv",
    "test_outpatient": DATA_DIR / "Test_Outpatientdata-1542969243754.csv",
    "test_beneficiary": DATA_DIR / "Test_Beneficiarydata-1542969243754.csv",
}

print("==================================================")
print("PHASE 0 — DATASET PROFILING & DISCOVERY")
print("==================================================")

profiles = {}

for key, path in files.items():
    if not path.exists():
        print(f"File missing: {path}")
        continue
    
    df = pd.read_csv(path, nrows=5000)  # quick sample for schema inspection
    full_len = len(pd.read_csv(path, usecols=[df.columns[0]]))
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2).to_dict()
    
    profiles[key] = {
        "file_name": path.name,
        "total_rows": full_len,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "sample_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_pct": {col: pct for col, pct in missing_pct.items() if pct > 0}
    }
    
    print(f"\n--- {key.upper()} ({path.name}) ---")
    print(f"Total Rows: {full_len:,} | Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

# Target Analysis
train_prov_df = pd.read_csv(files["train_provider"])
print("\n==================================================")
print("TARGET DISTRIBUTION (Train Provider Level)")
print("==================================================")
print(train_prov_df["PotentialFraud"].value_counts(dropna=False))
print(train_prov_df["PotentialFraud"].value_counts(normalize=True).round(4) * 100)

# Check Provider overlaps
inpatient_df = pd.read_csv(files["train_inpatient"], usecols=["Provider", "ClaimID", "BeneID"])
outpatient_df = pd.read_csv(files["train_outpatient"], usecols=["Provider", "ClaimID", "BeneID"])
bene_df = pd.read_csv(files["train_beneficiary"], usecols=["BeneID"])

inpatient_provs = set(inpatient_df["Provider"])
outpatient_provs = set(outpatient_df["Provider"])
target_provs = set(train_prov_df["Provider"])
all_claim_provs = inpatient_provs.union(outpatient_provs)

print("\n==================================================")
print("PROVIDER & CLAIMS RELATIONSHIPS")
print("==================================================")
print(f"Total Unique Target Providers: {len(target_provs):,}")
print(f"Providers in Inpatient Claims: {len(inpatient_provs):,}")
print(f"Providers in Outpatient Claims: {len(outpatient_provs):,}")
print(f"Providers in Claims Union: {len(all_claim_provs):,}")
print(f"Target Providers matching Claims: {len(target_provs.intersection(all_claim_provs)):,}")
print(f"Total Inpatient Claims: {len(inpatient_df):,}")
print(f"Total Outpatient Claims: {len(outpatient_df):,}")
print(f"Total Beneficiaries: {len(bene_df):,}")

# Save detailed profile JSON
with open("data/dataset_profile.json", "w") as f:
    json.dump(profiles, f, indent=2)

print("\nSaved profile to data/dataset_profile.json")
