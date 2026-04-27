import pandas as pd
import numpy as np
import pyreadr
import directories as dir
import matplotlib.pyplot as plt

# Create directory for data
data_fs = dir.ROOT / "data" / "raw_data" / "fs.csv"
data_suvr = dir.ROOT / "data" / "raw_data" / "suvr.csv"

def parse_csv():
    # Parase FS
    fs = pd.read_csv(data_fs, low_memory=False)

    # Parse SUVR
    suvr = pd.read_csv(data_suvr, low_memory=False)

    # Parse adnimerge2
    adnimerge2 = pyreadr.read_r(dir.ROOT / "data" / "raw_data" / "ADNIMERGE2" / "data" / "DXSUM.rda")
    dxsum = adnimerge2["DXSUM"]
    
    return fs, suvr, dxsum


def filter(fs, suvr, dxsum):
    # FS filter
    fs_filtered = fs[
        fs["OVERALLQC"].isin(["Pass", "Partial", "Hippocampus Only"]) |
        fs["OVERALLQC"].isna()
    ].copy()
    fs_filtered = fs_filtered[fs_filtered["FIELD_STRENGTH"] == "3T"].copy()

    # SUVR filter
    suvr_filtered = suvr[
        (suvr["qc_flag"] >= 1) &
        (suvr["TRACER"] == "FTP")
    ].copy()

    # Fix RID (Roster ID: primary patient key in ADNI study) types (since pandas only merges based on same datatype)
    fs_filtered["RID"] = fs_filtered["RID"].astype(int)
    suvr_filtered["RID"] = suvr_filtered["RID"].astype(int)
    dxsum["RID"] = dxsum["RID"].astype(float).astype(int)

    # Verify 
    print(f"FS after filer: {fs_filtered['RID'].nunique()} participants")
    print(f"SUVR after filer: {suvr_filtered['RID'].nunique()} participants")

    return fs_filtered, suvr_filtered, dxsum

def merge_data(fs_filtered, suvr_filtered, dxsum):
    # Inner merge on FS + SUVR on RID + VISCODE (visit code)
    paired = pd.merge(fs_filtered, suvr_filtered, on=["RID", "VISCODE"], how="inner")
    print(f"After FS + SUVR Merge: {paired['RID'].nunique()} participants")
    print(f"After FS + SUVR Merge: {len(paired)} total timestamps")

    # Left merge on paired + diagnosis
    paired = pd.merge(paired, dxsum[["RID", "VISCODE", "DIAGNOSIS"]], on=["RID", "VISCODE"], how="left")
    print(f"Missing diagnosis: {paired['DIAGNOSIS'].isna().sum()}")
    print(f"Diagnosis distribution: \n{paired['DIAGNOSIS'].value_counts()}")

    # Parse dates and calculate date difference
    paired["EXAMDATE"] = pd.to_datetime(paired["EXAMDATE"])
    paired["SCANDATE"] = pd.to_datetime(paired["SCANDATE"])
    paired["date_diff"] = (paired["SCANDATE"] - paired["EXAMDATE"]).dt.days

    # Drop rows where PET predates MRI (data leakage)
    paired = paired[paired["date_diff"] >= 0].copy()
    print(f"After removing negative date gaps: {paired['RID'].nunique()} participants, {len(paired)} rows")

    # Deduplicate
    paired = paired.sort_values("SCANDATE").groupby("RID").first().reset_index()
    print(f"After deduplication: {paired['RID'].nunique()} participants")

    # NaN filters on paired table
    st_cols = [c for c in paired.columns if c.startswith("ST")]
    feat_cols = (
        [c for c in st_cols if c.endswith("CV")] +
        [c for c in st_cols if c.endswith("SA")] +
        [c for c in st_cols if c.endswith("TA")]
    )
    fs_nan_frac = paired[feat_cols].isna().mean(axis=1)
    paired = paired[fs_nan_frac <= 0.20].copy()
    print(f"After FS NaN filter: {paired['RID'].nunique()} participants")

    suvr_target_cols = [c for c in paired.columns if c.endswith("_SUVR")
                        and "META" not in c
                        and "INFERIORCEREBELLUM" not in c]
    suvr_nan_frac = paired[suvr_target_cols].isna().mean(axis=1)
    paired = paired[suvr_nan_frac <= 0.20].copy()
    print(f"After SUVR NaN filter: {paired['RID'].nunique()} participants")

    return paired



