import pandas as pd
import numpy as np
import pyreadr
import directories as dir
import matplotlib.pyplot as plt
from braak_lookup import ST_TO_SUVR_REGION, SUVR_REGION_TO_BRAAK
import json
from sklearn.model_selection import train_test_split

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

def extract_features(paired):
    # Find ST codes that exist in BOTH the paired table AND the lookup
    valid_st = []
    for st in ST_TO_SUVR_REGION.keys():
        cv = f"{st}CV"
        sa = f"{st}SA"
        ta = f"{st}TA"

        if cv in paired.columns and sa in paired.columns and ta in paired.columns:
            valid_st.append(st)
    
    # Sort valid st
    valid_st = sorted(valid_st, key=lambda x: int(x.replace("ST", "")))

    # Select the valid region names based on valid st
    region_names = [ST_TO_SUVR_REGION[st] for st in valid_st]

    # Select the valid braak stages based on valid region
    braak_stages = np.array([SUVR_REGION_TO_BRAAK[stage] for stage in region_names])

    print(f"Valid regions: {len(valid_st)}") 
    print(f"Braak stage distribution: {np.bincount(braak_stages)}")  

    # Node matrix initialization 
    n_subjects = len(paired)
    n_regions = len(valid_st)
    X = np.zeros((n_subjects, n_regions, 3))

    # Populate the node matrix 
    for i, st in enumerate(valid_st): 
        X[:, i, 0] = paired[f"{st}CV"].values
        X[:, i, 1] = paired[f"{st}SA"].values
        X[:, i, 2] = paired[f"{st}TA"].values

    # SUVR target matrix 
    Y = np.zeros((n_subjects, n_regions))
    for i, region in enumerate(region_names):
        col = f"{region}_SUVR"
        if col in paired.columns:
            Y[:, i] = paired[col].values
        else:
            print(f"WARNING: {col} missing in paired table")
            Y[:, i] = np.nan
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X NaN count: {np.isnan(X).sum()}")
    print(f"Y NaN count: {np.isnan(Y).sum()}")
    
    return X, Y, region_names, braak_stages

def split_and_save(paired, X, Y, region_names, braak_stages, seed=1):
    # Drop subjects with missing diagnoses
    valid_mask = paired["DIAGNOSIS"].notna().values
    nan_mask = paired["DIAGNOSIS"].isna().sum()
    paired = paired[valid_mask].reset_index(drop=True)
    
    # Filter X and Y
    X = X[valid_mask].copy()
    Y = Y[valid_mask].copy()
    print(f"Dropped {nan_mask} subjects with missing diagnosis")
    print(f"Remaining: {len(paired)} subjects")

    # Stratify by diagnosis - fill NaN with "Unknown" so they get split too
    rids = paired["RID"].values
    labels = paired["DIAGNOSIS"].values
    
    # 70/15/15 split via two-step stratified split
    train_idx, temp_idx = train_test_split(
        np.arange(len(rids)),
        test_size=0.30,
        stratify=labels,
        random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=labels[temp_idx],
        random_state=seed
    )
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Z-score normalize node features using train statistics only
    train_mean = np.nanmean(X[train_idx], axis=0)  # (n_regions, 3)
    train_std = np.nanstd(X[train_idx], axis=0)
    train_std[train_std == 0] = 1  # safeguard breaking code by avoid divide by zero (later will not have impact) 
    
    X_norm = (X - train_mean) / train_std
    
    # Impute remaining NaN with 0 (which is train mean after normalization)
    X_norm = np.nan_to_num(X_norm, nan=0.0)
    
    # For SUVR targets, impute NaN with train mean per region (no normalization)
    train_suvr_mean = np.nanmean(Y[train_idx], axis=0)  # (n_regions,)
    for r in range(Y.shape[1]):
        nan_mask = np.isnan(Y[:, r])
        Y[nan_mask, r] = train_suvr_mean[r]
    
    # Build splits dict with RID lists
    splits = {
        "train": rids[train_idx].tolist(),
        "val": rids[val_idx].tolist(),
        "test": rids[test_idx].tolist()
    }
    
    norm_params = {
        "feature_mean": train_mean.tolist(),
        "feature_std": train_std.tolist(),
        "suvr_mean": train_suvr_mean.tolist()
    }
    
    # Save everything to data/processed/
    PROCESSED = dir.ROOT / "data" / "processed"
    PROCESSED.mkdir(exist_ok=True)
    
    np.save(PROCESSED / "node_features.npy", X_norm)
    np.save(PROCESSED / "suvr_targets.npy", Y)
    np.save(PROCESSED / "braak_stages.npy", braak_stages)
    np.save(PROCESSED / "region_names.npy", np.array(region_names))
    np.save(PROCESSED / "subject_ids.npy", rids)
    np.save(PROCESSED / "diagnosis.npy", paired["DIAGNOSIS"].values)
    
    with open(PROCESSED / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    with open(PROCESSED / "norm_params.json", "w") as f:
        json.dump(norm_params, f, indent=2)
    
    print(f"\nSaved to {PROCESSED}")
    print(f"node_features: {X_norm.shape}")
    print(f"suvr_targets: {Y.shape}")
    print(f"braak_stages: {braak_stages.shape}")
    print(f"subjects: {len(rids)}")

if __name__ == "__main__":
    print("Loading data...")
    fs, suvr, dxsum = parse_csv()
    
    print("\nFiltering...")
    fs_filtered, suvr_filtered, dxsum = filter(fs, suvr, dxsum)
    
    print("\nMerging...")
    paired = merge_data(fs_filtered, suvr_filtered, dxsum)
    
    print("\nExtracting features...")
    X, Y, region_names, braak_stages = extract_features(paired)
    
    print("\nSplitting and saving...")
    split_and_save(paired, X, Y, region_names, braak_stages)
    
    print("\nDone.")

# Final check 

X = np.load("/Users/andykim/Documents/2. 2026 Spring/DLBI/Project/data/processed/node_features.npy")
Y = np.load("/Users/andykim/Documents/2. 2026 Spring/DLBI/Project/data/processed/suvr_targets.npy")

with open("/Users/andykim/Documents/2. 2026 Spring/DLBI/Project/data/processed/splits.json") as f:
    splits = json.load(f)

print(f"X: {X.shape}, NaN: {np.isnan(X).sum()}")
print(f"Y: {Y.shape}, NaN: {np.isnan(Y).sum()}, range: {Y.min():.2f} to {Y.max():.2f}")
print(f"Splits: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")