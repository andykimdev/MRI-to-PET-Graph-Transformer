import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
from sklearn.model_selection import train_test_split

from braak_lookup import ST_TO_SUVR_REGION, SUVR_REGION_TO_BRAAK
import paths

# Create directory for data
data_fs = paths.ROOT / "data" / "raw_data" / "fs.csv"
data_suvr = paths.ROOT / "data" / "raw_data" / "suvr.csv"

def parse_csv():
    """Load raw FS, SUVR, and diagnosis data from disk.

    Returns
        Tuple of (fs, suvr, dxsum) as pandas DataFrames
    """
    fs = pd.read_csv(data_fs, low_memory=False)

    suvr = pd.read_csv(data_suvr, low_memory=False)

    adnimerge2 = pyreadr.read_r(paths.ROOT / "data" / "raw_data" / "ADNIMERGE2" / "data" / "DXSUM.rda")
    dxsum = adnimerge2["DXSUM"]

    return fs, suvr, dxsum


def filter(fs, suvr, dxsum):
    """Apply quality control filters to FS and SUVR data and normalise RID types.

    Args
        fs - raw FreeSurfer DataFrame
        suvr - raw SUVR DataFrame
        dxsum - diagnosis summary DataFrame

    Returns
        Tuple of (fs_filtered, suvr_filtered, dxsum) after applying QC filters
    """
    fs_filtered = fs[
        fs["OVERALLQC"].isin(["Pass", "Partial", "Hippocampus Only"]) |
        fs["OVERALLQC"].isna()
    ].copy()
    fs_filtered = fs_filtered[fs_filtered["FIELD_STRENGTH"] == "3T"].copy()

    suvr_filtered = suvr[
        (suvr["qc_flag"] >= 1) &
        (suvr["TRACER"] == "FTP")
    ].copy()

    # Fix RID (Roster ID - primary patient key in ADNI study) types (since pandas only merges based on same datatype)
    fs_filtered["RID"] = fs_filtered["RID"].astype(int)
    suvr_filtered["RID"] = suvr_filtered["RID"].astype(int)
    dxsum["RID"] = dxsum["RID"].astype(float).astype(int)

    print(f"FS after filter: {fs_filtered['RID'].nunique()} participants")
    print(f"SUVR after filter: {suvr_filtered['RID'].nunique()} participants")

    return fs_filtered, suvr_filtered, dxsum

def merge_data(fs_filtered, suvr_filtered, dxsum):
    """Merge FS, SUVR, and diagnosis tables and apply date and NaN filters.

    Args
        fs_filtered - quality-controlled FreeSurfer DataFrame
        suvr_filtered - quality-controlled SUVR DataFrame
        dxsum - diagnosis summary DataFrame

    Returns
        Merged and deduplicated DataFrame with one row per subject
    """
    paired = pd.merge(fs_filtered, suvr_filtered, on=["RID", "VISCODE"], how="inner")
    print(f"After FS + SUVR Merge: {paired['RID'].nunique()} participants")
    print(f"After FS + SUVR Merge: {len(paired)} total timestamps")

    paired = pd.merge(paired, dxsum[["RID", "VISCODE", "DIAGNOSIS"]], on=["RID", "VISCODE"], how="left")
    print(f"Missing diagnosis: {paired['DIAGNOSIS'].isna().sum()}")
    print(f"Diagnosis distribution: \n{paired['DIAGNOSIS'].value_counts()}")

    paired["EXAMDATE"] = pd.to_datetime(paired["EXAMDATE"])
    paired["SCANDATE"] = pd.to_datetime(paired["SCANDATE"])
    paired["date_diff"] = (paired["SCANDATE"] - paired["EXAMDATE"]).dt.days

    # negative diff means PET was acquired before MRI, so MRI cannot reflect tau captured by PET
    paired = paired[paired["date_diff"] >= 0].copy()
    print(f"After removing negative date gaps: {paired['RID'].nunique()} participants, {len(paired)} rows")

    # keep earliest scan per subject so repeated visits do not inflate the effective sample size
    paired = paired.sort_values("SCANDATE").groupby("RID").first().reset_index()
    print(f"After deduplication: {paired['RID'].nunique()} participants")

    st_cols = [c for c in paired.columns if c.startswith("ST")]
    feat_cols = (
        [c for c in st_cols if c.endswith("CV")] +
        [c for c in st_cols if c.endswith("SA")] +
        [c for c in st_cols if c.endswith("TA")]
    )
    fs_nan_frac = paired[feat_cols].isna().mean(axis=1)
    paired = paired[fs_nan_frac <= 0.20].copy()
    print(f"After FS NaN filter: {paired['RID'].nunique()} participants")

    # exclude META composites and the cerebellum reference region used for SUVR normalisation
    suvr_target_cols = [c for c in paired.columns if c.endswith("_SUVR")
                        and "META" not in c
                        and "INFERIORCEREBELLUM" not in c]
    suvr_nan_frac = paired[suvr_target_cols].isna().mean(axis=1)
    paired = paired[suvr_nan_frac <= 0.20].copy()
    print(f"After SUVR NaN filter: {paired['RID'].nunique()} participants")

    return paired

def extract_features(paired):
    """Build node feature matrix X and SUVR target matrix Y from the paired table.

    Args
        paired - merged and filtered subject DataFrame

    Returns
        Tuple of (X, Y, region_names, braak_stages) where X is shape
        (n_subjects, n_regions, 3), Y is shape (n_subjects, n_regions),
        region_names is a list of region name strings, and braak_stages is
        an integer array of Braak stage per region
    """
    valid_st = []
    for st in ST_TO_SUVR_REGION.keys():
        cv = f"{st}CV"
        sa = f"{st}SA"
        ta = f"{st}TA"

        if cv in paired.columns and sa in paired.columns and ta in paired.columns:
            valid_st.append(st)

    valid_st = sorted(valid_st, key=lambda x: int(x.replace("ST", "")))

    region_names = [ST_TO_SUVR_REGION[st] for st in valid_st]

    braak_stages = np.array([SUVR_REGION_TO_BRAAK[stage] for stage in region_names])

    print(f"Valid regions: {len(valid_st)}")
    print(f"Braak stage distribution: {np.bincount(braak_stages)}")

    n_subjects = len(paired)
    n_regions = len(valid_st)
    X = np.zeros((n_subjects, n_regions, 3))

    for i, st in enumerate(valid_st):
        X[:, i, 0] = paired[f"{st}CV"].values
        X[:, i, 1] = paired[f"{st}SA"].values
        X[:, i, 2] = paired[f"{st}TA"].values

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
    """Split subjects into train/val/test sets, normalise features, and save outputs.

    Args
        paired - merged subject DataFrame with DIAGNOSIS column
        X - raw node feature array of shape (n_subjects, n_regions, 3)
        Y - SUVR target array of shape (n_subjects, n_regions)
        region_names - list of region name strings
        braak_stages - integer array of Braak stage per region
        seed - random seed used for reproducible splitting

    Returns
        None
    """
    valid_mask = paired["DIAGNOSIS"].notna().values
    nan_mask = paired["DIAGNOSIS"].isna().sum()
    paired = paired[valid_mask].reset_index(drop=True)

    X = X[valid_mask].copy()
    Y = Y[valid_mask].copy()
    print(f"Dropped {nan_mask} subjects with missing diagnosis")
    print(f"Remaining: {len(paired)} subjects")

    rids = paired["RID"].values
    labels = paired["DIAGNOSIS"].values

    # two-step split gives 70/15/15 while preserving diagnosis proportions in every partition
    train_idx, temp_idx = train_test_split(
        np.arange(len(rids)),
        test_size=0.30,
        # stratify so CN/MCI/AD ratios are balanced across train, val, and test
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

    # fit normalisation on train only so val/test subjects contribute no information to the statistics
    train_mean = np.nanmean(X[train_idx], axis=0)  # (n_regions, 3)
    train_std = np.nanstd(X[train_idx], axis=0)
    train_std[train_std == 0] = 1  # safeguard to avoid divide by zero (will not affect values later)

    X_norm = (X - train_mean) / train_std

    # replace remaining NaNs with zero because zero is the mean after standardisation
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    # impute SUVR targets with train mean rather than normalising because SUVR is the prediction target
    train_suvr_mean = np.nanmean(Y[train_idx], axis=0)  # (n_regions,)
    for r in range(Y.shape[1]):
        nan_mask = np.isnan(Y[:, r])
        Y[nan_mask, r] = train_suvr_mean[r]

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

    PROCESSED = paths.ROOT / "data" / "processed"
    PROCESSED.mkdir(exist_ok=True)

    # save raw features separately so k-fold CV can recompute fold-specific normalisation
    np.save(PROCESSED / "node_features_raw.npy", X)
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

    PROCESSED = paths.ROOT / "data" / "processed"
    X = np.load(PROCESSED / "node_features.npy")
    Y = np.load(PROCESSED / "suvr_targets.npy")
    with open(PROCESSED / "splits.json") as f:
        splits = json.load(f)
    print(f"X: {X.shape}, NaN: {np.isnan(X).sum()}")
    print(f"Y: {Y.shape}, NaN: {np.isnan(Y).sum()}, range: {Y.min():.2f} to {Y.max():.2f}")
    print(f"Splits: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
