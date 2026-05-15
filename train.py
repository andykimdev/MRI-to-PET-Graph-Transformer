"""
CLI commands

python train.py --model baseline                                        5-fold CV, seed 42
python train.py --model baseline --k 10 --seed 1                      10-fold CV, seed 1
python plot_training.py --model baseline                              loss + correlation plots with std bands across folds
python plot_training.py --model compare                                 baseline vs braak comparison with bands
python plot_training.py --model baseline --summary                      prints per-fold table + aggregate stats
python plot_training.py --model compare --summary                       prints both models' summaries + paired t-test
"""

# train.py
import sys
sys.path.append("preprocessing")

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

import paths
from dataset import BrainTauDataset
from model import BaselineGraphTransformer, BraakGraphTransformer


def train_one_epoch(model, loader, optimizer, criterion, device, use_braak):
    """Run a single training epoch and return average loss.

Args
    model - the graph transformer model to train
    loader - DataLoader providing batched training graphs
    optimizer - optimizer used to update model parameters
    criterion - loss function applied to predictions and targets
    device - torch device to move batches onto
    use_braak - whether to pass Braak stage features to the model

Returns
    average training loss over all samples in the epoch
"""
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_braak:
            preds = model(batch.x, batch.edge_index, batch.braak)
        else:
            preds = model(batch.x, batch.edge_index)

        loss = criterion(preds, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / total_samples


def evaluate(model, loader, criterion, device, use_braak):
    """Evaluate the model on a data split and return loss and correlation metrics.

Args
    model - the graph transformer model to evaluate
    loader - DataLoader providing batched graphs for evaluation
    criterion - loss function applied to predictions and targets
    device - torch device to move batches onto
    use_braak - whether to pass Braak stage features to the model

Returns
    dict with keys loss, mean_correlation, and median_correlation
"""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if use_braak:
                preds = model(batch.x, batch.edge_index, batch.braak)
            else:
                preds = model(batch.x, batch.edge_index)

            loss = criterion(preds, batch.y)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    n_regions = 66
    preds_reshaped = all_preds.reshape(-1, n_regions)
    targets_reshaped = all_targets.reshape(-1, n_regions)

    region_corrs = []
    for r in range(n_regions):
        # skip constant columns because corrcoef is undefined when std is zero
        if np.std(preds_reshaped[:, r]) > 0 and np.std(targets_reshaped[:, r]) > 0:
            corr = np.corrcoef(preds_reshaped[:, r], targets_reshaped[:, r])[0, 1]
            region_corrs.append(corr)
    # per-region correlation alongside MSE catches spatial mismatch that MSE alone can miss

    return {
        "loss": total_loss / total_samples,
        "mean_correlation": np.mean(region_corrs),
        "median_correlation": np.median(region_corrs)
    }


def _build_model(model_type, config):
    """Instantiate and return the model and a use_braak flag for the given model type.

Args
    model_type - either "baseline" or "braak" selecting the architecture
    config - dict of hyperparameters including hidden_dim, n_layers, n_heads, dropout

Returns
    tuple of (model instance, use_braak bool)
"""
    if model_type == "baseline":
        return BaselineGraphTransformer(
            in_dim=3,
            hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            dropout=config["dropout"]
        ), False
    elif model_type == "braak":
        return BraakGraphTransformer(
            in_dim=3,
            hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            dropout=config["dropout"]
        ), True
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _run_fold(model_type, config, train_ds, val_ds, device, ckpt_path):
    """Train one cross-validation fold and return the history and best entry.

Args
    model_type - either "baseline" or "braak" selecting the architecture
    config - dict of hyperparameters
    train_ds - BrainTauDataset for the fold training split
    val_ds - BrainTauDataset for the fold validation split
    device - torch device to run training on
    ckpt_path - Path where the best checkpoint will be saved

Returns
    tuple of (full history list, best epoch entry dict, trained model, use_braak bool)
"""
    model, use_braak = _build_model(model_type, config)
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_braak)
        val_metrics = evaluate(model, val_loader, criterion, device, use_braak)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mean_corr": val_metrics["mean_correlation"],
        })

        print(f"  Epoch {epoch:3d} | train {train_loss:.4f} | val {val_metrics['loss']:.4f} | corr {val_metrics['mean_correlation']:.3f}")

        # early stopping tracks val loss rather than val correlation because
        # MSE is the training objective and loss is more stable to optimise against
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0  # reset only on genuine improvement so a plateau still counts
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_corr": val_metrics["mean_correlation"],
                "config": config,
                "model_type": model_type,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    best_entry = min(history, key=lambda h: h["val_loss"])
    return history, best_entry, model, use_braak


def train_kfold(model_type, config, k=5, seed=42):
    """Train a model with k-fold cross-validation and run test evaluation.

Args
    model_type - either "baseline" or "braak" selecting the architecture
    config - dict of hyperparameters
    k - number of cross-validation folds
    seed - integer random seed for fold splitting and reproducibility

Returns
    None
"""
    processed = paths.ROOT / "data" / "processed"
    log_dir = paths.ROOT / "outputs" / "logs"
    ckpt_dir = paths.ROOT / "outputs" / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    X_raw = np.load(processed / "node_features_raw.npy")
    Y = np.load(processed / "suvr_targets.npy")
    subject_ids = np.load(processed / "subject_ids.npy")
    diagnosis = np.load(processed / "diagnosis.npy", allow_pickle=True)

    with open(processed / "splits.json") as f:
        splits = json.load(f)

    trainval_rids = set(splits["train"]) | set(splits["val"])
    test_rids = set(splits["test"])

    trainval_idx = np.array([i for i, rid in enumerate(subject_ids) if rid in trainval_rids])
    test_idx = np.array([i for i, rid in enumerate(subject_ids) if rid in test_rids])

    trainval_labels = diagnosis[trainval_idx]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_summaries = []
    fold_histories = []

    # diagnosis labels drive stratification so each fold mirrors the overall CN/MCI/AD ratio
    for fold, (rel_train_idx, rel_val_idx) in enumerate(skf.split(trainval_idx, trainval_labels), start=1):
        print(f"\n{'='*50}")
        print(f"{model_type.upper()} | Fold {fold}/{k}")
        print(f"{'='*50}")

        abs_train_idx = trainval_idx[rel_train_idx]
        abs_val_idx = trainval_idx[rel_val_idx]

        # compute normalization from this fold's train split only so val/test see no future information
        train_mean = np.nanmean(X_raw[abs_train_idx], axis=0)  # (n_regions, 3)
        train_std = np.nanstd(X_raw[abs_train_idx], axis=0)
        norm_params = {"mean": train_mean, "std": train_std}

        Y_fold = Y.copy()
        # impute missing SUVR with train-fold mean only so val/test targets stay uncontaminated
        train_suvr_mean = np.nanmean(Y_fold[abs_train_idx], axis=0)
        for r in range(Y_fold.shape[1]):
            nan_mask = np.isnan(Y_fold[:, r])
            Y_fold[nan_mask, r] = train_suvr_mean[r]

        train_ds = BrainTauDataset(processed, split="train", norm_params=norm_params, indices=abs_train_idx)
        val_ds = BrainTauDataset(processed, split="val", norm_params=norm_params, indices=abs_val_idx)

        train_ds.Y = Y_fold
        val_ds.Y = Y_fold

        ckpt_path = ckpt_dir / f"{model_type}_fold{fold}.pt"
        history, best_entry, _, _ = _run_fold(
            model_type, config, train_ds, val_ds, device, ckpt_path
        )

        fold_summaries.append({
            "fold": fold,
            "best_epoch": best_entry["epoch"],
            "best_val_loss": best_entry["val_loss"],
            "best_val_corr": best_entry["val_mean_corr"],
            "train_size": len(abs_train_idx),
            "val_size": len(abs_val_idx),
        })
        fold_histories.append(history)

        with open(log_dir / f"{model_type}_fold{fold}_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"Fold {fold} best -> val_loss: {best_entry['val_loss']:.4f}, val_corr: {best_entry['val_mean_corr']:.3f}")

    _run_test_eval(
        model_type, config, k, seed,
        processed, ckpt_dir, log_dir, device,
        X_raw, trainval_idx, trainval_labels, test_idx,
        fold_summaries
    )


def _run_test_eval(model_type, config, k, seed,
                   processed, ckpt_dir, log_dir, device,
                   X_raw, trainval_idx, trainval_labels, test_idx,
                   fold_summaries):
    """Evaluate each fold checkpoint on the held-out test set and save aggregate results.

Args
    model_type - either "baseline" or "braak" selecting the architecture
    config - dict of hyperparameters
    k - number of cross-validation folds
    seed - integer random seed used when splitting folds
    processed - Path to the directory containing processed data files
    ckpt_dir - Path to the directory containing saved fold checkpoints
    log_dir - Path to the directory where logs and results are written
    device - torch device to run evaluation on
    X_raw - raw node feature array of shape (n_subjects, n_regions, n_features)
    trainval_idx - array of absolute indices belonging to the train+val pool
    trainval_labels - diagnosis labels aligned to trainval_idx for stratification
    test_idx - array of absolute indices belonging to the test set
    fold_summaries - list of per-fold summary dicts collected during training

Returns
    None
"""
    print(f"\n{'='*50}")
    print(f"Test evaluation across {k} fold checkpoints")
    print(f"{'='*50}")

    test_losses, test_corrs = [], []
    # precompute all fold splits before the loop so each fold's train indices
    # match exactly what was used when the checkpoint was saved during training
    skf_splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                      .split(trainval_idx, trainval_labels))

    for fold in range(1, k + 1):
        ckpt = torch.load(ckpt_dir / f"{model_type}_fold{fold}.pt",
                          map_location=device, weights_only=False)
        model, use_braak = _build_model(model_type, config)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)

        rel_train_idx = skf_splits[fold - 1][0]
        abs_train_idx = trainval_idx[rel_train_idx]
        # recompute this fold's train-only statistics so test normalisation is consistent with training
        train_mean = np.nanmean(X_raw[abs_train_idx], axis=0)
        train_std = np.nanstd(X_raw[abs_train_idx], axis=0)
        norm_params = {"mean": train_mean, "std": train_std}

        test_ds = BrainTauDataset(processed, split="test", norm_params=norm_params, indices=test_idx)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
        criterion = nn.MSELoss()

        test_metrics = evaluate(model, test_loader, criterion, device, use_braak)
        test_losses.append(test_metrics["loss"])
        test_corrs.append(test_metrics["mean_correlation"])
        print(f"  Fold {fold} test -> loss: {test_metrics['loss']:.4f}, corr: {test_metrics['mean_correlation']:.3f}")

    val_losses = [s["best_val_loss"] for s in fold_summaries]
    val_corrs = [s["best_val_corr"] for s in fold_summaries]

    for i, s in enumerate(fold_summaries):
        s["test_loss"] = test_losses[i]
        s["test_corr"] = test_corrs[i]

    aggregate = {
        "model_type": model_type,
        "k": k,
        "seed": seed,
        "folds": fold_summaries,
        "mean_val_loss": float(np.mean(val_losses)),
        "std_val_loss": float(np.std(val_losses)),
        "mean_val_corr": float(np.mean(val_corrs)),
        "std_val_corr":  float(np.std(val_corrs)),
        "mean_test_loss": float(np.mean(test_losses)),
        "std_test_loss": float(np.std(test_losses)),
        "mean_test_corr": float(np.mean(test_corrs)),
        "std_test_corr": float(np.std(test_corrs)),
    }

    with open(log_dir / f"{model_type}_aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{model_type.upper()} {k}-fold summary:")
    print(f"  val_loss  : {aggregate['mean_val_loss']:.4f} +/- {aggregate['std_val_loss']:.4f}")
    print(f"  val_corr  : {aggregate['mean_val_corr']:.3f} +/- {aggregate['std_val_corr']:.3f}")
    print(f"  test_loss : {aggregate['mean_test_loss']:.4f} +/- {aggregate['std_test_loss']:.4f}")
    print(f"  test_corr : {aggregate['mean_test_corr']:.3f} +/- {aggregate['std_test_corr']:.3f}")
    print(f"Saved: {log_dir / f'{model_type}_aggregate.json'}")


def test_only(model_type, config, k=5, seed=42):
    """Re-run test evaluation using existing fold checkpoints and history files.

Args
    model_type - either "baseline" or "braak" selecting the architecture
    config - dict of hyperparameters
    k - number of cross-validation folds matching the saved checkpoints
    seed - integer random seed used when the checkpoints were trained

Returns
    None
"""
    processed = paths.ROOT / "data" / "processed"
    log_dir = paths.ROOT / "outputs" / "logs"
    ckpt_dir = paths.ROOT / "outputs" / "checkpoints"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    X_raw = np.load(processed / "node_features_raw.npy")
    subject_ids = np.load(processed / "subject_ids.npy")
    diagnosis = np.load(processed / "diagnosis.npy", allow_pickle=True)

    with open(processed / "splits.json") as f:
        splits = json.load(f)

    trainval_rids = set(splits["train"]) | set(splits["val"])
    test_rids = set(splits["test"])
    trainval_idx = np.array([i for i, rid in enumerate(subject_ids) if rid in trainval_rids])
    test_idx = np.array([i for i, rid in enumerate(subject_ids) if rid in test_rids])
    trainval_labels = diagnosis[trainval_idx]

    fold_summaries = []
    for fold in range(1, k + 1):
        p = log_dir / f"{model_type}_fold{fold}_history.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} -- cannot reconstruct fold summaries.")
        with open(p) as f:
            history = json.load(f)
        best_entry = min(history, key=lambda h: h["val_loss"])
        fold_summaries.append({
            "fold": fold,
            "best_epoch": best_entry["epoch"],
            "best_val_loss": best_entry["val_loss"],
            "best_val_corr": best_entry["val_mean_corr"],
            "train_size": 0,  # not stored in history; harmless placeholder
            "val_size": 0,
        })

    _run_test_eval(
        model_type, config, k, seed,
        processed, ckpt_dir, log_dir, device,
        X_raw, trainval_idx, trainval_labels, test_idx,
        fold_summaries
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "braak"], required=True)
    parser.add_argument("--k", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold splits (default: 42)")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip training; run test eval on existing fold checkpoints")
    args = parser.parse_args()

    config = {
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "batch_size": 16,
        "hidden_dim": 16,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.3,
        "max_epochs": 1000,
        "patience": 20
    }

    if args.test_only:
        test_only(args.model, config, k=args.k, seed=args.seed)
    else:
        train_kfold(args.model, config, k=args.k, seed=args.seed)
