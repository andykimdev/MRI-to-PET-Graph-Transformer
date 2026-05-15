"""
Visualize per-region correlation difference: BraakGT - Baseline.

Usage:
    python visualize_regions.py
    python visualize_regions.py --k 5 --seed 42
"""
import sys
sys.path.append("preprocessing")

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

import paths
from dataset import BrainTauDataset
from model import BaselineGraphTransformer, BraakGraphTransformer


def _build_model(model_type, config):
    """Instantiate and return a model along with a flag indicating Braak usage.

    Args
        model_type - string, either "baseline" or "braak"
        config - dict containing hyperparameters: hidden_dim, n_layers, n_heads, dropout

    Returns
        tuple of (model, use_braak) where model is the instantiated nn.Module
        and use_braak is a bool indicating whether the model expects braak stage input
    """
    if model_type == "baseline":
        return BaselineGraphTransformer(
            in_dim=3, hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"], n_heads=config["n_heads"],
            dropout=config["dropout"]
        ), False
    else:
        return BraakGraphTransformer(
            in_dim=3, hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"], n_heads=config["n_heads"],
            dropout=config["dropout"]
        ), True


def get_per_region_corrs(model_type, config, k, seed):
    """Return a per-region Pearson r array on the test set averaged across fold checkpoints.

    Args
        model_type - string, either "baseline" or "braak"
        config - dict containing hyperparameters: hidden_dim, n_layers, n_heads, dropout
        k - number of cross-validation folds
        seed - random seed used for the stratified k-fold split

    Returns
        numpy array of shape (n_regions,) containing mean Pearson r per region
    """
    processed = paths.ROOT / "data" / "processed"
    ckpt_dir = paths.ROOT / "outputs" / "checkpoints"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    skf_splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                      .split(trainval_idx, trainval_labels))

    n_regions = 66
    fold_corrs = []

    for fold in range(1, k + 1):
        ckpt = torch.load(ckpt_dir / f"{model_type}_fold{fold}.pt",
                          map_location=device, weights_only=False)
        model, use_braak = _build_model(model_type, config)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        model.eval()

        rel_train_idx = skf_splits[fold - 1][0]
        abs_train_idx = trainval_idx[rel_train_idx]
        train_mean = np.nanmean(X_raw[abs_train_idx], axis=0)
        train_std = np.nanstd(X_raw[abs_train_idx], axis=0)
        # recomputed per fold so test data is normalized with the same stats the fold's model was trained on
        norm_params = {"mean": train_mean, "std": train_std}

        test_ds = BrainTauDataset(processed, split="test", norm_params=norm_params, indices=test_idx)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.braak) if use_braak \
                        else model(batch.x, batch.edge_index)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())

        preds_mat = np.concatenate(all_preds).reshape(-1, n_regions)
        targets_mat = np.concatenate(all_targets).reshape(-1, n_regions)

        region_corrs = []
        for r in range(n_regions):
            # corrcoef returns NaN for a constant vector, so treat zero-std regions as 0 to keep the mean valid
            if np.std(preds_mat[:, r]) > 0 and np.std(targets_mat[:, r]) > 0:
                corr = np.corrcoef(preds_mat[:, r], targets_mat[:, r])[0, 1]
            else:
                corr = 0.0
            region_corrs.append(corr)

        fold_corrs.append(region_corrs)

    # averaged across folds rather than pooled so each fold contributes equally regardless of test-set size variance
    return np.mean(fold_corrs, axis=0)  # (n_regions,)


def plot_difference_map(baseline_corrs, braak_corrs, region_names, save_path):
    """Plot a bar chart of top 5 and bottom 5 per-region correlation differences (BraakGT minus Baseline).

    Args
        baseline_corrs - numpy array of per-region Pearson r for the baseline model
        braak_corrs - numpy array of per-region Pearson r for the BraakGT model
        region_names - list of region name strings corresponding to array indices
        save_path - file path where the figure will be saved

    Returns
        None
    """
    from matplotlib.patches import Patch
    from preprocessing.braak_lookup import SUVR_REGION_TO_BRAAK

    stage_colors = {0: "#2196F3", 1: "#FF9800", 2: "#9C27B0"}  # blue / orange / purple
    stage_names = {0: "Early (I/II)", 1: "Middle (III/IV)", 2: "Late (V/VI)"}

    diff = braak_corrs - baseline_corrs
    sort_idx = np.argsort(diff)

    # bottom 5 (worst) then top 5 (best), separated by a gap
    selected_idx = list(sort_idx[:5]) + list(sort_idx[-5:])

    labels_all = [r.replace("CTX_LH_", "L-").replace("CTX_RH_", "R-").lower() for r in region_names]
    labels = [labels_all[i] for i in selected_idx]
    diff_selected = diff[selected_idx]
    selected_regions = [region_names[i] for i in selected_idx]
    bar_colors = [stage_colors[SUVR_REGION_TO_BRAAK.get(r, 2)] for r in selected_regions]

    # insert a visual gap between the bottom 5 and top 5
    x_pos = list(range(5)) + [r + 1.5 for r in range(5, 10)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos, diff_selected, color=bar_colors, edgecolor="none", width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=15, rotation=45, ha="right")
    ax.set_ylabel("Δ Pearson r  (BraakGT − Baseline)", fontsize=11)
    ax.set_title("Top 5 and Bottom 5 Region Correlation Changes: BraakGT vs Baseline\n(test set, averaged across folds)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    for tick, color in zip(ax.get_xticklabels(), bar_colors):
        tick.set_color(color)

    # label the two groups
    ax.text(2, ax.get_ylim()[0] * 0.85, "Bottom 5", ha="center", fontsize=9, color="gray")
    ax.text(7, ax.get_ylim()[0] * 0.85, "Top 5", ha="center", fontsize=9, color="gray")

    legend_handles = [Patch(facecolor=stage_colors[s], label=stage_names[s]) for s in sorted(stage_names)]
    ax.legend(handles=legend_handles, title="Braak Stage", fontsize=9,
              title_fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_per_model_corrs(baseline_corrs, braak_corrs, region_names, save_path):
    """Plot side-by-side horizontal bars of per-region correlations for both models.

    Args
        baseline_corrs - numpy array of per-region Pearson r for the baseline model
        braak_corrs - numpy array of per-region Pearson r for the BraakGT model
        region_names - list of region name strings corresponding to array indices
        save_path - file path where the figure will be saved

    Returns
        None
    """
    n = len(region_names)
    x = np.arange(n)

    sort_idx = np.argsort(baseline_corrs)
    labels = [region_names[i].replace("CTX_LH_", "L-").replace("CTX_RH_", "R-").lower()
              for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.barh(x - 0.2, baseline_corrs[sort_idx], height=0.35, label="Baseline", color="#3878D8", alpha=0.8)
    ax.barh(x + 0.2, braak_corrs[sort_idx], height=0.35, label="BraakGT", color="#D85A30", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_title("Per-Region Test Correlation: Baseline vs BraakGT\n(averaged across folds)", fontsize=12)
    ax.legend(fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_braak_stage_breakdown(baseline_corrs, braak_corrs, region_names, save_path):
    """Plot box plots of per-region correlation grouped by Braak stage for both models.

    Args
        baseline_corrs - numpy array of per-region Pearson r for the baseline model
        braak_corrs - numpy array of per-region Pearson r for the BraakGT model
        region_names - list of region name strings corresponding to array indices
        save_path - file path where the figure will be saved

    Returns
        None
    """
    from preprocessing.braak_lookup import SUVR_REGION_TO_BRAAK

    stage_labels = {0: "Early (I/II)", 1: "Middle (III/IV)", 2: "Late (V/VI)"}
    stage_colors = {"Baseline": "#3878D8", "BraakGT": "#D85A30"}

    groups = {0: {"Baseline": [], "BraakGT": []},
              1: {"Baseline": [], "BraakGT": []},
              2: {"Baseline": [], "BraakGT": []}}

    for i, region in enumerate(region_names):
        stage = SUVR_REGION_TO_BRAAK.get(region)
        if stage is None:
            continue
        groups[stage]["Baseline"].append(baseline_corrs[i])
        groups[stage]["BraakGT"].append(braak_corrs[i])

    # sharey so all three stage panels share the same y-axis scale, making cross-stage comparisons valid
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    for ax, (stage, label) in zip(axes, stage_labels.items()):
        data = [groups[stage]["Baseline"], groups[stage]["BraakGT"]]
        n_regions = len(groups[stage]["Baseline"])

        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", linewidth=2))

        for patch, color in zip(bp["boxes"], stage_colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        for j, (bl, br) in enumerate(zip(groups[stage]["Baseline"], groups[stage]["BraakGT"])):
            ax.plot([1, 2], [bl, br], color="gray", alpha=0.3, linewidth=0.8)
            ax.scatter([1, 2], [bl, br], color=list(stage_colors.values()), s=20, zorder=3, alpha=0.6)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Baseline", "BraakGT"], fontsize=10)
        ax.set_title(f"{label}\n(n={n_regions} regions)", fontsize=11)
        ax.set_ylabel("Pearson r" if stage == 0 else "", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        for x, (model, vals) in zip([1, 2], groups[stage].items()):
            ax.text(x, ax.get_ylim()[0] if ax.get_ylim()[0] > -0.3 else -0.28,
                    f"μ={np.mean(vals):.2f}", ha="center", fontsize=8, color="black")

    fig.suptitle("Per-Region Correlation by Braak Stage: Baseline vs BraakGT\n(test set, averaged across folds)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = {
        "lr": 1e-4, "weight_decay": 1e-2, "batch_size": 16,
        "hidden_dim": 16, "n_layers": 2, "n_heads": 4,
        "dropout": 0.3, "max_epochs": 1000, "patience": 20
    }

    region_names = np.load(paths.ROOT / "data" / "processed" / "region_names.npy", allow_pickle=True)
    region_names = [str(r) for r in region_names]

    log_dir = paths.ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Computing per-region correlations for baseline...")
    baseline_corrs = get_per_region_corrs("baseline", config, args.k, args.seed)

    print("Computing per-region correlations for braak...")
    braak_corrs = get_per_region_corrs("braak", config, args.k, args.seed)

    print(f"\nBaseline mean corr: {baseline_corrs.mean():.3f}")
    print(f"BraakGT  mean corr: {braak_corrs.mean():.3f}")
    print(f"Mean diff: {(braak_corrs - baseline_corrs).mean():+.3f}")

    from preprocessing.braak_lookup import SUVR_REGION_TO_BRAAK
    stage_str = {0: "I/II", 1: "III/IV", 2: "V/VI"}

    diff = braak_corrs - baseline_corrs
    sort_idx = np.argsort(diff)[::-1]  # descending by improvement
    labels = [r.replace("CTX_LH_", "l-").replace("CTX_RH_", "r-").lower() for r in region_names]
    print(f"\n{'Region':<45} {'Baseline':>10} {'BraakGT':>10} {'Delta':>10}")
    print("-" * 77)
    for i in sort_idx:
        stage = stage_str.get(SUVR_REGION_TO_BRAAK.get(region_names[i], 2), "V/VI")
        label = f"{labels[i]} ({stage})"
        print(f"{label:<45} {baseline_corrs[i]:>10.3f} {braak_corrs[i]:>10.3f} {diff[i]:>+10.3f}")

    table_path = log_dir / "region_corrs_table.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "braak_stage", "baseline_r", "braak_r", "delta"])
        for i in sort_idx:
            stage = stage_str.get(SUVR_REGION_TO_BRAAK.get(region_names[i], 2), "V/VI")
            writer.writerow([labels[i], stage, f"{baseline_corrs[i]:.4f}", f"{braak_corrs[i]:.4f}", f"{diff[i]:+.4f}"])
    print(f"\nSaved: {table_path}")

    plot_difference_map(baseline_corrs, braak_corrs, region_names,
                        log_dir / "region_diff_map.png")

    plot_per_model_corrs(baseline_corrs, braak_corrs, region_names,
                         log_dir / "region_corrs.png")

    plot_braak_stage_breakdown(baseline_corrs, braak_corrs, region_names,
                               log_dir / "braak_stage_breakdown.png")
