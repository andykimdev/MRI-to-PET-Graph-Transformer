# plot_training.py
import sys
sys.path.append("preprocessing")

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import paths


def _load_fold_histories(model_type, k):
    """Load per-fold history JSON files from the log directory.

    Args
        model_type - string identifier for the model, e.g. "baseline" or "braak"
        k - total number of folds expected

    Returns
        tuple of (histories, loaded_folds) where histories is a list of dicts
        and loaded_folds is a list of fold indices that were successfully loaded
    """
    log_dir = paths.ROOT / "outputs" / "logs"
    histories = []
    loaded_folds = []
    for fold in range(1, k + 1):
        p = log_dir / f"{model_type}_fold{fold}_history.json"
        if not p.exists():
            print(f"  Missing: {p.name} — skipping fold {fold}")
            continue
        with open(p) as f:
            histories.append(json.load(f))
        loaded_folds.append(fold)
    return histories, loaded_folds


def _align_and_stack(histories, key):
    """Pad shorter runs with their last value and stack into a 2-D array.

    Args
        histories - list of per-fold history dicts, each a list of epoch-level dicts
        key - metric key to extract from each epoch dict

    Returns
        numpy array of shape (n_folds, max_epochs)
    """
    max_len = max(len(h) for h in histories)
    rows = []
    for h in histories:
        vals = [entry[key] for entry in h]
        vals += [vals[-1]] * (max_len - len(vals))
        rows.append(vals)
    return np.array(rows)


def _plot_band(ax, epochs, mat, color, label):
    """Plot a mean line with a shaded standard deviation band.

    Args
        ax - matplotlib Axes object to draw on
        epochs - 1-D array of epoch indices
        mat - 2-D array of shape (n_folds, n_epochs) to summarize
        color - hex color string for the line and band
        label - legend label for the mean line

    Returns
        None
    """
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    ax.plot(epochs, mean, linewidth=2, color=color, label=label)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)


def plot_history(model_type, k=5):
    """Plot mean and std bands across folds for a single model type.

    Args
        model_type - string identifier for the model, e.g. "baseline" or "braak"
        k - number of folds to load (default 5)

    Returns
        None
    """
    log_dir = paths.ROOT / "outputs" / "logs"
    histories, loaded_folds = _load_fold_histories(model_type, k)

    if not histories:
        print(f"No fold histories found for {model_type}.")
        return

    train_mat = _align_and_stack(histories, "train_loss")
    val_mat = _align_and_stack(histories, "val_loss")
    corr_mat = _align_and_stack(histories, "val_mean_corr")
    epochs = np.arange(1, train_mat.shape[1] + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    _plot_band(axes[0], epochs, train_mat, "#3878D8", f"Train (n={len(loaded_folds)} folds)")
    _plot_band(axes[0], epochs, val_mat, "#D85A30", "Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title(f"{model_type.capitalize()}: Loss (mean ± std)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    _plot_band(axes[1], epochs, corr_mat, "#1D9E75", f"Val corr (n={len(loaded_folds)} folds)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean Per-Region Pearson r")
    axes[1].set_title(f"{model_type.capitalize()}: Validation Correlation (mean ± std)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = log_dir / f"{model_type}_training.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_comparison(k=5):
    """Compare baseline vs braak with mean and std bands across folds.

    Args
        k - number of folds to load (default 5)

    Returns
        None
    """
    log_dir = paths.ROOT / "outputs" / "logs"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for model_type, color in [("baseline", "#3878D8"), ("braak", "#D85A30")]:
        histories, loaded_folds = _load_fold_histories(model_type, k)
        if not histories:
            print(f"Skipping {model_type} — no fold histories found")
            continue

        val_mat = _align_and_stack(histories, "val_loss")
        corr_mat = _align_and_stack(histories, "val_mean_corr")
        epochs = np.arange(1, val_mat.shape[1] + 1)
        label = f"{model_type.capitalize()} (n={len(loaded_folds)} folds)"

        for mat, ax in [(val_mat, axes[0]), (corr_mat, axes[1])]:
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            ax.plot(epochs, mean, linewidth=2, color=color, label=label)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation MSE Loss")
    axes[0].set_title("Baseline vs Braak: Validation Loss (mean ± std)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean Per-Region Pearson r")
    axes[1].set_title("Baseline vs Braak: Validation Correlation (mean ± std)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = log_dir / "comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def print_aggregate(model_type):
    """Print the aggregate summary JSON for a model.

    Args
        model_type - string identifier for the model, e.g. "baseline" or "braak"

    Returns
        None
    """
    p = paths.ROOT / "outputs" / "logs" / f"{model_type}_aggregate.json"
    if not p.exists():
        print(f"No aggregate file for {model_type}. Run train.py first.")
        return
    with open(p) as f:
        agg = json.load(f)
    k = agg["k"]
    print(f"\n{model_type.upper()} {k}-fold CV aggregate:")
    print(f"  val_loss  : {agg['mean_val_loss']:.4f} ± {agg['std_val_loss']:.4f}")
    print(f"  val_corr  : {agg['mean_val_corr']:.3f} ± {agg['std_val_corr']:.3f}")
    print(f"  test_loss : {agg['mean_test_loss']:.4f} ± {agg['std_test_loss']:.4f}")
    print(f"  test_corr : {agg['mean_test_corr']:.3f} ± {agg['std_test_corr']:.3f}")
    print("\nPer-fold results:")
    for r in agg["folds"]:
        print(f"  fold {r['fold']} | epoch {r['best_epoch']:3d} | "
              f"val_loss {r['best_val_loss']:.4f} | val_corr {r['best_val_corr']:.3f} | "
              f"train_n {r['train_size']} val_n {r['val_size']}")


def _load_test_values(model_type):
    """Return test loss and correlation arrays across folds from the aggregate JSON.

    Args
        model_type - string identifier for the model, e.g. "baseline" or "braak"

    Returns
        tuple of (test_losses, test_corrs), each a numpy array of length k,
        or (None, None) if the aggregate file does not exist
    """
    p = paths.ROOT / "outputs" / "logs" / f"{model_type}_aggregate.json"
    if not p.exists():
        return None, None
    with open(p) as f:
        agg = json.load(f)
    losses = np.array([r["test_loss"] for r in agg["folds"]])
    corrs = np.array([r["test_corr"] for r in agg["folds"]])
    return losses, corrs


def print_paired_ttest():
    """Run and print a paired t-test between baseline and braak on test metrics across folds.

    Args
        None

    Returns
        None
    """
    bl_losses, bl_corrs = _load_test_values("baseline")
    br_losses, br_corrs = _load_test_values("braak")

    if bl_losses is None or br_losses is None:
        print("\nNeed aggregate files for both models to run paired t-test.")
        return
    if len(bl_losses) != len(br_losses):
        print(f"\nFold counts differ ({len(bl_losses)} vs {len(br_losses)}) — cannot run paired t-test.")
        return

    print(f"\n{'='*50}")
    print(f"Paired t-test on test set  (n={len(bl_losses)} folds, two-tailed)")
    print(f"{'='*50}")

    for metric, bl, br in [("test_loss", bl_losses, br_losses),
                            ("test_corr", bl_corrs, br_corrs)]:
        t, p = stats.ttest_rel(bl, br)
        diff = bl - br
        print(f"\n  {metric}:")
        print(f"    baseline : {bl.mean():.4f} ± {bl.std():.4f}")
        print(f"    braak    : {br.mean():.4f} ± {br.std():.4f}")
        print(f"    mean diff (baseline - braak) : {diff.mean():+.4f}")
        print(f"    t = {t:.3f},  p = {p:.4f}", end="")
        print("  *" if p < 0.05 else "  (n.s.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "braak", "compare"], default="baseline")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--summary", action="store_true",
                        help="Print aggregate summary instead of plotting")
    args = parser.parse_args()

    if args.summary:
        targets = ["baseline", "braak"] if args.model == "compare" else [args.model]
        for m in targets:
            print_aggregate(m)
        if args.model == "compare":
            print_paired_ttest()
    elif args.model == "compare":
        plot_comparison(args.k)
    else:
        plot_history(args.model, args.k)
