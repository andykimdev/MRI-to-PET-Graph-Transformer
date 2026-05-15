import sys
sys.path.insert(0, "preprocessing")

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import Data

import paths


class BrainTauDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed brain tau graph data.

    Args
        processed_dir - path to data/processed/
        split - "train", "val", or "test", used only when indices is None
        norm_params - dict with "mean" (n_regions, 3) and "std" (n_regions, 3)
            arrays; when provided, normalizes raw features on the fly for k-fold mode
        indices - explicit integer indices into the full arrays; overrides split lookup

    Returns
        A Dataset instance whose items are PyG Data objects.
    """

    def __init__(self, processed_dir, split, norm_params=None, indices=None):
        """Initialize BrainTauDataset by loading arrays and building edge structure.

        Args
            processed_dir - path to data/processed/
            split - "train", "val", or "test", used only when indices is None
            norm_params - dict with "mean" (n_regions, 3) and "std" (n_regions, 3)
                arrays; when provided, normalizes raw features on the fly for k-fold mode
            indices - explicit integer indices into the full arrays; overrides split lookup

        Returns
            None
        """
        super().__init__()

        self.processed_dir = Path(processed_dir)
        self.split = split

        if norm_params is not None:
            X_raw = np.load(self.processed_dir / "node_features_raw.npy")
            mean = norm_params["mean"]
            std = norm_params["std"]
            std[std == 0] = 1  # avoid division by zero for constant features while keeping their normalized value at 0
            X_norm = (X_raw - mean) / std
            self.X = np.nan_to_num(X_norm, nan=0.0)
        else:
            self.X = np.load(self.processed_dir / "node_features.npy")

        self.Y = np.load(self.processed_dir / "suvr_targets.npy")
        self.braak = np.load(self.processed_dir / "braak_stages.npy")
        self.subject_ids = np.load(self.processed_dir / "subject_ids.npy")

        if indices is not None:
            self.indices = np.array(indices)
        else:
            with open(self.processed_dir / "splits.json") as f:
                splits = json.load(f)
            split_rids = set(splits[split]) 
            self.indices = np.array([i for i, rid in enumerate(self.subject_ids)
                                     if rid in split_rids])

        n_regions = self.X.shape[1]
        rows, cols = [], []
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long)  # built once and shared so every subject uses the same atlas topology

        self.braak_tensor = torch.tensor(self.braak, dtype=torch.long)  # pre-converted so __getitem__ never re-allocates a tensor on each access

    def __len__(self):
        """Return the number of samples in this split.

        Returns
            Integer count of samples.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """Retrieve a single graph sample by split-local index.

        Args
            idx - integer index within this split

        Returns
            PyG Data object with fields x, edge_index, y, braak, and subject_id.
        """
        global_idx = self.indices[idx]
        x = torch.tensor(self.X[global_idx], dtype=torch.float)
        y = torch.tensor(self.Y[global_idx], dtype=torch.float)

        return Data(x=x, edge_index=self.edge_index, y=y, braak=self.braak_tensor, subject_id=int(self.subject_ids[global_idx]))


if __name__ == "__main__":
    train_ds = BrainTauDataset(paths.ROOT / "data" / "processed", "train")
    val_ds = BrainTauDataset(paths.ROOT / "data" / "processed", "val")
    test_ds = BrainTauDataset(paths.ROOT / "data" / "processed", "test")

    sample = train_ds[0]
    print(f"\nSample shapes:")
    print(f"  x: {sample.x.shape}")
    print(f"  y: {sample.y.shape}")
    print(f"  edge_index: {sample.edge_index.shape}")
    print(f"  braak: {sample.braak.shape}")
    print(f"  subject_id: {sample.subject_id}")
