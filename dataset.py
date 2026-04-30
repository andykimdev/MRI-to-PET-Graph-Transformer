import sys
sys.path.insert(0, "preprocessing") # since directories.py is in "preprocessing" folder

import torch 
import numpy as np
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
import directories as dir 
print(dir.__file__)

class BrainTauDataset(Dataset):
    def __init__(self, processed_dir, split):
        super().__init__()
        
        # Directory to saved numpy files via preprocess.py
        self.processed_dir = Path(processed_dir)

        # Train / Val / Test dataset
        self.split = split 

        # Load all data
        self.X = np.load(self.processed_dir / "node_features.npy")
        self.Y = np.load(self.processed_dir / "suvr_targets.npy")
        self.braak = np.load(self.processed_dir / "braak_stages.npy")        
        self.subject_ids = np.load(self.processed_dir / "subject_ids.npy")

        # Filtering 
        with open(self.processed_dir / "splits.json") as f:
            splits = json.load(f)
        
        split_rids = set(splits[split]) # setting it as set() for performance boosting 
        self.indices = np.array([i for i, rid in enumerate(self.subject_ids)
                                 if rid in split_rids])
        
        # Building edge structure
        n_regions = self.X.shape[1]
        rows, cols = [], []
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long)

        # Converting braak to tensor
        self.braak_tensor = torch.tensor(self.braak, dtype=torch.long)
        
    # lenth method
    def __len__(self):
        return len(self.indices)
    
    # split method 
    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        x = torch.tensor(self.X[global_idx], dtype=torch.float)
        y = torch.tensor(self.Y[global_idx], dtype=torch.float)

        return Data(x=x, edge_index=self.edge_index, y=y, braak=self.braak_tensor, subject_id=int(self.subject_ids[global_idx]))
    
if __name__ == "__main__":
    train_ds = BrainTauDataset(dir.ROOT / "data" / "processed", "train")
    val_ds = BrainTauDataset(dir.ROOT / "data" / "processed", "val")
    test_ds = BrainTauDataset(dir.ROOT / "data" / "processed", "test")

    sample = train_ds[0]
    print(f"\nSample shapes:")
    print(f"  x: {sample.x.shape}")
    print(f"  y: {sample.y.shape}")
    print(f"  edge_index: {sample.edge_index.shape}")
    print(f"  braak: {sample.braak.shape}")
    print(f"  subject_id: {sample.subject_id}")








        

        

