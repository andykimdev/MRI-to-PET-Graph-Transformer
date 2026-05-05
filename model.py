import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class BaselineGraphTransformer(nn.Module):
    """Graph transformer without Braak prior. Predicts SUVR per node."""
    def __init__(self, in_dim=3, hidden_dim=64, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__() 
        
        # Project node features to hidden dimension
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Graph transformer layers
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Regression head per node 
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, braak=None):
        h = self.input_proj(x)

        for layer in self.layers:
            h_new = layer(h, edge_index)
            h = h + F.relu(h_new) # Residual connection
        
        out = self.head(h).squeeze(-1)
        return out
    
class BraakGraphTransformer(nn.Module):
    """Graph transformer without Braak prior. Predicts SUVR per node."""
    def __init__(self, in_dim=3, hidden_dim=64, n_layers=4, n_heads=4, n_braak_stages=3, dropout=0.1):
        super().__init__()

        # Project node features to hidden dimension
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Learnable Braak stage embedding 
        self.braak_embedding = nn.Embedding(n_braak_stages, hidden_dim)

        # Graph transformer layers
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Regression head per node
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, braak):
        h = self.input_proj(x)

        h = h + self.braak_embedding(braak)

        for layer in self.layers:
            h_new = layer(h, edge_index)
            h = h + F.relu(h_new)

        out = self.head(h).squeeze(-1)

        return out
        
if __name__ == "__main__":
    n_nodes = 66
    batch_size = 4

    # Simulate a batch of 4 graphs concatenated together
    x = torch.randn(batch_size * n_nodes, 3)
    braak = torch.randint(0, 3, (batch_size * n_nodes,))
    
    # Fully connected edges within each graph
    rows, cols = [], []
    for b in range(batch_size):
        offset = b * n_nodes
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(offset + i)
                    cols.append(offset + j)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    # Test baseline
    baseline = BaselineGraphTransformer()
    out_baseline = baseline(x, edge_index)
    n_params_baseline = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline output: {out_baseline.shape}")
    print(f"Baseline parameters: {n_params_baseline:,}")
    
    # Test Braak model
    braak_model = BraakGraphTransformer()
    out_braak = braak_model(x, edge_index, braak)
    n_params_braak = sum(p.numel() for p in braak_model.parameters())
    print(f"\nBraak model output: {out_braak.shape}")
    print(f"Braak model parameters: {n_params_braak:,}")
    
    # Confirm shapes match
    assert out_baseline.shape == (batch_size * n_nodes,)
    assert out_braak.shape == (batch_size * n_nodes,)
    print("\nBoth models produce correct output shape.")






        