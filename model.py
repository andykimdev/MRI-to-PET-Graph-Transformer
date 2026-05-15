import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class BaselineGraphTransformer(nn.Module):
    """Graph transformer without Braak prior that predicts SUVR per node.

    Args
        in_dim - number of input node feature dimensions
        hidden_dim - size of the hidden representation
        n_layers - number of graph transformer layers
        n_heads - number of attention heads per layer
        dropout - dropout probability used in transformer layers and the regression head

    Returns
        nn.Module instance ready for training
    """
    def __init__(self, in_dim=3, hidden_dim=64, n_layers=4, n_heads=4, dropout=0.1):
        """Initialize layers for the baseline graph transformer.

        Args
            in_dim - number of input node feature dimensions
            hidden_dim - size of the hidden representation
            n_layers - number of graph transformer layers
            n_heads - number of attention heads per layer
            dropout - dropout probability used in transformer layers and the regression head

        Returns
            None
        """
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, braak=None):
        """Run a forward pass through the baseline graph transformer.

        Args
            x - node feature matrix of shape (num_nodes, in_dim)
            edge_index - graph connectivity tensor of shape (2, num_edges)
            braak - unused Braak stage tensor included for interface compatibility

        Returns
            predicted SUVR values per node as a tensor of shape (num_nodes,)
        """
        h = self.input_proj(x)

        for layer in self.layers:
            h_new = layer(h, edge_index)
            h = h + F.relu(h_new)  # residual keeps early spatial structure from vanishing in deep stacks

        out = self.head(h).squeeze(-1)
        return out

class BraakGraphTransformer(nn.Module):
    """Graph transformer with Braak stage embeddings that predicts SUVR per node.

    Args
        in_dim - number of input node feature dimensions
        hidden_dim - size of the hidden representation
        n_layers - number of graph transformer layers
        n_heads - number of attention heads per layer
        n_braak_stages - number of distinct Braak stages to embed
        dropout - dropout probability used in transformer layers and the regression head

    Returns
        nn.Module instance ready for training
    """
    def __init__(self, in_dim=3, hidden_dim=64, n_layers=4, n_heads=4, n_braak_stages=3, dropout=0.1):
        """Initialize layers for the Braak-conditioned graph transformer.

        Args
            in_dim - number of input node feature dimensions
            hidden_dim - size of the hidden representation
            n_layers - number of graph transformer layers
            n_heads - number of attention heads per layer
            n_braak_stages - number of distinct Braak stages to embed
            dropout - dropout probability used in transformer layers and the regression head

        Returns
            None
        """
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.braak_embedding = nn.Embedding(n_braak_stages, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // n_heads,
                heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, braak):
        """Run a forward pass through the Braak-conditioned graph transformer.

        Args
            x - node feature matrix of shape (num_nodes, in_dim)
            edge_index - graph connectivity tensor of shape (2, num_edges)
            braak - integer Braak stage index per node as a tensor of shape (num_nodes,)

        Returns
            predicted SUVR values per node as a tensor of shape (num_nodes,)
        """
        h = self.input_proj(x)

        h = h + self.braak_embedding(braak)  # additive so Braak modulates existing features rather than expanding the dimension

        for layer in self.layers:
            h_new = layer(h, edge_index)
            h = h + F.relu(h_new)  # residual keeps early spatial structure from vanishing in deep stacks

        out = self.head(h).squeeze(-1)

        return out

if __name__ == "__main__":
    n_nodes = 66
    batch_size = 4

    x = torch.randn(batch_size * n_nodes, 3)
    braak = torch.randint(0, 3, (batch_size * n_nodes,))

    rows, cols = [], []
    for b in range(batch_size):
        offset = b * n_nodes
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(offset + i)
                    cols.append(offset + j)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)

    baseline = BaselineGraphTransformer()
    out_baseline = baseline(x, edge_index)
    n_params_baseline = sum(p.numel() for p in baseline.parameters())
    print(f"Baseline output: {out_baseline.shape}")
    print(f"Baseline parameters: {n_params_baseline:,}")

    braak_model = BraakGraphTransformer()
    out_braak = braak_model(x, edge_index, braak)
    n_params_braak = sum(p.numel() for p in braak_model.parameters())
    print(f"\nBraak model output: {out_braak.shape}")
    print(f"Braak model parameters: {n_params_braak:,}")

    assert out_baseline.shape == (batch_size * n_nodes,)
    assert out_braak.shape == (batch_size * n_nodes,)
    print("\nBoth models produce correct output shape.")
