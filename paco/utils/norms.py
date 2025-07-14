import torch
from torch import nn

class LayerNorm1d(nn.Module):
    """LayerNorm for 1D Conv features of shape (B, C, N)."""
    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)
