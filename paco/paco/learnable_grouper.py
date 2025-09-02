import torch
import torch.nn as nn
import torch.nn.functional as F


class _DecoderLayer(nn.Module):
    """A simplified transformer decoder layer for plane queries."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, points: torch.Tensor):
        attn_out, attn = self.cross_attn(
            queries, points, points, need_weights=True
        )
        queries = self.norm1(queries + attn_out)
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)
        return queries, attn


class LearnableQueryGrouper(nn.Module):
    """Group points to plane proxies with learnable queries."""

    def __init__(
        self,
        num_queries: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_features = embed_dim
        self.temperature = temperature

        self.point_proj = nn.Linear(6, embed_dim)
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.layers = nn.ModuleList(
            [_DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.confidence_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, num=None):
        b, n, _ = x.shape
        coor = x[:, :, :3]
        normal = x[:, :, 3:6]

        point_feat = self.point_proj(torch.cat([coor, normal], dim=-1))

        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        attn = None
        for layer in self.layers:
            queries, attn = layer(queries, point_feat)

        assign = F.softmax(attn / self.temperature, dim=1)
        feat = torch.einsum("bqn,bnd->bqd", assign, point_feat)
        agg_coor = torch.einsum("bqn,bnd->bqd", assign, coor)
        agg_normal = torch.einsum("bqn,bnd->bqd", assign, normal)
        plane_idx = (
            torch.arange(self.num_queries, device=x.device)
            .view(1, self.num_queries, 1)
            .expand(b, -1, -1)
        )
        confidence = torch.sigmoid(self.confidence_head(queries)).squeeze(-1)
        return agg_coor, feat, agg_normal, plane_idx, assign, confidence

