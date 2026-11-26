"""
models/social_gat.py

Purpose:
    Graph Attention Network (GAT) that models social interactions between
    agents. Produces interaction-aware embeddings for each agent.

Inputs:
    node_features : (B, N, model_dim) — per-agent temporal embeddings
    past          : (B, N, T, 4)      — for computing edge features
    mask          : (B, N) bool       — True for valid agents

Outputs:
    social_emb : (B, N, model_dim) — interaction-aware embedding per agent

Public interface:
    SocialGAT(config, device) -> Module
    forward(node_features, past, mask) -> (B, N, model_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GATLayer(nn.Module):
    """
    Single Graph Attention layer.

    Each node attends to all other valid nodes in the graph.
    Edge features (relative distance, relative velocity) are concatenated
    with the node pair representation before computing attention weights.

    Args:
        in_dim   : input feature dimension per node
        out_dim  : output feature dimension per node
        num_heads: number of attention heads
        edge_dim : edge feature dimension (3: dist + rel_vx + rel_vy)
        dropout  : attention dropout
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        edge_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.num_heads  = num_heads
        self.head_dim   = out_dim // num_heads
        self.out_dim    = out_dim
        self.scale      = self.head_dim ** -0.5

        # Linear projections for queries, keys, values
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)       # Q projection
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)       # K projection
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)       # V projection

        # Edge feature MLP: maps edge_dim → num_heads (one scalar per head)
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, num_heads * 4),
            nn.ReLU(),
            nn.Linear(num_heads * 4, num_heads),
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,            # (B, N, in_dim)
        edge_features: torch.Tensor, # (B, N, N, edge_dim)
        attn_mask: torch.Tensor,    # (B, N, N) bool — True = valid edge
    ) -> torch.Tensor:              # (B, N, out_dim)
        """
        Runs one GAT layer with edge-feature-augmented attention.
        """
        B, N, _ = x.shape                                       # unpack

        Q = self.W_q(x).reshape(B, N, self.num_heads, self.head_dim)  # (B, N, H, Dh)
        K = self.W_k(x).reshape(B, N, self.num_heads, self.head_dim)  # (B, N, H, Dh)
        V = self.W_v(x).reshape(B, N, self.num_heads, self.head_dim)  # (B, N, H, Dh)

        # Compute raw attention scores: Q_i · K_j for all i, j
        # einsum avoids materialising a (B, N, N, H, Dh) broadcast product.
        scores = torch.einsum("bnhd,bmhd->bnmh", Q, K) * self.scale  # (B, N, N, H)

        # Add edge feature contribution (one scalar per head per edge)
        edge_bias = self.edge_proj(edge_features)               # (B, N, N, H)
        scores = scores + edge_bias                             # (B, N, N, H)

        # Mask invalid edges (padding agents)
        mask_exp = attn_mask.unsqueeze(-1)                      # (B, N, N, 1)
        scores = scores.masked_fill(~mask_exp, float("-inf"))   # (B, N, N, H)

        # Softmax over source nodes dimension (dim=2)
        attn = F.softmax(scores, dim=2)                         # (B, N, N, H)
        attn = self.dropout(attn)
        # Replace nan from all-masked rows with 0
        attn = torch.nan_to_num(attn, nan=0.0)

        # Aggregate values
        agg = torch.einsum("bnmh,bmhd->bnhd", attn, V)          # (B, N, H, Dh)
        agg = agg.reshape(B, N, self.out_dim)                   # (B, N, out_dim)

        # Output projection + residual + norm
        out = self.out_proj(agg)                                # (B, N, out_dim)
        out = self.norm(out + x if x.shape[-1] == self.out_dim else out)
        return out                                              # (B, N, out_dim)


class SocialGAT(nn.Module):
    """
    Two-layer Graph Attention Network for social interaction encoding.

    Constructs a fully connected graph over valid agents. Edge features
    capture relative positions and velocities between agent pairs.

    Args:
        config : full config dict
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device

        m = config["model"]
        model_dim  = m["model_dim"]   # 128
        num_heads  = m["gat_heads"]   # 4
        num_layers = m["gat_layers"]  # 2
        edge_dim   = m["edge_dim"]    # 3

        # Stack GAT layers
        layers = []
        for i in range(num_layers):
            layers.append(
                GATLayer(
                    in_dim=model_dim,
                    out_dim=model_dim,
                    num_heads=num_heads,
                    edge_dim=edge_dim,
                )
            )
        self.gat_layers = nn.ModuleList(layers)

        # Final layer norm
        self.norm = nn.LayerNorm(model_dim)

        self.to(device)

    def _compute_edge_features(
        self,
        past: torch.Tensor,  # (B, N, T, 4)
    ) -> torch.Tensor:       # returns (B, N, N, 3)
        """
        Compute pairwise edge features from the last observed timestep.

        Edge feature for (i → j): [relative_distance, rel_vx, rel_vy]
            relative_distance = ||pos_j - pos_i||
            rel_vx            = vx_j - vx_i
            rel_vy            = vy_j - vy_i
        """
        # Use last observed timestep
        pos = past[:, :, -1, :2]    # (B, N, 2)  positions at t=-1
        vel = past[:, :, -1, 2:]    # (B, N, 2)  velocities at t=-1

        # Pairwise relative positions
        pos_i = pos.unsqueeze(2)    # (B, N, 1, 2)
        pos_j = pos.unsqueeze(1)    # (B, 1, N, 2)
        rel_pos = pos_j - pos_i     # (B, N, N, 2)  pos_j - pos_i

        # NOTE: use sqrt(sum(x^2) + eps) rather than .norm() — at zero distance
        # (self-edges, co-located agents) norm/sqrt has a NaN gradient in the
        # backward pass, which corrupts the whole GAT. The eps keeps it finite.
        dist = torch.sqrt(rel_pos.pow(2).sum(-1, keepdim=True) + 1e-6)  # (B, N, N, 1)

        # Pairwise relative velocities
        vel_i = vel.unsqueeze(2)    # (B, N, 1, 2)
        vel_j = vel.unsqueeze(1)    # (B, 1, N, 2)
        rel_vel = vel_j - vel_i     # (B, N, N, 2)

        edge_features = torch.cat([dist, rel_vel], dim=-1)  # (B, N, N, 3)
        return edge_features

    def forward(
        self,
        node_features: torch.Tensor,        # (B, N, model_dim)
        past: torch.Tensor,                 # (B, N, T, 4)
        mask: Optional[torch.Tensor] = None,  # (B, N) bool
    ) -> torch.Tensor:                      # returns (B, N, model_dim)
        """
        Args:
            node_features : (B, N, D) — input node embeddings (from temporal encoder)
            past          : (B, N, T, 4) — for edge feature computation
            mask          : (B, N) bool — True for valid agents

        Returns:
            social_emb : (B, N, D)
        """
        B, N, D = node_features.shape                           # unpack

        # Build edge validity mask: edge (i,j) valid iff both i and j are valid
        if mask is not None:
            mask_i = mask.unsqueeze(2)                          # (B, N, 1)
            mask_j = mask.unsqueeze(1)                          # (B, 1, N)
            edge_mask = mask_i & mask_j                         # (B, N, N) bool
        else:
            edge_mask = torch.ones(B, N, N, dtype=torch.bool, device=self.device)

        # Compute edge features
        edge_features = self._compute_edge_features(past)       # (B, N, N, 3)

        x = node_features                                       # (B, N, D)
        for layer in self.gat_layers:
            x = layer(x, edge_features, edge_mask)             # (B, N, D)

        social_emb = self.norm(x)                               # (B, N, D)
        return social_emb