"""
models/map_encoder.py

Purpose:
    Encodes lane polylines into a per-agent map context vector.
    Uses a shared MLP applied pointwise, max-pooling per polyline, then
    cross-attention with agent embeddings as queries.

    If map_polylines is None or all-zero, returns a zero vector of shape
    (B, N, model_dim) with no gradient.

Inputs:
    agent_emb     : (B, N, model_dim) — fused agent embedding (query)
    map_polylines : (B, P, L, 2)      — P polylines, L points, (x, y)

Outputs:
    map_emb : (B, N, model_dim)

Public interface:
    MapEncoder(config, device) -> Module
    forward(agent_emb, map_polylines) -> (B, N, model_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MapEncoder(nn.Module):
    """
    Polyline encoder with agent-conditioned cross-attention aggregation.

    Architecture:
        1. Shared MLP applied pointwise to each (x, y) coordinate → point features
        2. Max-pool over points within each polyline → polyline features
        3. Cross-attention: agent embeddings attend over polyline features
        4. Output per-agent map context vector

    Args:
        config : full config dict
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device

        m = config["model"]
        model_dim = m["model_dim"]  # 128
        point_dim = 2               # (x, y) per point

        # 1. Pointwise MLP: maps (x, y) → model_dim features
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, model_dim),
            nn.ReLU(),
        )

        # 2. Polyline-level projection after max-pool
        self.polyline_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
        )

        # 3. Cross-attention: agent (query) attends over polylines (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=m["num_heads"],
            dropout=0.1,
            batch_first=True,
        )

        # 4. Output layer norm
        self.norm = nn.LayerNorm(model_dim)

        self.model_dim = model_dim
        self.to(device)

    def forward(
        self,
        agent_emb: torch.Tensor,            # (B, N, model_dim)
        map_polylines: torch.Tensor,        # (B, P, L, 2)
    ) -> torch.Tensor:                      # returns (B, N, model_dim)
        """
        Args:
            agent_emb     : (B, N, D) — agent embeddings used as cross-attn queries
            map_polylines : (B, P, L, 2) — lane polylines in agent-centric frame

        Returns:
            map_emb : (B, N, D) — per-agent map context vector.
                      Zero vector if map data is empty or unavailable.
        """
        B, N, D = agent_emb.shape                               # unpack

        # --- handle missing or empty map data ---
        if map_polylines is None:
            return torch.zeros(B, N, self.model_dim, device=self.device)

        map_polylines = map_polylines.to(self.device)

        P, L = map_polylines.shape[1], map_polylines.shape[2]  # num polylines, num points

        if P == 0 or L == 0:
            # No polylines: return zero map embedding
            return torch.zeros(B, N, self.model_dim, device=self.device)

        # Check if all polylines are zero (dummy padding)
        if map_polylines.abs().sum() < 1e-8:
            return torch.zeros(B, N, self.model_dim, device=self.device)

        # 1. Apply pointwise MLP to each (x, y) point
        flat_points = map_polylines.reshape(B * P, L, 2)       # (B*P, L, 2)
        point_feats = self.point_mlp(flat_points)               # (B*P, L, D)

        # 2. Max-pool over points within each polyline
        poly_feats = point_feats.max(dim=1).values              # (B*P, D)
        poly_feats = poly_feats.reshape(B, P, D)                # (B, P, D)

        # 3. Polyline projection
        poly_feats = self.polyline_proj(poly_feats)             # (B, P, D)

        # 4. Cross-attention: each agent attends over all polylines.
        # batch_first MultiheadAttention accepts query (B, N, D) against
        # key/value (B, P, D) directly — no need to replicate the polyline
        # features N times into a (B*N, P, D) tensor.
        attn_out, _ = self.cross_attn(
            query=agent_emb,  # (B, N, D)
            key=poly_feats,   # (B, P, D)
            value=poly_feats, # (B, P, D)
        )                                                        # (B, N, D)

        # 5. Residual + norm
        map_emb = self.norm(attn_out + agent_emb)               # (B, N, D)
        return map_emb