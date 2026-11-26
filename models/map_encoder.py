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
        agent_emb: torch.Tensor,                    # (B, N, model_dim)
        map_polylines: torch.Tensor,                # (B, P, L, 2)
        agent_pos: torch.Tensor = None,             # (B, N, 2) last observed positions
        polyline_mask: torch.Tensor = None,         # (B, P) bool, True = real polyline
    ) -> torch.Tensor:                              # returns (B, N, model_dim)
        """
        Args:
            agent_emb     : (B, N, D) — agent embeddings used as cross-attn queries
            map_polylines : (B, P, L, 2) — lane polylines in the scene frame
            agent_pos     : (B, N, 2) — if given, every agent encodes the
                            polylines relative to its own position, so the
                            attention knows where the lanes are *for that agent*
            polyline_mask : (B, P) — padded polylines are ignored

        Returns:
            map_emb : (B, N, D) — per-agent map context vector. Zero for scenes
                      without any polyline (and for the whole batch when the
                      map is unavailable).
        """
        B, N, D = agent_emb.shape
        zeros = agent_emb.new_zeros(B, N, self.model_dim)
        if map_polylines is None:
            return zeros
        map_polylines = map_polylines.to(agent_emb.device)
        P, L = map_polylines.shape[1], map_polylines.shape[2]
        if P == 0 or L == 0:
            return zeros
        if polyline_mask is None:
            # Legacy inputs: an all-zero polyline is padding.
            polyline_mask = map_polylines.abs().sum(dim=(-1, -2)) > 1e-8   # (B, P)
        polyline_mask = polyline_mask.to(agent_emb.device)
        has_map = polyline_mask.any(dim=1)                                  # (B,)
        if not bool(has_map.any()):
            return zeros

        # 1-3. Point MLP -> max-pool -> projection, per agent-relative polyline.
        if agent_pos is not None:
            rel = map_polylines.unsqueeze(1) - agent_pos[:, :, None, None, :]  # (B, N, P, L, 2)
        else:
            rel = map_polylines.unsqueeze(1).expand(B, N, P, L, 2)
        poly_feats = self.point_mlp(rel).max(dim=3).values                  # (B, N, P, D)
        poly_feats = self.polyline_proj(poly_feats).reshape(B * N, P, D)    # (B*N, P, D)

        # 4. Cross-attention: each agent attends over its own polyline set.
        # Scenes without polylines get a dummy all-valid mask (no NaN rows);
        # their output is zeroed below.
        key_pad = ~polyline_mask
        key_pad = key_pad & has_map.unsqueeze(1)
        key_pad = key_pad.unsqueeze(1).expand(B, N, P).reshape(B * N, P)
        attn_out, _ = self.cross_attn(
            query=agent_emb.reshape(B * N, 1, D),
            key=poly_feats,
            value=poly_feats,
            key_padding_mask=key_pad,
        )                                                                   # (B*N, 1, D)
        attn_out = attn_out.reshape(B, N, D)

        # 5. Residual + norm; scenes without a map contribute nothing.
        map_emb = self.norm(attn_out + agent_emb)
        return map_emb * has_map.view(B, 1, 1).to(map_emb.dtype)
