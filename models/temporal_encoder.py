"""
models/temporal_encoder.py

Purpose:
    Encodes the 8-step past trajectory of each agent into a fixed-size
    temporal embedding using a Transformer encoder with sinusoidal
    positional encoding.

Inputs:
    past : (B, N, T, state_dim) — agent states (x, y, vx, vy)
    mask : (B, N) bool          — True for valid agents

Outputs:
    temporal_emb : (B, N, model_dim) — per-agent temporal embedding

Public interface:
    TemporalEncoder(config, device) -> Module
    forward(past, mask) -> (B, N, model_dim)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding as in 'Attention Is All You Need'.

    Args:
        model_dim : embedding dimension (must be even)
        max_len   : maximum sequence length
    """

    def __init__(self, model_dim: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, model_dim)                    # (max_len, D)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float()
            * (-math.log(10000.0) / model_dim)
        )                                                        # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)            # even dims
        pe[:, 1::2] = torch.cos(position * div_term)            # odd dims
        pe = pe.unsqueeze(0)                                     # (1, max_len, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, seq_len, model_dim)
        Returns:
            x + positional encoding, same shape
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]                      # (batch, T, D)


class TemporalEncoder(nn.Module):
    """
    Transformer-based encoder that produces a per-agent temporal embedding
    from the 8-step motion history.

    Architecture:
        1. Linear projection: state_dim → model_dim
        2. Sinusoidal positional encoding
        3. TransformerEncoder (num_layers layers, num_heads heads)
        4. Mean pooling over the time dimension → final embedding

    Args:
        config : full config dict
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device

        m = config["model"]
        state_dim   = m["state_dim"]                            # 4
        model_dim   = m["model_dim"]                            # 128
        num_heads   = m["num_heads"]                            # 4
        num_layers  = m["num_transformer_layers"]               # 2
        ff_dim      = model_dim * 4                             # 512

        # 1. Input projection
        self.input_proj = nn.Linear(state_dim, model_dim)       # (state_dim) → (D)

        # 2. Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(model_dim)

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,  # expects (batch, seq, dim)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 4. Layer norm on output
        self.norm = nn.LayerNorm(model_dim)

        self.to(device)

    def forward(
        self,
        past: torch.Tensor,          # (B, N, T, state_dim)
        mask: Optional[torch.Tensor] = None,  # (B, N) bool
    ) -> torch.Tensor:               # returns (B, N, model_dim)
        """
        Args:
            past : (B, N, T, state_dim) — agent histories
            mask : (B, N) bool — True for valid agents (unused here,
                   padding is handled by collation; kept for API consistency)

        Returns:
            temporal_emb : (B, N, model_dim)
        """
        B, N, T, S = past.shape                                 # unpack

        # Flatten batch and agent dims for parallel processing
        x = past.reshape(B * N, T, S)                          # (B*N, T, state_dim)

        # 1. Project to model dim
        x = self.input_proj(x)                                  # (B*N, T, D)

        # 2. Add positional encoding
        x = self.pos_enc(x)                                     # (B*N, T, D)

        # 3. Transformer encoding
        x = self.transformer(x)                                 # (B*N, T, D)

        # 4. Mean pool over time dimension → single embedding per agent
        x = x.mean(dim=1)                                       # (B*N, D)

        # 5. Layer norm
        x = self.norm(x)                                        # (B*N, D)

        # Reshape back to (B, N, D)
        temporal_emb = x.reshape(B, N, -1)                     # (B, N, D)
        return temporal_emb