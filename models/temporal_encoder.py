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
    Per-agent temporal embedding of the motion history.

    Input positions are expected relative to each agent's own last observed
    position (the model does this), so the encoder sees motion, not location.

    Architectures (config["model"]["temporal_encoder"]):
        transformer (default):
            1. Linear projection: state_dim -> model_dim
            2. Sinusoidal positional encoding
            3. TransformerEncoder with padded history steps masked out
            4. Masked mean pooling over time
        lstm (ablation): single-layer LSTM, final hidden state.

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
        self.kind   = m.get("temporal_encoder", "transformer")
        if self.kind not in ("transformer", "lstm"):
            raise ValueError(f"Unknown temporal_encoder '{self.kind}'")

        self.input_proj = nn.Linear(state_dim, model_dim)       # (state_dim) -> (D)
        if self.kind == "transformer":
            self.pos_enc = SinusoidalPositionalEncoding(model_dim)
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
                enable_nested_tensor=False,
            )
        else:
            self.lstm = nn.LSTM(model_dim, model_dim, batch_first=True)

        self.norm = nn.LayerNorm(model_dim)
        self.to(device)

    def forward(
        self,
        past: torch.Tensor,                       # (B, N, T, state_dim)
        mask: Optional[torch.Tensor] = None,      # (B, N) bool (unused; API compat)
        past_mask: Optional[torch.Tensor] = None, # (B, N, T) bool, True = observed
    ) -> torch.Tensor:                            # returns (B, N, model_dim)
        B, N, T, S = past.shape
        x = self.input_proj(past.reshape(B * N, T, S))         # (B*N, T, D)
        if past_mask is None:
            pm = torch.ones(B * N, T, dtype=torch.bool, device=past.device)
        else:
            pm = past_mask.reshape(B * N, T).clone()
            pm[:, -1] = True          # current step is always observed

        if self.kind == "transformer":
            x = self.pos_enc(x)
            x = self.transformer(x, src_key_padding_mask=~pm)  # (B*N, T, D)
            w = pm.unsqueeze(-1).to(x.dtype)
            x = (x * w).sum(1) / w.sum(1).clamp(min=1.0)       # masked mean
        else:
            # Padded steps are the oldest ones; zero them so the LSTM sees a
            # neutral prefix, then take the final hidden state.
            x = x * pm.unsqueeze(-1).to(x.dtype)
            _, (h, _) = self.lstm(x)
            x = h[-1]                                          # (B*N, D)

        x = self.norm(x)
        return x.reshape(B, N, -1)                             # (B, N, D)
