"""
models/decoder.py

Purpose:
    GRU-based autoregressive decoder that generates future trajectories
    conditioned on context, goal embedding, and latent variable z.
    Includes a social attention mechanism at each decoding step.

Inputs:
    context      : (B, N, 256)    — fused context vector
    goal_coords  : (B, N, 2)      — single goal coordinate per agent
    z            : (B, N, latent_dim) — latent variable
    teacher_forcing_ratio : float — probability of using GT at each step

Outputs:
    trajectory : (B, N, T_future, 2) — predicted positions (accumulated deltas)

Public interface:
    Decoder(config, device) -> Module
    forward(context, goal_coords, z, future_gt=None, teacher_forcing_ratio=0.5)
        -> (B, N, T_future, 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SocialAttentionStep(nn.Module):
    """
    Lightweight social attention at each GRU decoding step.
    Each agent's GRU input is augmented with a weighted sum of
    neighbor context embeddings.

    Args:
        query_dim   : dimension of the per-agent query (GRU hidden state)
        context_dim : dimension of the social context embeddings
    """

    def __init__(self, query_dim: int, context_dim: int):
        super().__init__()
        self.query_proj   = nn.Linear(query_dim, context_dim)
        self.scale        = context_dim ** -0.5
        self.output_proj  = nn.Linear(context_dim, context_dim)

    def forward(
        self,
        query: torch.Tensor,    # (B, N, query_dim)   — current GRU hidden
        keys: torch.Tensor,     # (B, N, context_dim) — social context
        mask: torch.Tensor,     # (B, N) bool
    ) -> torch.Tensor:          # returns (B, N, context_dim)
        """
        Scaled dot-product attention over agents.

        Each agent attends to all other valid agents' context embeddings.
        """
        B, N, _ = query.shape

        Q = self.query_proj(query)          # (B, N, C)
        K = keys                            # (B, N, C)
        V = keys                            # (B, N, C)

        # Attention scores: Q_i · K_j
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # (B, N, N)

        # Mask: each agent should not strongly attend to invalid agents
        if mask is not None:
            mask_j = mask.unsqueeze(1).expand(B, N, N)   # (B, N, N)
            scores = scores.masked_fill(~mask_j, float("-inf"))

        attn = F.softmax(scores, dim=-1)                 # (B, N, N)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Weighted sum of V
        out = torch.bmm(attn, V)                                  # (B, N, C)
        out = self.output_proj(out)                               # (B, N, C)
        return out


class StaticInputGRUCell(nn.Module):
    """GRU cell whose input splits into a per-step part and a part that is
    constant over the sequence. Call project_const() once, then step with the
    per-step input. Equivalent to nn.GRUCell on the concatenated input."""

    def __init__(self, var_dim: int, const_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_var = nn.Linear(var_dim, 3 * hidden_dim, bias=False)
        self.w_const = nn.Linear(const_dim, 3 * hidden_dim)
        self.w_hh = nn.Linear(hidden_dim, 3 * hidden_dim)

    def project_const(self, const_input: torch.Tensor) -> torch.Tensor:
        return self.w_const(const_input)

    def forward(self, var_input, const_gates, hidden):
        gi = self.w_var(var_input) + const_gates
        gh = self.w_hh(hidden)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * hidden


class Decoder(nn.Module):
    """
    GRU autoregressive decoder with intent (goal) and latent conditioning.

    At each step t, the GRU receives:
        [prev_delta (2) || context (256) || goal_emb (64) || z (32) || social_attn (D)]

    Predicts delta displacements (dx, dy) which are accumulated into positions.

    Args:
        config : full config dict
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device

        m = config["model"]
        self.latent_dim    = m["latent_dim"]            # 32
        self.goal_emb_dim  = m["goal_embed_dim"]        # 64
        self.hidden_dim    = m["decoder_hidden_dim"]    # 256
        model_dim          = m["model_dim"]             # 128 (social context dim)
        self.future_steps  = config["dataset"]["future_steps"]  # 12
        # Ablation switch: without it the decoder gets no neighbour information.
        self.use_social    = m.get("use_social", True)

        context_dim = 256   # from fusion MLP

        # Goal embedding: project 2D goal → goal_emb_dim
        self.goal_proj = nn.Linear(2, self.goal_emb_dim)

        # Social attention at each decode step
        self.social_attn = SocialAttentionStep(
            query_dim=self.hidden_dim,
            context_dim=model_dim,
        )

        # GRU input = [prev_delta (2) || social_attn (D)]  (changes every step)
        #           + [context (256) || goal_emb (G) || z (D_z)] (constant)
        # The constant part's contribution to the gates is computed once per
        # sequence instead of at every step; otherwise this is a standard GRU.
        const_dim = context_dim + self.goal_emb_dim + self.latent_dim
        self.gru = StaticInputGRUCell(
            var_dim=2 + model_dim,
            const_dim=const_dim,
            hidden_dim=self.hidden_dim,
        )

        # Output head: hidden → (dx, dy)
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # Initial hidden state projection: context → hidden
        self.hidden_init = nn.Linear(context_dim, self.hidden_dim)

        self.to(device)

    def forward(
        self,
        context: torch.Tensor,          # (B, N, 256)
        social_emb: torch.Tensor,       # (B, N, model_dim) — for social attention
        goal_coords: torch.Tensor,      # (B, N, 2)
        z: torch.Tensor,                # (B, N, latent_dim)
        mask: torch.Tensor,             # (B, N) bool
        future_gt: Optional[torch.Tensor] = None,  # (B, N, T, 2) for teacher forcing
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:                  # returns (B, N, T_future, 2)
        """
        Autoregressive GRU decoding with social attention and teacher forcing.

        Args:
            context              : (B, N, 256) fused context
            social_emb           : (B, N, D)   interaction embeddings for social attn
            goal_coords          : (B, N, 2)   goal destination per agent
            z                    : (B, N, D_z) latent variable
            mask                 : (B, N) bool valid agent mask
            future_gt            : (B, N, T, 2) ground truth (for teacher forcing)
            teacher_forcing_ratio: float in [0,1], 0 = free-running

        Returns:
            positions : (B, N, T_future, 2) — positions relative to each
                        agent's last observed position (goal_coords and
                        future_gt must use the same convention)
        """
        B, N, _ = context.shape                                 # unpack
        T = self.future_steps                                   # 12
        device = context.device

        # Initialize GRU hidden state from context
        ctx_flat    = context.reshape(B * N, -1)                # (B*N, 256)
        hidden      = self.hidden_init(ctx_flat)                # (B*N, H)
        hidden      = torch.tanh(hidden)

        # Prepare fixed inputs (constant across time steps)
        goal_emb    = self.goal_proj(goal_coords)               # (B, N, G)
        goal_flat   = goal_emb.reshape(B * N, -1)              # (B*N, G)
        z_flat      = z.reshape(B * N, -1)                     # (B*N, D_z)

        # Starting position: last observed position
        prev_pos    = context.new_zeros(B, N, 2)               # (B, N, 2) zeros (agent-centric)
        prev_delta  = context.new_zeros(B * N, 2)              # (B*N, 2)

        predicted_positions = []

        # Draw all T teacher-forcing decisions once, up front, on the CPU.
        # Doing this per-step with `torch.rand(1).item()` forced a device->host
        # sync every timestep. When teacher forcing is inactive (eval, ratio 0,
        # or no ground truth) we skip sampling entirely and never teacher-force,
        # which matches the previous behaviour exactly (rand in [0,1) is never
        # < 0.0).
        if self.training and future_gt is not None and teacher_forcing_ratio > 0.0:
            tf_flips = torch.rand(T) < teacher_forcing_ratio    # (T,) bool, CPU
        else:
            tf_flips = None

        const_gates = self.gru.project_const(
            torch.cat([ctx_flat, goal_flat, z_flat], dim=-1))  # (B*N, 3H)

        for t in range(T):
            # Social attention: query = current hidden, keys = social embeddings
            hidden_3d   = hidden.reshape(B, N, -1)              # (B, N, H)
            if self.use_social:
                social_ctx = self.social_attn(hidden_3d, social_emb, mask)  # (B, N, D)
                social_flat = social_ctx.reshape(B * N, -1)    # (B*N, D)
            else:
                social_flat = social_emb.new_zeros(B * N, social_emb.shape[-1])

            # GRU step (constant inputs pre-projected in const_gates)
            var_input = torch.cat([prev_delta, social_flat], dim=-1)   # (B*N, 2+D)
            hidden = self.gru(var_input, const_gates, hidden)   # (B*N, H)

            # Predict delta displacement
            delta = self.output_head(hidden)                    # (B*N, 2)
            delta_3d = delta.reshape(B, N, 2)                   # (B, N, 2)

            # Accumulate positions
            pos = prev_pos + delta_3d                           # (B, N, 2)
            predicted_positions.append(pos)

            # Prepare next step input (decision drawn before the loop)
            use_gt = tf_flips is not None and bool(tf_flips[t])
            if use_gt and t < T - 1:
                # Teacher forcing: use ground-truth delta
                if t == 0:
                    gt_prev = context.new_zeros(B, N, 2)
                else:
                    gt_prev = future_gt[:, :, t - 1, :]
                gt_delta = future_gt[:, :, t, :] - gt_prev     # (B, N, 2)
                prev_delta = gt_delta.reshape(B * N, 2)
                prev_pos   = future_gt[:, :, t, :]
            else:
                prev_delta = delta                              # (B*N, 2)
                prev_pos   = pos                               # (B, N, 2)

        # Stack all timesteps
        positions = torch.stack(predicted_positions, dim=2)    # (B, N, T, 2)
        return positions