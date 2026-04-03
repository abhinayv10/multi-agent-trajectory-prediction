"""
models/goal_predictor.py

Purpose:
    Predicts probable destination (goal) points for each agent.
    Samples candidate goals from neighbor endpoints and map polylines,
    scores them using a neural network, and selects top-K.

Inputs:
    context       : (B, N, 256)  — fused context vector
    past          : (B, N, T, 4) — for sampling neighbor-based goals
    map_polylines : (B, P, L, 2) — for sampling polyline endpoints

Outputs:
    goal_coords    : (B, N, K, 2) — top-K goal coordinates
    goal_log_probs : (B, N, K)    — log-probabilities of top-K goals

Public interface:
    GoalPredictor(config, device) -> Module
    forward(context, past, map_polylines) -> (goal_coords, goal_log_probs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GoalPredictor(nn.Module):
    """
    Intent-first goal prediction module.

    Candidate generation strategy:
        - Final observed positions of all neighbor agents + Gaussian noise
        - Endpoints of map polylines (first and last point of each lane)
        - Random Gaussian samples around the ego agent's last position

    Scoring:
        - MLP over concatenation of (goal_coord, context_vector)

    Args:
        config : full config dict
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device

        m = config["model"]
        self.num_goals      = m["num_goals"]        # K = 6
        context_dim         = 256                   # from fusion MLP output
        goal_dim            = 2                     # (x, y)

        # MLP scorer: takes [goal_coord || context] → scalar score
        self.scorer = nn.Sequential(
            nn.Linear(goal_dim + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.to(device)

    def _sample_candidates(
        self,
        past: torch.Tensor,           # (B, N, T, 4)
        map_polylines: torch.Tensor,  # (B, P, L, 2)
        num_candidates: int = 50,
    ) -> torch.Tensor:                # returns (B, N, C, 2)
        """
        Sample C goal candidates per agent.

        Sources (combined up to num_candidates):
            1. Neighbor last positions + Gaussian noise (std=2.0m)
            2. Map polyline endpoints (first and last point)
            3. Random Gaussian around ego's last position
        """
        B, N, T, _ = past.shape                                 # unpack
        device = past.device

        all_candidates = []

        # Source 1: neighbor endpoints with noise
        last_pos = past[:, :, -1, :2]                          # (B, N, 2)
        for _ in range(10):
            noise = torch.randn_like(last_pos) * 2.0            # (B, N, 2)
            # For each agent, use a different neighbor's position
            perm = torch.randperm(N, device=device)
            neighbor_pos = last_pos[:, perm, :]                 # (B, N, 2)
            all_candidates.append(neighbor_pos + noise)         # (B, N, 2)

        # Source 2: map polyline endpoints
        if map_polylines is not None and map_polylines.shape[1] > 0:
            P = map_polylines.shape[1]
            poly_starts = map_polylines[:, :, 0, :]             # (B, P, 2)
            poly_ends   = map_polylines[:, :, -1, :]            # (B, P, 2)

            # Expand polyline endpoints to per-agent candidates
            # Each agent gets all polyline endpoints (use subset if too many)
            max_poly = min(P, 10)
            for p_idx in range(max_poly):
                # Expand (B, 2) → (B, N, 2)
                ep_start = poly_starts[:, p_idx, :].unsqueeze(1).expand(B, N, 2)
                ep_end   = poly_ends[:, p_idx, :].unsqueeze(1).expand(B, N, 2)
                all_candidates.append(ep_start)
                all_candidates.append(ep_end)

        # Source 3: random Gaussian around each agent's last position
        ego_last = past[:, :, -1, :2]                          # (B, N, 2)
        for _ in range(20):
            rand = ego_last + torch.randn_like(ego_last) * 3.0  # (B, N, 2)
            all_candidates.append(rand)

        # Stack all candidates: list of (B, N, 2) → (B, N, C_total, 2)
        candidates = torch.stack(all_candidates, dim=2)         # (B, N, C_total, 2)

        # Subsample to num_candidates
        C_total = candidates.shape[2]
        if C_total > num_candidates:
            idx = torch.randperm(C_total, device=device)[:num_candidates]
            candidates = candidates[:, :, idx, :]               # (B, N, C, 2)
        elif C_total < num_candidates:
            # Pad by repeating with noise
            deficit = num_candidates - C_total
            pad = candidates[:, :, :deficit, :] + torch.randn(
                B, N, deficit, 2, device=device
            ) * 1.0
            candidates = torch.cat([candidates, pad], dim=2)    # (B, N, C, 2)

        return candidates                                        # (B, N, C, 2)

    def forward(
        self,
        context: torch.Tensor,          # (B, N, 256)
        past: torch.Tensor,             # (B, N, T, 4)
        map_polylines: torch.Tensor,    # (B, P, L, 2) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context       : (B, N, 256) — fused context vector per agent
            past          : (B, N, T, 4) — agent histories for candidate sampling
            map_polylines : (B, P, L, 2) — map data for goal sampling

        Returns:
            goal_coords    : (B, N, K, 2) — top-K goal coordinates
            goal_log_probs : (B, N, K)    — log-probabilities (sum to ~1 over K)
        """
        B, N, C_dim = context.shape                             # unpack

        # 1. Sample candidate goals
        candidates = self._sample_candidates(past, map_polylines, num_candidates=50)
        # candidates: (B, N, C, 2)
        C = candidates.shape[2]

        # 2. Score each candidate
        # Expand context to match candidates: (B, N, 256) → (B, N, C, 256)
        ctx_exp = context.unsqueeze(2).expand(B, N, C, C_dim)  # (B, N, C, 256)

        # Concatenate goal and context
        scorer_input = torch.cat([candidates, ctx_exp], dim=-1)  # (B, N, C, 258)
        flat_input = scorer_input.reshape(B * N * C, -1)         # (B*N*C, 258)

        scores = self.scorer(flat_input)                         # (B*N*C, 1)
        scores = scores.reshape(B, N, C)                        # (B, N, C)

        # 3. Softmax to get probability distribution over C candidates
        log_probs_all = F.log_softmax(scores, dim=-1)           # (B, N, C)

        # 4. Select top-K candidates by score
        K = self.num_goals
        _, top_idx = scores.topk(K, dim=-1)                    # (B, N, K)

        # Gather top-K goal coordinates
        top_idx_exp = top_idx.unsqueeze(-1).expand(B, N, K, 2)  # (B, N, K, 2)
        goal_coords = candidates.gather(2, top_idx_exp)          # (B, N, K, 2)

        # Gather top-K log-probs
        goal_log_probs = log_probs_all.gather(2, top_idx)        # (B, N, K)

        return goal_coords, goal_log_probs