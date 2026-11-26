"""
models/goal_predictor.py

Purpose:
    Predicts probable destination (goal) points for each agent, TNT-style:
    a fixed set of kinematic + map candidates is scored by a network, each
    candidate gets a learned offset, and the top-K refined goals are kept.

    All goal coordinates are relative to the agent's last observed position.

Inputs:
    context       : (B, N, 256)  — fused context vector
    last_vel      : (B, N, 2)    — last observed velocity (scene frame)
    map_polylines : (B, P, L, 2) — lane polylines (scene frame), may have P = 0
    agent_pos     : (B, N, 2)    — last observed positions (scene frame)
    polyline_mask : (B, P) bool

Outputs (dict):
    goal_coords     : (B, N, K, 2) — top-K refined goals (relative)
    goal_log_probs  : (B, N, K)    — log-probabilities of the top-K goals
    candidates      : (B, N, C, 2) — raw candidates (relative)
    refined         : (B, N, C, 2) — candidates + learned offsets
    logits          : (B, N, C)    — candidate scores

Public interface:
    GoalPredictor(config, device) -> Module
    forward(context, last_vel, map_polylines, agent_pos, polyline_mask, k) -> dict
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Kinematic candidate grid around the constant-velocity endpoint.
SPEED_SCALES = (0.5, 0.8, 1.0, 1.2, 1.5)
HEADING_OFFSETS_DEG = (-90, -45, -20, 0, 20, 45, 90)
# Rings (metres) for slow agents whose heading is unreliable.
RING_RADII = (1.0, 3.0, 6.0)
RING_DIRECTIONS = 8
COORD_SCALE = 10.0   # network inputs are divided by this (metres -> ~unit range)


class GoalPredictor(nn.Module):
    """
    Args:
        config : full config dict (reads model.num_goals, dataset.dt,
                 dataset.future_steps)
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device
        m = config["model"]
        self.num_goals = m["num_goals"]
        self.horizon = config["dataset"]["future_steps"] * config["dataset"].get("dt", 0.4)
        context_dim = 256

        self.scorer = nn.Sequential(
            nn.Linear(2 + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(2 + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.to(device)

    def _candidates(self, last_vel, map_polylines, agent_pos, polyline_mask):
        """Deterministic candidate set, relative to each agent: (B, N, C, 2)."""
        B, N, _ = last_vel.shape
        dev, dt = last_vel.device, last_vel.dtype
        speed = last_vel.norm(dim=-1, keepdim=True)                  # (B, N, 1)
        heading = torch.atan2(last_vel[..., 1], last_vel[..., 0]).unsqueeze(-1)

        # Stop + speed x heading grid around the CV endpoint.
        scales = torch.tensor(SPEED_SCALES, device=dev, dtype=dt)
        offs = torch.tensor(HEADING_OFFSETS_DEG, device=dev, dtype=dt) * math.pi / 180
        s_grid, o_grid = torch.meshgrid(scales, offs, indexing="ij")
        s_grid, o_grid = s_grid.reshape(-1), o_grid.reshape(-1)       # (S*H,)
        dist = speed * self.horizon * s_grid                          # (B, N, S*H)
        ang = heading + o_grid
        kin = torch.stack([dist * torch.cos(ang), dist * torch.sin(ang)], -1)
        stop = last_vel.new_zeros(B, N, 1, 2)

        # Rings, aligned with the heading so they rotate with the scene.
        radii = torch.tensor(RING_RADII, device=dev, dtype=dt)
        dirs = torch.arange(RING_DIRECTIONS, device=dev, dtype=dt) * (2 * math.pi / RING_DIRECTIONS)
        r_grid, d_grid = torch.meshgrid(radii, dirs, indexing="ij")
        r_grid, d_grid = r_grid.reshape(-1), d_grid.reshape(-1)
        ring_ang = heading + d_grid
        ring = torch.stack([r_grid * torch.cos(ring_ang), r_grid * torch.sin(ring_ang)], -1)

        parts = [stop, kin, ring.expand(B, N, -1, 2)]

        # Lane endpoints (start and end of each polyline), relative to the agent.
        if map_polylines is not None and map_polylines.shape[1] > 0:
            ends = torch.cat([map_polylines[:, :, 0], map_polylines[:, :, -1]], 1)  # (B, 2P, 2)
            rel = ends.unsqueeze(1) - agent_pos.unsqueeze(2)                         # (B, N, 2P, 2)
            if polyline_mask is not None:
                valid = torch.cat([polyline_mask, polyline_mask], 1)                 # (B, 2P)
                # Padded polylines collapse to the "stop" candidate.
                rel = rel * valid[:, None, :, None].to(rel.dtype)
            parts.append(rel)
        return torch.cat(parts, dim=2)

    def forward(
        self,
        context: torch.Tensor,              # (B, N, 256)
        last_vel: torch.Tensor,             # (B, N, 2)
        map_polylines: torch.Tensor = None, # (B, P, L, 2)
        agent_pos: torch.Tensor = None,     # (B, N, 2)
        polyline_mask: torch.Tensor = None, # (B, P)
        k: int = None,
    ) -> dict:
        B, N, C_dim = context.shape
        K = k or self.num_goals
        if agent_pos is None:
            agent_pos = context.new_zeros(B, N, 2)
        cand = self._candidates(last_vel, map_polylines, agent_pos, polyline_mask)  # (B, N, C, 2)
        C = cand.shape[2]
        if K > C:
            raise ValueError(f"GoalPredictor: asked for {K} goals but only {C} candidates exist")

        ctx_exp = context.unsqueeze(2).expand(B, N, C, C_dim)
        inp = torch.cat([cand / COORD_SCALE, ctx_exp], dim=-1)       # (B, N, C, 2+256)
        logits = self.scorer(inp).squeeze(-1)                         # (B, N, C)
        refined = cand + self.offset_head(inp) * COORD_SCALE * 0.1    # offsets ~ metres

        log_probs_all = F.log_softmax(logits, dim=-1)
        _, top_idx = logits.topk(K, dim=-1)                           # sorted, best first
        goal_coords = refined.gather(2, top_idx.unsqueeze(-1).expand(B, N, K, 2))
        goal_log_probs = log_probs_all.gather(2, top_idx)
        return {
            "goal_coords": goal_coords,
            "goal_log_probs": goal_log_probs,
            "candidates": cand,
            "refined": refined,
            "logits": logits,
        }
