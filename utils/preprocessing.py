"""
utils/preprocessing.py

Purpose:
    Coordinate frame normalization for multi-agent trajectory data.
    All model inputs must be in the ego-agent's local frame.

Inputs:
    past          : (N, T_past, 4)  — (x, y, vx, vy) in world frame
    future        : (N, T_future, 2) — (x, y) in world frame
    map_polylines : (P, L, 2)       — (x, y) in world frame

Outputs:
    Transformed versions of all three tensors in agent-centric frame,
    plus the (origin, angle) tuple needed for inversion.

Public interface:
    normalize_to_agent_frame(past, future, map_polylines)
        -> (past_norm, future_norm, poly_norm, origin, angle)
    denormalize_from_agent_frame(positions, origin, angle)
        -> positions in world frame
"""

import numpy as np
import torch
from typing import Tuple, Union

Tensor = Union[np.ndarray, torch.Tensor]


def _get_rotation_matrix(angle: float) -> np.ndarray:
    """
    2D rotation matrix for rotating by -angle (to align heading with +x axis).

    Args:
        angle : heading angle in radians

    Returns:
        R : (2, 2) rotation matrix
    """
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)   # (2, 2)
    return R


def normalize_to_agent_frame(
    past: np.ndarray,
    future: np.ndarray,
    map_polylines: np.ndarray,
    ego_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Transforms all coordinates into the ego agent's local frame at its
    last observed timestep.

    Convention:
        - Origin   : ego agent's position at t=-1 (last observed step)
        - Heading  : ego agent's velocity direction at t=-1 aligns with +x

    Args:
        past          : (N, T_past, 4)   — world frame (x, y, vx, vy)
        future        : (N, T_future, 2) — world frame (x, y)
        map_polylines : (P, L, 2)        — world frame (x, y)
        ego_idx       : index of the ego agent (default 0)

    Returns:
        past_norm   : (N, T_past, 4)   in agent-centric frame
        future_norm : (N, T_future, 2) in agent-centric frame
        poly_norm   : (P, L, 2)        in agent-centric frame
        origin      : (2,)  world-frame origin used for inversion
        angle       : float  heading angle used for inversion
    """
    past = np.array(past, dtype=np.float32)         # (N, T_past, 4)
    future = np.array(future, dtype=np.float32)     # (N, T_future, 2)
    map_polylines = np.array(map_polylines, dtype=np.float32)  # (P, L, 2)

    # --- determine origin and heading ---
    origin = past[ego_idx, -1, :2].copy()           # (2,)  last position of ego
    vx, vy = past[ego_idx, -1, 2], past[ego_idx, -1, 3]
    angle = float(np.arctan2(vy, vx)) if (abs(vx) + abs(vy)) > 1e-6 else 0.0

    R = _get_rotation_matrix(angle)                 # (2, 2)

    # --- normalize past positions and velocities ---
    past_norm = past.copy()
    past_norm[:, :, :2] = ((past[:, :, :2] - origin) @ R.T)   # positions
    past_norm[:, :, 2:] = (past[:, :, 2:] @ R.T)              # velocities

    # --- normalize future positions ---
    future_norm = (future - origin) @ R.T           # (N, T_future, 2)

    # --- normalize map polylines ---
    P, L, _ = map_polylines.shape
    flat = map_polylines.reshape(-1, 2)             # (P*L, 2)
    flat_norm = (flat - origin) @ R.T               # (P*L, 2)
    poly_norm = flat_norm.reshape(P, L, 2)          # (P, L, 2)

    return past_norm, future_norm, poly_norm, origin, angle


def denormalize_from_agent_frame(
    positions: Union[np.ndarray, torch.Tensor],
    origin: np.ndarray,
    angle: float,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Inverts normalize_to_agent_frame for predicted position outputs.

    Args:
        positions : (..., 2) predicted positions in agent-centric frame
        origin    : (2,) world-frame origin
        angle     : float heading angle from normalization

    Returns:
        positions in world frame, same shape and type as input
    """
    is_tensor = isinstance(positions, torch.Tensor)
    if is_tensor:
        device = positions.device
        positions = positions.detach().cpu().numpy()

    # Inverse rotation: rotate by +angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R_inv = np.array([[cos_a, -sin_a],
                      [sin_a,  cos_a]], dtype=np.float32)       # (2, 2)

    shape = positions.shape
    flat = positions.reshape(-1, 2)                             # (*, 2)
    world = (flat @ R_inv.T) + origin                          # (*, 2)
    result = world.reshape(shape)                               # (..., 2)

    if is_tensor:
        result = torch.from_numpy(result).to(device)

    return result