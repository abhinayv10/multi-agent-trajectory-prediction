"""
utils/visualization.py

Purpose:
    Top-down visualization of past trajectories, ground-truth futures,
    K predicted trajectories, and map polylines.

Inputs:
    past          : (N, T_past, 4) or (N, T_past, 2) — agent history
    future_gt     : (N, T_future, 2) — ground truth future
    predictions   : (N, K, T_future, 2) — K predicted trajectories
    map_polylines : (P, L, 2) — map polylines
    scores        : (N, K) — confidence scores (optional)
    save_path     : str or None — if None, calls plt.show()

Public interface:
    plot_scene(past, future_gt, predictions, map_polylines,
               scores=None, agent_idx=0, save_path=None) -> None
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Union
import os


def plot_scene(
    past: np.ndarray,           # (N, T_past, 4) or (N, T_past, 2)
    future_gt: np.ndarray,      # (N, T_future, 2)
    predictions: np.ndarray,    # (N, K, T_future, 2)
    map_polylines: np.ndarray,  # (P, L, 2)
    scores: Optional[np.ndarray] = None,  # (N, K) confidence scores
    agent_idx: int = 0,
    save_path: Optional[str] = None,
    title: str = "Trajectory Prediction",
) -> None:
    """
    Renders a top-down scene for a single agent with all other agents shown
    as context.

    Visual conventions:
        Past trajectories         : grey dashed lines
        Ground truth future       : solid green line with circular marker at end
        Predicted trajectories    : blue colormap (darker = higher confidence)
        Map polylines             : light grey dashed lines
        Agent start positions     : small black dots

    Args:
        past          : agent history, uses only first 2 dims (x, y)
        future_gt     : ground truth future positions
        predictions   : K predicted trajectories for each agent
        map_polylines : lane polylines
        scores        : per-prediction confidence (used for colormap if provided)
        agent_idx     : which agent to highlight (default 0 = ego)
        save_path     : file path to save figure; if None, calls plt.show()
        title         : figure title
    """
    past          = np.array(past, dtype=np.float32)
    future_gt     = np.array(future_gt, dtype=np.float32)
    predictions   = np.array(predictions, dtype=np.float32)
    map_polylines = np.array(map_polylines, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # --- map polylines ---
    for p in range(map_polylines.shape[0]):
        ax.plot(
            map_polylines[p, :, 0],
            map_polylines[p, :, 1],
            color="lightgrey",
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )

    N = past.shape[0]

    # --- context agents (not ego) ---
    for n in range(N):
        if n == agent_idx:
            continue
        ax.plot(
            past[n, :, 0],
            past[n, :, 1],
            color="lightblue",
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
        ax.scatter(
            past[n, -1, 0],
            past[n, -1, 1],
            color="blue",
            s=20,
            zorder=3,
        )

    # --- ego agent past ---
    ax.plot(
        past[agent_idx, :, 0],
        past[agent_idx, :, 1],
        color="grey",
        linestyle="--",
        linewidth=1.5,
        label="Past",
        zorder=4,
    )
    ax.scatter(
        past[agent_idx, -1, 0],
        past[agent_idx, -1, 1],
        color="black",
        s=50,
        zorder=5,
    )

    # --- ground truth future ---
    ax.plot(
        future_gt[agent_idx, :, 0],
        future_gt[agent_idx, :, 1],
        color="green",
        linewidth=2.0,
        label="Ground Truth",
        zorder=6,
    )
    ax.scatter(
        future_gt[agent_idx, -1, 0],
        future_gt[agent_idx, -1, 1],
        color="green",
        s=80,
        marker="o",
        zorder=7,
    )

    # --- predicted trajectories ---
    K = predictions.shape[1]
    cmap = cm.Blues

    # normalize scores to [0.3, 1.0] for colormap
    if scores is not None:
        s = scores[agent_idx]  # (K,)
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
        alphas = 0.3 + 0.7 * s_norm
    else:
        alphas = np.linspace(0.4, 1.0, K)

    for k in range(K):
        color_val = 0.3 + 0.7 * (k / max(K - 1, 1))
        color = cmap(color_val)
        ax.plot(
            predictions[agent_idx, k, :, 0],
            predictions[agent_idx, k, :, 1],
            color=color,
            linewidth=1.5,
            alpha=float(alphas[k]),
            label="Prediction" if k == 0 else None,
            zorder=8,
        )
        ax.scatter(
            predictions[agent_idx, k, -1, 0],
            predictions[agent_idx, k, -1, 1],
            color=color,
            s=40,
            alpha=float(alphas[k]),
            zorder=9,
        )

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)