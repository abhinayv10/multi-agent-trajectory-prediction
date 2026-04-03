"""
utils/metrics.py

Purpose:
    Evaluation metrics for multi-modal trajectory prediction.
    All functions operate on numpy arrays for compatibility with
    both training loops and standalone evaluation scripts.

Inputs / Outputs documented per function.

Public interface:
    compute_ade(predictions, probabilities, ground_truth, mask) -> float
    compute_fde(predictions, probabilities, ground_truth, mask) -> float
    compute_min_ade(predictions, ground_truth, mask) -> float
    compute_min_fde(predictions, ground_truth, mask) -> float
    compute_miss_rate(predictions, ground_truth, mask, threshold) -> float
    compute_oracle_fde(predictions, ground_truth, mask) -> float
    compute_diversity(predictions, mask) -> float
    compute_all_metrics(predictions, probabilities, ground_truth, mask) -> dict
"""

import numpy as np
import torch
from typing import Union

Array = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: Array) -> np.ndarray:
    """Convert tensor or array to numpy float32."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.array(x, dtype=np.float32)


def compute_min_ade(
    predictions: Array,  # (B, N, K, T, 2)
    ground_truth: Array, # (B, N, T, 2)
    mask: Array,         # (B, N) bool
) -> float:
    """
    Minimum Average Displacement Error: mean L2 distance over all timesteps,
    minimized over K predictions per agent.

    Returns:
        ADE in meters (float).
    """
    predictions  = _to_numpy(predictions)   # (B, N, K, T, 2)
    ground_truth = _to_numpy(ground_truth)  # (B, N, T, 2)
    mask         = _to_numpy(mask).astype(bool)  # (B, N)

    gt_exp = ground_truth[:, :, np.newaxis, :, :]  # (B, N, 1, T, 2)
    l2 = np.linalg.norm(predictions - gt_exp, axis=-1)  # (B, N, K, T)
    ade_per_k = l2.mean(axis=-1)                         # (B, N, K)
    min_ade = ade_per_k.min(axis=-1)                     # (B, N)

    return float(min_ade[mask].mean())


def compute_min_fde(
    predictions: Array,  # (B, N, K, T, 2)
    ground_truth: Array, # (B, N, T, 2)
    mask: Array,         # (B, N) bool
) -> float:
    """
    Minimum Final Displacement Error: L2 distance at the last timestep,
    minimized over K predictions per agent.

    Returns:
        FDE in meters (float).
    """
    predictions  = _to_numpy(predictions)
    ground_truth = _to_numpy(ground_truth)
    mask         = _to_numpy(mask).astype(bool)

    pred_final = predictions[:, :, :, -1, :]            # (B, N, K, 2)
    gt_final   = ground_truth[:, :, -1, :]              # (B, N, 2)
    gt_exp     = gt_final[:, :, np.newaxis, :]           # (B, N, 1, 2)

    fde_per_k = np.linalg.norm(pred_final - gt_exp, axis=-1)  # (B, N, K)
    min_fde   = fde_per_k.min(axis=-1)                         # (B, N)

    return float(min_fde[mask].mean())


def compute_ade(
    predictions: Array,    # (B, N, K, T, 2)
    probabilities: Array,  # (B, N, K)
    ground_truth: Array,   # (B, N, T, 2)
    mask: Array,           # (B, N) bool
) -> float:
    """
    Most-Probable Average Displacement Error (pi-ADE).
    Calculates ADE of the single trajectory assigned the highest probability.
    """
    predictions   = _to_numpy(predictions)
    probabilities = _to_numpy(probabilities)
    ground_truth  = _to_numpy(ground_truth)
    mask          = _to_numpy(mask).astype(bool)

    B, N, K, T, _ = predictions.shape
    best_idx = np.argmax(probabilities, axis=-1)  # (B, N)
    
    # Gather the best prediction
    best_idx_exp = best_idx[:, :, np.newaxis, np.newaxis, np.newaxis]
    best_idx_exp = np.broadcast_to(best_idx_exp, (B, N, 1, T, 2))
    best_pred = np.take_along_axis(predictions, best_idx_exp, axis=2).squeeze(2)  # (B, N, T, 2)

    l2 = np.linalg.norm(best_pred - ground_truth, axis=-1)  # (B, N, T)
    ade = l2.mean(axis=-1)                                  # (B, N)
    
    return float(ade[mask].mean())


def compute_fde(
    predictions: Array,    # (B, N, K, T, 2)
    probabilities: Array,  # (B, N, K)
    ground_truth: Array,   # (B, N, T, 2)
    mask: Array,           # (B, N) bool
) -> float:
    """
    Most-Probable Final Displacement Error (pi-FDE).
    Calculates FDE of the single trajectory assigned the highest probability.
    """
    predictions   = _to_numpy(predictions)
    probabilities = _to_numpy(probabilities)
    ground_truth  = _to_numpy(ground_truth)
    mask          = _to_numpy(mask).astype(bool)

    B, N, K, T, _ = predictions.shape
    best_idx = np.argmax(probabilities, axis=-1)  # (B, N)
    
    best_idx_exp = best_idx[:, :, np.newaxis, np.newaxis]  # (B, N, 1, 1)
    best_idx_exp = np.broadcast_to(best_idx_exp, (B, N, 1, 2))
    pred_final = predictions[:, :, :, -1, :]               # (B, N, K, 2)
    best_final = np.take_along_axis(pred_final, best_idx_exp, axis=2).squeeze(2)  # (B, N, 2)

    gt_final = ground_truth[:, :, -1, :]                   # (B, N, 2)
    fde = np.linalg.norm(best_final - gt_final, axis=-1)   # (B, N)

    return float(fde[mask].mean())


def compute_miss_rate(
    predictions: Array,  # (B, N, K, T, 2)
    ground_truth: Array, # (B, N, T, 2)
    mask: Array,         # (B, N) bool
    threshold: float = 2.0,
) -> float:
    """
    Miss Rate: fraction of agents where minFDE@K > threshold meters.

    Args:
        threshold : distance threshold in meters (default 2.0)

    Returns:
        Miss rate in [0, 1] (float).
    """
    predictions  = _to_numpy(predictions)
    ground_truth = _to_numpy(ground_truth)
    mask         = _to_numpy(mask).astype(bool)

    pred_final = predictions[:, :, :, -1, :]
    gt_final   = ground_truth[:, :, -1, :]
    gt_exp     = gt_final[:, :, np.newaxis, :]

    fde_per_k = np.linalg.norm(pred_final - gt_exp, axis=-1)  # (B, N, K)
    min_fde   = fde_per_k.min(axis=-1)                         # (B, N)

    missed = (min_fde > threshold).astype(float)
    return float(missed[mask].mean())


def compute_oracle_fde(
    predictions: Array,  # (B, N, K, T, 2)
    ground_truth: Array, # (B, N, T, 2)
    mask: Array,         # (B, N) bool
) -> float:
    """
    Oracle FDE: the FDE of the prediction closest to the ground truth
    at the final timestep. Represents upper-bound model capability.

    Returns:
        Oracle FDE in meters (float).
    """
    predictions  = _to_numpy(predictions)
    ground_truth = _to_numpy(ground_truth)
    mask         = _to_numpy(mask).astype(bool)

    pred_final = predictions[:, :, :, -1, :]
    gt_final   = ground_truth[:, :, -1, :]
    gt_exp     = gt_final[:, :, np.newaxis, :]

    fde_per_k  = np.linalg.norm(pred_final - gt_exp, axis=-1)  # (B, N, K)
    oracle_fde = fde_per_k.min(axis=-1)                         # (B, N)

    return float(oracle_fde[mask].mean())


def compute_diversity(
    predictions: Array,  # (B, N, K, T, 2)
    mask: Array,         # (B, N) bool
) -> float:
    """
    Diversity Score: mean pairwise L2 distance between the K predictions
    per agent, averaged over the dataset. Higher = more diverse.

    Returns:
        Diversity score in meters (float).
    """
    predictions = _to_numpy(predictions)   # (B, N, K, T, 2)
    mask        = _to_numpy(mask).astype(bool)

    B, N, K, T, _ = predictions.shape
    flat = predictions.reshape(B, N, K, -1)  # (B, N, K, T*2)

    # pairwise distances
    scores = []
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            diff = flat[:, :, k1, :] - flat[:, :, k2, :]  # (B, N, T*2)
            dist = np.linalg.norm(diff, axis=-1)            # (B, N)
            scores.append(dist[mask])

    if len(scores) == 0:
        return 0.0

    all_scores = np.concatenate(scores)
    return float(all_scores.mean())


def compute_all_metrics(
    predictions: Array,    # (B, N, K, T, 2)
    probabilities: Array,  # (B, N, K)
    ground_truth: Array,   # (B, N, T, 2)
    mask: Array,           # (B, N) bool
    miss_threshold: float = 2.0,
) -> dict:
    """
    Compute all metrics in one call.

    Returns:
        dict with keys: ade, fde, min_ade, min_fde, miss_rate, oracle_fde, diversity
    """
    return {
        "ade":        compute_ade(predictions, probabilities, ground_truth, mask),
        "fde":        compute_fde(predictions, probabilities, ground_truth, mask),
        "min_ade":    compute_min_ade(predictions, ground_truth, mask),
        "min_fde":    compute_min_fde(predictions, ground_truth, mask),
        "miss_rate":  compute_miss_rate(predictions, ground_truth, mask, miss_threshold),
        "oracle_fde": compute_oracle_fde(predictions, ground_truth, mask),
        "diversity":  compute_diversity(predictions, mask),
    }