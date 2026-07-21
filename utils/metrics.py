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


def _safe_masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of `values[mask]`, returning 0.0 when no agents are valid.

    Plain `values[mask].mean()` returns NaN on an all-False mask (e.g. an
    empty/fallback scene), which then silently poisons epoch averages.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return 0.0
    return float(values[mask].mean())


def _temporal_mean(l2: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
    """Mean over the time axis (-1), excluding padded steps when valid_mask
    (..., T) is given. l2 has shape (..., T)."""
    if valid_mask is None:
        return l2.mean(axis=-1)
    vm = valid_mask.astype(np.float32)
    # broadcast vm (B, N, T) against l2 (B, N, K, T) if a K axis is present
    if l2.ndim == vm.ndim + 1:
        vm = vm[:, :, np.newaxis, :]
    return (l2 * vm).sum(axis=-1) / (vm.sum(axis=-1) + 1e-8)


def _last_valid_index(valid_mask: np.ndarray, T: int) -> np.ndarray:
    """Index of each agent's last real future step from valid_mask (B, N, T).
    Returns (B, N) int array; falls back to T-1 when a row is all padding."""
    steps = np.arange(T, dtype=np.float32)
    idx = (valid_mask.astype(np.float32) * steps).argmax(axis=-1)  # (B, N)
    return idx.astype(np.intp)


def _final_pred_gt(predictions: np.ndarray, ground_truth: np.ndarray,
                   valid_mask: np.ndarray = None):
    """Return (pred_final (B,N,K,2), gt_final (B,N,2)) at each agent's final
    step — the last *valid* step when valid_mask is given, else the tensor end."""
    if valid_mask is None:
        return predictions[:, :, :, -1, :], ground_truth[:, :, -1, :]
    B, N, K, T, _ = predictions.shape
    idx = _last_valid_index(valid_mask, T)                          # (B, N)
    pi = np.broadcast_to(idx[:, :, None, None, None], (B, N, K, 1, 2))
    pred_final = np.take_along_axis(predictions, pi, axis=3).squeeze(3)  # (B,N,K,2)
    gi = np.broadcast_to(idx[:, :, None, None], (B, N, 1, 2))
    gt_final = np.take_along_axis(ground_truth, gi, axis=2).squeeze(2)   # (B,N,2)
    return pred_final, gt_final


def compute_min_ade(
    predictions: Array,  # (B, N, K, T, 2)
    ground_truth: Array, # (B, N, T, 2)
    mask: Array,         # (B, N) bool
    valid_mask: Array = None,  # (B, N, T) bool — real vs padded future steps
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
    vm = _to_numpy(valid_mask) if valid_mask is not None else None

    gt_exp = ground_truth[:, :, np.newaxis, :, :]  # (B, N, 1, T, 2)
    l2 = np.linalg.norm(predictions - gt_exp, axis=-1)  # (B, N, K, T)
    ade_per_k = _temporal_mean(l2, vm)                   # (B, N, K)
    min_ade = ade_per_k.min(axis=-1)                     # (B, N)

    return _safe_masked_mean(min_ade, mask)


def compute_min_fde(
    predictions: Array,  # (B, N, K, T, 2)
    ground_truth: Array, # (B, N, T, 2)
    mask: Array,         # (B, N) bool
    valid_mask: Array = None,  # (B, N, T) bool
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
    vm = _to_numpy(valid_mask) if valid_mask is not None else None

    pred_final, gt_final = _final_pred_gt(predictions, ground_truth, vm)
    gt_exp     = gt_final[:, :, np.newaxis, :]           # (B, N, 1, 2)

    fde_per_k = np.linalg.norm(pred_final - gt_exp, axis=-1)  # (B, N, K)
    min_fde   = fde_per_k.min(axis=-1)                         # (B, N)

    return _safe_masked_mean(min_fde, mask)


def compute_ade(
    predictions: Array,    # (B, N, K, T, 2)
    probabilities: Array,  # (B, N, K)
    ground_truth: Array,   # (B, N, T, 2)
    mask: Array,           # (B, N) bool
    valid_mask: Array = None,  # (B, N, T) bool
) -> float:
    """
    Most-Probable Average Displacement Error (pi-ADE).
    Calculates ADE of the single trajectory assigned the highest probability.
    """
    predictions   = _to_numpy(predictions)
    probabilities = _to_numpy(probabilities)
    ground_truth  = _to_numpy(ground_truth)
    mask          = _to_numpy(mask).astype(bool)
    vm = _to_numpy(valid_mask) if valid_mask is not None else None

    B, N, K, T, _ = predictions.shape
    best_idx = np.argmax(probabilities, axis=-1)  # (B, N)
    
    # Gather the best prediction
    best_idx_exp = best_idx[:, :, np.newaxis, np.newaxis, np.newaxis]
    best_idx_exp = np.broadcast_to(best_idx_exp, (B, N, 1, T, 2))
    best_pred = np.take_along_axis(predictions, best_idx_exp, axis=2).squeeze(2)  # (B, N, T, 2)

    l2 = np.linalg.norm(best_pred - ground_truth, axis=-1)  # (B, N, T)
    ade = _temporal_mean(l2, vm)                            # (B, N)

    return _safe_masked_mean(ade, mask)


def compute_fde(
    predictions: Array,    # (B, N, K, T, 2)
    probabilities: Array,  # (B, N, K)
    ground_truth: Array,   # (B, N, T, 2)
    mask: Array,           # (B, N) bool
    valid_mask: Array = None,  # (B, N, T) bool
) -> float:
    """
    Most-Probable Final Displacement Error (pi-FDE).
    Calculates FDE of the single trajectory assigned the highest probability.
    """
    predictions   = _to_numpy(predictions)
    probabilities = _to_numpy(probabilities)
    ground_truth  = _to_numpy(ground_truth)
    mask          = _to_numpy(mask).astype(bool)
    vm = _to_numpy(valid_mask) if valid_mask is not None else None

    B, N, K, T, _ = predictions.shape
    best_idx = np.argmax(probabilities, axis=-1)  # (B, N)

    pred_final, gt_final = _final_pred_gt(predictions, ground_truth, vm)  # (B,N,K,2),(B,N,2)
    best_idx_exp = np.broadcast_to(best_idx[:, :, np.newaxis, np.newaxis], (B, N, 1, 2))
    best_final = np.take_along_axis(pred_final, best_idx_exp, axis=2).squeeze(2)  # (B, N, 2)

    fde = np.linalg.norm(best_final - gt_final, axis=-1)   # (B, N)

    return _safe_masked_mean(fde, mask)


def compute_miss_rate(
    predictions: Array,  # (B, N, K, T, 2)
    ground_truth: Array, # (B, N, T, 2)
    mask: Array,         # (B, N) bool
    threshold: float = 2.0,
    valid_mask: Array = None,  # (B, N, T) bool
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
    vm = _to_numpy(valid_mask) if valid_mask is not None else None

    pred_final, gt_final = _final_pred_gt(predictions, ground_truth, vm)
    gt_exp     = gt_final[:, :, np.newaxis, :]

    fde_per_k = np.linalg.norm(pred_final - gt_exp, axis=-1)  # (B, N, K)
    min_fde   = fde_per_k.min(axis=-1)                         # (B, N)

    missed = (min_fde > threshold).astype(float)
    return _safe_masked_mean(missed, mask)


def compute_oracle_fde(
    predictions: Array,  # (B, N, K, T, 2)
    ground_truth: Array, # (B, N, T, 2)
    mask: Array,         # (B, N) bool
    valid_mask: Array = None,  # (B, N, T) bool
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
    vm = _to_numpy(valid_mask) if valid_mask is not None else None

    pred_final, gt_final = _final_pred_gt(predictions, ground_truth, vm)
    gt_exp     = gt_final[:, :, np.newaxis, :]

    fde_per_k  = np.linalg.norm(pred_final - gt_exp, axis=-1)  # (B, N, K)
    oracle_fde = fde_per_k.min(axis=-1)                         # (B, N)

    return _safe_masked_mean(oracle_fde, mask)


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
    if all_scores.size == 0:          # no valid agents in this batch
        return 0.0
    return float(all_scores.mean())


def compute_all_metrics(
    predictions: Array,    # (B, N, K, T, 2)
    probabilities: Array,  # (B, N, K)
    ground_truth: Array,   # (B, N, T, 2)
    mask: Array,           # (B, N) bool
    miss_threshold: float = 2.0,
    valid_mask: Array = None,  # (B, N, T) bool — real vs padded future steps
) -> dict:
    """
    Compute all metrics in one call.

    Returns:
        dict with keys: ade, fde, min_ade, min_fde, miss_rate, oracle_fde, diversity
    """
    return {
        "ade":        compute_ade(predictions, probabilities, ground_truth, mask, valid_mask),
        "fde":        compute_fde(predictions, probabilities, ground_truth, mask, valid_mask),
        "min_ade":    compute_min_ade(predictions, ground_truth, mask, valid_mask),
        "min_fde":    compute_min_fde(predictions, ground_truth, mask, valid_mask),
        "miss_rate":  compute_miss_rate(predictions, ground_truth, mask, miss_threshold, valid_mask),
        "oracle_fde": compute_oracle_fde(predictions, ground_truth, mask, valid_mask),
        "diversity":  compute_diversity(predictions, mask),
    }