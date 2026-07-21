"""
utils/losses.py

Purpose:
    All loss functions for multi-modal trajectory prediction.
    Every function is standalone and independently testable.
    compute_total_loss combines them using weights from config.

Inputs / Outputs documented per function below.

Public interface:
    min_ade_loss(predictions, ground_truth, mask) -> scalar tensor
    min_fde_loss(predictions, ground_truth, mask) -> scalar tensor
    goal_loss(predicted_goals, goal_probs, ground_truth_endpoints) -> scalar tensor
    kl_loss(prior_params, posterior_params, kl_weight) -> scalar tensor
    diversity_loss(predictions) -> scalar tensor
    nll_loss(log_probs, predictions, ground_truth) -> scalar tensor
    compute_total_loss(outputs, batch, config, epoch) -> (total, loss_dict)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def min_ade_loss(
    predictions: torch.Tensor,  # (B, N, K, T, 2) — K candidates
    ground_truth: torch.Tensor, # (B, N, T, 2)
    mask: torch.Tensor,         # (B, N) bool
    valid_mask: torch.Tensor = None,  # (B, N, T) bool — real vs padded future steps
) -> torch.Tensor:
    """
    Minimum Average Displacement Error over K predictions.

    For each agent, picks the best prediction (closest to GT over all steps)
    and averages the per-step L2 distances.

    If valid_mask is given, padded/repeated future timesteps are excluded from
    the per-step average so fabricated padding does not bias the error.

    Returns:
        Scalar loss averaged over valid agents.
    """
    B, N, K, T, _ = predictions.shape                          # unpack dims
    gt_exp = ground_truth.unsqueeze(2).expand_as(predictions)  # (B, N, K, T, 2)

    # per-step L2 distance
    l2 = torch.norm(predictions - gt_exp, dim=-1)              # (B, N, K, T)

    if valid_mask is not None:
        vm = valid_mask.to(l2.dtype).unsqueeze(2)             # (B, N, 1, T)
        ade_per_k = (l2 * vm).sum(dim=-1) / (vm.sum(dim=-1) + 1e-8)  # (B, N, K)
    else:
        ade_per_k = l2.mean(dim=-1)                            # (B, N, K)

    # min over K candidates
    min_ade, _ = ade_per_k.min(dim=-1)                        # (B, N)

    # mask and average
    masked = min_ade * mask.float()                            # (B, N)
    loss = masked.sum() / (mask.float().sum() + 1e-8)
    return loss


def min_fde_loss(
    predictions: torch.Tensor,  # (B, N, K, T, 2)
    ground_truth: torch.Tensor, # (B, N, T, 2)
    mask: torch.Tensor,         # (B, N) bool
    valid_mask: torch.Tensor = None,  # (B, N, T) bool — real vs padded future steps
) -> torch.Tensor:
    """
    Minimum Final Displacement Error over K predictions.

    If valid_mask is given, the "final" step is each agent's last REAL future
    timestep rather than the padded tensor end.

    Returns:
        Scalar loss averaged over valid agents.
    """
    B, N, K, T, _ = predictions.shape
    if valid_mask is not None:
        # index of the last valid (True) timestep per agent; arange makes argmax
        # pick the largest index that is still valid. Padded-only agents get 0,
        # but they are removed by the agent mask below anyway.
        steps = torch.arange(T, device=predictions.device).float()   # (T,)
        last_idx = (valid_mask.to(steps.dtype) * steps).argmax(dim=-1)  # (B, N)
        pidx = last_idx.view(B, N, 1, 1, 1).expand(B, N, K, 1, 2)
        pred_final = predictions.gather(3, pidx).squeeze(3)          # (B, N, K, 2)
        gidx = last_idx.view(B, N, 1, 1).expand(B, N, 1, 2)
        gt_final = ground_truth.gather(2, gidx).squeeze(2)           # (B, N, 2)
    else:
        pred_final = predictions[:, :, :, -1, :]                  # (B, N, K, 2)
        gt_final   = ground_truth[:, :, -1, :]                    # (B, N, 2)
    gt_exp     = gt_final.unsqueeze(2).expand_as(pred_final)   # (B, N, K, 2)

    fde_per_k = torch.norm(pred_final - gt_exp, dim=-1)        # (B, N, K)
    min_fde, _ = fde_per_k.min(dim=-1)                        # (B, N)

    masked = min_fde * mask.float()
    loss = masked.sum() / (mask.float().sum() + 1e-8)
    return loss


def goal_loss(
    predicted_goals: torch.Tensor,  # (B, N, K, 2) — top-K goal coords
    goal_log_probs: torch.Tensor,   # (B, N, K)    — log-probabilities
    ground_truth_endpoints: torch.Tensor,  # (B, N, 2)
    mask: torch.Tensor,             # (B, N) bool
) -> torch.Tensor:
    """
    Encourages the highest-probability goal to be close to the true endpoint.

    Uses the probability-weighted sum of L2 distances across goals as a
    soft version of the loss (differentiable through the probabilities).

    Returns:
        Scalar loss averaged over valid agents.
    """
    # probability-weighted distance
    probs = goal_log_probs.exp()                                # (B, N, K)
    gt_exp = ground_truth_endpoints.unsqueeze(2).expand(
        *ground_truth_endpoints.shape[:2], predicted_goals.shape[2], 2
    )                                                           # (B, N, K, 2)

    dist = torch.norm(predicted_goals - gt_exp, dim=-1)        # (B, N, K)
    weighted = (probs * dist).sum(dim=-1)                       # (B, N)

    masked = weighted * mask.float()
    loss = masked.sum() / (mask.float().sum() + 1e-8)
    return loss


def kl_loss(
    prior_params: Tuple[torch.Tensor, torch.Tensor],      # (mean, log_var) each (B*N, D)
    posterior_params: Tuple[torch.Tensor, torch.Tensor],  # (mean, log_var) each (B*N, D)
    kl_weight: float,
    mask: torch.Tensor,  # (B, N) bool, flattened to (B*N,)
) -> torch.Tensor:
    """
    KL divergence KL(q(z|c,future) || p(z|c)) with annealing weight.

    Args:
        prior_params     : (mu_p, logvar_p) from prior network
        posterior_params : (mu_q, logvar_q) from posterior network
        kl_weight        : annealing scalar in [0, 1]
        mask             : (B*N,) bool

    Returns:
        Scalar KL loss.
    """
    mu_p, logvar_p = prior_params        # (B*N, D)
    mu_q, logvar_q = posterior_params    # (B*N, D)

    # KL(N(mu_q, sigma_q) || N(mu_p, sigma_p)) analytically
    kl = 0.5 * (
        logvar_p - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / (logvar_p.exp() + 1e-8)
        - 1.0
    )                                    # (B*N, D)
    kl_per_agent = kl.sum(dim=-1)        # (B*N,)

    flat_mask = mask.reshape(-1).float() # (B*N,)
    loss = (kl_per_agent * flat_mask).sum() / (flat_mask.sum() + 1e-8)
    return kl_weight * loss


# Saturating cap on the per-agent mean pairwise distance. Without it the
# diversity reward (-distance) is unbounded below and can dominate ADE/FDE,
# pushing modes infinitely apart and destabilizing training.
DIVERSITY_CLAMP = 10.0


def diversity_loss(
    predictions: torch.Tensor,  # (B, N, K, T, 2)
    mask: torch.Tensor,         # (B, N) bool
    clamp: float = DIVERSITY_CLAMP,
) -> torch.Tensor:
    """
    Penalizes low diversity by negating mean pairwise L2 distance between
    the K predicted trajectories. Encourages spread.

    The mean pairwise distance is clamped to `clamp` so the reward saturates
    and cannot be driven to -inf.

    Returns:
        Scalar loss (negative diversity, to be minimized).
    """
    B, N, K, T, _ = predictions.shape
    flat = predictions.reshape(B, N, K, -1)             # (B, N, K, T*2)

    # pairwise distances between all K predictions
    # expand for broadcasting
    p1 = flat.unsqueeze(3)                              # (B, N, K, 1, T*2)
    p2 = flat.unsqueeze(2)                              # (B, N, 1, K, T*2)
    pairwise = torch.norm(p1 - p2, dim=-1)              # (B, N, K, K)

    # upper triangle only (exclude diagonal)
    triu_mask = torch.triu(torch.ones(K, K, device=predictions.device), diagonal=1)  # (K, K)
    div = (pairwise * triu_mask).sum(dim=(-1, -2))      # (B, N)
    num_pairs = K * (K - 1) / 2.0
    div = div / (num_pairs + 1e-8)                      # mean pairwise dist
    div = div.clamp(max=clamp)                           # saturate the reward

    # negate: we want to maximize diversity = minimize negative diversity
    neg_div = -div * mask.float()
    loss = neg_div.sum() / (mask.float().sum() + 1e-8)
    return loss


def nll_loss(
    log_probs: torch.Tensor,    # (B, N, K) — log probabilities of K trajectories
    predictions: torch.Tensor,  # (B, N, K, T, 2)
    ground_truth: torch.Tensor, # (B, N, T, 2)
    mask: torch.Tensor,         # (B, N) bool
    valid_mask: torch.Tensor = None,  # (B, N, T) bool — real vs padded future steps
) -> torch.Tensor:
    """
    Negative log-likelihood of the ground truth under the predicted mixture.

    Uses a soft assignment: weight each trajectory's log-likelihood by its
    predicted probability. This is a FIXED-variance (sigma^2 = 1) Gaussian
    surrogate — the 2*pi normalization constant is dropped, so the absolute
    value is not a calibrated NLL; it is used only as a relative training
    objective. `log_probs` must be normalized (log-softmax) for the mixture
    to be valid; the caller guarantees this.

    Returns:
        Scalar NLL loss averaged over valid agents.
    """
    B, N, K, T, _ = predictions.shape
    # log_probs must align with the candidate axis of predictions.
    assert log_probs.shape[-1] == K, (
        f"nll_loss: log_probs K={log_probs.shape[-1]} does not match "
        f"predictions K={K}; pass a matched (candidates, log_probs) pair."
    )
    gt_exp = ground_truth.unsqueeze(2).expand_as(predictions)  # (B, N, K, T, 2)

    # Gaussian log-likelihood (variance=1 for simplicity)
    sq = (predictions - gt_exp).pow(2).sum(dim=-1)             # (B, N, K, T)
    if valid_mask is not None:
        sq = sq * valid_mask.to(sq.dtype).unsqueeze(2)        # zero padded steps
    log_likelihood = -0.5 * sq.sum(dim=-1)                     # (B, N, K)

    # log-sum-exp over K: log p(gt) = log sum_k p(k) * p(gt|k)
    log_mixture = torch.logsumexp(log_probs + log_likelihood, dim=-1)         # (B, N)

    masked = -log_mixture * mask.float()
    loss = masked.sum() / (mask.float().sum() + 1e-8)
    return loss


# ---------------------------------------------------------------------------
# KL annealing schedule
# ---------------------------------------------------------------------------

def get_kl_weight(epoch: int, anneal_epochs: int) -> float:
    """
    Linear KL annealing: weight goes from 0 → 1 over anneal_epochs epochs.

    Args:
        epoch        : current epoch (0-indexed — pass epoch-1 for a 1-indexed
                       training loop so the first epoch gets weight 0)
        anneal_epochs: number of epochs to reach full weight

    Returns:
        kl_weight in [0.0, 1.0]
    """
    return min(1.0, max(0, epoch) / max(anneal_epochs, 1))


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def compute_total_loss(
    outputs: dict,
    batch: dict,
    config: dict,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combines all individual losses using weights from config.

    Args:
        outputs : dict from model forward pass with keys:
            predictions      : (B, N, K, T, 2)
            goal_coords      : (B, N, K_g, 2)
            goal_log_probs   : (B, N, K_g)
            prior_params     : (mu, logvar) each (B*N, D)
            posterior_params : (mu, logvar) each (B*N, D)  — None at inference
            log_probs        : (B, N, K)
        batch  : dict from DataLoader with keys: future, mask
        config : full config dict
        epoch  : current epoch for KL annealing

    Returns:
        total_loss : scalar tensor
        loss_dict  : dict of individual float values for logging
    """
    w = config["training"]["loss_weights"]
    anneal_epochs = config["training"]["kl_anneal_epochs"]

    predictions = outputs["predictions"]         # (B, N, K, T, 2)
    
    # Use all candidates for full trajectory training if available
    train_candidates = outputs.get("all_candidates", predictions)
    
    ground_truth = batch["future"]               # (B, N, T, 2)
    mask = batch["mask"]                         # (B, N)

    # Optional per-timestep future validity mask (real vs padded steps)
    valid_mask = batch.get("valid_mask")         # (B, N, T) bool or None

    # Move ground truth to same device as predictions
    ground_truth = ground_truth.to(predictions.device)
    mask = mask.to(predictions.device)
    if valid_mask is not None:
        valid_mask = valid_mask.to(predictions.device)

    loss_dict = {}
    total = torch.tensor(0.0, device=predictions.device)

    # minADE
    l_ade = min_ade_loss(train_candidates, ground_truth, mask, valid_mask)
    loss_dict["min_ade"] = l_ade.item()
    total = total + w["min_ade"] * l_ade

    # minFDE
    l_fde = min_fde_loss(train_candidates, ground_truth, mask, valid_mask)
    loss_dict["min_fde"] = l_fde.item()
    total = total + w["min_fde"] * l_fde

    # Goal loss
    if "goal_coords" in outputs and outputs["goal_coords"] is not None:
        gt_endpoints = ground_truth[:, :, -1, :]             # (B, N, 2)
        l_goal = goal_loss(
            outputs["goal_coords"],
            outputs["goal_log_probs"],
            gt_endpoints,
            mask,
        )
        loss_dict["goal"] = l_goal.item()
        total = total + w["goal"] * l_goal

    # KL divergence (only during training when posterior is available)
    if (outputs.get("posterior_params") is not None and
            outputs.get("prior_params") is not None):
        # epoch is 1-indexed in the training loop; pass epoch-1 so the first
        # epoch starts at weight 0 (true warm-start) rather than 1/anneal_epochs.
        kl_w = get_kl_weight(epoch - 1, anneal_epochs)
        l_kl = kl_loss(
            outputs["prior_params"],
            outputs["posterior_params"],
            kl_weight=kl_w,
            mask=mask,
        )
        loss_dict["kl"] = l_kl.item()
        loss_dict["kl_weight"] = kl_w
        total = total + w["kl"] * l_kl

    # Diversity loss
    l_div = diversity_loss(predictions, mask)
    loss_dict["diversity"] = l_div.item()
    total = total + w["diversity"] * l_div

    # NLL (optional). The candidate set and its log-probs MUST be a matched
    # pair on the K axis, otherwise logsumexp broadcasts incorrectly. Prefer the
    # full candidate pool (all_candidates + all_log_probs); fall back to the
    # top-K pair (predictions + log_probs) when the full pool is unavailable.
    if w.get("nll", 0.0) > 0.0:
        if "all_candidates" in outputs and "all_log_probs" in outputs:
            nll_candidates, nll_log_probs = outputs["all_candidates"], outputs["all_log_probs"]
        elif "log_probs" in outputs:
            nll_candidates, nll_log_probs = predictions, outputs["log_probs"]
        else:
            nll_candidates = nll_log_probs = None

        if nll_log_probs is not None:
            l_nll = nll_loss(
                nll_log_probs,
                nll_candidates,
                ground_truth,
                mask,
                valid_mask,
            )
            loss_dict["nll"] = l_nll.item()
            total = total + w["nll"] * l_nll

    loss_dict["total"] = total.item()
    return total, loss_dict