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
    goal_loss(goal_candidates, goal_refined, goal_logits, gt_endpoints, mask) -> scalar tensor
    ranking_loss(all_log_probs, candidates, ground_truth, mask) -> scalar tensor
    recon_loss(recon, ground_truth, mask) -> scalar tensor
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
    goal_candidates: torch.Tensor,  # (B, N, C, 2) raw candidates
    goal_refined: torch.Tensor,     # (B, N, C, 2) candidates + predicted offsets
    goal_logits: torch.Tensor,      # (B, N, C)
    gt_endpoints: torch.Tensor,     # (B, N, 2) last valid ground-truth position
    mask: torch.Tensor,             # (B, N) bool
) -> torch.Tensor:
    """
    TNT-style goal loss: cross-entropy towards the candidate nearest to the
    true endpoint, plus smooth-L1 regression of that candidate's offset.

    Returns:
        Scalar loss averaged over valid agents.
    """
    d = torch.norm(goal_candidates - gt_endpoints.unsqueeze(2), dim=-1)    # (B, N, C)
    target = d.argmin(dim=-1)                                              # (B, N)
    ce = F.cross_entropy(goal_logits.flatten(0, 1), target.flatten(), reduction="none")
    ce = ce.view_as(target)
    idx = target[..., None, None].expand(-1, -1, 1, 2)
    best = goal_refined.gather(2, idx).squeeze(2)                          # (B, N, 2)
    reg = F.smooth_l1_loss(best, gt_endpoints, reduction="none").sum(-1)   # (B, N)
    m = mask.float()
    return ((ce + reg) * m).sum() / (m.sum() + 1e-8)


def ranking_loss(
    all_log_probs: torch.Tensor,  # (B, N, C) log-softmax scores of the candidates
    candidates: torch.Tensor,     # (B, N, C, T, 2)
    ground_truth: torch.Tensor,   # (B, N, T, 2)
    mask: torch.Tensor,           # (B, N) bool
    valid_mask: torch.Tensor = None,
    temperature: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy that teaches the ranker to put the candidates with the
    lowest ADE first. This is what makes minADE@1 / minFDE@1 meaningful.

    temperature > 0 uses soft targets softmax(-ADE / temperature): among many
    similar samples the single argmin is close to a random label, while the
    soft target says "closer is better" consistently. temperature = 0 is the
    hard argmin target.
    """
    l2 = torch.norm(candidates - ground_truth.unsqueeze(2), dim=-1)        # (B, N, C, T)
    if valid_mask is not None:
        vm = valid_mask.to(l2.dtype).unsqueeze(2)
        ade = (l2 * vm).sum(-1) / (vm.sum(-1) + 1e-8)
    else:
        ade = l2.mean(-1)
    if temperature > 0:
        target = F.softmax(-ade.detach() / temperature, dim=-1)            # (B, N, C)
        nll = -(target * all_log_probs).sum(-1)                            # (B, N)
    else:
        target = ade.argmin(dim=-1).detach()                               # (B, N)
        nll = -all_log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)   # (B, N)
    m = mask.float()
    return (nll * m).sum() / (m.sum() + 1e-8)


def recon_loss(
    recon: torch.Tensor,          # (B, N, T, 2) guided (posterior) reconstruction
    ground_truth: torch.Tensor,   # (B, N, T, 2)
    mask: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """ADE of the guided candidate (trains the CVAE posterior + decoder)."""
    return min_ade_loss(recon.unsqueeze(2), ground_truth, mask, valid_mask)


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
    Combines all losses using config["training"]["loss_weights"]. Terms whose
    inputs a model does not produce (e.g. goals for the LSTM baseline, KL with
    the CVAE ablated) are skipped.

    Losses are averaged over *target* agents (batch["target_mask"]); other
    valid agents only serve as social context.

    Returns:
        total_loss : scalar tensor
        loss_dict  : dict of individual float values for logging
    """
    w = config["training"]["loss_weights"]
    anneal_epochs = config["training"]["kl_anneal_epochs"]

    predictions = outputs["predictions"]                                   # (B, N, K, T, 2)
    dev = predictions.device
    candidates = outputs.get("all_candidates")
    if candidates is None:
        candidates = predictions
    ground_truth = batch["future"].to(dev)
    mask = batch.get("target_mask", batch["mask"]).to(dev)
    valid_mask = batch.get("valid_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.to(dev)

    loss_dict = {}
    total = torch.zeros((), device=dev)

    def add(name, value, weight):
        nonlocal total
        loss_dict[name] = value.item()
        total = total + weight * value

    # Winner-takes-all over the (prior) candidate pool.
    add("min_ade", min_ade_loss(candidates, ground_truth, mask, valid_mask), w["min_ade"])
    add("min_fde", min_fde_loss(candidates, ground_truth, mask, valid_mask), w["min_fde"])

    if outputs.get("recon") is not None and w.get("recon", 0.0) > 0:
        add("recon", recon_loss(outputs["recon"], ground_truth, mask, valid_mask), w["recon"])

    if outputs.get("goal_logits") is not None and w.get("goal", 0.0) > 0:
        from models.trajectory import last_valid_future
        gt_end = last_valid_future(ground_truth, valid_mask)
        add("goal", goal_loss(outputs["goal_candidates"], outputs["goal_refined"],
                              outputs["goal_logits"], gt_end, mask), w["goal"])

    if outputs.get("posterior_params") is not None and outputs.get("prior_params") is not None:
        # epoch is 1-indexed; epoch-1 makes the first epoch a true warm start.
        kl_w = get_kl_weight(epoch - 1, anneal_epochs)
        add("kl", kl_loss(outputs["prior_params"], outputs["posterior_params"],
                          kl_weight=kl_w, mask=mask), w["kl"])
        loss_dict["kl_weight"] = kl_w

    if w.get("diversity", 0.0) > 0 and predictions.shape[2] > 1:
        clamp = config["training"].get("diversity_clamp", DIVERSITY_CLAMP)
        add("diversity", diversity_loss(predictions, mask, clamp=clamp), w["diversity"])

    log_probs_all = outputs.get("all_log_probs")
    if log_probs_all is not None:
        if w.get("rank", 0.0) > 0:
            tau = config["training"].get("rank_temperature", 0.0)
            add("rank", ranking_loss(log_probs_all, candidates, ground_truth, mask, valid_mask,
                                     temperature=tau), w["rank"])
        if w.get("nll", 0.0) > 0:
            add("nll", nll_loss(log_probs_all, candidates, ground_truth, mask, valid_mask),
                w["nll"])

    loss_dict["total"] = total.item()
    return total, loss_dict
