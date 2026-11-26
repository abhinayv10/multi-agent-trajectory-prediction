"""
models/trajectory.py

Purpose:
    Top-level model that wires the sub-modules into one forward pass and
    handles candidate generation, ranking and output assembly.

Pipeline:
    past (relative to each agent) -> TemporalEncoder -> temporal_emb (B, N, D)
    temporal_emb + past           -> SocialGAT       -> social_emb   (B, N, D)
    temporal_emb + polylines      -> MapEncoder      -> map_emb      (B, N, D)
    concat + MLP                                     -> context      (B, N, 256)
    context -> GoalPredictor -> goals (B, N, K_g, 2)
    context -> CVAE prior    -> z samples
    (context, goal, z) -> Decoder -> C candidates
    candidates -> RankingNet -> top-K

    Every agent is decoded in its own translated frame (origin = its last
    observed position) and shifted back, so the decoder's zero start state is
    correct for every agent, not only the one at the scene origin.

Training-only "guided" candidate:
    one extra candidate per agent decoded with the posterior latent, the
    ground-truth goal and teacher forcing. It trains the CVAE posterior / KL
    and the decoder's goal-following through a reconstruction loss. It is
    kept out of the best-of-C pool so that the winner-takes-all losses train
    the prior samples that are actually used at test time.

Ablation switches (config["model"]):
    temporal_encoder : "transformer" | "lstm"
    use_social       : GAT + decoder social attention
    use_map          : map encoder
    use_cvae         : latent sampling (off -> one deterministic decode per goal)
    use_goals        : goal predictor (off -> CVAE samples only)
    use_ranking      : learned ranking (off -> one sample per goal, by goal score)

Outputs:
    dict with predictions (B, N, K, T, 2) sorted by confidence (scene frame),
    log_probs (B, N, K), all_candidates (B, N, C, T, 2), all_log_probs,
    recon (B, N, T, 2) or None, goal outputs, prior / posterior params.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_encoder import TemporalEncoder
from .social_gat import SocialGAT
from .map_encoder import MapEncoder
from .goal_predictor import GoalPredictor, COORD_SCALE
from .cvae import CVAE
from .decoder import Decoder


def last_valid_future(future: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Each agent's last real future position: (B, N, T, 2) -> (B, N, 2)."""
    if valid_mask is None:
        return future[:, :, -1]
    T = future.shape[2]
    steps = torch.arange(T, device=future.device, dtype=torch.float32)
    idx = (valid_mask.to(torch.float32) * steps).argmax(dim=-1)            # (B, N)
    return future.gather(2, idx[:, :, None, None].expand(-1, -1, 1, 2)).squeeze(2)


class TrajectoryModel(nn.Module):
    """
    Intent-aware, socially-adaptive, multi-modal trajectory predictor.

    Args:
        config : full config dict
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        m = config["model"]
        self.model_dim       = m["model_dim"]
        self.num_goals       = m["num_goals"]
        self.num_candidates  = m["num_candidates"]
        self.num_predictions = m["num_predictions"]
        self.latent_dim      = m["latent_dim"]
        self.future_steps    = config["dataset"]["future_steps"]

        self.use_social  = m.get("use_social", True)
        self.use_map     = m.get("use_map", True)
        self.use_cvae    = m.get("use_cvae", True)
        self.use_goals   = m.get("use_goals", True)
        self.use_ranking = m.get("use_ranking", True)
        if not (self.use_cvae or self.use_goals):
            raise ValueError("use_cvae and use_goals cannot both be off: "
                             "the model would have no source of multi-modality.")

        # Candidate plan. Without the CVAE every candidate is a distinct goal.
        if self.use_goals:
            self.goals_used = self.num_goals if self.use_cvae else self.num_candidates
        else:
            self.goals_used = 1
        self.samples_per_goal = (math.ceil(self.num_candidates / self.goals_used)
                                 if self.use_cvae else 1)
        self.num_pool = min(self.num_candidates, self.goals_used * self.samples_per_goal)
        if self.num_predictions > self.num_pool:
            raise ValueError(f"num_predictions={self.num_predictions} exceeds the "
                             f"{self.num_pool} candidates this configuration generates")

        # --- Encoders ---
        self.temporal_encoder = TemporalEncoder(config, device)
        self.social_encoder = SocialGAT(config, device) if self.use_social else None
        self.map_encoder = MapEncoder(config, device) if self.use_map else None

        # --- Fusion MLP: temporal(D) + social(D) + map(D) -> 256 ---
        # Disabled branches contribute zeros so parameter shapes stay fixed.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.model_dim * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        self.goal_predictor = GoalPredictor(config, device) if self.use_goals else None
        self.cvae = CVAE(config, device) if self.use_cvae else None
        self.decoder = Decoder(config, device)

        # --- Trajectory ranking network ---
        # The trajectory gets its own encoder (a 24-d input concatenated to a
        # 256-d context was ignored: scores collapsed to uniform). LayerNorm +
        # GELU keep units from dying.
        if self.use_ranking:
            self.rank_traj = nn.Sequential(
                nn.Linear(self.future_steps * 2, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 128),
            )
            self.rank_ctx = nn.Linear(256, 128)
            self.rank_head = nn.Sequential(
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        self.to(device)

    # ------------------------------------------------------------------
    def _encode(self, past, past_rel, mask, past_mask, map_polylines, polyline_mask, last_pos):
        """Run the encoders and fuse into the context vector.

        Returns:
            context    : (B, N, 256)
            social_emb : (B, N, D) — kept separately for decoder social attn
        """
        temporal_emb = self.temporal_encoder(past_rel, mask, past_mask)       # (B, N, D)
        if self.use_social:
            social_emb = self.social_encoder(temporal_emb, past, mask)        # (B, N, D)
        else:
            social_emb = torch.zeros_like(temporal_emb)
        if self.use_map:
            map_emb = self.map_encoder(temporal_emb, map_polylines, last_pos, polyline_mask)
        else:
            map_emb = torch.zeros_like(temporal_emb)
        fused = torch.cat([temporal_emb, social_emb, map_emb], dim=-1)        # (B, N, 3D)
        return self.fusion_mlp(fused), social_emb

    def _decode(self, context, social_emb, mask, goals, z, future_rel=None, tf_ratio=0.0):
        """Decode G candidates in one decoder call by folding the candidate
        axis into the batch: goals (B, N, G, 2), z (B, N, G, Dz) or None.
        Returns relative trajectories (B, N, G, T, 2)."""
        B, N, G, _ = goals.shape
        T = self.future_steps

        def fold(x):          # (B, N, ...) -> (B*G, N, ...)
            return x.unsqueeze(1).expand(B, G, *x.shape[1:]).reshape(B * G, *x.shape[1:])

        goals_f = goals.permute(0, 2, 1, 3).reshape(B * G, N, 2)
        if z is None:
            z_f = context.new_zeros(B * G, N, self.latent_dim)
        else:
            z_f = z.permute(0, 2, 1, 3).reshape(B * G, N, -1)
        traj = self.decoder(
            context=fold(context),
            social_emb=fold(social_emb),
            goal_coords=goals_f,
            z=z_f,
            mask=fold(mask),
            future_gt=fold(future_rel) if future_rel is not None else None,
            teacher_forcing_ratio=tf_ratio,
        )                                                                     # (B*G, N, T, 2)
        return traj.reshape(B, G, N, T, 2).permute(0, 2, 1, 3, 4)

    def _rank(self, candidates_rel, context, K):
        """Score candidates and select the top-K, best first.

        Returns:
            top_idx       : (B, N, K)
            log_probs     : (B, N, K)
            log_probs_all : (B, N, C) or None (ranking disabled)
        """
        B, N, C, T, _ = candidates_rel.shape
        if not self.use_ranking:
            # One candidate per goal in goal-score order, then second samples...
            idx = [(i % self.goals_used) * self.samples_per_goal + i // self.goals_used
                   for i in range(K)]
            top_idx = torch.as_tensor(idx, device=context.device).expand(B, N, K)
            log_probs = context.new_full((B, N, K), -math.log(K))
            return top_idx, log_probs, None

        # Candidates are detached: the ranking loss must score trajectories,
        # not reshape them to be easy to score.
        traj_flat = candidates_rel.detach().reshape(B, N, C, T * 2)
        h = self.rank_traj(traj_flat) + self.rank_ctx(context).unsqueeze(2)    # (B, N, C, 128)
        scores = self.rank_head(h).squeeze(-1)                                 # (B, N, C)
        log_probs_all = F.log_softmax(scores, dim=-1)                          # (B, N, C)
        _, top_idx = scores.topk(K, dim=-1)                                    # sorted
        return top_idx, log_probs_all.gather(2, top_idx), log_probs_all

    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 0.5,
        num_predictions: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch : collated dict (see data/dataset.py); future / valid_mask
                    are only needed in training mode.
            teacher_forcing_ratio : for the guided training candidate
            num_predictions : override K (e.g. 20 for best-of-20 evaluation)
        """
        dev = self.device
        past = batch["past"].to(dev)                                          # (B, N, T, 4)
        mask = batch["mask"].to(dev)                                          # (B, N)
        map_polylines = batch["map_polylines"].to(dev)                        # (B, P, L, 2)
        past_mask = batch.get("past_mask")
        past_mask = past_mask.to(dev) if past_mask is not None else None
        polyline_mask = batch.get("polyline_mask")
        polyline_mask = polyline_mask.to(dev) if polyline_mask is not None else None
        future = batch.get("future")
        valid_mask = batch.get("valid_mask")

        B, N = mask.shape
        K = num_predictions or self.num_predictions
        if K > self.num_pool:
            raise ValueError(f"Requested {K} predictions but only {self.num_pool} candidates exist")

        last_pos = past[:, :, -1, :2]                                         # (B, N, 2)
        past_rel = past.clone()
        past_rel[..., :2] = past[..., :2] - last_pos.unsqueeze(2)
        last_vel = past[:, :, -1, 2:]

        # 1. Encode
        context, social_emb = self._encode(past, past_rel, mask, past_mask,
                                           map_polylines, polyline_mask, last_pos)

        # 2. Goals (relative to each agent)
        goal_out = None
        if self.use_goals:
            goal_out = self.goal_predictor(context, last_vel, map_polylines, last_pos,
                                           polyline_mask, k=self.goals_used)
            goals = goal_out["goal_coords"]                                   # (B, N, G, 2)
        else:
            goals = context.new_zeros(B, N, 1, 2)

        # 3. Prior candidates, goal-major: candidate g * S + s.
        G, S = self.goals_used, self.samples_per_goal
        goals_c = goals.repeat_interleave(S, dim=2)[:, :, :self.num_pool]
        z = None
        prior_params = post_params = None
        if self.use_cvae:
            ctx_flat = context.reshape(B * N, -1)
            prior_params = self.cvae.encode_prior(ctx_flat)
            z = self.cvae.sample_prior(ctx_flat, self.num_pool).reshape(B, N, self.num_pool, -1)
            # The first candidate of every goal uses the prior mean: the most
            # likely trajectory for that goal, which the ranker can put first.
            mu = prior_params[0].reshape(B, N, 1, -1)
            first = torch.zeros(self.num_pool, dtype=torch.bool, device=z.device)
            first[::S] = True
            z = torch.where(first.view(1, 1, -1, 1), mu, z)
        cand_rel = self._decode(context, social_emb, mask, goals_c, z)        # (B, N, C, T, 2)

        # 4. Guided reconstruction candidate (training only).
        recon = None
        if self.training and future is not None:
            future_rel = future.to(dev) - last_pos.unsqueeze(2)
            vm = valid_mask.to(dev) if valid_mask is not None else None
            gt_goal = last_valid_future(future_rel, vm).unsqueeze(2)           # (B, N, 1, 2)
            z_post = None
            if self.use_cvae:
                post_mu, post_logvar = self.cvae.encode_posterior(
                    context.reshape(B * N, -1), future_rel.reshape(B * N, self.future_steps, 2))
                post_params = (post_mu, post_logvar)
                z_post = self.cvae._reparameterize(post_mu, post_logvar).reshape(B, N, 1, -1)
            recon = self._decode(context, social_emb, mask,
                                 gt_goal if self.use_goals else goals,
                                 z_post, future_rel, teacher_forcing_ratio)[:, :, 0]
            # recon: (B, N, T, 2)
            recon = recon + last_pos.unsqueeze(2)

        # 5. Rank and select top-K
        top_idx, log_probs, log_probs_all = self._rank(cand_rel, context, K)
        T = self.future_steps
        pred_rel = cand_rel.gather(2, top_idx[..., None, None].expand(B, N, K, T, 2))

        offset = last_pos[:, :, None, None, :]
        out = {
            "predictions":      pred_rel + offset,                 # (B, N, K, T, 2)
            "log_probs":        log_probs,                         # (B, N, K)
            "all_candidates":   cand_rel + offset,                 # (B, N, C, T, 2)
            "all_log_probs":    log_probs_all,                     # (B, N, C) or None
            "recon":            recon,                             # (B, N, T, 2) or None
            "prior_params":     prior_params,
            "posterior_params": post_params,
            "goal_coords":      None,
            "goal_log_probs":   None,
        }
        if goal_out is not None:
            lp = last_pos.unsqueeze(2)
            out.update({
                "goal_coords":     goal_out["goal_coords"] + lp,   # (B, N, G, 2)
                "goal_log_probs":  goal_out["goal_log_probs"],
                "goal_candidates": goal_out["candidates"] + lp,    # (B, N, Cg, 2)
                "goal_refined":    goal_out["refined"] + lp,
                "goal_logits":     goal_out["logits"],
            })
        return out
