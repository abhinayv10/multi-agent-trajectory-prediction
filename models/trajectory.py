"""
models/trajectory_model.py

Purpose:
    Top-level model that wires together all sub-modules into a single
    forward pass. Handles candidate generation, ranking, and output assembly.

Pipeline:
    past → TemporalEncoder → temporal_emb (B, N, D)
    past → SocialGAT       → social_emb   (B, N, D)
    map  → MapEncoder      → map_emb      (B, N, D)
    concat + MLP           → context      (B, N, 256)
    context → GoalPredictor → goals       (B, N, K_g, 2)
    context → CVAE         → z            (B*N, D_z)
    (context, goal, z) → Decoder → 50 candidates
    candidates → RankingNet → top-K=6

Inputs:
    batch : dict with keys past (B,N,T,4), future (B,N,T,2),
            mask (B,N), map_polylines (B,P,L,2)

Outputs:
    dict with predictions (B,N,K,T,2), goal_coords, goal_log_probs,
         log_probs, prior_params, posterior_params

Public interface:
    TrajectoryModel(config, device) -> Module
    forward(batch, teacher_forcing_ratio=0.5) -> dict
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .temporal_encoder import TemporalEncoder
from .social_gat import SocialGAT
from .map_encoder import MapEncoder
from .goal_predictor import GoalPredictor
from .cvae import CVAE
from .decoder import Decoder


class TrajectoryModel(nn.Module):
    """
    Full intent-aware, socially-adaptive, multi-modal trajectory predictor.

    Args:
        config : full config dict
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        m = config["model"]
        self.model_dim       = m["model_dim"]          # 128
        self.num_goals       = m["num_goals"]          # K_g = 6
        self.num_candidates  = m["num_candidates"]     # 50
        self.num_predictions = m["num_predictions"]    # K = 6
        self.latent_dim      = m["latent_dim"]         # 32

        # --- Encoders ---
        self.temporal_encoder = TemporalEncoder(config, device)
        self.social_encoder   = SocialGAT(config, device)
        self.map_encoder      = MapEncoder(config, device)

        # --- Fusion MLP ---
        # Input: temporal(D) + social(D) + map(D) = 3D
        # Output: context vector of dim 256
        fusion_in = self.model_dim * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # --- Goal Predictor ---
        self.goal_predictor = GoalPredictor(config, device)

        # --- CVAE ---
        self.cvae = CVAE(config, device)

        # --- Decoder ---
        self.decoder = Decoder(config, device)

        # --- Trajectory Ranking Network ---
        future_steps = config["dataset"]["future_steps"]  # 12
        ranking_in = future_steps * 2 + 256               # flattened traj + context
        self.ranking_net = nn.Sequential(
            nn.Linear(ranking_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.to(device)

    def _encode(
        self,
        past: torch.Tensor,           # (B, N, T, 4)
        mask: torch.Tensor,           # (B, N) bool
        map_polylines: torch.Tensor,  # (B, P, L, 2)
    ):
        """
        Run all three encoders and fuse into context vector.

        Returns:
            context    : (B, N, 256)
            social_emb : (B, N, D) — kept separately for decoder social attn
        """
        # Temporal encoding
        temporal_emb = self.temporal_encoder(past, mask)        # (B, N, D)

        # Social encoding
        social_emb = self.social_encoder(temporal_emb, past, mask)  # (B, N, D)

        # Map encoding (uses temporal_emb as query)
        map_emb = self.map_encoder(temporal_emb, map_polylines) # (B, N, D)

        # Fusion: concatenate all three embeddings → context
        fused = torch.cat([temporal_emb, social_emb, map_emb], dim=-1)  # (B, N, 3D)
        context = self.fusion_mlp(fused)                         # (B, N, 256)

        return context, social_emb

    def _generate_candidates(
        self,
        context: torch.Tensor,      # (B, N, 256)
        social_emb: torch.Tensor,   # (B, N, D)
        goal_coords: torch.Tensor,  # (B, N, K_g, 2)
        mask: torch.Tensor,         # (B, N) bool
        future_gt: Optional[torch.Tensor] = None,   # (B, N, T, 2)
        teacher_forcing_ratio: float = 0.0,
    ):
        """
        Generate exactly num_candidates trajectory candidates over
        (goal, latent_z) pairs.

        Candidate plan = num_goals * ceil(num_candidates/num_goals), truncated
        to exactly num_candidates. Decoding is done in at most TWO batched
        decoder calls (posterior/teacher-forced group + prior/free group) by
        folding the candidate axis into the batch dimension. At eval there is
        only the prior group -> exactly ONE decoder call.

        Returns:
            candidates  : (B, N, C, T, 2)
            prior_params: (mu, logvar) each (B*N, D_z)
            post_params : (mu, logvar) or None
        """
        B, N, _ = context.shape
        T = self.config["dataset"]["future_steps"]              # 12
        K_g = self.num_goals
        C = self.num_candidates                                 # exact total, e.g. 50

        # Latent samples per goal. ceil so K_g * samples_per_goal >= C; the
        # candidate plan below is truncated to exactly C afterwards (Task #6).
        samples_per_goal = math.ceil(C / K_g)

        # Flatten for CVAE
        ctx_flat = context.reshape(B * N, -1)                   # (B*N, 256)

        # CVAE: get prior/posterior params and sample z
        post_z, prior_params, post_params = self.cvae(context, future_gt)
        use_posterior = self.training and post_z is not None

        # ------------------------------------------------------------------
        # Build the per-candidate plan in the ORIGINAL goal-major order,
        # capped at exactly C candidates. This is cheap (MLP + randn only);
        # the expensive 12-step GRU decoding is done in at most two batched
        # calls below.  For each candidate we record its goal index, its
        # latent z (B, N, D_z), and whether it is the teacher-forced
        # posterior candidate (s == 0 of each goal, training only).
        # ------------------------------------------------------------------
        goal_ids: list = []            # int goal index per candidate
        z_list: list = []              # (B, N, D_z) latent per candidate
        is_post: list = []             # bool: posterior (teacher-forced) candidate

        for g in range(K_g):
            # Fresh prior samples per goal (preserves per-goal independence).
            z_samples = self.cvae.sample_prior(ctx_flat, samples_per_goal)
            # z_samples: (B*N, S, D_z)
            for s in range(samples_per_goal):
                if len(goal_ids) >= C:
                    break
                posterior = use_posterior and s == 0
                if posterior:
                    z_s = post_z.reshape(B, N, -1)                  # (B, N, D_z)
                else:
                    z_s = z_samples[:, s, :].reshape(B, N, -1)      # (B, N, D_z)
                goal_ids.append(g)
                z_list.append(z_s)
                is_post.append(posterior)
            if len(goal_ids) >= C:
                break

        # Partition candidate positions into the two groups.
        #   POSTERIOR group: teacher-forced, prior=posterior latent, tf_ratio.
        #   PRIOR/free group: prior latents, teacher_forcing_ratio = 0.0.
        # Only teacher-force the posterior candidates. Prior samples must run
        # free — teacher-forcing every candidate pulls them all toward the
        # single ground-truth future and collapses the candidate diversity the
        # best-of-K / diversity losses rely on.
        post_positions  = [i for i, p in enumerate(is_post) if p]
        prior_positions = [i for i, p in enumerate(is_post) if not p]

        def _run_group(positions, tf_ratio, use_tf):
            """Decode all candidates in `positions` with ONE decoder call by
            folding the candidate axis into the batch dim: (B, Cg, N, ...) ->
            (B*Cg, N, ...). Social attention inside the decoder stays over the
            N-agent axis within each folded row, so every candidate copy is
            independent and correct. Returns (B, Cg, N, T, 2)."""
            Cg = len(positions)
            # Expand context / social / mask across the candidate axis.
            ctx_fold = (
                context.unsqueeze(1)
                .expand(B, Cg, N, context.shape[-1])
                .reshape(B * Cg, N, -1)
            )
            soc_fold = (
                social_emb.unsqueeze(1)
                .expand(B, Cg, N, social_emb.shape[-1])
                .reshape(B * Cg, N, -1)
            )
            mask_fold = mask.unsqueeze(1).expand(B, Cg, N).reshape(B * Cg, N)

            # Per-candidate goals: gather (B, N, Cg, 2) then fold candidate axis.
            gid = torch.as_tensor(
                [goal_ids[i] for i in positions],
                device=context.device,
                dtype=torch.long,
            )
            goals_fold = (
                goal_coords[:, :, gid, :]        # (B, N, Cg, 2)
                .permute(0, 2, 1, 3)             # (B, Cg, N, 2)
                .reshape(B * Cg, N, 2)
            )

            # Per-candidate latents: (B, Cg, N, D_z) -> (B*Cg, N, D_z).
            z_fold = torch.stack(
                [z_list[i] for i in positions], dim=1
            ).reshape(B * Cg, N, -1)

            # Teacher forcing only for the posterior group during training.
            if use_tf and future_gt is not None:
                fut_fold = (
                    future_gt.unsqueeze(1)
                    .expand(B, Cg, N, T, 2)
                    .reshape(B * Cg, N, T, 2)
                )
            else:
                fut_fold = None

            traj = self.decoder(
                context=ctx_fold,
                social_emb=soc_fold,
                goal_coords=goals_fold,
                z=z_fold,
                mask=mask_fold,
                future_gt=fut_fold,
                teacher_forcing_ratio=tf_ratio,
            )                                                    # (B*Cg, N, T, 2)
            return traj.reshape(B, Cg, N, T, 2)

        # At most two decoder calls: posterior (train only) + prior. At eval
        # post_positions is empty, so this is exactly ONE decoder call.
        group_trajs = []
        group_order: list = []
        if post_positions:
            group_trajs.append(
                _run_group(post_positions, teacher_forcing_ratio, True)
            )
            group_order.extend(post_positions)
        if prior_positions:
            group_trajs.append(_run_group(prior_positions, 0.0, False))
            group_order.extend(prior_positions)

        # Concatenate along candidate axis (grouped order), then reorder back
        # to the original goal-major candidate order via inverse permutation.
        grouped = torch.cat(group_trajs, dim=1)                 # (B, C, N, T, 2)
        inv = [0] * C
        for p, orig in enumerate(group_order):
            inv[orig] = p
        inv_idx = torch.as_tensor(inv, device=context.device, dtype=torch.long)
        candidates_bcnt = grouped.index_select(1, inv_idx)      # (B, C, N, T, 2)

        # (B, C, N, T, 2) -> (B, N, C, T, 2) as the rest of the code expects.
        candidates = candidates_bcnt.permute(0, 2, 1, 3, 4).contiguous()
        return candidates, prior_params, post_params

    def _rank_and_select(
        self,
        candidates: torch.Tensor,   # (B, N, C, T, 2)
        context: torch.Tensor,      # (B, N, 256)
    ):
        """
        Score all candidates and select top-K predictions.

        Returns:
            predictions : (B, N, K, T, 2)
            log_probs   : (B, N, K)
        """
        B, N, C, T, _ = candidates.shape                       # unpack
        K = self.num_predictions

        # Flatten trajectory for ranking: (B, N, C, T*2)
        traj_flat = candidates.reshape(B, N, C, T * 2)         # (B, N, C, T*2)

        # Expand context to match candidates
        ctx_exp = context.unsqueeze(2).expand(B, N, C, 256)    # (B, N, C, 256)

        # Concatenate and score
        rank_input = torch.cat([traj_flat, ctx_exp], dim=-1)   # (B, N, C, T*2+256)
        flat_input = rank_input.reshape(B * N * C, -1)         # (B*N*C, *)

        scores = self.ranking_net(flat_input)                   # (B*N*C, 1)
        scores = scores.reshape(B, N, C)                        # (B, N, C)
        log_probs_all = F.log_softmax(scores, dim=-1)           # (B, N, C)

        # Select top-K
        _, top_idx = scores.topk(K, dim=-1)                    # (B, N, K)

        # Gather predictions
        top_idx_exp = top_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, K, T, 2)
        predictions = candidates.gather(2, top_idx_exp)         # (B, N, K, T, 2)

        log_probs = log_probs_all.gather(2, top_idx)            # (B, N, K)

        return predictions, log_probs, log_probs_all

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            batch : dict with:
                past          : (B, N, T, 4)
                future        : (B, N, T, 2)  — None at inference
                mask          : (B, N) bool
                map_polylines : (B, P, L, 2)
            teacher_forcing_ratio : float

        Returns:
            dict with:
                predictions      : (B, N, K, T, 2)   — top-K trajectories
                log_probs        : (B, N, K)          — log-confidence scores
                goal_coords      : (B, N, K_g, 2)     — predicted goals
                goal_log_probs   : (B, N, K_g)        — goal log-probs
                prior_params     : (mu, logvar) each (B*N, D_z)
                posterior_params : (mu, logvar) or None
        """
        # Move all inputs to device
        past          = batch["past"].to(self.device)           # (B, N, T, 4)
        mask          = batch["mask"].to(self.device)           # (B, N)
        map_polylines = batch["map_polylines"].to(self.device)  # (B, P, L, 2)
        future_gt     = batch.get("future")
        if future_gt is not None:
            future_gt = future_gt.to(self.device)              # (B, N, T, 2)

        # 1. Encode
        context, social_emb = self._encode(past, mask, map_polylines)

        # 2. Predict goals
        goal_coords, goal_log_probs = self.goal_predictor(
            context, past, map_polylines
        )                                                        # (B,N,K_g,2), (B,N,K_g)

        # 3. Generate candidates
        candidates, prior_params, post_params = self._generate_candidates(
            context=context,
            social_emb=social_emb,
            goal_coords=goal_coords,
            mask=mask,
            future_gt=future_gt if self.training else None,
            teacher_forcing_ratio=teacher_forcing_ratio if self.training else 0.0,
        )                                                        # (B, N, C, T, 2)

        # 4. Rank and select top-K
        predictions, log_probs, log_probs_all = self._rank_and_select(candidates, context)
        # predictions: (B, N, K, T, 2)
        # log_probs:   (B, N, K)

        return {
            "predictions":      predictions,      # (B, N, K, T, 2)
            "log_probs":        log_probs,        # (B, N, K)
            "all_candidates":   candidates,       # (B, N, C, T, 2)
            "all_log_probs":    log_probs_all,    # (B, N, C)
            "goal_coords":      goal_coords,      # (B, N, K_g, 2)
            "goal_log_probs":   goal_log_probs,   # (B, N, K_g)
            "prior_params":     prior_params,     # (mu, logvar)
            "posterior_params": post_params,      # (mu, logvar) or None
        }