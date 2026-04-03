"""
models/cvae.py

Purpose:
    Conditional Variational Autoencoder (CVAE) for latent trajectory modes.

    Prior   p(z | c)         : MLP from context → (mu, log_var)
    Posterior q(z | c, future): MLP from (context + future encoding) → (mu, log_var)

    At training: sample from posterior q(z|c, future)
    At inference: sample from prior p(z|c)

Inputs (prior):
    context : (B*N, context_dim) — fused context vector

Inputs (posterior, training only):
    context : (B*N, context_dim)
    future  : (B*N, T_future, 2)

Outputs:
    z            : (B*N, latent_dim) — sampled latent vector
    prior_params : (mu, log_var) each (B*N, latent_dim)
    post_params  : (mu, log_var) each (B*N, latent_dim) or None at inference

Public interface:
    CVAE(config, device) -> Module
    forward(context, future=None) -> (z, prior_params, posterior_params)
    sample_prior(context, n_samples) -> (B*N, n_samples, latent_dim)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CVAE(nn.Module):
    """
    Conditional VAE for multi-modal trajectory prediction.

    Args:
        config : full config dict
        device : torch.device
    """

    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device

        m = config["model"]
        self.latent_dim  = m["latent_dim"]          # 32
        context_dim      = 256                       # from fusion MLP

        future_steps = config["dataset"]["future_steps"]  # 12
        future_dim   = future_steps * 2                   # 24  (flattened x,y)

        # Prior p(z|c): context → (mu, log_var)
        self.prior_net = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.prior_mu     = nn.Linear(128, self.latent_dim)
        self.prior_logvar = nn.Linear(128, self.latent_dim)

        # Future encoder for posterior (GRU over future trajectory)
        self.future_encoder = nn.GRU(
            input_size=2,              # (x, y) per step
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        # Posterior q(z|c, future): (context + future_encoding) → (mu, log_var)
        posterior_input_dim = context_dim + 64
        self.posterior_net = nn.Sequential(
            nn.Linear(posterior_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.posterior_mu     = nn.Linear(128, self.latent_dim)
        self.posterior_logvar = nn.Linear(128, self.latent_dim)

        self.to(device)

    def _reparameterize(
        self,
        mu: torch.Tensor,       # (*, latent_dim)
        log_var: torch.Tensor,  # (*, latent_dim)
    ) -> torch.Tensor:          # returns (*, latent_dim)
        """
        Reparameterization trick: z = mu + eps * sigma, eps ~ N(0, I).
        Gradient flows through mu and sigma (not eps).
        """
        if not self.training:
            return mu  # deterministic during eval for stability
        std = (0.5 * log_var).exp()                             # (*, D)
        eps = torch.randn_like(std)                             # (*, D)
        return mu + eps * std                                   # (*, D)

    def encode_prior(
        self,
        context: torch.Tensor,  # (M, context_dim) where M = B*N
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prior distribution parameters p(z|c).

        Returns:
            mu      : (M, latent_dim)
            log_var : (M, latent_dim)
        """
        h = self.prior_net(context)                             # (M, 128)
        mu      = self.prior_mu(h)                              # (M, D)
        log_var = self.prior_logvar(h).clamp(-10, 2)           # (M, D)
        return mu, log_var

    def encode_posterior(
        self,
        context: torch.Tensor,  # (M, context_dim)
        future: torch.Tensor,   # (M, T_future, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior distribution parameters q(z|c, future).

        Returns:
            mu      : (M, latent_dim)
            log_var : (M, latent_dim)
        """
        # Encode future trajectory with GRU
        _, h_n = self.future_encoder(future)                    # h_n: (1, M, 64)
        future_enc = h_n.squeeze(0)                             # (M, 64)

        # Concatenate with context
        posterior_input = torch.cat([context, future_enc], dim=-1)  # (M, context_dim+64)

        h = self.posterior_net(posterior_input)                 # (M, 128)
        mu      = self.posterior_mu(h)                          # (M, D)
        log_var = self.posterior_logvar(h).clamp(-10, 2)       # (M, D)
        return mu, log_var

    def forward(
        self,
        context: torch.Tensor,                   # (B, N, 256) or (M, 256)
        future: Optional[torch.Tensor] = None,   # (B, N, T, 2) or None
    ) -> Tuple[
        torch.Tensor,                            # z: (M, D)
        Tuple[torch.Tensor, torch.Tensor],       # prior_params: (mu, logvar)
        Optional[Tuple[torch.Tensor, torch.Tensor]],  # posterior_params or None
    ]:
        """
        Args:
            context : (B, N, 256) — fused context vectors
            future  : (B, N, T, 2) — ground truth future (training only)

        Returns:
            z                : (B*N, latent_dim) — sampled latent
            prior_params     : (mu, logvar) each (B*N, D)
            posterior_params : (mu, logvar) each (B*N, D) or None
        """
        # Flatten batch and agent dims
        if context.dim() == 3:
            B, N, D = context.shape                             # unpack
            ctx_flat = context.reshape(B * N, D)               # (B*N, 256)
        else:
            ctx_flat = context                                  # already (M, 256)
            B, N = None, None

        # Prior parameters
        prior_mu, prior_logvar = self.encode_prior(ctx_flat)   # (M, D)
        prior_params = (prior_mu, prior_logvar)

        # Posterior parameters (training only)
        posterior_params = None
        if future is not None and self.training:
            if future.dim() == 4:
                BN = future.shape[0] * future.shape[1]
                fut_flat = future.reshape(BN, future.shape[2], 2)  # (B*N, T, 2)
            else:
                fut_flat = future
            post_mu, post_logvar = self.encode_posterior(ctx_flat, fut_flat)
            posterior_params = (post_mu, post_logvar)

            # Sample from posterior during training
            z = self._reparameterize(post_mu, post_logvar)      # (M, D)
        else:
            # Sample from prior during inference
            z = self._reparameterize(prior_mu, prior_logvar)    # (M, D)

        return z, prior_params, posterior_params

    def sample_prior(
        self,
        context: torch.Tensor,  # (M, context_dim)
        n_samples: int,
    ) -> torch.Tensor:          # returns (M, n_samples, latent_dim)
        """
        Draw multiple samples from the prior for candidate trajectory generation.

        Args:
            context   : (M, 256) — context vectors (M = B*N)
            n_samples : number of latent samples per agent

        Returns:
            z_samples : (M, n_samples, latent_dim)
        """
        mu, log_var = self.encode_prior(context)                # (M, D)
        std = (0.5 * log_var).exp()                             # (M, D)

        # Expand for n_samples
        mu_exp  = mu.unsqueeze(1).expand(-1, n_samples, -1)    # (M, S, D)
        std_exp = std.unsqueeze(1).expand(-1, n_samples, -1)   # (M, S, D)
        eps     = torch.randn_like(std_exp)                     # (M, S, D)

        return mu_exp + eps * std_exp                           # (M, S, D)