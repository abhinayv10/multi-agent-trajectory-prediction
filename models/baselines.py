"""
models/baselines.py

Reference models used to put the full model's numbers in context. Both
return the same output dict as TrajectoryModel (predictions sorted best-first),
so train.py / eval.py treat all models alike.

ConstantVelocity (no parameters, no training)
    Extrapolates the last observed velocity, as in Schöller et al., "What the
    Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction"
    (RA-L 2020). Mode 0 is the deterministic CV prediction, so minADE@1 /
    minFDE@1 are the plain CV numbers. Further modes rotate the velocity by a
    random heading offset ~ N(0, heading_noise_deg), the paper's sampling
    variant, which gives a best-of-K score for the same model.

LSTMEncDec
    A plain sequence-to-sequence baseline: an LSTM encodes each agent's
    history (positions relative to its last position + velocities), an
    LSTMCell decodes displacements autoregressively. K learned mode
    embeddings make it multi-modal; it is trained winner-takes-all with a
    confidence head. No social, map, goal or latent components.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstantVelocity(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device
        self.future_steps = config["dataset"]["future_steps"]
        self.dt = config["dataset"].get("dt", 0.4)
        b = config.get("baseline", {})
        self.num_predictions = config["model"]["num_predictions"]
        self.heading_noise = math.radians(b.get("heading_noise_deg", 25.0))
        self.to(device)

    def forward(self, batch, teacher_forcing_ratio=0.0, num_predictions=None):
        past = batch["past"].to(self.device)                       # (B, N, T, 4)
        past_mask = batch.get("past_mask")
        K = num_predictions or self.num_predictions
        B, N, T_p, _ = past.shape

        last = past[:, :, -1, :2]
        # Last observed displacement / dt (falls back to the given velocity
        # when the previous step is padded).
        vel = past[:, :, -1, 2:]
        if T_p > 1:
            disp_vel = (past[:, :, -1, :2] - past[:, :, -2, :2]) / self.dt
            prev_ok = past_mask[:, :, -2].to(self.device) if past_mask is not None else \
                torch.ones(B, N, dtype=torch.bool, device=self.device)
            vel = torch.where(prev_ok.unsqueeze(-1), disp_vel, vel)

        angles = torch.randn(B, N, K, device=self.device) * self.heading_noise
        angles[:, :, 0] = 0.0                                      # mode 0 = plain CV
        c, s = torch.cos(angles), torch.sin(angles)
        vx, vy = vel[..., 0:1], vel[..., 1:2]
        v_rot = torch.stack([c * vx - s * vy, s * vx + c * vy], -1)  # (B, N, K, 2)
        t = torch.arange(1, self.future_steps + 1, device=self.device, dtype=past.dtype) * self.dt
        preds = last[:, :, None, None, :] + v_rot[:, :, :, None, :] * t[None, None, None, :, None]

        # Confidence: CV first, the rest equally likely.
        log_probs = past.new_full((B, N, K), math.log(0.5 / max(K - 1, 1)))
        log_probs[:, :, 0] = math.log(0.5) if K > 1 else 0.0
        return {"predictions": preds, "log_probs": log_probs,
                "all_candidates": preds, "all_log_probs": None}


class LSTMEncDec(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        self.device = device
        m = config["model"]
        self.future_steps = config["dataset"]["future_steps"]
        self.num_predictions = m["num_predictions"]
        self.num_modes = m.get("lstm_modes", self.num_predictions)
        hidden = m.get("lstm_hidden", 128)

        self.input_proj = nn.Linear(m["state_dim"], 64)
        self.encoder = nn.LSTM(64, hidden, batch_first=True)
        self.mode_emb = nn.Embedding(self.num_modes, 32)
        self.decoder = nn.LSTMCell(2 + 32, hidden)
        self.out = nn.Linear(hidden, 2)
        self.conf = nn.Sequential(nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Linear(64, 1))
        self.to(device)

    def forward(self, batch, teacher_forcing_ratio=0.0, num_predictions=None):
        past = batch["past"].to(self.device)                      # (B, N, T, 4)
        past_mask = batch.get("past_mask")
        K = num_predictions or self.num_predictions
        if K > self.num_modes:
            raise ValueError(f"LSTMEncDec has {self.num_modes} modes, {K} requested")
        B, N, T_p, S = past.shape
        M, T = self.num_modes, self.future_steps

        last = past[:, :, -1, :2]
        x = past.clone()
        x[..., :2] = x[..., :2] - last.unsqueeze(2)
        if past_mask is not None:
            x = x * past_mask.to(self.device).unsqueeze(-1).to(x.dtype)
        _, (h, c) = self.encoder(self.input_proj(x.reshape(B * N, T_p, S)))
        h, c = h[-1], c[-1]                                       # (B*N, H)

        # Decode all modes at once: (B*N*M, ...)
        modes = self.mode_emb.weight                              # (M, 32)
        h = h.repeat_interleave(M, 0)
        c = c.repeat_interleave(M, 0)
        mode_in = modes.repeat(B * N, 1)                          # (B*N*M, 32)
        conf = self.conf(torch.cat([h, mode_in], -1)).reshape(B, N, M)

        pos = h.new_zeros(B * N * M, 2)
        delta = h.new_zeros(B * N * M, 2)
        steps = []
        for _ in range(T):
            h, c = self.decoder(torch.cat([delta, mode_in], -1), (h, c))
            delta = self.out(h)
            pos = pos + delta
            steps.append(pos)
        traj = torch.stack(steps, 1).reshape(B, N, M, T, 2) + last[:, :, None, None, :]

        log_probs_all = F.log_softmax(conf, -1)                   # (B, N, M)
        _, idx = conf.topk(K, -1)
        preds = traj.gather(2, idx[..., None, None].expand(B, N, K, T, 2))
        return {"predictions": preds, "log_probs": log_probs_all.gather(2, idx),
                "all_candidates": traj, "all_log_probs": log_probs_all}
