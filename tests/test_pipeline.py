"""
Fast checks of the pipeline invariants (run: python -m pytest tests -q).
Uses the dummy dataset and a small model, so no data download is needed;
ETH/UCY tests are skipped when the data is absent.
"""

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataset import (AgentBudgetBatchSampler, DatasetFactory,  # noqa: E402
                          build_dataloader, collate_scenes)
from models import build_model                                       # noqa: E402
from utils.config import load_config                                 # noqa: E402
from utils.losses import compute_total_loss                          # noqa: E402
from utils.metrics import per_agent_errors                           # noqa: E402
from utils.preprocessing import (denormalize_from_agent_frame,       # noqa: E402
                                 normalize_to_agent_frame)

SMALL = [
    "dataset.type=dummy", "dataset.num_agents_max=5", "dataset.dt=0.4",
    "dataset.num_polylines=3", "dataset.dummy_train_size=8", "dataset.dummy_val_size=4",
    "model.model_dim=32", "model.num_heads=2", "model.num_transformer_layers=1",
    "model.latent_dim=8", "model.goal_embed_dim=16", "model.decoder_hidden_dim=32",
    "model.gat_heads=2", "model.num_candidates=12", "model.lstm_modes=8",
]

VARIANTS = {
    "ours": [],
    "no_transformer": ["model.temporal_encoder=lstm"],
    "no_social": ["model.use_social=false"],
    "no_map": ["model.use_map=false"],
    "no_cvae": ["model.use_cvae=false"],
    "no_goals": ["model.use_goals=false"],
    "no_ranking": ["model.use_ranking=false"],
    "lstm": ["model.type=lstm"],
    "cv": ["model.type=cv"],
}


def small_config(extra=()):
    return load_config(os.path.join(ROOT, "configs", "eth_ucy.yaml"), SMALL + list(extra))


def dummy_batch(config, n=4):
    ds = DatasetFactory(config, "train")
    return collate_scenes([ds[i] for i in range(n)])


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_forward_backward(name):
    torch.manual_seed(0)
    cfg = small_config(VARIANTS[name])
    model = build_model(cfg, torch.device("cpu"))
    batch = dummy_batch(cfg)
    B, N = batch["mask"].shape
    T = cfg["dataset"]["future_steps"]

    model.train()
    out = model(batch, teacher_forcing_ratio=0.5)
    loss, parts = compute_total_loss(out, batch, cfg, epoch=3)
    assert torch.isfinite(loss)
    if loss.requires_grad:
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    model.eval()
    with torch.no_grad():
        out = model(batch, teacher_forcing_ratio=0.0)
    K = cfg["model"]["num_predictions"]
    assert out["predictions"].shape == (B, N, K, T, 2)
    assert torch.isfinite(out["predictions"]).all()
    # Predictions are sorted best-first.
    lp = out["log_probs"]
    assert (lp[..., :-1] >= lp[..., 1:] - 1e-6).all()


def test_constant_velocity_exact_on_linear_motion():
    cfg = small_config(["model.type=cv"])
    model = build_model(cfg, torch.device("cpu"))
    dt, T_p, T_f = 0.4, 8, 12
    t = np.arange(-T_p + 1, T_f + 1) * dt
    vel = np.array([[1.2, -0.3], [0.0, 0.8]], dtype=np.float32)
    pos = (vel[:, None, :] * t[None, :, None]).astype(np.float32)
    past = np.concatenate([pos[:, :T_p], np.repeat(vel[:, None], T_p, 1)], -1)
    scene = {"past": past, "future": pos[:, T_p:], "mask": np.ones(2, bool),
             "map_polylines": np.zeros((0, 20, 2), np.float32)}
    out = model(collate_scenes([scene]), num_predictions=3)
    err = per_agent_errors(out["predictions"], torch.as_tensor(pos[None, :, T_p:]),
                           torch.ones(1, 2, dtype=torch.bool), ks=(1,))
    assert err["min_fde_1"].max() < 1e-4


def test_per_agent_errors_uses_ranked_order():
    gt = np.zeros((1, 1, 4, 2), np.float32)
    preds = np.stack([np.full((4, 2), 3.0), np.zeros((4, 2))])[None, None].astype(np.float32)
    e = per_agent_errors(preds, gt, np.ones((1, 1), bool), ks=(1, 2))
    assert e["min_fde_1"][0] == pytest.approx(np.hypot(3, 3))   # top-ranked is wrong
    assert e["min_fde_2"][0] == pytest.approx(0.0)               # best-of-2 is exact


def test_metrics_respect_valid_mask():
    gt = np.zeros((1, 1, 4, 2), np.float32)
    preds = np.zeros((1, 1, 1, 4, 2), np.float32)
    preds[..., 3, :] = 10.0                     # error only on a padded step
    vm = np.array([[[True, True, True, False]]])
    e = per_agent_errors(preds, gt, np.ones((1, 1), bool), vm, ks=(1,))
    assert e["min_ade_1"][0] == pytest.approx(0.0)
    assert e["min_fde_1"][0] == pytest.approx(0.0)


def test_frame_roundtrip():
    rng = np.random.default_rng(0)
    past = rng.normal(size=(3, 8, 4)).astype(np.float32)
    fut = rng.normal(size=(3, 12, 2)).astype(np.float32)
    poly = np.zeros((0, 20, 2), np.float32)
    _, fut_n, poly_n, origin, angle = normalize_to_agent_frame(past, fut, poly,
                                                               origin=[5.0, -2.0], angle=0.7)
    assert poly_n.shape == (0, 20, 2)
    back = denormalize_from_agent_frame(fut_n, origin, angle)
    assert np.allclose(back, fut, atol=1e-5)


def test_budget_sampler_covers_every_scene_once():
    sizes = np.random.default_rng(0).integers(1, 40, size=200)
    s = AgentBudgetBatchSampler(sizes, max_agents=64, max_scenes=16, shuffle=True,
                                generator=torch.Generator().manual_seed(0))
    idx = [i for b in s for i in b]
    assert sorted(idx) == list(range(200))
    for b in s:
        assert len(b) == 1 or max(sizes[b]) * len(b) <= 64


def test_padded_agents_do_not_change_real_predictions():
    """Adding padding agents to a scene must not change real agents' outputs."""
    torch.manual_seed(0)
    cfg = small_config(["model.use_cvae=false"])     # deterministic decoding
    model = build_model(cfg, torch.device("cpu")).eval()
    ds = DatasetFactory(cfg, "val")
    a, b = ds[0], ds[1]
    n = a["past"].shape[0]
    with torch.no_grad():
        alone = model(collate_scenes([a]))["predictions"][0, :n]
        padded = model(collate_scenes([a, b] if b["past"].shape[0] > n else [b, a]))
        idx = 0 if b["past"].shape[0] > n else 1
        together = padded["predictions"][idx, :n]
    assert torch.allclose(alone, together, atol=1e-4)


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "data", "eth_ucy", "zara1")),
                    reason="ETH/UCY not downloaded")
def test_eth_ucy_windows():
    cfg = load_config(os.path.join(ROOT, "configs", "eth_ucy.yaml"),
                      [f"dataset.data_root={os.path.join(ROOT, 'data', 'eth_ucy')}"])
    ds = DatasetFactory(cfg, "test")
    s = ds[0]
    assert s["past"].shape[1:] == (8, 4) and s["future"].shape[1:] == (12, 2)
    # Test windows are not rotated: velocities match position differences.
    v = (s["past"][:, -1, :2] - s["past"][:, -2, :2]) / 0.4
    assert np.allclose(v, s["past"][:, -1, 2:], atol=1e-4)
