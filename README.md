# Multi-Agent Trajectory Prediction

Intent-aware, socially-adaptive, multi-modal trajectory forecasting for autonomous driving — built in PyTorch.

---

## Overview

This project predicts **K = 6 future trajectories** for pedestrians and cyclists observed in a driving scene.  
Given **2 seconds of motion history** (8 timesteps at 4 Hz), the model forecasts **3 seconds of future motion** (12 timesteps) in the ego-agent's local coordinate frame.

The architecture combines a Transformer-based temporal encoder, a Graph Attention Network (GAT) social encoder, a cross-attention map encoder, a CVAE latent sampler, a goal predictor, and an autoregressive GRU decoder — all fused through a shared context vector.

The codebase incorporates native **NuScenes v1.0-mini dataset integration**, or can optionally use a `DummyDataset` for **zero-friction iteration**, enabling full training, evaluation, and inference with no external data. A single config change seamlessly swaps between NuScenes and dummy datasets without touching any other file.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (uses NuScenes v1.0-mini by default; set dataset.type to dummy for testing)
python train.py --config config.yaml

# 3. Evaluate a checkpoint
python eval.py --config config.yaml --checkpoint checkpoints/best_model.pt

# 4. Run inference on a random scene
python inference.py --config config.yaml --checkpoint checkpoints/best_model.pt

# 5. Inference with visualization saved to file
python inference.py --config config.yaml --checkpoint checkpoints/best_model.pt \
    --save-plot outputs/scene.png

# 6. Inference from a JSON scene file
python inference.py --config config.yaml --checkpoint checkpoints/best_model.pt \
    --scene my_scene.json --visualize
```

All commands above are heavily tested out of the box with the **NuScenes v1.0-mini dataset** and the internal `DummyDataset`.

---

## Project Structure

```
.
├── config.yaml                 # All hyperparameters; only file you ever need to edit
├── requirements.txt
├── train.py                    # Training loop
├── eval.py                     # Full metric evaluation
├── inference.py                # Single-scene inference + optional visualization
├── README.md
│
├── data/
│   ├── __init__.py
│   └── dataset.py              # DummyDataset, NuScenesDataset, DatasetFactory
│
├── models/
│   ├── __init__.py
│   ├── temporal_encoder.py     # Transformer over 8-step history → (N, D)
│   ├── social_gat.py           # 2-layer GAT over agent graph → (N, D)
│   ├── map_encoder.py          # MLP + cross-attention over polylines → (N, D)
│   ├── goal_predictor.py       # Scored candidate goals → (N, K, 2)
│   ├── cvae.py                 # Prior / posterior networks + KL loss
│   ├── decoder.py              # GRU decoder with social attention → (N, 12, 2)
│   └── trajectory_model.py     # Full pipeline + candidate ranking
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # normalize_to_agent_frame / denormalize_from_agent_frame
│   ├── losses.py               # minADE, minFDE, goal, KL, diversity, NLL losses
│   ├── metrics.py              # ADE, FDE, MissRate, OracleFDE, Diversity
│   └── visualization.py        # plot_scene — top-down matplotlib render
│
└── checkpoints/                # Created automatically; best_model.pt saved here
```

---

## Configuration (`config.yaml`)

All hyperparameters live in one file. Key sections:

```yaml
dataset:
  type: nuscenes       # uses 'nuscenes' by default; set to 'dummy' for no-data testing
  data_root: ./        # path to your nuScenes v1.0-mini or full dataset
  num_agents_max: 10
  past_steps: 8
  future_steps: 12
  num_polylines: 6
  polyline_points: 20

model:
  model_dim: 128       # D — shared embedding dimension
  num_heads: 4
  num_transformer_layers: 2
  latent_dim: 32       # CVAE latent dimension
  num_goals: 6         # K goals
  num_candidates: 50   # total candidates before ranking
  num_predictions: 6   # final top-K output

training:
  epochs: 200
  batch_size: 32
  learning_rate: 1e-4
  grad_clip: 1.0
  teacher_forcing_ratio: 0.5   # 0.0 forced at eval/inference
  kl_anneal_epochs: 20
  loss_weights:
    min_ade: 1.0
    min_fde: 1.0
    goal: 0.5
    kl: 0.1
    diversity: 0.5
    nll: 0.5           # optional; enable by setting > 0
  checkpoint_path: checkpoints/best_model.pt

logging:
  use_wandb: false     # set true if wandb is installed
  project_name: trajectory_prediction
```

No other file needs editing for standard runs.

---

## Dataset Layer

### DummyDataset

Generates structurally valid random scenes entirely in memory — no files, no downloads.  
Each scene contains 2–10 agents and returns:

| Key | Shape | Description |
|---|---|---|
| `past` | `(N, 8, 4)` | History: (x, y, vx, vy) per timestep |
| `future` | `(N, 12, 2)` | Ground truth future: (x, y) |
| `mask` | `(N,)` | Bool — True for valid agents |
| `map_polylines` | `(P, L, 2)` | P polylines of L points each |

All coordinates are already in **agent-centric normalized frame**.

### NuScenes Integration

The codebase comes complete with a native `NuScenesDataset` implementation tightly integrated with the NuScenes devkit.

1. Set `dataset.type: nuscenes` and `dataset.data_root: /path/to/nuscenes` in `config.yaml`.
2. Ensure you have the `v1.0-mini` (or standard `v1.0`) NuScenes dataset ready at your data_root path.

**No other file needs modification.**  
`train.py`, `eval.py`, and `inference.py` all go through `DatasetFactory` and seamlessly switch between `nuscenes` and `dummy` dataset generation.

### Scene JSON Format (for `inference.py --scene`)

```json
{
  "past":          [[[x, y, vx, vy], ...], ...],
  "future":        [[[x, y], ...], ...],
  "mask":          [true, false, ...],
  "map_polylines": [[[x, y], ...], ...]
}
```

Shapes must match `(N, 8, 4)`, `(N, 12, 2)`, `(N,)`, `(P, L, 2)` respectively.

---

## Model Architecture

```
past (N, 8, 4)
    │
    ├─► Temporal Encoder (Transformer, 2 layers, 4 heads)
    │       └─► temporal_emb  (N, D=128)
    │
    ├─► Social Encoder (2-layer GAT, 4 heads)
    │       └─► social_emb    (N, D)
    │
    └─► Map Encoder (MLP + cross-attention)
            └─► map_emb       (N, D)
                    │
              Fusion MLP (LayerNorm + ReLU)
                    │
              context c       (N, 256)
                    │
          ┌─────────┴─────────┐
          │                   │
    Goal Predictor        CVAE Prior
    top-K goals + probs   z ~ p(z|c)
          │                   │
          └─────────┬─────────┘
                    │
              GRU Decoder
         (conditioned on c, goal, z)
         (social attention at each step)
                    │
          50 candidate trajectories
                    │
            Ranking Network (MLP)
                    │
          top-K=6 predictions + scores
                 (N, 6, 12, 2)
```

### Key design choices

- **Temporal Encoder**: sinusoidal positional encoding + Transformer encoder; projects state dim 4 → model dim D.
- **Social GAT**: fully connected agent graph; edge features are relative distance (1D) + relative velocity (2D).
- **Map Encoder**: returns `zeros(N, D)` gracefully when map is absent — documented explicitly.
- **CVAE**: KL weight anneals linearly from 0 → 1 over `kl_anneal_epochs` epochs.
- **Decoder**: teacher forcing ratio configurable in config; forced to 0.0 at eval/inference; social attention added at every GRU step.
- **Candidate generation**: outer loop over K goals × M latent samples → 50 candidates → scored → top-6 kept.

---

## Losses

| Loss | Description | Weight key |
|---|---|---|
| `min_ade_loss` | minADE@K — L2 over all 12 steps | `min_ade` |
| `min_fde_loss` | minFDE@K — L2 at final step | `min_fde` |
| `goal_loss` | L2 from best goal to true endpoint | `goal` |
| `kl_loss` | KL divergence with linear annealing | `kl` |
| `diversity_loss` | Negated mean pairwise L2 among K preds | `diversity` |
| `nll_loss` | NLL of GT under predicted distribution (optional) | `nll` |

All weights are read from `config.yaml → training.loss_weights`.  
Each individual loss is logged separately every epoch.

---

## Metrics

| Metric | Description |
|---|---|
| `ADE` | Average displacement error (all steps, best-of-K) |
| `FDE` | Final displacement error (last step, best-of-K) |
| `MissRate` | Fraction of agents where minFDE@K > 2.0 m |
| `OracleFDE` | FDE of the closest prediction to GT (oracle) |
| `Diversity` | Mean pairwise L2 distance among K predictions |

---

## Training Details

- **Optimizer**: Adam with cosine annealing LR scheduler and 5-epoch warmup.
- **Gradient clipping**: max norm 1.0 (configurable).
- **Checkpointing**: best validation loss saved to `config.training.checkpoint_path`.
- **WandB**: enabled via `logging.use_wandb: true`; skipped gracefully if `wandb` is not installed.
- **Stdout logging**: tabular per-epoch format, all losses and metrics printed.

---

## Coordinate Frame Convention

All coordinates passed to any model component are in the **ego-agent's local frame**:

- Origin: agent's last observed position.
- X-axis: agent's last observed heading direction.

`utils/preprocessing.py` provides:
- `normalize_to_agent_frame(past, future, map_polylines)` — world → agent frame.
- `denormalize_from_agent_frame(predictions, origin, heading)` — agent frame → world.

`DummyDataset` returns data already normalized.  
`NuScenesDataset` stubs call `normalize_to_agent_frame` explicitly.

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
wandb>=0.15.0   # optional
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Expected Training Output

```
[train] Epoch  1/50 | loss=4.3821 | ade=3.1045 | fde=5.2231 | miss=0.8412
         losses → min_ade=1.2103  min_fde=2.0541  goal=0.4311  kl=0.0000  div=-0.3134
[train] Epoch  2/50 | loss=4.1033 | ade=2.9832 | fde=5.0112 | miss=0.8201
...
[val]   Epoch  2/50 | ADE=2.9104  FDE=4.9830  MissRate=0.8100  OracleFDE=3.1205  Diversity=1.2304
...
[train] Best model saved → checkpoints/best_model.pt
```

---

