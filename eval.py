"""
eval.py
-------
Purpose : Load a saved checkpoint and run the full metric suite on the
          validation split produced by DatasetFactory.
Inputs  : --config <path>       path to config.yaml
          --checkpoint <path>   path to a .pt checkpoint saved by train.py
          --device <str>        'cpu' | 'cuda'  (default: auto-detect)
Outputs : Prints a formatted summary table of ADE, FDE, MissRate, OracleFDE,
          and Diversity metrics to stdout.  Returns nothing.

Public interface
----------------
evaluate(model, loader, device, cfg) -> dict[str, float]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from data.dataset import DatasetFactory
from models.trajectory import TrajectoryModel
from utils.metrics import (
    compute_ade,
    compute_diversity,
    compute_fde,
    compute_min_ade,
    compute_min_fde,
    compute_miss_rate,
    compute_oracle_fde,
)


# ---------------------------------------------------------------------------
# Collate helper (mirrors train.py)
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of scene dicts into a batched dict.

    Each value is either stacked (tensors with a leading batch dim) or kept as
    a list when shapes differ across scenes (e.g. variable-agent tensors).
    We return lists here and let the model handle per-scene iteration.
    """
    keys = batch[0].keys()
    collated: dict[str, Any] = {}
    for k in keys:
        collated[k] = [item[k] for item in batch]
    return collated


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: TrajectoryModel,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Run one full pass over *loader* and return aggregated metric dict.

    Parameters
    ----------
    model   : trained TrajectoryModel (already moved to *device*)
    loader  : DataLoader yielding batched scene dicts
    device  : torch.device
    cfg     : full config dict (used for future_steps, num_predictions, etc.)

    Returns
    -------
    metrics : dict with keys
        'ADE', 'FDE', 'MissRate', 'OracleFDE', 'Diversity'
    """
    model.eval()

    num_predictions: int = cfg["model"]["num_predictions"]
    miss_threshold: float = 2.0  # metres – standard nuScenes threshold

    # Running sums for micro-averaging across scenes
    sum_ade = 0.0
    sum_fde = 0.0
    sum_min_ade = 0.0
    sum_min_fde = 0.0
    sum_miss = 0.0
    sum_oracle_fde = 0.0
    sum_diversity = 0.0
    num_scenes = 0

    with torch.no_grad():
        for batch in loader:
            # Each element in the batch list is one scene
            for scene_idx in range(len(batch["past"])):
                past = torch.tensor(batch["past"][scene_idx], dtype=torch.float32, device=device).unsqueeze(0)           # (1, N, 8, 4)
                future = torch.tensor(batch["future"][scene_idx], dtype=torch.float32, device=device).unsqueeze(0)       # (1, N, 12, 2)
                mask = torch.tensor(batch["mask"][scene_idx], dtype=torch.bool, device=device).unsqueeze(0)              # (1, N)
                map_poly = torch.tensor(batch["map_polylines"][scene_idx], dtype=torch.float32, device=device).unsqueeze(0)  # (1, P, L, 2)

                batch_dict = {
                    "past": past,
                    "mask": mask,
                    "map_polylines": map_poly,
                }
                
                # Forward pass (teacher_forcing_ratio=0.0 forced at eval time)
                outputs = model(
                    batch=batch_dict,
                    teacher_forcing_ratio=0.0,
                )
                predictions = outputs["predictions"]
                scores = outputs["log_probs"]

                ade = compute_ade(predictions, scores, future, mask)
                fde = compute_fde(predictions, scores, future, mask)
                min_ade = compute_min_ade(predictions, future, mask)
                min_fde = compute_min_fde(predictions, future, mask)
                miss = compute_miss_rate(
                    predictions, future, mask, threshold=miss_threshold
                )
                oracle = compute_oracle_fde(predictions, future, mask)
                div = compute_diversity(predictions, mask)

                sum_ade += ade
                sum_fde += fde
                sum_min_ade += min_ade
                sum_min_fde += min_fde
                sum_miss += miss
                sum_oracle_fde += oracle
                sum_diversity += div
                num_scenes += 1

    if num_scenes == 0:
        raise RuntimeError("Validation loader returned zero scenes.")

    metrics = {
        "ADE": sum_ade / num_scenes,
        "FDE": sum_fde / num_scenes,
        "minADE_6": sum_min_ade / num_scenes,
        "minFDE_6": sum_min_fde / num_scenes,
        "MissRate_6": sum_miss / num_scenes,
        "OracleFDE": sum_oracle_fde / num_scenes,
        "Diversity": sum_diversity / num_scenes,
    }
    return metrics


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict[str, float]) -> None:
    """Print a tidy two-column table of metric name → value."""
    col_w = max(len(k) for k in metrics) + 2
    sep = "+" + "-" * (col_w + 2) + "+" + "-" * 14 + "+"
    print(sep)
    print(f"| {'Metric':<{col_w}} | {'Value':>12} |")
    print(sep)
    for name, val in metrics.items():
        print(f"| {name:<{col_w}} | {val:>12.4f} |")
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved TrajectoryModel checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cpu' or 'cuda'. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    if not os.path.isfile(args.config):
        print(f"[ERROR] Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset & loader  (val split)
    # ------------------------------------------------------------------
    val_cfg = dict(cfg)  # shallow copy so we can override split
    val_dataset = DatasetFactory(val_cfg, split="val")

    batch_size: int = (
        args.batch_size
        if args.batch_size is not None
        else cfg["training"]["batch_size"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TrajectoryModel(config=cfg, device=device).to(device)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        print(
            f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr
        )
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Support both raw state-dict saves and richer dicts written by train.py
    if isinstance(checkpoint, dict) and ("model_state" in checkpoint or "model_state_dict" in checkpoint):
        state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
        saved_epoch = checkpoint.get("epoch", "?")
        saved_loss = checkpoint.get("val_loss", "?")
        print(
            f"[eval] Checkpoint: epoch={saved_epoch}, val_loss={saved_loss:.4f}"
            if isinstance(saved_loss, float)
            else f"[eval] Checkpoint: epoch={saved_epoch}"
        )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print("[eval] Checkpoint loaded successfully.")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("[eval] Running evaluation …")
    metrics = evaluate(model, val_loader, device, cfg)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n=== Evaluation Results ===")
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()