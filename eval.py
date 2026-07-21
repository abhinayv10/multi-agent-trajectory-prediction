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
from data.dataset import DatasetFactory, build_dataloader
from models.trajectory import TrajectoryModel
from utils.metrics import compute_all_metrics


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

    Uses the same batched/padded collate path as training (via build_dataloader
    / collate_scenes) and aggregates each batch's metrics weighted by its number
    of valid agents. This gives a true agent-weighted average — a 10-agent scene
    counts ten times a 1-agent scene — instead of the old per-scene micro-average.

    Returns
    -------
    metrics : dict with display keys ADE, FDE, minADE_6, minFDE_6, MissRate_6,
        OracleFDE, Diversity
    """
    model.eval()
    miss_threshold: float = 2.0  # metres – standard nuScenes threshold

    keys = ["ade", "fde", "min_ade", "min_fde", "miss_rate", "oracle_fde", "diversity"]
    sums = {k: 0.0 for k in keys}
    total_agents = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch, teacher_forcing_ratio=0.0)

            preds = outputs["predictions"].cpu().numpy()
            probs = outputs["log_probs"].cpu().numpy()
            gt = batch["future"].cpu().numpy()
            mask = batch["mask"].cpu().numpy().astype(bool)
            vm = batch.get("valid_mask")
            valid = vm.cpu().numpy() if vm is not None else None

            n_valid = int(mask.sum())
            if n_valid == 0:
                continue

            m = compute_all_metrics(preds, probs, gt, mask, miss_threshold, valid_mask=valid)
            for k in keys:
                sums[k] += m[k] * n_valid       # weight by valid-agent count
            total_agents += n_valid

    if total_agents == 0:
        raise RuntimeError("Validation loader returned zero valid agents.")

    return {
        "ADE":        sums["ade"] / total_agents,
        "FDE":        sums["fde"] / total_agents,
        "minADE_6":   sums["min_ade"] / total_agents,
        "minFDE_6":   sums["min_fde"] / total_agents,
        "MissRate_6": sums["miss_rate"] / total_agents,
        "OracleFDE":  sums["oracle_fde"] / total_agents,
        "Diversity":  sums["diversity"] / total_agents,
    }


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

    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
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