"""
inference.py
------------
Purpose : Run single-scene inference with a trained TrajectoryModel.
          Accepts input either from DummyDataset (default) or from a JSON
          file whose schema matches the dataset dict returned by DatasetFactory.
Inputs  : --config <path>       path to config.yaml
          --checkpoint <path>   path to a .pt checkpoint
          --scene <path>        optional JSON file with one scene dict
                                (omit to generate a random DummyDataset scene)
          --visualize           if set, render a top-down matplotlib plot
          --save-plot <path>    path to save the plot (PNG/PDF); implies --visualize
          --device <str>        'cpu' | 'cuda'  (default: auto-detect)
          --top-k <int>         number of predictions to keep (default: 6)
Outputs : Prints per-agent top-K trajectories and confidence scores to stdout.
          Optionally saves / displays a visualization.

Public interface
----------------
run_inference(model, scene, device, top_k) -> dict[str, torch.Tensor]
    Returns {'predictions': (N, K, 12, 2), 'scores': (N, K)}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from data.dataset import DatasetFactory
from models.trajectory import TrajectoryModel


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_inference(
    model: TrajectoryModel,
    scene: dict[str, torch.Tensor],
    device: torch.device,
    top_k: int = 6,
) -> dict[str, torch.Tensor]:
    """Run one-scene inference and return the top-K predictions.

    Parameters
    ----------
    model   : TrajectoryModel (eval mode, already on *device*)
    scene   : dict with keys
                'past'          (N, 8, 4)  agent histories
                'future'        (N, 12, 2) ground-truth (may be zeros/None)
                'mask'          (N,)       bool tensor of valid agents
                'map_polylines' (P, L, 2)  map context
    device  : torch.device
    top_k   : number of predictions to keep per agent (≤ model's num_predictions)

    Returns
    -------
    dict with
        'predictions'  (N, top_k, 12, 2)  predicted positions
        'scores'       (N, top_k)          log-probabilities / ranking scores
        'mask'         (N,)                valid-agent mask (passed through)
    """
    model.eval()

    past: torch.Tensor = scene["past"].to(device)           # (N, 8, 4)
    mask: torch.Tensor = scene["mask"].to(device)           # (N,)
    map_poly: torch.Tensor = scene["map_polylines"].to(device)  # (P, L, 2)

    batch_dict = {
        "past": past.unsqueeze(0),
        "mask": mask.unsqueeze(0),
        "map_polylines": map_poly.unsqueeze(0),
    }

    with torch.no_grad():
        outputs = model(
            batch=batch_dict,
            teacher_forcing_ratio=0.0,
        )
        predictions = outputs["predictions"][0]
        scores = outputs["log_probs"][0]

    # Keep only the requested top_k (already ranked by the model)
    k = min(top_k, predictions.shape[1])
    predictions = predictions[:, :k, :, :]   # (N, k, 12, 2)
    scores = scores[:, :k]                   # (N, k)

    return {
        "predictions": predictions,
        "scores": scores,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# Scene loading helpers
# ---------------------------------------------------------------------------

def load_scene_from_json(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a scene from a JSON file.

    The JSON must contain the following keys (matching the DatasetFactory
    output schema):
        past            list[list[list[float]]]  shape (N, 8, 4)
        future          list[list[list[float]]]  shape (N, 12, 2)
        mask            list[bool]               shape (N,)
        map_polylines   list[list[list[float]]]  shape (P, L, 2)

    Parameters
    ----------
    path   : path to the JSON file
    device : torch.device to place the tensors on

    Returns
    -------
    dict[str, torch.Tensor] matching the DatasetFactory schema
    """
    with open(path, "r") as fh:
        raw: dict[str, Any] = json.load(fh)

    required_keys = {"past", "future", "mask", "map_polylines"}
    missing = required_keys - raw.keys()
    if missing:
        raise ValueError(
            f"JSON scene file is missing required keys: {missing}"
        )

    scene: dict[str, torch.Tensor] = {
        "past": torch.tensor(raw["past"], dtype=torch.float32, device=device),           # (N, 8, 4)
        "future": torch.tensor(raw["future"], dtype=torch.float32, device=device),       # (N, 12, 2)
        "mask": torch.tensor(raw["mask"], dtype=torch.bool, device=device),              # (N,)
        "map_polylines": torch.tensor(
            raw["map_polylines"], dtype=torch.float32, device=device
        ),  # (P, L, 2)
    }
    return scene


def load_dummy_scene(cfg: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Generate one random scene from DummyDataset.

    Parameters
    ----------
    cfg    : full config dict
    device : torch.device

    Returns
    -------
    dict[str, torch.Tensor] – one scene dict placed on *device*
    """
    dummy_cfg = dict(cfg)
    dummy_cfg["dataset"] = dict(cfg["dataset"])
    dummy_cfg["dataset"]["type"] = "dummy"

    dataset = DatasetFactory(dummy_cfg, split="val")
    scene_raw = dataset[0]  # values are numpy arrays from DummyDataset

    # DummyDataset returns numpy arrays; convert to tensors with correct dtypes
    dtype_map = {
        "past":          torch.float32,
        "future":        torch.float32,
        "mask":          torch.bool,
        "map_polylines": torch.float32,
    }
    return {
        k: torch.tensor(v, dtype=dtype_map.get(k, torch.float32)).to(device)
        for k, v in scene_raw.items()
    }


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

def print_inference_results(
    results: dict[str, torch.Tensor],
    top_k: int,
) -> None:
    """Print per-agent, per-mode predicted final positions and scores."""

    predictions = results["predictions"].cpu()   # (N, K, 12, 2)
    scores = results["scores"].cpu()             # (N, K)
    mask = results["mask"].cpu()                 # (N,)

    N = predictions.shape[0]
    # Clamp to the number of modes the model actually returned — run_inference
    # already slices to min(top_k, num_predictions), so top_k may exceed K here.
    K = predictions.shape[1]
    top_k = min(top_k, K)
    print(f"\n{'='*60}")
    print(f"  Inference Results  —  {N} agents, top-{top_k} predictions")
    print(f"{'='*60}")

    for agent_idx in range(N):
        if not mask[agent_idx].item():
            print(f"\n  Agent {agent_idx:2d} : [MASKED – skipped]")
            continue

        print(f"\n  Agent {agent_idx:2d}:")
        print(f"    {'Mode':<6} {'Score':>10}  {'Final (x, y)':>22}")
        print(f"    {'-'*44}")

        for k in range(top_k):
            score = scores[agent_idx, k].item()
            final_x = predictions[agent_idx, k, -1, 0].item()
            final_y = predictions[agent_idx, k, -1, 1].item()
            print(
                f"    {k:>4d}   {score:>10.4f}  ({final_x:>10.3f}, {final_y:>10.3f})"
            )

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-scene inference with a trained TrajectoryModel."
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
        help="Path to the .pt checkpoint file.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing one scene dict.  "
            "Omit to use a randomly generated DummyDataset scene."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render a top-down matplotlib plot of the scene.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save the plot (PNG/PDF). Implies --visualize.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cpu' or 'cuda'. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of top predictions to report (default: 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve visualization flag
    # ------------------------------------------------------------------
    do_visualize: bool = args.visualize or (args.save_plot is not None)

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
    print(f"[inference] Using device: {device}")

    # ------------------------------------------------------------------
    # Load scene
    # ------------------------------------------------------------------
    if args.scene is not None:
        if not os.path.isfile(args.scene):
            print(
                f"[ERROR] Scene file not found: {args.scene}", file=sys.stderr
            )
            sys.exit(1)
        print(f"[inference] Loading scene from {args.scene} …")
        scene = load_scene_from_json(args.scene, device)
    else:
        print("[inference] No --scene provided. Generating a random DummyDataset scene …")
        scene = load_dummy_scene(cfg, device)

    N = scene["past"].shape[0]
    valid = scene["mask"].sum().item()
    print(f"[inference] Scene loaded: {N} agents total, {int(valid)} valid.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TrajectoryModel(config=cfg, device=device).to(device)

    if not os.path.isfile(args.checkpoint):
        print(
            f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr
        )
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and ("model_state" in checkpoint or "model_state_dict" in checkpoint):
        state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
        epoch_info = checkpoint.get("epoch", "?")
        print(f"[inference] Checkpoint epoch: {epoch_info}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print("[inference] Model loaded successfully.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("[inference] Running forward pass …")
    results = run_inference(model, scene, device, top_k=args.top_k)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print_inference_results(results, top_k=args.top_k)

    # ------------------------------------------------------------------
    # Optional visualization
    # ------------------------------------------------------------------
    if do_visualize:
        try:
            from utils.visualization import plot_scene
        except ImportError as exc:
            print(
                f"[WARNING] Could not import visualization module: {exc}",
                file=sys.stderr,
            )
            return

        past_cpu = scene["past"].cpu()                         # (N, 8, 4)
        future_cpu = scene["future"].cpu()                     # (N, 12, 2)
        preds_cpu = results["predictions"].cpu()               # (N, K, 12, 2)
        scores_cpu = results["scores"].cpu()                   # (N, K)
        map_poly_cpu = scene["map_polylines"].cpu()            # (P, L, 2)

        save_path: str | None = args.save_plot
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        plot_scene(
            past=past_cpu,
            future_gt=future_cpu,
            predictions=preds_cpu,
            scores=scores_cpu,
            map_polylines=map_poly_cpu,
            save_path=save_path,
        )

        if save_path:
            print(f"[inference] Plot saved to: {save_path}")
        else:
            print("[inference] Plot displayed inline.")


if __name__ == "__main__":
    main()