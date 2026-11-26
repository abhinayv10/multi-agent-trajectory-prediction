"""
inference.py
------------
Purpose : Run single-scene inference with a trained model.
          The scene comes from (in order of precedence):
            --scene <json>  a JSON file with one scene dict,
            --index <i>     scene i of the checkpoint's dataset split,
            otherwise       a random DummyDataset scene.
Inputs  : --checkpoint <path>   .pt checkpoint (its stored config builds the model)
          --set KEY=VALUE ...   config overrides
          --split <name>        split used with --index (default: test)
          --visualize           render a top-down matplotlib plot
          --save-plot <path>    save the plot (PNG/PDF); implies --visualize
          --top-k <int>         number of predictions to keep (default: 6)
Outputs : Per-agent top-K final positions and confidences; optional plot.

JSON scene schema (coordinates in a local metric frame):
    past          (N, T_past, 4)   x, y, vx, vy
    mask          (N,)             valid agents
    map_polylines (P, L, 2)        may be an empty list
    future        (N, T_future, 2) optional, only for plotting
    past_mask     (N, T_past)      optional

Public interface
----------------
run_inference(model, scene, device, top_k) -> dict
    {'predictions': (N, K, T, 2), 'scores': (N, K), 'mask': (N,)}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch

from data.dataset import DatasetFactory, collate_scenes
from eval import load_model, max_predictions
from utils.config import apply_overrides


def run_inference(model, scene: dict, device: torch.device, top_k: int = 6) -> dict:
    """Run one-scene inference and return the top-K predictions (best first)."""
    model.eval()
    batch = collate_scenes([scene])
    with torch.no_grad():
        out = model(batch, teacher_forcing_ratio=0.0, num_predictions=top_k)
    return {
        "predictions": out["predictions"][0].cpu(),     # (N, K, T, 2)
        "scores": out["log_probs"][0].cpu(),            # (N, K)
        "mask": batch["mask"][0],
    }


def load_scene_from_json(path: str, config: dict) -> dict:
    with open(path, "r") as fh:
        raw: dict[str, Any] = json.load(fh)
    missing = {"past", "mask"} - raw.keys()
    if missing:
        raise ValueError(f"JSON scene file is missing required keys: {missing}")
    past = np.asarray(raw["past"], dtype=np.float32)
    N = past.shape[0]
    L = config["dataset"].get("polyline_points", 20)
    polys = np.asarray(raw.get("map_polylines") or np.zeros((0, L, 2)), dtype=np.float32)
    future = raw.get("future")
    scene = {
        "past": past,
        "mask": np.asarray(raw["mask"], dtype=bool),
        "map_polylines": polys.reshape(-1, polys.shape[-2] if polys.ndim == 3 else L, 2),
        "future": (np.asarray(future, dtype=np.float32) if future is not None else
                   np.zeros((N, config["dataset"]["future_steps"], 2), dtype=np.float32)),
    }
    if "past_mask" in raw:
        scene["past_mask"] = np.asarray(raw["past_mask"], dtype=bool)
    return scene


def print_inference_results(results: dict, top_k: int) -> None:
    preds, scores, mask = results["predictions"], results["scores"], results["mask"]
    N, K = preds.shape[:2]
    top_k = min(top_k, K)
    print(f"\n{'=' * 60}\n  Inference results: {N} agents, top-{top_k} predictions\n{'=' * 60}")
    for i in range(N):
        if not bool(mask[i]):
            continue
        print(f"\n  Agent {i:2d}:")
        print(f"    {'Mode':<6} {'Prob':>8}  {'Final (x, y)':>22}")
        for k in range(top_k):
            x, y = preds[i, k, -1].tolist()
            print(f"    {k:>4d}   {float(scores[i, k].exp()):>8.3f}  ({x:>10.3f}, {y:>10.3f})")
    print()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Single-scene trajectory inference.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--scene", default=None, help="JSON scene file")
    ap.add_argument("--index", type=int, default=None, help="scene index in --split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--save-plot", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--top-k", type=int, default=6)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    model, config = load_model(args.checkpoint, None, args.set, device)
    top_k = min(args.top_k, max_predictions(model, config))

    if args.scene:
        scene = load_scene_from_json(args.scene, config)
        source = args.scene
    elif args.index is not None:
        split = "val" if config["dataset"]["type"] == "nuscenes" and args.split == "test" else args.split
        dataset = DatasetFactory(config, split=split)
        scene = dataset[args.index]
        source = f"{config['dataset']['type']} {split}[{args.index}]"
    else:
        dummy = apply_overrides(config, ["dataset.type=dummy"])
        scene = DatasetFactory(dummy, split="val")[0]
        source = "random DummyDataset scene"
    print(f"[inference] {source}: {scene['past'].shape[0]} agents")

    results = run_inference(model, scene, device, top_k=top_k)
    print_inference_results(results, top_k)

    if args.visualize or args.save_plot:
        from utils.visualization import plot_scene
        if args.save_plot:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_plot)), exist_ok=True)
        plot_scene(
            past=torch.as_tensor(scene["past"]),
            future_gt=torch.as_tensor(scene["future"]),
            predictions=results["predictions"],
            scores=results["scores"],
            map_polylines=torch.as_tensor(scene["map_polylines"]),
            save_path=args.save_plot,
        )
        if args.save_plot:
            print(f"[inference] plot saved to {args.save_plot}")


if __name__ == "__main__":
    main()
