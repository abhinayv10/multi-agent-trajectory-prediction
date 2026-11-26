"""
eval.py

Purpose:
    Evaluate a checkpoint (or the parameter-free constant-velocity baseline)
    on a dataset split and report:
      * minADE@k / minFDE@k / miss rate@k for k in evaluation.ks
        (predictions are sorted by confidence, so @1 measures the ranking),
      * the same metrics broken down by agent type,
      * inference latency per scene (batch size 1).

    The model architecture comes from the config stored in the checkpoint;
    --set overrides are applied on top (e.g. a different test scene).

Usage:
    python eval.py --checkpoint runs/eth_ucy/zara1/ours/seed0/best.pt
    python eval.py --config configs/eth_ucy.yaml --set model.type=cv
    python eval.py --checkpoint ... --split val --out results/run.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from data.dataset import AGENT_TYPE_NAMES, DatasetFactory, build_dataloader
from models import build_model
from utils.config import apply_overrides, load_config
from utils.metrics import per_agent_errors, summarize_errors


def max_predictions(model, config) -> int:
    """Largest K the model can produce."""
    kind = config["model"].get("type", "trajectory")
    if kind == "trajectory":
        return model.num_pool
    if kind == "lstm":
        return model.num_modes
    return 10_000   # constant velocity: any number of samples


@torch.no_grad()
def collect(model, loader, ks, seed=0, keep_predictions=False):
    """Run the model over a loader and gather per-agent errors.

    Returns dict with per-agent arrays "min_ade_<k>", "min_fde_<k>",
    "agent_type" and, if keep_predictions, the batches + predictions.
    """
    model.eval()
    torch.manual_seed(seed)
    k_max = max(ks)
    errors, types, kept = {}, [], []
    for batch in loader:
        out = model(batch, teacher_forcing_ratio=0.0, num_predictions=k_max)
        preds = out["predictions"].cpu()
        tm = batch["target_mask"]
        e = per_agent_errors(preds, batch["future"], tm, batch.get("valid_mask"), ks=ks)
        for k, v in e.items():
            errors.setdefault(k, []).append(v)
        types.append(batch["agent_type"][tm].numpy())
        if keep_predictions:
            kept.append((batch, preds, out["log_probs"].cpu()))
    res = {k: np.concatenate(v) for k, v in errors.items()}
    res["agent_type"] = np.concatenate(types) if types else np.zeros(0, dtype=np.int64)
    if keep_predictions:
        res["batches"] = kept
    return res


@torch.no_grad()
def time_inference(model, dataset, n_scenes, device, num_predictions=None):
    """Mean / p95 forward latency per scene (batch size 1, after warm-up)."""
    model.eval()
    n = min(n_scenes, len(dataset))
    if n == 0:
        return {}
    loader = build_dataloader(torch.utils.data.Subset(dataset, range(n)), batch_size=1)
    batches = list(loader)
    for b in batches[:5]:                     # warm-up
        model(b, teacher_forcing_ratio=0.0, num_predictions=num_predictions)
    times, agents = [], []
    for b in batches:
        t0 = time.perf_counter()
        model(b, teacher_forcing_ratio=0.0, num_predictions=num_predictions)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        agents.append(int(b["mask"].sum()))
    times = np.asarray(times)
    return {
        "ms_per_scene_mean": float(times.mean()),
        "ms_per_scene_p95": float(np.percentile(times, 95)),
        "agents_per_scene_mean": float(np.mean(agents)),
        "scenes_timed": n,
        "device": str(device),
        "threads": torch.get_num_threads(),
    }


def summarize(res, ks, miss_threshold):
    keys = [f"min_{m}_{k}" for k in ks for m in ("ade", "fde") if f"min_{m}_{k}" in res]
    overall = summarize_errors({k: res[k] for k in keys}, miss_threshold)
    by_type = {}
    for t in np.unique(res["agent_type"]):
        sel = res["agent_type"] == t
        name = AGENT_TYPE_NAMES.get(int(t), str(int(t)))
        by_type[name] = {"count": int(sel.sum()),
                         **summarize_errors({k: res[k][sel] for k in keys}, miss_threshold)}
    return overall, by_type


def load_model(checkpoint, config_path, overrides, device):
    """Build the model + config from a checkpoint (or a config, for CV)."""
    if checkpoint:
        if not os.path.isfile(checkpoint):
            sys.exit(f"[eval] checkpoint not found: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        config = apply_overrides(ckpt["config"], overrides)
        model = build_model(config, device)
        model.load_state_dict(ckpt["model_state"])
    else:
        config = load_config(config_path, overrides)
        if config["model"].get("type", "trajectory") != "cv":
            sys.exit("[eval] --checkpoint is required for trainable models")
        model = build_model(config, device)
    config.setdefault("evaluation", {})
    return model.to(device), config


def print_table(overall, by_type, ks):
    cols = [f"{m}@{k}" for k in ks for m in ("minADE", "minFDE", "MR")]
    def row(name, d, count=""):
        vals = []
        for k in ks:
            for key in (f"min_ade_{k}", f"min_fde_{k}", f"miss_rate_{k}"):
                vals.append(f"{d[key]:.3f}" if key in d else "-")
        return f"| {name:<12} | {str(count):>6} | " + " | ".join(f"{v:>9}" for v in vals) + " |"
    print("| " + "group".ljust(12) + " | " + "agents".rjust(6) + " | "
          + " | ".join(c.rjust(9) for c in cols) + " |")
    print(row("all", overall, sum(v["count"] for v in by_type.values())))
    for name, d in by_type.items():
        print(row(name, d, d["count"]))


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trajectory model.")
    ap.add_argument("--config", default="configs/eth_ucy.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--split", default="test", help="test | val | train")
    ap.add_argument("--out", default=None, help="Write results JSON here")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(args.checkpoint, args.config, args.set, device)
    ev = config["evaluation"]

    split = args.split
    if (split == "test" and config["dataset"]["type"] == "nuscenes"
            and config["dataset"].get("version", "v1.0-mini") != "v1.0-mini"):
        split = "val"     # full nuScenes: test annotations are not public
    dataset = DatasetFactory(config, split=split)
    loader = build_dataloader(dataset, batch_size=args.batch_size or config["training"]["batch_size"],
                              max_agents_per_batch=config["training"].get("max_agents_per_batch"))

    k_cap = max_predictions(model, config)
    ks = [k for k in ev.get("ks", [1, 6]) if k <= k_cap]
    res = collect(model, loader, ks, seed=ev.get("seed", 0))
    overall, by_type = summarize(res, ks, ev.get("miss_threshold", 2.0))

    timing = {}
    if not args.no_timing:
        timing = time_inference(model, dataset, ev.get("timing_scenes", 200), device)

    print(f"[eval] {config['dataset']['type']} split={split} "
          f"model={config['model'].get('type', 'trajectory')} scenes={len(dataset)}")
    print_table(overall, by_type, ks)
    if timing:
        print(f"[eval] latency: {timing['ms_per_scene_mean']:.1f} ms/scene mean, "
              f"{timing['ms_per_scene_p95']:.1f} ms p95 "
              f"({timing['agents_per_scene_mean']:.1f} agents/scene, {timing['device']}, "
              f"{timing['threads']} threads)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"split": split, "ks": ks, "overall": overall, "by_type": by_type,
                       "timing": timing, "checkpoint": args.checkpoint,
                       "n_scenes": len(dataset), "config": config}, fh, indent=1)


if __name__ == "__main__":
    main()
