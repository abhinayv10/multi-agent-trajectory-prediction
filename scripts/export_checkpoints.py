"""
scripts/export_checkpoints.py

Copy trained checkpoints into release/ without optimizer / scheduler state
(about a third of the size). The exported files load with eval.py,
inference.py and the analysis scripts exactly like the originals.

Usage:
    python scripts/export_checkpoints.py                # seed-0 full models
    python scripts/export_checkpoints.py --variant lstm --seed 1
"""

import argparse
import glob
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = ("epoch", "model_state", "best_score", "val_metrics", "config")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="ours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "release"))
    args = ap.parse_args()

    pattern = os.path.join(ROOT, "runs", "*", "**", args.variant, f"seed{args.seed}", "best.pt")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise SystemExit(f"no checkpoints match {pattern}")
    for path in paths:
        rel = os.path.relpath(os.path.dirname(os.path.dirname(os.path.dirname(path))),
                              os.path.join(ROOT, "runs"))          # e.g. eth_ucy/zara1
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        slim = {k: ckpt[k] for k in KEEP if k in ckpt}
        dest = os.path.join(args.out, rel, f"{args.variant}_seed{args.seed}.pt")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        torch.save(slim, dest)
        print(f"{os.path.relpath(path, ROOT)} -> {os.path.relpath(dest, ROOT)} "
              f"({os.path.getsize(dest) / 1e6:.1f} MB, epoch {slim.get('epoch')})")


if __name__ == "__main__":
    main()
