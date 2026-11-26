"""
scripts/make_gif.py

Animate predictions over a real ETH/UCY test sequence: each frame is one
prediction window (0.4 s apart) showing every pedestrian's observed track,
the ground-truth future and the model's top-K predictions, in world metres.

Usage:
    python scripts/make_gif.py --checkpoint runs/eth_ucy/zara1/ours/seed0/best.pt \
        --out docs/media/zara1_predictions.gif
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DatasetFactory, collate_scenes   # noqa: E402
from eval import load_model                               # noqa: E402
from utils.preprocessing import denormalize_from_agent_frame  # noqa: E402


def pick_sequence(dataset, length, max_agents):
    """Start index of the busiest run of `length` consecutive windows from
    one recording."""
    info = dataset.window_info
    best, best_score = 0, -1
    for s in range(len(info) - length):
        f0, fr0 = info[s]
        if info[s + length - 1][0] != f0:
            continue
        sizes = [dataset.windows[s + k].shape[0] for k in range(length)]
        if max(sizes) > max_agents:
            continue
        score = sum(sizes)
        if score > best_score:
            best, best_score = s, score
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="docs/media/predictions.gif")
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-agents", type=int, default=12,
                    help="cap on window size when choosing the sequence (readability)")
    ap.add_argument("--fps", type=float, default=2.5)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    torch.manual_seed(0)
    device = torch.device("cpu")
    model, config = load_model(args.checkpoint, None, args.set, device)
    model.eval()
    dataset = DatasetFactory(config, split="test")
    if not hasattr(dataset, "window_info"):
        sys.exit("make_gif.py needs the ETH/UCY dataset")

    # Busiest stretch of consecutive windows that stays readable.
    start = pick_sequence(dataset, args.frames, args.max_agents)
    idxs = range(start, start + args.frames)
    frames = []
    with torch.no_grad():
        for i in idxs:
            sample = dataset[i]
            batch = collate_scenes([sample])
            out = model(batch, teacher_forcing_ratio=0.0, num_predictions=args.top_k)
            o, a = sample["origin"], sample["angle"]
            frames.append({
                "past": denormalize_from_agent_frame(sample["past"][..., :2], o, a),
                "future": denormalize_from_agent_frame(sample["future"], o, a),
                "preds": denormalize_from_agent_frame(out["predictions"][0].numpy(), o, a),
                "probs": out["log_probs"][0].exp().numpy(),
            })

    pts = np.concatenate([np.concatenate([f["past"].reshape(-1, 2), f["future"].reshape(-1, 2)])
                          for f in frames])
    lo, hi = np.percentile(pts, 1, axis=0) - 1.0, np.percentile(pts, 99, axis=0) + 1.0

    fig, ax = plt.subplots(figsize=(7, 7 * (hi[1] - lo[1]) / max(hi[0] - lo[0], 1e-3) + 0.6))
    scene = config["dataset"]["test_scene"].upper()

    def draw(n):
        ax.clear()
        f = frames[n]
        for j in range(f["past"].shape[0]):
            ax.plot(f["past"][j, :, 0], f["past"][j, :, 1], color="#1b6ec2", lw=2)
            ax.scatter(*f["past"][j, -1], color="#1b6ec2", s=18, zorder=4)
            g = np.concatenate([f["past"][j, -1:], f["future"][j]])
            ax.plot(g[:, 0], g[:, 1], color="#1a9850", lw=1.6, ls="--")
            for k in range(f["preds"].shape[1]):
                p = np.concatenate([f["past"][j, -1:], f["preds"][j, k]])
                ax.plot(p[:, 0], p[:, 1], color="#d95f02",
                        lw=2.2 if k == 0 else 1.0, alpha=1.0 if k == 0 else 0.5)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"{scene} test set: t = {n * 0.4:4.1f} s   "
                     f"blue = observed 3.2 s, green = true 4.8 s, orange = top-{args.top_k}",
                     fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    anim = FuncAnimation(fig, draw, frames=len(frames))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=80)
    plt.close(fig)
    print(f"[gif] {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
