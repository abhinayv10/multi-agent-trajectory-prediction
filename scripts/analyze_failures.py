"""
scripts/analyze_failures.py

Where does the model fail, and is it worse than constant velocity there?

Every scored test agent is tagged with the motion patterns below (an agent
can have several). For each tag the script reports the model's minADE@1 /
minFDE@1 / minFDE@6 and the constant-velocity FDE on the same agents, then
plots the worst case of each tag.

Tags (thresholds in metres / seconds, ETH/UCY scale):
    straight      moving (> 0.3 m/s) and future heading within 20 deg of the past heading
    sharp_turn    moving and future displacement > 45 deg away from the last velocity
    start         nearly still in the last observed step (< 0.2 m/s), later moves > 0.5 m/s
    stop          moving > 0.5 m/s, average future speed < 40 % of it
    group         a neighbour within 1.0 m walking with a velocity difference < 0.3 m/s
    dense         5 or more other agents within 3 m at the last observed step

Usage:
    python scripts/analyze_failures.py --checkpoint runs/eth_ucy/zara1/ours/seed0/best.pt \
        --out-dir docs/failures
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DatasetFactory, build_dataloader          # noqa: E402
from eval import collect, load_model                              # noqa: E402
from models import build_model                                    # noqa: E402
from utils.config import apply_overrides                          # noqa: E402

TAGS = ["all", "straight", "sharp_turn", "start", "stop", "group", "dense"]


def angle_between(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = (a * b).sum(-1) / np.maximum(na * nb, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def agent_tags(past, future, mask, i, dt):
    """Motion-pattern tags for agent i of one scene (numpy arrays)."""
    v_last = past[i, -1, 2:]
    speed = np.linalg.norm(v_last)
    disp = future[i, -1] - past[i, -1, :2]
    fut_speed = np.linalg.norm(np.diff(np.concatenate([past[i, -1:, :2], future[i]]), axis=0),
                               axis=-1) / dt
    tags = {"all"}
    moving = speed > 0.3
    if moving and np.linalg.norm(disp) > 0.5:
        ang = angle_between(v_last, disp)
        if ang > 45:
            tags.add("sharp_turn")
        elif ang < 20:
            tags.add("straight")
    if speed < 0.2 and fut_speed.max() > 0.5:
        tags.add("start")
    if speed > 0.5 and fut_speed.mean() < 0.4 * speed:
        tags.add("stop")
    others = [j for j in np.where(mask)[0] if j != i]
    if others:
        d = np.linalg.norm(past[others, -1, :2] - past[i, -1, :2], axis=-1)
        dv = np.linalg.norm(past[others, -1, 2:] - v_last, axis=-1)
        if np.any((d < 1.0) & (dv < 0.3) & moving):
            tags.add("group")
        if (d < 3.0).sum() >= 5:
            tags.add("dense")
    return tags


def plot_case(case, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    past, fut, preds, probs, mask, i = (case[k] for k in
                                        ("past", "future", "preds", "probs", "mask", "i"))
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for j in np.where(mask)[0]:
        if j == i:
            continue
        ax.plot(past[j, :, 0], past[j, :, 1], color="0.75", lw=1.2)
        ax.plot(fut[j, :, 0], fut[j, :, 1], color="0.85", lw=1.0, ls="--")
        ax.scatter(past[j, -1, 0], past[j, -1, 1], s=12, color="0.6", zorder=3)
    order = np.argsort(-probs)
    for rank, k in enumerate(order):
        p = np.concatenate([past[i, -1:, :2], preds[k]])
        ax.plot(p[:, 0], p[:, 1], color="#d95f02", lw=2.6 if rank == 0 else 1.2,
                alpha=1.0 if rank == 0 else 0.55, zorder=4,
                label="top-ranked prediction" if rank == 0 else
                ("other predictions" if rank == 1 else None))
    ax.plot(past[i, :, 0], past[i, :, 1], color="#1b6ec2", lw=2.2, label="observed (3.2 s)", zorder=5)
    g = np.concatenate([past[i, -1:, :2], fut[i]])
    ax.plot(g[:, 0], g[:, 1], color="#1a9850", lw=2.2, ls="--", label="ground truth (4.8 s)", zorder=5)
    ax.scatter(*past[i, -1, :2], color="#1b6ec2", s=30, zorder=6)

    pts = np.concatenate([past[i, :, :2], fut[i], preds.reshape(-1, 2)])
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, max((hi - lo).max() / 2 + 0.8, 3.0)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="docs/failures")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, config = load_model(args.checkpoint, None, args.set, device)
    dataset = DatasetFactory(config, split=args.split)
    loader = build_dataloader(dataset, batch_size=config["training"]["batch_size"],
                              max_agents_per_batch=config["training"].get("max_agents_per_batch"))
    ks = [1, 6]
    res = collect(model, loader, ks, seed=0, keep_predictions=True)

    cv_cfg = apply_overrides(config, ["model.type=cv"])
    cv = build_model(cv_cfg, device)
    dt = config["dataset"].get("dt", 0.4)

    rows, cases = [], []
    for batch, preds, log_probs in res["batches"]:
        cv_pred = cv(batch, num_predictions=1)["predictions"][:, :, 0].numpy()
        past, fut = batch["past"].numpy(), batch["future"].numpy()
        tmask, mask = batch["target_mask"].numpy(), batch["mask"].numpy()
        P, LP = preds.numpy(), log_probs.numpy()
        for b in range(past.shape[0]):
            for i in np.where(tmask[b])[0]:
                l2 = np.linalg.norm(P[b, i, :6] - fut[b, i][None], axis=-1)        # (K, T)
                row = {
                    "tags": agent_tags(past[b], fut[b], mask[b], i, dt),
                    "ade1": float(l2[0].mean()), "fde1": float(l2[0, -1]),
                    "ade6": float(l2.mean(-1).min()), "fde6": float(l2[:, -1].min()),
                    "cv_fde": float(np.linalg.norm(cv_pred[b, i, -1] - fut[b, i, -1])),
                    "cv_ade": float(np.linalg.norm(cv_pred[b, i] - fut[b, i], axis=-1).mean()),
                }
                rows.append(row)
                cases.append({"past": past[b], "future": fut[b], "preds": P[b, i, :6],
                              "probs": np.exp(LP[b, i, :6]), "mask": mask[b], "i": i})

    table = {}
    for tag in TAGS:
        sel = [r for r in rows if tag in r["tags"]]
        if not sel:
            continue
        table[tag] = {
            "count": len(sel),
            "share": len(sel) / len(rows),
            **{k: float(np.mean([r[k] for r in sel]))
               for k in ("ade1", "fde1", "ade6", "fde6", "cv_ade", "cv_fde")},
            "beats_cv_fde1": float(np.mean([r["fde1"] < r["cv_fde"] for r in sel])),
        }

    scene = config["dataset"].get("test_scene", config["dataset"]["type"])
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Worst case per tag (by minFDE@6), each agent used at most once.
    used, picked = set(), []
    for tag in ["sharp_turn", "start", "stop", "group", "dense", "straight"]:
        idx = [n for n, r in enumerate(rows) if tag in r["tags"] and n not in used]
        if not idx:
            continue
        n = max(idx, key=lambda n: rows[n]["fde6"])
        used.add(n)
        r = rows[n]
        name = f"{scene}_{tag}.png"
        plot_case(cases[n], os.path.join(args.out_dir, name),
                  f"{scene.upper()} / {tag.replace('_', ' ')}: minFDE@6 {r['fde6']:.2f} m, "
                  f"top-1 FDE {r['fde1']:.2f} m, CV FDE {r['cv_fde']:.2f} m")
        picked.append({"tag": tag, "file": name, **{k: v for k, v in r.items() if k != "tags"},
                       "all_tags": sorted(r["tags"])})

    out = {"scene": scene, "checkpoint": args.checkpoint, "n_agents": len(rows),
           "by_tag": table, "plotted": picked}
    with open(os.path.join(args.results_dir, f"failures_{scene}.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"| tag | agents | share | minADE@1 | minFDE@1 | minFDE@6 | CV FDE | top-1 beats CV |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, t in table.items():
        print(f"| {tag} | {t['count']} | {t['share']:.0%} | {t['ade1']:.2f} | {t['fde1']:.2f} | "
              f"{t['fde6']:.2f} | {t['cv_fde']:.2f} | {t['beats_cv_fde1']:.0%} |")
    print(f"[failures] plots -> {args.out_dir}")


if __name__ == "__main__":
    main()
