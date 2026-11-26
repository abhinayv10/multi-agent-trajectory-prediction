"""
demo.py
-------
Runs the trained model on every nuScenes v1.0-mini test scene (official mini_val) and writes a
self-contained interactive page (demo/index.html) showing the inputs, the
model's predictions and the error against ground truth.

Usage:  python demo.py --checkpoint runs/nuscenes_mini/ours/seed0/best.pt
        python -m http.server 8765 --directory demo
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from data.dataset import DatasetFactory, collate_scenes
from eval import load_model

MISS_THRESHOLD = 2.0  # metres, same as eval.py


def r(a, n=2):
    return np.round(np.asarray(a, dtype=np.float64), n).tolist()


def build(ckpt_path: str) -> dict:
    device = torch.device("cpu")
    model, cfg = load_model(ckpt_path, None, [], device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ds = DatasetFactory(cfg, split="test")
    model.eval()

    scenes, all_ade, all_fde, all_miss = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        batch = collate_scenes([s])
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(batch=batch, teacher_forcing_ratio=0.0)
        pred = out["predictions"][0].numpy()                  # (N,K,12,2)
        lp = out["log_probs"][0].numpy()                      # (N,K)
        cand = out["all_candidates"][0].numpy()               # (N,C,12,2)
        goals = (out["goal_coords"][0].numpy() if out.get("goal_coords") is not None
                 else np.zeros((cand.shape[0], 0, 2)))       # (N,Kg,2)
        fut = s["future"]
        vm = s.get("valid_mask", np.ones(fut.shape[:2], bool))
        agents = []
        for n in range(fut.shape[0]):
            if not s["target_mask"][n]:
                continue                                       # context agent, not scored
            v = vm[n]
            if v.sum() == 0:
                continue                                       # no real future to score
            err = np.linalg.norm(pred[n] - fut[n][None], axis=-1)   # (K,12)
            ade = (err * v).sum(1) / v.sum()
            last = np.where(v)[0][-1]
            fde = err[:, last]
            best = int(np.argmin(ade))
            p = np.exp(lp[n] - lp[n].max()); p = p / p.sum()
            all_ade.append(float(ade.min())); all_fde.append(float(fde.min()))
            all_miss.append(float(fde.min() > MISS_THRESHOLD))
            agents.append({
                "id": n,
                "past": r(s["past"][n][:, :2]),
                "future": r(fut[n]),
                "valid": v.astype(int).tolist(),
                "pred": r(pred[n]),
                "prob": r(p, 3),
                "cand": r(cand[n][::3], 1),
                "goals": r(goals[n]),
                "ade": r(ade), "fde": r(fde), "best": best,
                "minADE": float(round(ade.min(), 2)), "minFDE": float(round(fde.min(), 2)),
            })
        scenes.append({"id": i, "agents": agents})
    summary = {
        "scenes": len(scenes), "agents": len(all_ade),
        "minADE": round(float(np.mean(all_ade)), 2),
        "minFDE": round(float(np.mean(all_fde)), 2),
        "medADE": round(float(np.median(all_ade)), 2),
        "medFDE": round(float(np.median(all_fde)), 2),
        "miss": round(float(np.mean(all_miss)) * 100, 1),
        "epoch": int(ck.get("epoch", -1)),
        "K": int(cfg["model"]["num_predictions"]),
        "C": int(cfg["model"]["num_candidates"]),
        "params": int(sum(p.numel() for p in model.parameters())),
    }
    return {"summary": summary, "scenes": scenes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/nuscenes_mini/ours/seed0/best.pt")
    ap.add_argument("--out", default="demo/index.html")
    a = ap.parse_args()
    data = build(a.checkpoint)
    tpl = open("demo/template.html", encoding="utf-8").read()
    html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    open(a.out, "w", encoding="utf-8").write(html)
    print("wrote", a.out, data["summary"])


if __name__ == "__main__":
    main()
