"""
demo_pedestrian.py
------------------
Builds demo/index.html: a side-by-side view of one walking pedestrian.
Left shows what actually happened, right shows what the trained model predicted.

Cases are moving pedestrians (speed > 0.5 m/s) from the nuScenes-mini
test scenes (official mini_val); every scored agent has a full 6 s future. Eight cases are
chosen evenly across the accuracy ranking so the page is not cherry-picked.

Usage: python demo_pedestrian.py [--checkpoint runs/nuscenes_mini/ours/seed0/best.pt]
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from data.dataset import AGENT_TYPES, DatasetFactory, collate_scenes
from demo import r
from eval import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/nuscenes_mini/ours/seed0/best.pt")
    args = ap.parse_args()
    model, cfg = load_model(args.checkpoint, None, [], torch.device("cpu"))
    model.eval()
    ds = DatasetFactory(cfg, split="test")

    cases = []
    for i in range(len(ds)):
        s = ds[i]
        peds = [n for n in range(len(s["mask"]))
                if s["target_mask"][n] and s["agent_type"][n] == AGENT_TYPES["pedestrian"]]
        if not peds:
            continue
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(collate_scenes([s]), teacher_forcing_ratio=0.0)
        pred = out["predictions"][0].numpy()
        lp = out["log_probs"][0].numpy()
        for n in peds:
            if np.linalg.norm(s["past"][n][-1][2:]) < 0.5:
                continue
            fut = s["future"][n]
            err = np.linalg.norm(pred[n] - fut[None], axis=-1)
            ade, fde = err.mean(1), err[:, -1]
            p = np.exp(lp[n] - lp[n].max()); p /= p.sum()
            others = []
            for m in range(len(s["mask"])):
                if m != n and np.linalg.norm(s["past"][m][-1][:2] - s["past"][n][-1][:2]) < 30:
                    others.append({"past": r(s["past"][m][:, :2]), "future": r(s["future"][m])})
            cases.append({
                "sample": i, "agent": n, "category": "pedestrian",
                "speed": round(float(np.linalg.norm(s["past"][n][-1][2:])), 2),
                "past": r(s["past"][n][:, :2]), "future": r(fut), "pred": r(pred[n]),
                "prob": r(p, 3), "ade": r(ade), "fde": r(fde), "best": int(np.argmin(ade)),
                "minADE": round(float(ade.min()), 2), "minFDE": round(float(fde.min()), 2),
                "others": others,
            })
    cases.sort(key=lambda c: c["minADE"])
    picks = [cases[int(round(k))] for k in np.linspace(0, len(cases) - 1, 8)]
    # keep distinct scenes
    seen, final = set(), []
    for c in picks:
        if c["sample"] not in seen:
            seen.add(c["sample"]); final.append(c)
    data = {"cases": final, "pool": len(cases)}
    tpl = open("demo/pedestrian_template.html", encoding="utf-8").read()
    open("demo/index.html", "w", encoding="utf-8").write(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    print("pool", len(cases), [(c["sample"], c["minADE"]) for c in final])


if __name__ == "__main__":
    main()
