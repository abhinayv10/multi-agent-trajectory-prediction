"""
scripts/investigate_ranking.py

Why does the full model lose to the LSTM at best-of-6, and why does its
top-ranked prediction lose to constant velocity? This script looks inside the
trained models on the VALIDATION split only (the test split is never read), so
whatever it suggests can be tried without tuning on the test set.

For every checkpoint it scores all candidates of every target agent and then
compares ways of choosing from the same candidates:

  best-of-6 rules
    ranker6      the model's own choice (top 6 by ranking score)
    first6       candidates 0-5, i.e. one per goal in goal-score order (the
                 "no ranking" rule, applied to the trained full model)
    random6      6 candidates drawn at random (mean of 20 draws)
    oracle20     best of all 20 candidates (upper bound for any rule)
  top-1 rules
    ranker1      the model's top-ranked candidate
    first1       candidate 0 (best-scored goal, CVAE prior mean)
    cv           constant velocity (mode 0 of the CV baseline)
    near_cv1     the candidate whose end point is closest to the CV end point

It also measures the ranking itself (how often the top-1 is among the 3 best
candidates; rank correlation between score and error) and the speed bias of the
top-1 (predicted / true end-point displacement).

Usage:
    python scripts/investigate_ranking.py                       # ours + lstm, 5 scenes x 3 seeds
    python scripts/investigate_ranking.py --variants ours --scenes zara1 --seeds 0

Writes results/investigation_val_<date>.json and prints a summary table.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from data.dataset import DatasetFactory, build_dataloader   # noqa: E402
from eval import load_model                                 # noqa: E402
from models.baselines import ConstantVelocity               # noqa: E402
from run_benchmark import git_commit                        # noqa: E402

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def endpoint_and_ade(cands, gt, valid):
    """cands (A, C, T, 2), gt (A, T, 2), valid (A, T) bool -> fde (A, C), ade (A, C)."""
    d = np.linalg.norm(cands - gt[:, None], axis=-1)                     # (A, C, T)
    v = valid[:, None, :].astype(float)
    ade = (d * v).sum(-1) / np.maximum(v.sum(-1), 1)
    last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)        # last valid step
    fde = d[np.arange(len(d)), :, last]
    return fde, ade, last


def spearman_rows(a, b):
    """Mean Spearman correlation between matching rows of a and b."""
    ra = a.argsort(1).argsort(1).astype(float)
    rb = b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    ok = den > 0
    return float(np.mean((ra * rb).sum(1)[ok] / den[ok]))


@torch.no_grad()
def analyse(checkpoint, device, rng):
    model, config = load_model(checkpoint, None, [], device)
    model.eval()
    cv = ConstantVelocity(config, device)
    ds = DatasetFactory(config, split="val")
    loader = build_dataloader(ds, batch_size=config["training"]["batch_size"],
                              max_agents_per_batch=config["training"].get("max_agents_per_batch"))
    torch.manual_seed(config["evaluation"].get("seed", 0))

    rows = {k: [] for k in ("cands", "scores", "gt", "valid", "cv", "last")}
    for batch in loader:
        out = model(batch, teacher_forcing_ratio=0.0)
        tm = batch["target_mask"].bool()
        rows["cands"].append(out["all_candidates"].cpu()[tm].numpy())
        s = out.get("all_log_probs")
        rows["scores"].append(None if s is None else s.cpu()[tm].numpy())
        rows["gt"].append(batch["future"][tm].numpy())
        vm = batch.get("valid_mask")
        rows["valid"].append((vm[tm] if vm is not None else torch.ones(batch["future"][tm].shape[:2])).bool().numpy())
        rows["cv"].append(cv(batch, num_predictions=1)["predictions"].cpu()[tm][:, 0].numpy())
        rows["last"].append(batch["past"][tm][:, -1, :2].numpy())

    cands = np.concatenate(rows["cands"])
    gt, valid = np.concatenate(rows["gt"]), np.concatenate(rows["valid"])
    cvp, last = np.concatenate(rows["cv"]), np.concatenate(rows["last"])
    has_scores = rows["scores"][0] is not None
    scores = np.concatenate(rows["scores"]) if has_scores else None
    A, C = cands.shape[:2]

    fde, ade, lastidx = endpoint_and_ade(cands, gt, valid)
    cv_fde, _, _ = endpoint_and_ade(cvp[:, None], gt, valid)
    cv_fde = cv_fde[:, 0]
    r = {"agents": int(A), "candidates": int(C)}

    order = np.argsort(-scores, 1) if has_scores else np.tile(np.arange(C), (A, 1))
    pick = lambda idx: fde[np.arange(A)[:, None], idx]                   # noqa: E731
    r["minFDE6_ranker6"] = float(pick(order[:, :6]).min(1).mean())
    r["minFDE6_first6"] = float(fde[:, :6].min(1).mean())
    r["minFDE6_random6"] = float(np.mean([pick(np.stack([rng.choice(C, 6, replace=False) for _ in range(A)])).min(1).mean()
                                          for _ in range(20)]))
    r["minFDE20_oracle"] = float(fde.min(1).mean())

    # Spread of the chosen six: mean pairwise distance between their end points.
    def spread(idx):
        e = cands[np.arange(A)[:, None], idx][np.arange(A), :, lastidx]      # (A, 6, 2)
        d = np.linalg.norm(e[:, :, None] - e[:, None], axis=-1)
        return float((d.sum((1, 2)) / (6 * 5)).mean())
    r["spread6_ranker6"] = spread(order[:, :6])
    r["spread6_first6"] = spread(np.tile(np.arange(6), (A, 1)))

    top1 = order[:, 0]
    r["FDE1_ranker1"] = float(fde[np.arange(A), top1].mean())
    r["FDE1_first1"] = float(fde[:, 0].mean())
    r["FDE1_cv"] = float(cv_fde.mean())
    end_cv = cvp[np.arange(A), lastidx]
    end_c = cands[np.arange(A), :, lastidx]                              # (A, C, 2)
    near = np.linalg.norm(end_c - end_cv[:, None], axis=-1).argmin(1)
    r["FDE1_near_cv1"] = float(fde[np.arange(A), near].mean())
    r["ranker1_beats_cv"] = float(np.mean(fde[np.arange(A), top1] < cv_fde))

    best3 = np.argsort(fde, 1)[:, :3]
    r["top1_in_best3"] = float(np.mean((best3 == top1[:, None]).any(1)))
    r["top1_in_best3_chance"] = 3 / C
    if has_scores:
        r["spearman_score_vs_neg_fde"] = spearman_rows(scores, -fde)

    # Speed bias: end-point displacement from the last observed position.
    true_disp = np.linalg.norm(gt[np.arange(A), lastidx] - last, axis=-1)
    moving = true_disp > 1.0
    top1_disp = np.linalg.norm(end_c[np.arange(A), top1] - last, axis=-1)
    cv_disp = np.linalg.norm(end_cv - last, axis=-1)
    r["moving_agents"] = int(moving.sum())
    r["disp_ratio_top1_median"] = float(np.median(top1_disp[moving] / true_disp[moving]))
    r["disp_ratio_cv_median"] = float(np.median(cv_disp[moving] / true_disp[moving]))
    r["top1_shorter_than_truth"] = float(np.mean(top1_disp[moving] < true_disp[moving]))
    r["cv_shorter_than_truth"] = float(np.mean(cv_disp[moving] < true_disp[moving]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["ours", "lstm"])
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    results = {}
    for v in args.variants:
        for sc in args.scenes:
            for seed in args.seeds:
                ck = os.path.join(ROOT, "runs", "eth_ucy", sc, v, f"seed{seed}", "best.pt")
                if not os.path.isfile(ck):
                    print(f"[skip] {ck} missing")
                    continue
                key = f"{v}/{sc}/seed{seed}"
                results[key] = analyse(ck, device, np.random.default_rng(seed))
                print(f"[done] {key}: ranker1 {results[key]['FDE1_ranker1']:.3f}  cv {results[key]['FDE1_cv']:.3f}  "
                      f"ranker6 {results[key]['minFDE6_ranker6']:.3f}  first6 {results[key]['minFDE6_first6']:.3f}",
                      flush=True)

    # Mean over scenes and seeds, per variant.
    summary = {}
    for v in args.variants:
        rs = [r for k, r in results.items() if k.startswith(v + "/")]
        if rs:
            summary[v] = {m: float(np.mean([r[m] for r in rs])) for m in rs[0]
                          if all(m in r for r in rs)}
            summary[v]["runs"] = len(rs)

    date = time.strftime("%Y-%m-%d")
    out = args.out or os.path.join(ROOT, "results", f"investigation_val_{date}.json")
    meta = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit(),
            "split": "val", "variants": args.variants, "scenes": args.scenes, "seeds": args.seeds}
    with open(out, "w") as fh:
        json.dump({"_meta": meta, "summary": summary, "runs": results}, fh, indent=1)

    print("\nMean over scenes and seeds (validation split, metres):")
    for v, s in summary.items():
        print(f"\n[{v}] ({s['runs']} runs)")
        for m, val in s.items():
            if m != "runs":
                print(f"  {m:<28} {val:.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
