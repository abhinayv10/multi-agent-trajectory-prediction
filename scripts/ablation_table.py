"""
scripts/ablation_table.py

Build the ablation table for one scene / seed from runs/**/test.json and
write it into README.md between the <!-- ablation:start/end --> markers.
Mean ± std over the seeds that finished; variants with no finished run are
listed as "not run". Raw per-seed metrics, the date and the git commit are
saved to results/ablation_<scene>.json.

Usage:
    python scripts/ablation_table.py [--scene zara1] [--seeds 0 1 2]
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = [
    ("ours", "Full model"),
    ("no_transformer", "− Transformer (LSTM history encoder)"),
    ("no_social", "− social (GAT + decoder social attention)"),
    ("no_cvae", "− CVAE (one deterministic decode per goal)"),
    ("no_goals", "− goal predictor (CVAE samples only)"),
    ("no_ranking", "− ranking network (one sample per goal)"),
]
METRICS = ["min_ade_1", "min_fde_1", "min_ade_6", "min_fde_6", "min_ade_20", "min_fde_20"]
START, END = "<!-- ablation:start -->", "<!-- ablation:end -->"


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def fmt(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console: print Δ and ±
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="zara1")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    res = {}   # variant -> {seed: overall metrics}
    for v, _ in ROWS:
        for seed in args.seeds:
            p = os.path.join(ROOT, "runs", "eth_ucy", args.scene, v, f"seed{seed}", "test.json")
            if os.path.isfile(p):
                with open(p) as fh:
                    res.setdefault(v, {})[seed] = json.load(fh)["overall"]

    seed_note = ", ".join(str(s) for s in args.seeds)
    lines = [f"| Variant | Seeds | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | minADE@20 | minFDE@20 | ΔminFDE@6 vs full |",
             "|---|---|---|---|---|---|---|---|---|"]
    base = res.get("ours", {})
    for v, label in ROWS:
        runs = res.get(v, {})
        if not runs:
            lines.append(f"| {label} | 0 | *not run* | | | | | | |")
            continue
        cells = [fmt([r[m] for r in runs.values() if m in r]) for m in METRICS]
        # paired delta: only seeds where both this variant and the full model finished
        paired = [runs[s]["min_fde_6"] - base[s]["min_fde_6"] for s in runs if s in base]
        delta = "-" if v == "ours" or not paired else (
            f"{np.mean(paired):+.2f}" if len(paired) == 1 else
            f"{np.mean(paired):+.2f} ± {np.std(paired, ddof=1):.2f}")
        lines.append(f"| {label} | {len(runs)} | " + " | ".join(cells) + f" | {delta} |")
    table = "\n".join(lines)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"ablation_{args.scene}.json"), "w") as fh:
        json.dump({"generated": datetime.datetime.now().isoformat(timespec="seconds"),
                   "git_commit": git_commit(), "scene": args.scene, "seeds_requested": args.seeds,
                   "runs": {v: {str(k): r for k, r in d.items()} for v, d in res.items()}}, fh, indent=1)

    path = os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    block = f"{START}\n{table}\n{END}"
    if START in text:
        a, b = text.index(START), text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    elif "ABLATION_TABLE" in text:
        text = text.replace("ABLATION_TABLE", block)
    else:
        raise SystemExit("README has no ablation markers")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(table)


if __name__ == "__main__":
    main()
