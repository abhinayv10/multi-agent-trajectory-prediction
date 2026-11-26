"""
scripts/ablation_table.py

Build the ablation table for one scene / seed from runs/**/test.json and
write it into README.md between the <!-- ablation:start/end --> markers.
Variants that have not finished are listed as "running".

Usage:
    python scripts/ablation_table.py [--scene zara1] [--seed 0]
"""

import argparse
import json
import os

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="zara1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = {}
    for v, _ in ROWS:
        p = os.path.join(ROOT, "runs", "eth_ucy", args.scene, v, f"seed{args.seed}", "test.json")
        if os.path.isfile(p):
            with open(p) as fh:
                res[v] = json.load(fh)["overall"]
    base = res.get("ours")

    lines = ["| Variant | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | minADE@20 | minFDE@20 | ΔminFDE@6 vs full |",
             "|---|---|---|---|---|---|---|---|"]
    for v, label in ROWS:
        if v not in res:
            lines.append(f"| {label} | *running* | | | | | | |")
            continue
        r = res[v]
        cells = [f"{r[m]:.2f}" if m in r else "-" for m in METRICS]
        delta = "-" if v == "ours" or base is None else f"{r['min_fde_6'] - base['min_fde_6']:+.2f}"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {delta} |")
    table = "\n".join(lines)

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
