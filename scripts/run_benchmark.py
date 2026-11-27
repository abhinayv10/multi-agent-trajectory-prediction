"""
scripts/run_benchmark.py

One command to reproduce every reported number:

    python scripts/run_benchmark.py                       # full ETH/UCY study
    python scripts/run_benchmark.py --dataset nuscenes_mini
    python scripts/run_benchmark.py --eval-only           # re-evaluate existing runs
    python scripts/run_benchmark.py --aggregate-only      # rebuild tables only

For every (test scene x variant x seed) it trains a model (skipped when the
run already finished), evaluates it on the held-out split, then aggregates
mean +/- std over seeds into results/<dataset>_results.json and
results/<dataset>_results.md.

Variants:
    cv              constant-velocity baseline (no training; seed = sampling seed)
    lstm            plain LSTM encoder-decoder baseline
    ours            full model
    no_transformer  temporal Transformer replaced by an LSTM
    no_social       GAT + decoder social attention removed
    no_cvae         CVAE removed (one deterministic decode per goal)
    no_goals        goal predictor removed (CVAE samples only)
    no_ranking      ranking network removed (one sample per goal, by goal score)
The map encoder cannot be ablated on ETH/UCY (no map data exists).

Runs are independent subprocesses; --jobs controls parallelism and
--threads the torch threads of each job.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = {
    "cv":             ["model.type=cv"],
    "lstm":           ["model.type=lstm"],
    "ours":           [],
    "no_transformer": ["model.temporal_encoder=lstm"],
    "no_social":      ["model.use_social=false"],
    "no_cvae":        ["model.use_cvae=false"],
    "no_goals":       ["model.use_goals=false"],
    "no_ranking":     ["model.use_ranking=false"],
    "ours_per_goal":  [],
}
# Evaluation-only variants: re-evaluate another variant's trained checkpoint
# with different inference settings (no training of their own).
EVAL_ONLY = {
    "ours_per_goal": ("ours", ["model.selection=per_goal"]),
}
MAIN_VARIANTS = ["cv", "lstm", "ours", "ours_per_goal"]
ABLATIONS = ["ours", "no_transformer", "no_social", "no_cvae", "no_goals", "no_ranking"]
LABELS = {
    "cv": "Constant velocity",
    "lstm": "LSTM enc-dec",
    "ours": "Ours (full)",
    "ours_per_goal": "Ours, one candidate per goal",
    "no_transformer": "- Transformer (LSTM encoder)",
    "no_social": "- social (GAT + decoder attn)",
    "no_cvae": "- CVAE",
    "no_goals": "- goal predictor",
    "no_ranking": "- ranking network",
}

DATASETS = {
    "eth_ucy": {"config": "configs/eth_ucy.yaml",
                "scenes": ["eth", "hotel", "univ", "zara1", "zara2"]},
    "nuscenes_mini": {"config": "configs/nuscenes_mini.yaml", "scenes": [None]},
}


def run_dir(dataset, scene, variant, seed):
    parts = ["runs", dataset] + ([scene] if scene else []) + [variant, f"seed{seed}"]
    return os.path.join(ROOT, *parts)


def scene_overrides(scene):
    return [f"dataset.test_scene={scene}"] if scene else []


def run(cmd, log_path):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONUTF8": "1"})
    return proc.returncode


def job(args, dataset, cfg_path, scene, variant, seed):
    """Train (if needed) and evaluate one run. Returns (key, status)."""
    d = run_dir(dataset, scene, variant, seed)
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "log.txt")
    ckpt = os.path.join(d, "best.pt")
    done = os.path.join(d, "train_done")
    result = os.path.join(d, "test.json")
    overrides = scene_overrides(scene) + VARIANTS[variant] + list(args.set)
    key = f"{scene or dataset}/{variant}/seed{seed}"
    t0 = time.time()

    eval_set = []
    if variant in EVAL_ONLY:
        base, eval_set = EVAL_ONLY[variant]
        ckpt = os.path.join(run_dir(dataset, scene, base, seed), "best.pt")

    if variant != "cv" and variant not in EVAL_ONLY and not args.eval_only and not os.path.isfile(done):
        cmd = [sys.executable, "train.py", "--config", cfg_path, "--seed", str(seed),
               "--checkpoint", ckpt, "--threads", str(args.threads), "--set", *overrides]
        if run(cmd, log) != 0:
            return key, f"TRAIN FAILED (see {log})"
        open(done, "w").close()

    if args.skip_existing_eval and os.path.isfile(result):
        return key, "cached"
    if variant == "cv":
        cmd = [sys.executable, "eval.py", "--config", cfg_path, "--threads", str(args.threads),
               "--out", result, "--set", *overrides, f"evaluation.seed={seed}"]
    else:
        if not os.path.isfile(ckpt):
            return key, "no checkpoint"
        cmd = [sys.executable, "eval.py", "--checkpoint", ckpt, "--threads", str(args.threads),
               "--out", result, "--set", f"evaluation.seed={seed}", *eval_set]
    if run(cmd, log) != 0:
        return key, f"EVAL FAILED (see {log})"
    return key, f"ok ({(time.time() - t0) / 60:.1f} min)"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
METRICS = ["min_ade_1", "min_fde_1", "min_ade_6", "min_fde_6", "miss_rate_6",
           "min_ade_20", "min_fde_20"]


def load_results(dataset, scenes, variants, seeds):
    """{variant: {scene: {seed: result_json}}} for runs that have results."""
    out = {}
    for v in variants:
        for s in scenes:
            for seed in seeds:
                p = os.path.join(run_dir(dataset, s, v, seed), "test.json")
                if os.path.isfile(p):
                    with open(p) as fh:
                        out.setdefault(v, {}).setdefault(s, {})[seed] = json.load(fh)
    return out


def git_commit():
    """Short HEAD hash, with "-dirty" when tracked code or configs differ from it,
    so a result is never credited to a commit that did not produce it."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "*.py", "configs"],
                                 cwd=ROOT).returncode != 0
        return commit + ("-dirty" if changed else "")
    except Exception:
        return None


def fmt(vals, digits=2):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}"


def scene_avg(res_v, scenes, seed, metric):
    """Mean over scenes for one seed (None if any scene is missing)."""
    vals = []
    for s in scenes:
        r = res_v.get(s, {}).get(seed)
        if r is None or metric not in r["overall"]:
            return None
        vals.append(r["overall"][metric])
    return float(np.mean(vals))


def aggregate(dataset, scenes, seeds):
    res = load_results(dataset, scenes, list(VARIANTS), seeds)
    summary = {}
    for v, by_scene in res.items():
        summary[v] = {"per_scene": {}, "average": {}}
        for s in scenes:
            runs = by_scene.get(s, {})
            summary[v]["per_scene"][s or dataset] = {
                m: [runs[k]["overall"].get(m) for k in sorted(runs)] for m in METRICS}
            summary[v]["per_scene"][s or dataset]["seeds"] = sorted(runs)
        for m in METRICS:
            summary[v]["average"][m] = [scene_avg(by_scene, scenes, seed, m) for seed in seeds]
        timings = [r["timing"] for runs in by_scene.values() for r in runs.values() if r.get("timing")]
        if timings:
            summary[v]["latency_ms"] = float(np.mean([t["ms_per_scene_mean"] for t in timings]))
            summary[v]["latency_agents"] = float(np.mean([t["agents_per_scene_mean"] for t in timings]))
            summary[v]["latency_threads"] = timings[0]["threads"]

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    commit = git_commit()
    summary["_meta"] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit,
        "config": DATASETS[dataset]["config"], "seeds": seeds, "scenes": scenes,
        "variant_overrides": VARIANTS,
        "note": "each run's full resolved config is in runs/<dataset>/.../test.json under 'config'",
    }
    with open(os.path.join(ROOT, "results", f"{dataset}_results.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    md = render_markdown(dataset, scenes, seeds, {k: v for k, v in summary.items() if k != "_meta"}, res)
    md += f"\nGenerated {summary['_meta']['generated']} from git commit {commit}.\n"
    with open(os.path.join(ROOT, "results", f"{dataset}_results.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


def render_markdown(dataset, scenes, seeds, summary, res):
    L = [f"# Results: {dataset}", "",
         f"Mean ± std over seeds {seeds} (sample std). Errors in metres; "
         "MR@6 = fraction of agents with minFDE@6 > 2 m. "
         "Predictions are ranked by model confidence, so @1 is the top-ranked mode.", ""]

    def has(v):
        return v in summary

    # --- headline table (scene average) ---
    L += ["## Main comparison (average over test scenes)", "",
          "| Method | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | MR@6 | minADE@20 | minFDE@20 | ms / scene |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in MAIN_VARIANTS + [a for a in ABLATIONS if a != "ours"]:
        if not has(v):
            continue
        if v == ABLATIONS[1]:
            L.append("| *Ablations* | | | | | | | | |")
        a = summary[v]["average"]
        lat = summary[v].get("latency_ms")
        lat_cell = f"{lat:.1f}" if lat is not None else "-"
        L.append(f"| {LABELS[v]} | " + " | ".join(fmt(a[m]) for m in METRICS) + f" | {lat_cell} |")
    L.append("")

    # --- per-scene table (minADE@6 / minFDE@6) ---
    if len(scenes) > 1:
        L += ["## Per scene: minADE@6 / minFDE@6", "",
              "| Method | " + " | ".join(s.upper() for s in scenes) + " | AVG |",
              "|---|" + "---|" * (len(scenes) + 1)]
        for v in MAIN_VARIANTS + [a for a in ABLATIONS if a != "ours"]:
            if not has(v):
                continue
            cells = []
            for s in scenes:
                ps = summary[v]["per_scene"][s]
                cells.append(f"{fmt(ps['min_ade_6'])} / {fmt(ps['min_fde_6'])}")
            a = summary[v]["average"]
            cells.append(f"{fmt(a['min_ade_6'])} / {fmt(a['min_fde_6'])}")
            L.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
        L.append("")

        L += ["## Per scene: minADE@20 / minFDE@20 (the protocol most ETH/UCY papers report)", "",
              "| Method | " + " | ".join(s.upper() for s in scenes) + " | AVG |",
              "|---|" + "---|" * (len(scenes) + 1)]
        for v in MAIN_VARIANTS:
            if not has(v):
                continue
            cells = []
            for s in scenes:
                ps = summary[v]["per_scene"][s]
                cells.append(f"{fmt(ps['min_ade_20'])} / {fmt(ps['min_fde_20'])}")
            a = summary[v]["average"]
            cells.append(f"{fmt(a['min_ade_20'])} / {fmt(a['min_fde_20'])}")
            L.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
        L.append("")

    # --- per agent type (nuScenes) ---
    if dataset == "nuscenes_mini":
        L += ["## By agent type", "",
              "| Method | Type | agents | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | MR@6 |",
              "|---|---|---|---|---|---|---|---|"]
        for v in MAIN_VARIANTS:
            runs = res.get(v, {}).get(None, {})
            if not runs:
                continue
            types = sorted({t for r in runs.values() for t in r["by_type"]})
            for t in types:
                rows = [r["by_type"][t] for r in runs.values() if t in r["by_type"]]
                cnt = rows[0]["count"]
                cells = [fmt([r.get(m) for r in rows]) for m in
                         ["min_ade_1", "min_fde_1", "min_ade_6", "min_fde_6", "miss_rate_6"]]
                L.append(f"| {LABELS[v]} | {t} | {cnt} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def keep_awake():
    """Ask Windows not to sleep while the benchmark runs (released when the
    process exits; power settings are not changed). No-op elsewhere."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    keep_awake()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="eth_ucy", choices=list(DATASETS))
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=3, help="parallel runs")
    ap.add_argument("--threads", type=int, default=4, help="torch threads per run")
    ap.add_argument("--set", nargs="*", default=[], help="extra config overrides for every run")
    ap.add_argument("--eval-only", action="store_true", help="never train; evaluate what exists")
    ap.add_argument("--skip-existing-eval", action="store_true",
                    help="do not re-evaluate runs that already have test.json")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    scenes = args.scenes or spec["scenes"]
    if args.dataset == "eth_ucy" and not args.aggregate_only:
        subprocess.run([sys.executable, "scripts/download_eth_ucy.py"], cwd=ROOT, check=True)

    if not args.aggregate_only:
        # Cheap / headline runs first so partial results are useful early.
        order = sorted(args.variants, key=lambda v: (list(VARIANTS).index(v)))
        jobs = [(s, v, seed) for v in order for seed in args.seeds for s in scenes]
        print(f"[benchmark] {len(jobs)} runs, {args.jobs} in parallel x {args.threads} threads")
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(job, args, args.dataset, spec["config"], s, v, seed)
                    for s, v, seed in jobs]
            for f in cf.as_completed(futs):
                key, status = f.result()
                print(f"[benchmark] {key}: {status}", flush=True)

    aggregate(args.dataset, scenes, args.seeds)


if __name__ == "__main__":
    main()
