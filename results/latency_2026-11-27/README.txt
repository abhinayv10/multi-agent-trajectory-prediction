Inference latency, measured 2026-11-27 at git commit 75e4dcb, with no other jobs running.
CPU: 12th Gen Intel Core i7-12700H, 4 threads, batch size 1, seed-0 checkpoints, test split.
Commands (from the repo root):
  python eval.py --config configs/eth_ucy.yaml --threads 4 --set model.type=cv dataset.test_scene=<scene> evaluation.seed=0 --out results/latency_2026-11-27/<scene>_cv.json
  python eval.py --checkpoint runs/eth_ucy/<scene>/<lstm|ours>/seed0/best.pt --threads 4 --out results/latency_2026-11-27/<scene>_<model>.json
Latency is in each file under "timing". The metrics in these files equal the matching runs/**/test.json.
The "ms / scene" column in eth_ucy_results.md was measured with 3 training jobs sharing the CPU and is not reliable.
