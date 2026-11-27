# Results: nuscenes_mini

Mean ± std over seeds [0, 1, 2] (sample std). Errors in metres; MR@6 = fraction of agents with minFDE@6 > 2 m. Predictions are ranked by model confidence, so @1 is the top-ranked mode.

## Main comparison (average over test scenes)

| Method | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | MR@6 | minADE@20 | minFDE@20 | ms / scene |
|---|---|---|---|---|---|---|---|---|
| Constant velocity | 0.68 ± 0.00 | 1.48 ± 0.00 | 0.53 ± 0.00 | 1.07 ± 0.01 | 0.16 ± 0.00 | 0.45 ± 0.00 | 0.85 ± 0.00 | 0.2 |
| LSTM enc-dec | 1.00 ± 0.09 | 2.05 ± 0.16 | 0.61 ± 0.02 | 1.14 ± 0.06 | 0.10 ± 0.03 | 0.57 ± 0.02 | 1.04 ± 0.05 | 26.9 |
| Ours (full) | 1.22 ± 0.14 | 2.36 ± 0.19 | 0.85 ± 0.11 | 1.64 ± 0.14 | 0.20 ± 0.03 | 0.60 ± 0.05 | 1.12 ± 0.07 | 52.2 |

## By agent type

| Method | Type | agents | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | MR@6 |
|---|---|---|---|---|---|---|---|
| Constant velocity | cyclist | 6 | 0.93 ± 0.00 | 1.82 ± 0.00 | 0.93 ± 0.00 | 1.82 ± 0.00 | 0.50 ± 0.00 |
| Constant velocity | pedestrian | 594 | 0.67 ± 0.00 | 1.47 ± 0.00 | 0.52 ± 0.00 | 1.06 ± 0.01 | 0.15 ± 0.00 |
| LSTM enc-dec | cyclist | 6 | 12.77 ± 0.14 | 24.21 ± 0.26 | 12.77 ± 0.14 | 24.21 ± 0.26 | 1.00 ± 0.00 |
| LSTM enc-dec | pedestrian | 594 | 0.88 ± 0.09 | 1.82 ± 0.16 | 0.49 ± 0.02 | 0.91 ± 0.06 | 0.09 ± 0.03 |
| Ours (full) | cyclist | 6 | 14.75 ± 0.30 | 27.55 ± 0.56 | 13.58 ± 0.35 | 25.29 ± 0.69 | 1.00 ± 0.00 |
| Ours (full) | pedestrian | 594 | 1.09 ± 0.14 | 2.10 ± 0.19 | 0.72 ± 0.11 | 1.40 ± 0.14 | 0.20 ± 0.03 |

Generated 2026-11-27T04:26:27 from git commit 3af4679.
