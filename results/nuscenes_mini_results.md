# Results: nuscenes_mini

Mean ± std over seeds [0, 1, 2] (sample std). Errors in metres; MR@6 = fraction of agents with minFDE@6 > 2 m. Predictions are ranked by model confidence, so @1 is the top-ranked mode.

## Main comparison (average over test scenes)

| Method | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | MR@6 | minADE@20 | minFDE@20 | ms / scene |
|---|---|---|---|---|---|---|---|---|
| Constant velocity | 0.68 ± 0.00 | 1.48 ± 0.00 | 0.53 ± 0.00 | 1.07 ± 0.01 | 0.16 ± 0.00 | 0.45 ± 0.00 | 0.85 ± 0.00 | 0.4 |
| LSTM enc-dec | 1.00 ± 0.09 | 2.07 ± 0.18 | 0.57 ± 0.01 | 1.06 ± 0.01 | 0.07 ± 0.02 | 0.51 ± 0.04 | 0.94 ± 0.08 | 87.7 |
| Ours (full) | 1.05 | 2.09 | 0.78 | 1.56 | 0.19 | 0.60 | 1.11 | 149.1 |

## By agent type

| Method | Type | agents | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | MR@6 |
|---|---|---|---|---|---|---|---|
| Constant velocity | cyclist | 6 | 0.93 ± 0.00 | 1.82 ± 0.00 | 0.93 ± 0.00 | 1.82 ± 0.00 | 0.50 ± 0.00 |
| Constant velocity | pedestrian | 594 | 0.67 ± 0.00 | 1.47 ± 0.00 | 0.52 ± 0.00 | 1.06 ± 0.01 | 0.15 ± 0.00 |
| LSTM enc-dec | cyclist | 6 | 12.80 ± 0.33 | 24.25 ± 0.42 | 12.71 ± 0.26 | 24.06 ± 0.32 | 1.00 ± 0.00 |
| LSTM enc-dec | pedestrian | 594 | 0.88 ± 0.09 | 1.85 ± 0.18 | 0.44 ± 0.01 | 0.83 ± 0.02 | 0.07 ± 0.02 |
| Ours (full) | cyclist | 6 | 14.31 | 26.72 | 13.65 | 25.45 | 1.00 |
| Ours (full) | pedestrian | 594 | 0.91 | 1.84 | 0.65 | 1.32 | 0.18 |
