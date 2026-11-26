# Results: eth_ucy

Mean ± std over seeds [0] (sample std). Errors in metres; MR@6 = fraction of agents with minFDE@6 > 2 m. Predictions are ranked by model confidence, so @1 is the top-ranked mode.

## Main comparison (average over test scenes)

| Method | minADE@1 | minFDE@1 | minADE@6 | minFDE@6 | MR@6 | minADE@20 | minFDE@20 | ms / scene |
|---|---|---|---|---|---|---|---|---|
| Constant velocity | 0.43 | 0.95 | 0.34 | 0.73 | 0.05 | 0.30 | 0.61 | 0.1 |
| LSTM enc-dec | 0.43 | 0.97 | 0.31 | 0.65 | 0.05 | 0.20 | 0.37 | 15.6 |
| Ours (full) | 0.80 | 1.63 | 0.40 | 0.84 | 0.08 | 0.25 | 0.48 | 38.7 |
| *Ablations* | | | | | | | | |
| - Transformer (LSTM encoder) | 0.71 | 1.44 | 0.39 | 0.82 | 0.07 | 0.25 | 0.46 | 26.2 |
| - social (GAT + decoder attn) | 0.76 | 1.57 | 0.40 | 0.85 | 0.07 | 0.25 | 0.49 | 20.1 |
| - CVAE | 0.81 | 1.64 | 0.37 | 0.77 | 0.06 | 0.27 | 0.50 | 15.0 |
| - goal predictor | 0.60 | 1.28 | 0.34 | 0.71 | 0.06 | 0.24 | 0.45 | 7.5 |
