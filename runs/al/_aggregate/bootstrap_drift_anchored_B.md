# Bootstrap CI — drift_anchored_B

K = 5 seeds × n_bootstrap = 2000 per seed = 10,000 pooled bootstrap resamples

## Per-seed point estimates

| run_id | seed | n | F1w | R_phish | R_benign |
|---|---:|---:|---|---|---|
| `drift_anchored_B_seed2025` | 2025 | 956 | 0.7947 [0.7702, 0.8199] | 0.6253 [0.5824, 0.6702] | 0.9693 [0.9530, 0.9836] |
| `drift_anchored_B_seed2026` | 2026 | 956 | 0.7844 [0.7582, 0.8102] | 0.6081 [0.5632, 0.6531] | 0.9673 [0.9509, 0.9816] |
| `drift_anchored_B_seed2027` | 2027 | 956 | 0.7836 [0.7581, 0.8090] | 0.5996 [0.5546, 0.6445] | 0.9755 [0.9611, 0.9877] |
| `drift_anchored_B_seed2028` | 2028 | 956 | 0.7890 [0.7637, 0.8134] | 0.6039 [0.5610, 0.6488] | 0.9816 [0.9693, 0.9918] |
| `drift_anchored_B_seed2029` | 2029 | 956 | 0.8049 [0.7790, 0.8302] | 0.6424 [0.5953, 0.6874] | 0.9714 [0.9550, 0.9857] |

## Across-seed aggregate (pooled stratified bootstrap)

| Metric | mean of points | std (n-1, K seeds) | pooled mean ± half_CI |
|---|---:|---:|---:|
| f1_weighted | 0.7913 | 0.0088 | 0.7913 ± 0.0294 [0.7626, 0.8214] |
| f1_phish | 0.7490 | 0.0125 | 0.7490 ± 0.0407 [0.7090, 0.7903] |
| precision_phish | 0.9562 | 0.0085 | 0.9562 ± 0.0272 [0.9276, 0.9821] |
| recall_phish | 0.6158 | 0.0178 | 0.6161 ± 0.0546 [0.5632, 0.6724] |
| precision_benign | 0.7263 | 0.0088 | 0.7266 ± 0.0278 [0.7007, 0.7564] |
| recall_benign | 0.9730 | 0.0057 | 0.9729 ± 0.0174 [0.9550, 0.9898] |
| accuracy | 0.7985 | 0.0078 | 0.7986 ± 0.0267 [0.7730, 0.8264] |
| auc | 0.9329 | 0.0089 | 0.9328 ± 0.0214 [0.9111, 0.9538] |

**Reporting convention**: paper headline uses `pooled_mean ± pooled_ci_half` (stratified bootstrap across K seeds × 2000 resamples each). `std_across_seeds` is the seed-to-seed variability of the point estimates (sanity).