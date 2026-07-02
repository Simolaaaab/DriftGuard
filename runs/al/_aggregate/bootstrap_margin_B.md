# Bootstrap CI — margin_B

K = 5 seeds × n_bootstrap = 2000 per seed = 10,000 pooled bootstrap resamples

## Per-seed point estimates

| run_id | seed | n | F1w | R_phish | R_benign |
|---|---:|---:|---|---|---|
| `margin_B_seed2025` | 2025 | 956 | 0.7771 [0.7510, 0.8025] | 0.5931 [0.5482, 0.6381] | 0.9693 [0.9530, 0.9836] |
| `margin_B_seed2026` | 2026 | 956 | 0.7602 [0.7332, 0.7860] | 0.5610 [0.5161, 0.6039] | 0.9714 [0.9550, 0.9857] |
| `margin_B_seed2027` | 2027 | 956 | 0.7747 [0.7495, 0.8006] | 0.5889 [0.5460, 0.6338] | 0.9693 [0.9530, 0.9836] |
| `margin_B_seed2028` | 2028 | 956 | 0.7757 [0.7490, 0.8005] | 0.5782 [0.5332, 0.6231] | 0.9836 [0.9714, 0.9939] |
| `margin_B_seed2029` | 2029 | 956 | 0.8059 [0.7792, 0.8314] | 0.6424 [0.5953, 0.6874] | 0.9734 [0.9591, 0.9857] |

## Across-seed aggregate (pooled stratified bootstrap)

| Metric | mean of points | std (n-1, K seeds) | pooled mean ± half_CI |
|---|---:|---:|---:|
| f1_weighted | 0.7787 | 0.0167 | 0.7787 ± 0.0389 [0.7429, 0.8207] |
| f1_phish | 0.7312 | 0.0234 | 0.7310 ± 0.0543 [0.6804, 0.7890] |
| precision_phish | 0.9552 | 0.0099 | 0.9551 ± 0.0284 [0.9258, 0.9825] |
| recall_phish | 0.5927 | 0.0304 | 0.5929 ± 0.0696 [0.5310, 0.6702] |
| precision_benign | 0.7148 | 0.0154 | 0.7150 ± 0.0355 [0.6845, 0.7555] |
| recall_benign | 0.9734 | 0.0060 | 0.9733 ± 0.0174 [0.9550, 0.9898] |
| accuracy | 0.7874 | 0.0149 | 0.7875 ± 0.0345 [0.7563, 0.8253] |
| auc | 0.9316 | 0.0104 | 0.9315 ± 0.0231 [0.9089, 0.9551] |

**Reporting convention**: paper headline uses `pooled_mean ± pooled_ci_half` (stratified bootstrap across K seeds × 2000 resamples each). `std_across_seeds` is the seed-to-seed variability of the point estimates (sanity).