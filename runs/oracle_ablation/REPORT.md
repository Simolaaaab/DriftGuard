# Step 2 — Oracle Ablation Results

Test set: 100 samples (50 phish, 50 benign)
  - FN seeds: 50
  - FP seeds: 50
  - Uncertain phish fillers: 0
  - Uncertain benign fillers: 0

## Headline metrics (overall, n=test-set size)

| Variant | Acc | F1w | P_phish | R_phish | P_benign | R_benign | uncertain |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_html_probs_osint | 0.770 | 0.761 | 0.935 | 0.580 | 0.696 | 0.960 | 0 |
| A_html_probs | 0.690 | 0.676 | 0.828 | 0.480 | 0.634 | 0.900 | 0 |
| E_poc_2k | 0.820 | 0.814 | 1.000 | 0.640 | 0.735 | 1.000 | 0 |
| D_html_osint | 0.910 | 0.909 | 1.000 | 0.820 | 0.847 | 1.000 | 0 |
| C_html_only | 0.900 | 0.899 | 1.000 | 0.800 | 0.833 | 1.000 | 0 |

## FN recovery — how many of the ensemble's missed phishings the oracle catches

| Variant | FN recovery rate | uncertain |
|---|---:|---:|
| B_html_probs_osint | 0.580 (28 / 50) | 0 |
| A_html_probs | 0.480 (24 / 50) | 0 |
| E_poc_2k | 0.640 (32 / 50) | 0 |
| D_html_osint | 0.820 (41 / 50) | 0 |
| C_html_only | 0.800 (40 / 50) | 0 |

## FP recovery — how many of the ensemble's false alarms the oracle correctly downgrades to benign

| Variant | FP recovery rate | uncertain |
|---|---:|---:|
| B_html_probs_osint | 0.960 (48 / 50) | 0 |
| A_html_probs | 0.900 (45 / 50) | 0 |
| E_poc_2k | 1.000 (50 / 50) | 0 |
| D_html_osint | 1.000 (50 / 50) | 0 |
| C_html_only | 1.000 (50 / 50) | 0 |