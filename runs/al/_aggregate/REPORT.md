# Step 3 — Active Learning Headline Comparison

## Final-round results per run

| run_id | strategy | oracle | rounds | labels | LLM calls | PP F1w init→final | ΔPP | ΔKP | ΔDP |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| smoke_periodic_gt | margin | gt_dryrun | 25 | 1250 | 1250 | 0.623→0.873 | +0.250 | -0.002 | -0.158 |
| drift_anchored_llm_periodic_temporal_B_v4_hard_filter | drift_anchored | llm | 25 | 1049 | 1250 | 0.623→0.802 | +0.179 | -0.003 | +0.009 |
| drift_anchored_llm_periodic_temporal_B_v4_off | drift_anchored | llm | 25 | 1247 | 1250 | 0.623→0.800 | +0.177 | -0.002 | +0.113 |
| drift_anchored_llm_periodic_temporal_B | drift_anchored | llm | 25 | 1247 | 1250 | 0.623→0.800 | +0.177 | -0.002 | +0.113 |
| drift_anchored_llm_periodic_temporal_B_v4 | drift_anchored | llm | 25 | 1179 | 1250 | 0.623→0.799 | +0.175 | -0.002 | +0.062 |
| drift_anchored_B_seed2025 | drift_anchored | llm | 25 | 1249 | 1250 | 0.623→0.795 | +0.171 | -0.002 | +0.109 |
| drift_anchored_llm_periodic_temporal_B_v4_t1_only | drift_anchored | llm | 25 | 1224 | 1250 | 0.623→0.794 | +0.170 | -0.002 | +0.112 |
| drift_anchored_llm_periodic_shuffle_B | drift_anchored | llm | 25 | 1250 | 1250 | 0.623→0.792 | +0.169 | -0.003 | +0.198 |
| drift_anchored_B_seed2026 | drift_anchored | llm | 25 | 1249 | 1250 | 0.636→0.784 | +0.149 | -0.004 | +0.107 |
| drift_anchored_B_seed2027 | drift_anchored | llm | 25 | 1244 | 1250 | 0.612→0.784 | +0.172 | -0.002 | +0.086 |
| drift_anchored_llm_periodic_temporal_B_v4_t3_only | drift_anchored | llm | 25 | 1115 | 1250 | 0.623→0.782 | +0.159 | -0.003 | +0.010 |
| qbc_llm_periodic_temporal_B | qbc | llm | 25 | 1246 | 1250 | 0.623→0.779 | +0.155 | -0.008 | +0.301 |
| hybrid_llm_periodic_temporal_B | hybrid | llm | 25 | 1248 | 1250 | 0.623→0.778 | +0.155 | -0.008 | +0.299 |
| margin_B_seed2025 | margin | llm | 25 | 1247 | 1250 | 0.623→0.777 | +0.154 | -0.007 | +0.284 |
| margin_llm_periodic_temporal_B | margin | llm | 25 | 1246 | 1250 | 0.623→0.777 | +0.154 | -0.008 | +0.285 |
| core_set_llm_periodic_temporal_B | core_set | llm | 25 | 1245 | 1250 | 0.623→0.776 | +0.153 | -0.003 | +0.270 |
| margin_B_seed2027 | margin | llm | 25 | 1247 | 1250 | 0.612→0.775 | +0.163 | -0.005 | +0.265 |
| drift_anchored_llm_periodic_temporal | drift_anchored | llm | 25 | 1240 | 1250 | 0.623→0.773 | +0.150 | -0.001 | +0.138 |
| margin_llm_periodic_temporal | margin | llm | 25 | 1244 | 1250 | 0.623→0.771 | +0.147 | -0.007 | +0.278 |
| badge_llm_periodic_temporal_B | badge | llm | 25 | 1248 | 1250 | 0.623→0.769 | +0.146 | -0.006 | +0.290 |
| qbc_llm_periodic_temporal | qbc | llm | 25 | 1238 | 1250 | 0.623→0.768 | +0.145 | -0.007 | +0.274 |
| smoke_margin_gt | margin | gt_dryrun | 5 | 250 | 250 | 0.623→0.767 | +0.143 | +0.001 | -0.103 |
| badge_llm_periodic_temporal | badge | llm | 25 | 1246 | 1250 | 0.623→0.761 | +0.138 | -0.007 | +0.273 |
| drift_anchored_llm_periodic_temporal_B_v4_t2_only | drift_anchored | llm | 25 | 1116 | 1250 | 0.623→0.761 | +0.137 | -0.003 | +0.090 |
| margin_B_seed2026 | margin | llm | 25 | 1245 | 1250 | 0.636→0.760 | +0.124 | -0.008 | +0.312 |
| core_set_llm_periodic_temporal | core_set | llm | 25 | 1242 | 1250 | 0.623→0.760 | +0.136 | -0.005 | +0.262 |
| hybrid_llm_periodic_temporal | hybrid | llm | 25 | 1245 | 1250 | 0.623→0.759 | +0.135 | -0.007 | +0.273 |
| random_llm_periodic_temporal | random | llm | 25 | 1243 | 1250 | 0.623→0.745 | +0.122 | -0.002 | +0.210 |
| random_llm_periodic_temporal_B | random | llm | 25 | 1241 | 1250 | 0.623→0.745 | +0.121 | -0.002 | +0.232 |
| drift_anchored_B_kimi_seed2025 | drift_anchored | llm | 25 | 411 | 1250 | 0.623→0.676 | +0.053 | -0.001 | -0.004 |
| smoke_seed2026 | margin | gt_dryrun | 3 | 150 | 150 | 0.636→0.674 | +0.039 | -0.001 | -0.056 |
| smoke_b_gt | margin | gt_dryrun | 5 | 250 | 250 | 0.623→0.673 | +0.050 | -0.002 | +0.010 |
| smoke_step4_gt | margin | gt_dryrun | 3 | 150 | 150 | 0.623→0.645 | +0.022 | -0.002 | -0.003 |
| margin_llm_temporal | margin | llm | 3 | 150 | 150 | 0.623→0.613 | -0.010 | +0.000 | +0.075 |

## Notes

- **ΔKP near zero** confirms full-retrain prevents catastrophic forgetting.
- **ΔPP positive** = AL recovery on natural drift.
- **ΔDP negative** = adversarial regression — boundary of self-healing approach.

## Files
- `learning_curves_pp.csv` — per-round PP F1w per run.
- `pareto_labels_vs_f1.csv` — labels-vs-F1w Pareto data.
- `headline_table.csv` — this table.