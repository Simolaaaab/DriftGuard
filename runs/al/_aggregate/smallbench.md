# Small-Benchmark Comparative Bench (multi-LLM × strategy)

Baseline provider: **`deepseek_v4`** (absolute reference; Δ rows computed against this provider *per strategy* on the same small-benchmark manifest).

## Absolute results

| strategy | provider | seed | rounds | LLM calls | PP F1w | KP F1w | DP F1w | PP rec_phish |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| drift_anchored | `deepseek_v4` | 2025 | 12 | 235 | 0.6745 | 0.9795 | 0.6753 | 0.4016 |
| drift_anchored | `gpt_5_4` | 2025 | 12 | 218 | 0.6354 | 0.9801 | 0.6651 | 0.3525 |
| drift_anchored | `gpt_5_4_mini` | 2025 | 12 | 215 | 0.6354 | 0.9801 | 0.6502 | 0.3525 |
| drift_anchored | `mistral_large` | 2025 | 12 | 217 | 0.6708 | 0.9801 | 0.5640 | 0.4180 |
| pure_random | `deepseek_v4` | 2025 | 10 | 250 | 0.7009 | 0.9790 | 0.6715 | 0.4672 |
| pure_random | `gpt_5_4` | 2025 | 10 | 250 | 0.7009 | 0.9795 | 0.6826 | 0.4672 |
| pure_random | `gpt_5_4_mini` | 2025 | 10 | 250 | 0.7336 | 0.9795 | 0.6682 | 0.5164 |
| pure_random | `mistral_large` | 2025 | 10 | 250 | 0.7167 | 0.9790 | 0.6242 | 0.5000 |

## Δ vs `deepseek_v4` (same strategy, same bench)

| strategy | provider | seed | ΔPP F1w | ΔKP F1w | ΔDP F1w | ΔR_phish |
|---|---|---:|---:|---:|---:|---:|
| drift_anchored | `deepseek_v4` | 2025 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| drift_anchored | `gpt_5_4` | 2025 | -0.0392 | +0.0006 | -0.0102 | -0.0492 |
| drift_anchored | `gpt_5_4_mini` | 2025 | -0.0392 | +0.0006 | -0.0251 | -0.0492 |
| drift_anchored | `mistral_large` | 2025 | -0.0037 | +0.0006 | -0.1113 | +0.0164 |
| pure_random | `deepseek_v4` | 2025 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| pure_random | `gpt_5_4` | 2025 | +0.0000 | +0.0006 | +0.0110 | +0.0000 |
| pure_random | `gpt_5_4_mini` | 2025 | +0.0327 | +0.0006 | -0.0034 | +0.0492 |
| pure_random | `mistral_large` | 2025 | +0.0158 | +0.0000 | -0.0474 | +0.0328 |

## Full-PP reference (anchor, not a scale-generalization claim)

The small benchmark uses |PP_stream|=500 / |PP_test|=250 to make multi-LLM bench affordable. Absolute F1w numbers on this subset are NOT comparable to the full-PP Main Results because: (i) PP_test n=250 has SE ≈ ±2.5pp on binomial F1w; (ii) the AL loop sees ~250 labels vs ~1250 on full → less information; (iii) the pool fills more slowly at small stream sizes, biasing drift-triggered selection. The small bench is for **relative** comparisons across LLM providers at matched budget, not for reproducing absolute Main Results numbers.

Full-PP DeepSeek canonical (Step-3, K=5 seeds):

| strategy | full-PP PP F1w | note |
|---|---:|---|
| drift_anchored | 0.800 | drift_anchored+B+temporal canonical seed 2025; see Step-3 _aggregate/headline_table.csv |
| margin | 0.777 | margin+B+temporal canonical seed 2025 |