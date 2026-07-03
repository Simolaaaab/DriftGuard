# Cross-LLM hallucination ↔ correctness analysis

All providers: drift_anchored + Prompt B + K=20 + periodic_audit + temporal PP_stream, seed 2025.

## Per-model headline

| provider | n | accuracy | mean_trust | halluc_indicator_rate | pearson_trust_correct | phi_halluc_correct |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 479 | 0.8079 | 0.7588 | 0.0983 | 0.3035 | 0.2386 |
| GPT-5.4 | 482 | 0.8133 | 0.7501 | 0.0788 | 0.3695 | 0.2903 |
| GPT-5.4-mini | 481 | 0.8233 | 0.7634 | 0.1243 | 0.3091 | 0.2275 |
| GPT-5.1 | 484 | 0.7748 | 0.7513 | 0.0908 | 0.4291 | 0.2963 |
| Llama-3.3-70B | 323 | 0.8638 | 0.8048 | 0.1173 | 0.3847 | 0.0935 |
| Mistral-Large-3 | 490 | 0.8592 | 0.7801 | 0.0943 | 0.1573 | 0.2637 |


## 2×2 contingency  (hallucinated × correct)

Hallucinated = ≥1 indicator refuted by the verifier (`n_contradicted ≥ 1`). Higher Fisher odds ratio means hallucinated samples are MORE likely to be wrong.

| provider | n_halluc_correct | n_halluc_wrong | n_clean_correct | n_clean_wrong | acc_when_halluc | acc_when_clean | acc_drop_due_to_halluc | fisher_odds_ratio | fisher_p_two_sided |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 289 | 43 | 98 | 49 | 0.8705 | 0.6667 | -0.2038 | 3.3605 | 0.0000 |
| GPT-5.4 | 257 | 26 | 135 | 64 | 0.9081 | 0.6784 | -0.2297 | 4.6860 | 0.0000 |
| GPT-5.4-mini | 243 | 27 | 153 | 58 | 0.9000 | 0.7251 | -0.1749 | 3.4118 | 0.0000 |
| GPT-5.1 | 219 | 25 | 156 | 84 | 0.8975 | 0.6500 | -0.2475 | 4.7169 | 0.0000 |
| Llama-3.3-70B | 133 | 15 | 146 | 29 | 0.8986 | 0.8343 | -0.0644 | 1.7612 | 0.1049 |
| Mistral-Large-3 | 283 | 21 | 138 | 48 | 0.9309 | 0.7419 | -0.1890 | 4.6874 | 0.0000 |


## Trust-quartile accuracy table

Per provider × trust quartile (sample-relative). The headline finding is the *very-high trust paradox*: in most providers Q4 accuracy is LOWER than Q3, because the verifier confirms evidence-grounding but cannot discriminate sample-level ambiguity.

| provider | bucket | bucket_lo | bucket_hi | n | accuracy | mean_trust |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | Q1 | 0.0000 | 0.6350 | 123 | 0.5854 | 0.5036 |
| DeepSeek-V4-Flash | Q2 | 0.6350 | 0.8543 | 118 | 0.9492 | 0.7760 |
| DeepSeek-V4-Flash | Q3 | 0.8543 | 0.8647 | 147 | 0.9524 | 0.8605 |
| DeepSeek-V4-Flash | Q4 | 0.8647 | 1.0010 | 91 | 0.6923 | 0.9173 |
| GPT-5.4 | Q1 | 0.0000 | 0.6112 | 121 | 0.5537 | 0.4869 |
| GPT-5.4 | Q2 | 0.6112 | 0.8512 | 120 | 0.9417 | 0.7475 |
| GPT-5.4 | Q3 | 0.8512 | 0.8633 | 120 | 0.9750 | 0.8588 |
| GPT-5.4 | Q4 | 0.8633 | 1.0010 | 121 | 0.7851 | 0.9079 |
| GPT-5.4-mini | Q1 | 0.0000 | 0.6353 | 121 | 0.5868 | 0.5105 |
| GPT-5.4-mini | Q2 | 0.6353 | 0.8498 | 120 | 0.9500 | 0.7494 |
| GPT-5.4-mini | Q3 | 0.8498 | 0.8694 | 121 | 0.9587 | 0.8630 |
| GPT-5.4-mini | Q4 | 0.8694 | 1.0010 | 119 | 0.7983 | 0.9335 |
| GPT-5.1 | Q1 | 0.0000 | 0.5784 | 121 | 0.4380 | 0.4656 |
| GPT-5.1 | Q2 | 0.5784 | 0.8493 | 121 | 0.9256 | 0.7465 |
| GPT-5.1 | Q3 | 0.8493 | 0.8730 | 124 | 0.9597 | 0.8603 |
| GPT-5.1 | Q4 | 0.8730 | 1.0010 | 118 | 0.7712 | 0.9344 |
| Llama-3.3-70B | Q1 | 0.0000 | 0.8075 | 81 | 0.6543 | 0.5139 |
| Llama-3.3-70B | Q2 | 0.8075 | 0.8567 | 85 | 0.9765 | 0.8421 |
| Llama-3.3-70B | Q3 | 0.8567 | 0.9500 | 119 | 0.9160 | 0.9260 |
| Llama-3.3-70B | Q4 | 0.9500 | 1.0010 | 38 | 0.8947 | 0.9620 |
| Mistral-Large-3 | Q1 | 0.0000 | 0.6650 | 128 | 0.7656 | 0.5422 |
| Mistral-Large-3 | Q2 | 0.6650 | 0.8546 | 117 | 0.9402 | 0.8016 |
| Mistral-Large-3 | Q3 | 0.8546 | 0.8662 | 126 | 0.9683 | 0.8606 |
| Mistral-Large-3 | Q4 | 0.8662 | 1.0010 | 119 | 0.7647 | 0.9294 |

