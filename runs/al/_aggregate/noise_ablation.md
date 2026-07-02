# Noise Ablation — Causal Origin of DP-OOD Robustness

Strategies: ['drift_anchored', 'margin']
Arms: ['adversarial_b2p', 'adversarial_p2b', 'clean_gt', 'llm', 'random_flip']
K seeds per (strategy, arm): min=3 max=3

## Effective noise rate (audit)

| Strategy | Arm | seed | nominal_rate | n_flipped/n_cached | effective_rate | served_phish_frac | gt_phish_frac |
|---|---|---:|---:|---|---:|---:|---:|
| drift_anchored | adversarial_b2p | 2025 | 0.30 | 273/1250 | 0.218 | 0.506 | 0.287 |
| drift_anchored | adversarial_b2p | 2026 | 0.30 | 281/1250 | 0.225 | 0.498 | 0.274 |
| drift_anchored | adversarial_b2p | 2027 | 0.30 | 274/1250 | 0.219 | 0.500 | 0.281 |
| drift_anchored | adversarial_p2b | 2025 | 0.30 | 141/1250 | 0.113 | 0.283 | 0.396 |
| drift_anchored | adversarial_p2b | 2026 | 0.30 | 133/1250 | 0.106 | 0.280 | 0.386 |
| drift_anchored | adversarial_p2b | 2027 | 0.30 | 143/1250 | 0.114 | 0.297 | 0.411 |
| drift_anchored | clean_gt | 2025 | 0.00 | 0/1241 | 0.000 | 0.354 | 0.354 |
| drift_anchored | clean_gt | 2026 | 0.00 | 0/1241 | 0.000 | 0.359 | 0.359 |
| drift_anchored | clean_gt | 2027 | 0.00 | 0/1246 | 0.000 | 0.356 | 0.356 |
| drift_anchored | llm | 2025 | 0.00 | 0/1241 | 0.000 | 0.227 | 0.227 |
| drift_anchored | llm | 2026 | 0.00 | 0/1234 | 0.000 | 0.232 | 0.232 |
| drift_anchored | llm | 2027 | 0.00 | 0/1236 | 0.000 | 0.236 | 0.236 |
| drift_anchored | random_flip | 2025 | 0.30 | 390/1250 | 0.312 | 0.412 | 0.271 |
| drift_anchored | random_flip | 2026 | 0.30 | 372/1250 | 0.298 | 0.434 | 0.303 |
| drift_anchored | random_flip | 2027 | 0.30 | 381/1250 | 0.305 | 0.438 | 0.310 |
| margin | adversarial_b2p | 2025 | 0.30 | 262/1250 | 0.210 | 0.546 | 0.337 |
| margin | adversarial_b2p | 2026 | 0.30 | 259/1250 | 0.207 | 0.545 | 0.338 |
| margin | adversarial_b2p | 2027 | 0.30 | 272/1250 | 0.218 | 0.532 | 0.314 |
| margin | adversarial_p2b | 2025 | 0.30 | 208/1250 | 0.166 | 0.406 | 0.572 |
| margin | adversarial_p2b | 2026 | 0.30 | 201/1250 | 0.161 | 0.393 | 0.554 |
| margin | adversarial_p2b | 2027 | 0.30 | 197/1250 | 0.158 | 0.405 | 0.562 |
| margin | clean_gt | 2025 | 0.00 | 0/1227 | 0.000 | 0.507 | 0.507 |
| margin | clean_gt | 2026 | 0.00 | 0/1231 | 0.000 | 0.504 | 0.504 |
| margin | clean_gt | 2027 | 0.00 | 0/1231 | 0.000 | 0.511 | 0.511 |
| margin | llm | 2025 | 0.00 | 0/1232 | 0.000 | 0.304 | 0.304 |
| margin | llm | 2026 | 0.00 | 0/1232 | 0.000 | 0.296 | 0.296 |
| margin | llm | 2027 | 0.00 | 0/1232 | 0.000 | 0.313 | 0.313 |
| margin | random_flip | 2025 | 0.30 | 378/1250 | 0.302 | 0.463 | 0.390 |
| margin | random_flip | 2026 | 0.30 | 388/1250 | 0.310 | 0.458 | 0.383 |
| margin | random_flip | 2027 | 0.30 | 397/1250 | 0.318 | 0.463 | 0.386 |

> **Reading note**: the *nominal* rate is the Bernoulli parameter passed to the flipper. The *effective* rate is `n_flipped / n_cached`. For adversarial arms the effective rate is bounded above by the class share of the selected sub-population (adversarial_p2b only flips phish; if the selection is benign-dominated, the effective rate is much lower than the nominal).

## Per-arm headline (pooled stratified bootstrap)

| Strategy | Arm | Slice | F1w (pooled mean ± half_CI) | R_phish | R_benign |
|---|---|---|---|---|---|
| drift_anchored | adversarial_b2p | pp_test | 0.8085 ± 0.0299 | 0.8544 ± 0.0353 | 0.7653 ± 0.0450 |
| drift_anchored | adversarial_b2p | kp_test | 0.9802 ± 0.0080 | 0.9776 ± 0.0105 | 0.9823 ± 0.0100 |
| drift_anchored | adversarial_b2p | dp_probe | 0.1497 ± 0.0567 | 0.9928 ± 0.0072 | 0.0690 ± 0.0377 |
| drift_anchored | adversarial_p2b | pp_test | 0.8201 ± 0.0330 | 0.7003 ± 0.0664 | 0.9404 ± 0.0266 |
| drift_anchored | adversarial_p2b | kp_test | 0.9802 ± 0.0066 | 0.9744 ± 0.0111 | 0.9850 ± 0.0085 |
| drift_anchored | adversarial_p2b | dp_probe | 0.5453 ± 0.0520 | 0.9555 ± 0.0223 | 0.4108 ± 0.0572 |
| drift_anchored | clean_gt | pp_test | 0.8741 ± 0.0240 | 0.8431 ± 0.0354 | 0.9042 ± 0.0276 |
| drift_anchored | clean_gt | kp_test | 0.9789 ± 0.0072 | 0.9751 ± 0.0105 | 0.9820 ± 0.0085 |
| drift_anchored | clean_gt | dp_probe | 0.2996 ± 0.0258 | 0.9855 ± 0.0078 | 0.1771 ± 0.0205 |
| drift_anchored | llm | pp_test | 0.7876 ± 0.0273 | 0.6113 ± 0.0493 | 0.9706 ± 0.0164 |
| drift_anchored | llm | kp_test | 0.9800 ± 0.0072 | 0.9731 ± 0.0118 | 0.9856 ± 0.0080 |
| drift_anchored | llm | dp_probe | 0.6195 ± 0.0121 | 0.9313 ± 0.0274 | 0.4985 ± 0.0135 |
| drift_anchored | random_flip | pp_test | 0.7443 ± 0.0594 | 0.6447 ± 0.1060 | 0.8464 ± 0.0481 |
| drift_anchored | random_flip | kp_test | 0.9813 ± 0.0078 | 0.9756 ± 0.0118 | 0.9860 ± 0.0090 |
| drift_anchored | random_flip | dp_probe | 0.3116 ± 0.1082 | 0.9838 ± 0.0104 | 0.1905 ± 0.0854 |
| margin | adversarial_b2p | pp_test | 0.8223 ± 0.0308 | 0.8602 ± 0.0321 | 0.7864 ± 0.0481 |
| margin | adversarial_b2p | kp_test | 0.9789 ± 0.0083 | 0.9780 ± 0.0105 | 0.9796 ± 0.0105 |
| margin | adversarial_b2p | dp_probe | 0.1582 ± 0.0264 | 0.9897 ± 0.0067 | 0.0742 ± 0.0177 |
| margin | adversarial_p2b | pp_test | 0.8325 ± 0.0243 | 0.7104 ± 0.0428 | 0.9548 ± 0.0204 |
| margin | adversarial_p2b | kp_test | 0.9793 ± 0.0067 | 0.9710 ± 0.0124 | 0.9860 ± 0.0075 |
| margin | adversarial_p2b | dp_probe | 0.6211 ± 0.0667 | 0.9289 ± 0.0399 | 0.5033 ± 0.0832 |
| margin | clean_gt | pp_test | 0.8790 ± 0.0228 | 0.8387 ± 0.0353 | 0.9179 ± 0.0256 |
| margin | clean_gt | kp_test | 0.9790 ± 0.0080 | 0.9764 ± 0.0105 | 0.9810 ± 0.0095 |
| margin | clean_gt | dp_probe | 0.3532 ± 0.0310 | 0.9825 ± 0.0088 | 0.2214 ± 0.0263 |
| margin | llm | pp_test | 0.7707 ± 0.0300 | 0.5813 ± 0.0525 | 0.9699 ± 0.0153 |
| margin | llm | kp_test | 0.9760 ± 0.0083 | 0.9645 ± 0.0136 | 0.9853 ± 0.0090 |
| margin | llm | dp_probe | 0.8057 ± 0.0208 | 0.8698 ± 0.0424 | 0.7606 ± 0.0346 |
| margin | random_flip | pp_test | 0.7656 ± 0.0344 | 0.6796 ± 0.0535 | 0.8517 ± 0.0378 |
| margin | random_flip | kp_test | 0.9819 ± 0.0069 | 0.9760 ± 0.0111 | 0.9867 ± 0.0085 |
| margin | random_flip | dp_probe | 0.4104 ± 0.0725 | 0.9817 ± 0.0114 | 0.2732 ± 0.0670 |

## Headline pairwise tests (DP F1w)

| Strategy | A vs B | mean_diff (A−B) | 95% CI | p (two-sided) |
|---|---|---:|---|---:|
| drift_anchored | adversarial_b2p vs adversarial_p2b | -0.3956 | [-0.4818, -0.2785] | 0.000 |
| drift_anchored | adversarial_b2p vs clean_gt | -0.1499 | [-0.1932, -0.0786] | 0.000 |
| drift_anchored | adversarial_b2p vs llm | -0.4698 | [-0.5233, -0.4090] | 0.000 |
| drift_anchored | adversarial_b2p vs random_flip | -0.1619 | [-0.2587, -0.0466] | 0.000 |
| drift_anchored | adversarial_p2b vs clean_gt | +0.2458 | [+0.1947, +0.2969] | 0.000 |
| drift_anchored | adversarial_p2b vs llm | -0.0742 | [-0.1374, -0.0372] | 0.000 |
| drift_anchored | adversarial_p2b vs random_flip | +0.2337 | [+0.0948, +0.3846] | 0.000 |
| drift_anchored | clean_gt vs llm | -0.3199 | [-0.3436, -0.2873] | 0.000 |
| drift_anchored | clean_gt vs random_flip | -0.0121 | [-0.1082, +0.1366] | 0.667 |
| drift_anchored | llm vs random_flip | +0.3079 | [+0.2240, +0.4349] | 0.000 |
| margin | adversarial_b2p vs adversarial_p2b | -0.4629 | [-0.5292, -0.3562] | 0.000 |
| margin | adversarial_b2p vs clean_gt | -0.1951 | [-0.2387, -0.1674] | 0.000 |
| margin | adversarial_b2p vs llm | -0.6475 | [-0.6814, -0.6089] | 0.000 |
| margin | adversarial_b2p vs random_flip | -0.2522 | [-0.3435, -0.1626] | 0.000 |
| margin | adversarial_p2b vs clean_gt | +0.2678 | [+0.1805, +0.3346] | 0.000 |
| margin | adversarial_p2b vs llm | -0.1847 | [-0.2604, -0.1266] | 0.000 |
| margin | adversarial_p2b vs random_flip | +0.2106 | [+0.1780, +0.2600] | 0.000 |
| margin | clean_gt vs llm | -0.4525 | [-0.5039, -0.4127] | 0.000 |
| margin | clean_gt vs random_flip | -0.0572 | [-0.1118, +0.0140] | 0.587 |
| margin | llm vs random_flip | +0.3953 | [+0.3095, +0.4568] | 0.000 |

## Interpretation guide for the paper

- If **DP F1w (random_flip) ≈ DP F1w (llm)** with overlapping CIs ⇒ the DP-OOD lift is explained by generic label-noise regularisation; the LLM noise is *not* structurally privileged. The paper's Section 6 should be reframed accordingly.
- If **DP F1w (random_flip) < DP F1w (llm)** with p<0.05 ⇒ the LLM noise is *structured* and the structure is the contribution. Section 6 holds.
- If **DP F1w (adversarial_p2b) < DP F1w (random_flip)** ⇒ the noise *direction* matters; adversarial label-flipping is more destructive than symmetric noise, justifying the asymmetric verifier design.
- If **PP F1w (clean_gt) ≫ PP F1w (other arms)** ⇒ the AL machinery is correctly learning from labels; noise hurts in-distribution but helps OOD (classic bias-variance trade-off).