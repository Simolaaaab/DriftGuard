# Step 5 — Small Benchmark for Comparative Studies

This module exists for one reason: **multi-LLM benchmarking is expensive on
the full dataset and the supervisor is right that comparative claims
("LLM-A vs LLM-B", "prompt-X vs prompt-Y") only need a fraction of the
data**.

## Scope of this module

### What it does
- Deterministic stratified subsample of PP_stream + PP_test (default
  `--subset-stream 500 --subset-test 250`, stratified by class, temporal
  order preserved).
- Persists the subsample as a JSON manifest (`subset_manifest.json`) so
  every script that needs the small benchmark loads identical splits
  — even across re-runs and re-clones.
- Provides a **pure-random AL baseline** (`run_pure_random_al.py`):
  no drift trigger, no pool filtering, uniform Bernoulli sampling on
  the stream with a hard budget. This is the baseline the supervisor
  asked for and that Reviewer 2 will demand.
- Provides a **multi-LLM dispatcher** (`llm_providers.py`): extends
  `AsyncLLM` with provider-pluggable backends for OpenAI direct,
  Azure OpenAI, Mistral, xAI (Grok), beyond the existing DeepSeek +
  Azure-AI-Foundry routes.
- Provides a **multi-LLM orchestrator** (`run_multi_llm_bench.py`):
  launches one canonical config (drift_anchored+B+temporal) across
  the chosen LLM stable on the small benchmark, with shared seed.
- Provides a **comparative aggregator** (`aggregate_smallbench.py`):
  reports Δ vs the in-bench DeepSeek baseline (the supervisor's
  explicit reporting convention).

### What it does NOT do
- It does **not** replace the full-dataset Main Results from Step 3.
  Those (drift_anchored+B+temporal K=5 seeds → 0.800 ± CI) are the
  paper's absolute numbers. The small benchmark is for *relative*
  comparisons only.
- It does **not** redo the full ablation suite (selection strategies,
  trigger modes, verifier tiers). Those are already in Step 3/4 on
  full PP, with K-seed bands and pairwise tests. The small benchmark
  is only for things that *cost* (multi-LLM, multi-prompt with
  multi-LLM, etc.).
- It does **not** evaluate on DP/KP differently from Step 3. The
  KP_test (frozen, 2256 samples) and DP_probe (frozen, 6426 samples)
  remain identical full slices for every run — only PP_stream and
  PP_test are subsampled. This keeps OOD probes apples-to-apples
  across all experiments.

## The "scale generalization" sanity check

Every small-benchmark run includes a sanity row in the aggregator
output: drift_anchored+B+DeepSeek on the small benchmark, compared
against the same config on full PP. If they're within ~3-4pp F1w the
comparative ranking on small generalizes to full; the paper can quote
this as the validation evidence.

## Folder layout

```
step5_smallbench/
├── README.md                    (this file)
├── make_subset.py                deterministic stratified subsample + manifest
├── load_subset.py                helper: load splits from manifest
├── llm_providers.py              multi-provider AsyncLLM extension
├── run_pure_random_al.py         pure-Bernoulli AL baseline (small OR full)
├── run_smallbench_loop.py        wrapper around step3_al loop, uses subset splits
├── run_multi_llm_bench.py        orchestrator: launches canonical × N LLMs
├── aggregate_smallbench.py       Δ-vs-baseline comparative tables
└── manifests/
    └── smallbench_v1.json        the canonical manifest (committed)
```

## Quickstart

```bash
# 0) (one-time) build the small benchmark manifest
python3 reboot/step5_smallbench/make_subset.py \
    --stream 500 --test 250 --seed 2025

# 1) sanity baseline: drift_anchored+B+DeepSeek on the small benchmark
#    (~5 min, ~$0.50)
python3 reboot/step5_smallbench/run_smallbench_loop.py \
    --strategy drift_anchored --oracle-prompt B \
    --run-id smallbench_drift_anchored_B_deepseek \
    --model DeepSeek-V4-Flash

# 2) pure-random AL baseline (same budget, no drift logic, no pool)
#    Also runnable on full PP via --full-dataset
python3 reboot/step5_smallbench/run_pure_random_al.py \
    --run-id smallbench_pure_random_deepseek \
    --model DeepSeek-V4-Flash --budget 250

# 3) multi-LLM bench: canonical config across all configured LLMs
python3 reboot/step5_smallbench/run_multi_llm_bench.py \
    --models deepseek_v4,gpt_5_4,gpt_5_4_mini,mistral_large,grok_4 \
    --strategies drift_anchored,pure_random

# 4) comparative aggregation
python3 reboot/step5_smallbench/aggregate_smallbench.py
```

## Pure-random AL baseline mechanics

The existing `--strategy random` in Step 3 selects uniformly *from the
uncertainty pool* (which is itself filtered by the uncertainty band
(0.30, 0.70) ∪ proposer_disagreement ≥ 0.35). That's "random conditional
on being uncertain" — useful, but not the baseline the supervisor asked
for.

Pure-random AL ignores the pool entirely:

1. Sample `budget` row indices uniformly from `[0, len(stream))` once
   at the start, deterministic by `--seed`.
2. As the stream is iterated, when row_idx is in the sampled set, send
   it to the oracle. Otherwise, just score with the current model.
3. Retrain every `K_per_round=50` accumulated labels (matching the AL
   round cadence).

This is the **"label budget without selection logic"** condition. The
gap between drift_anchored+B and this baseline is the **selection-
strategy contribution** the paper needs.
