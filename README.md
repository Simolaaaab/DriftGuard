# DRIFTGUARD — Label-Free Self-Repair of Phishing Detectors under Drift

Artifact accompanying the WTMC submission *"DRIFTGUARD: Label-Free Self-Repair of
Phishing Detectors under Drift via LLM-Guided Active Learning."*

DRIFTGUARD is a closed-loop phishing detector that repairs itself under temporal
concept drift with **no human labels** and a small, predictable LLM budget. A
leakage-clean meta-learner scoring ~98% weighted F1 in-distribution decays to
~63% on temporally fresh phishing; DRIFTGUARD recovers ~18 points of weighted F1
using ~1,250 selective LLM-oracle queries, with no measurable forgetting.

This repository contains the curated pipeline, the reproducible inputs, and the
aggregated results behind every number and figure in the paper. It is
self-contained: **all headline results can be replayed from the shipped oracle
cache without any API key.**

## Structure

```
step1_ensemble/     Leakage-clean meta-learner + cross-dataset AUC leakage gate
step1_5_osint/      OSINT enrichment (Tranco / Wayback / CT / RDAP / BGP), zero-leakage
step2_oracle/       LLM-oracle prompt ablation (variants A–E) on hard FN/FP samples
step3_al/           The active-learning loop: pool, drift detector, selection,
                    oracle, class-aware verifier, retrain — plus cited analyses
step5_smallbench/   Multi-LLM oracle benchmark (6 providers) + health-monitor study
figures/            Figure generators + rendered PDFs/PNGs

datasets/           Base Z-matrices (KP / DP / PP) + source JSON
runs/osint/         OSINT-enriched Z-matrices (the state the AL loop consumes)
runs/ensemble/      Step-1 leakage-gate manifest + reports
runs/oracle_ablation/   Step-2 results + D_html_osint oracle cache
runs/al/_aggregate/     Headline tables, learning curves, Pareto data, paper figures
runs/al/_clean_shared_cache/   Shared oracle cache — replay every run, no API key
runs/al/drift_anchored_B_seed2025/   One canonical run as a worked example

docs/paper/         LaTeX sources + bibliography
```

## Datasets

- **KP** (KnowPhish) — training, in-distribution.
- **PP** (PhreshPhish) — temporally fresh phishing; the natural-drift test.
- **DP** (SpacePhish/DeltaPhish) — out-of-distribution adversarial probe; **never trained on**.

The `datasets/*.json` files carry the URLs and cleaned HTML used to derive
features. Full re-collection of live HTML / OSINT is not required to reproduce
the paper: the derived Z-matrices and the oracle cache are shipped.

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**1. Leakage-clean baseline (Step 1)**
```bash
python3 step1_ensemble/ensemble_reboot.py --mode osint
```

**2. Replay the canonical AL run from cache (no API key needed)**
```bash
python3 step3_al/run_al.py \
    --strategy drift_anchored --oracle llm --oracle-prompt B \
    --threshold-preset periodic_audit --max-rounds 30 --seed 2025 \
    --run-id drift_anchored_B_seed2025_replay
```
The oracle is served from `runs/al/_clean_shared_cache/`; every queried sample is
a cache hit, so the run is deterministic and free.

**3. Aggregate tables + figures**
```bash
python3 step3_al/analyze.py
python3 figures/make_all.py
```

Running the loop against a live provider (to extend beyond the cached samples)
requires credentials via environment variables:
```bash
export AZURE_API_KEY=...  AZURE_BASE_URL=...  AZURE_MODEL=DeepSeek-V4-Flash
# or: export DEEPSEEK_API_KEY=sk-...
```

## Key results (5-seed, canonical)

| Metric                | KP-only baseline | After AL (drift-anchored + B) |
|-----------------------|------------------|-------------------------------|
| PP weighted F1        | 0.629            | 0.800                         |
| PP phishing recall    | 0.553            | 0.634                         |
| KP sanity accuracy    | 0.982            | 0.980  (no forgetting)        |
| DP adversarial F1     | (probe)          | +11–30pp vs baseline          |

Full per-strategy / per-seed numbers are in `runs/al/_aggregate/`.
