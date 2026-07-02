# Step 3 — Active Learning Loop

The final module of the reboot. Simulates pool-based AL with drift
trigger on the PhreshPhish stream, using the leakage-clean Ensemble
from Step 1 and the D_html_osint oracle from Step 2. Per the Step 3
Specification Document (sign-off received).

## Architecture summary

```
KP_train (frozen)
   │
   ▼
StandardScaler (round-0, frozen forever)
   │
   ▼
LogReg (refit each round on KP ∪ accepted labels — no class_weight)
   │
   ├──── eval per round on KP_test, PP_test, DP_probe
   │
PP_stream (temporal order by JSON `date`)
   │
   ▼
score → admit-if-uncertain → UncertaintyPool (cap 2000, FIFO)
   │
   ▼
CompositeDriftDetector (delegate_ratio ∧ KL, 200-row window, cooldown 50)
   │
   ▼ triggered
Selection (margin | qbc | core_set | badge | drift_anchored | hybrid | random)
   │
   ▼ k=50
CachedOracle (Step-2 cache → Step-3 cache → LLM D_html_osint)
   │
   ▼ accepted labels
retrain(KP ∪ accepted) → new model
   │
   ▼ rescore pool, reset detector window, evaluate
loop
```

## Modules

| File | Responsibility |
|---|---|
| `common.py` | shared types (StreamRow, PoolSample, OracleLabel, …), paths, hyperparameters |
| `stream.py` | temporal PP iteration |
| `pool.py` | bounded uncertainty pool + living-pool rescore |
| `drift_detector.py` | AND-gated composite detector (delegate_ratio + KL) |
| `selection.py` | 7 strategies behind one dispatch |
| `oracle.py` | 3-tier cached oracle: Step-2 cache → Step-3 cache → LLM |
| `retrain.py` | full-union refit with hard-coded immutability invariants |
| `evaluator.py` | per-round metrics on KP/PP/DP including ECE |
| `loop.py` | `run_al(...)` orchestration |
| `run_al.py` | CLI |
| `analyze.py` | cross-run aggregation + Markdown report |

## Drift threshold calibration (important)

On the PP_stream with the leakage-clean LogReg:

```
delegate_ratio: min=0.005  median=0.040  max=0.100  p95=0.085
KL divergence:  min=0.049  median=0.286  max=0.561  p95=0.476
```

The spec-default AND `(0.10 ∧ 0.50)` fires **zero** times on the full PP
traversal — the gated LogReg is too confident on PP (predictions
saturate at the band edges, leaving few samples inside `[0.30, 0.70]`).
This is a **finding for the paper** (the leakage-clean model has paid
for cleanliness with reduced calibration on OOD), not a bug.

Three presets exposed via `--threshold-preset`:

| Preset | delegate | KL | Trigger behaviour on PP |
|---|---:|---:|---|
| `strict` (spec default) | 0.10 | 0.50 | 0 triggers — academically pure but inert |
| `permissive` (canonical for paper) | 0.05 | 0.30 | ~5–8 triggers per run, healthy pacing |
| `very_permissive` | 0.03 | 0.20 | more triggers, useful only for ablation |

The paper will report `strict` as the spec-canonical and `permissive`
as the calibrated runtime. Both belong in the ablation.

## Run commands

```bash
# 0. Smoke check with perfect oracle (no LLM key needed).
python3 reboot/step3_al/run_al.py \
    --strategy margin --oracle gt_dryrun \
    --threshold-preset permissive --max-rounds 5 \
    --run-id smoke_margin_gt

# 1. Set LLM credentials.
export AZURE_API_KEY=...
export AZURE_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1
export AZURE_MODEL=DeepSeek-V4-Flash

# 2. Canonical 30-round runs, all 7 strategies.
for STR in margin qbc core_set badge drift_anchored hybrid random; do
    python3 reboot/step3_al/run_al.py \
        --strategy $STR --oracle llm \
        --threshold-preset permissive --max-rounds 30 \
        --run-id "${STR}_llm_temporal"
done

# 3. Aggregate.
python3 reboot/step3_al/analyze.py
```

## Output layout

```
reboot/runs/al/
├── {run_id}/
│   ├── config.json
│   ├── summary.json
│   ├── reference_hist.npy
│   ├── eval_per_round.csv      # per-round metrics on KP/PP/DP
│   ├── rounds.jsonl            # per-trigger audit
│   ├── snapshots/
│   │   ├── scaler.joblib
│   │   └── model_round_{0,5,...,30}.joblib
│   └── oracle_cache/           # gzipped per-sample LLM payloads
└── _aggregate/
    ├── headline_table.csv
    ├── learning_curves_{pp,kp,dp}.csv
    ├── pareto_labels_vs_f1.csv
    └── REPORT.md
```

## Smoke test result (margin + gt_dryrun + permissive, 5 rounds)

```
            R0      R1      R2      R3      R4      R5
KP F1w     0.9806  0.9801  0.9817  0.9817  0.9817  0.9817   ← no forgetting
PP F1w     0.6234  0.6477  0.6696  0.6906  0.7184  0.7666   ← +14pp
DP F1w     0.5141  0.5251  0.5231  0.4971  0.4112  0.4111   ← OOD regression
```

This validates the pipeline end-to-end with a perfect oracle. The
real LLM ablation will show how well D_html_osint (Step-2 winner)
approximates this perfect-oracle ceiling.

## Invariants enforced

- `class_weight = None` (Spec §3.6, locked).
- `frozen_scaler` fit at round-0, never refit.
- DP labels never enter retrain (caller responsibility; documented).
- PP_test labels never read during selection or training.
- Same seed=2025 across all runs.
- Step-2 oracle cache reused verbatim → identical labels regardless of
  which selection strategy picked them. Strategy comparison is on
  *selection*, not on stochastic LLM output.

## Reproducibility

Set `--stream-order temporal --threshold-preset permissive --oracle llm`
with seed=2025 (hardcoded) → identical results modulo LLM
nondeterminism on first encounter. Once a sample's LLM response is
cached, every subsequent run is bit-for-bit deterministic.

To ship the public artifact: bundle `reboot/runs/al/*/oracle_cache/`
with the supplement; reviewers can re-run any strategy without an
API key by reading from the cache.
