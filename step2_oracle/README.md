# Step 2 — Oracle Ablation Lab

Goal: find the single most effective LLM oracle configuration *in
isolation*, before wiring it into the Active Learning loop in Step 3.

## Five variants tested

A 2×2 + 1 ablation that isolates **probabilities** and **OSINT
dossier** as orthogonal context axes, plus a faithful PoC v1
reproduction as a historical baseline.

|  Variant            | HTML | meta probs | OSINT dossier | Notes |
|---------------------|------|-----------|---------------|-------|
| `A_html_probs`      | 8K   | ✓         | ✗             | The user's "Variant A" — PoC-style content with probs |
| `B_html_probs_osint`| 8K   | ✓         | ✓             | The user's "Variant B" — full context |
| `C_html_only`       | 8K   | ✗         | ✗             | Naive LLM baseline |
| `D_html_osint`      | 8K   | ✗         | ✓             | Isolates OSINT signal (no proposer hint) |
| `E_poc_2k`          | 2K   | ✓         | ✗             | Faithful PoC v1 — same prompt, same 2K cap, original schema |

All four 8K variants share the same system prompt and the same
"decide and commit" Conservatism-Bug fix that we ported from
Agent Phase 14. They differ only in the user-message body. The PoC
variant uses the original 4-step prompt with no commit clause —
that's deliberate, so we measure the historical baseline honestly.

## Leakage-clean OSINT dossier

The dossier translates all 23 OSINT columns (Tier A only — Tranco,
Wayback, RDAP, CT analytics, DNS, ASN) into discursive English text
the LLM can reason over. Numbers are kept; interpretation is *not*
pre-applied. The LLM must draw its own conclusions.

Why this matters: in Step 1, eleven of these OSINT columns were
gated out of the meta-learner for **temporal leakage** (KP phishing
samples are dead → OSINT shows "alive vs dead" instead of "benign
vs phishing"). The same eleven columns are perfectly usable as
*contextual hints to an LLM* — the LLM can reason that "no Wayback
snapshots" is a phishing signal **only when combined** with "domain
age < 30 days", which is exactly the multi-hop reasoning a
statistical aggregator can't do safely.

## Sample selection

100 PhreshPhish samples, drawn to *concentrate on ensemble errors*:

- 50 phishing samples, preferring False Negatives (true phish that
  the leakage-clean LogReg from Step 1 missed). Fillers from
  smallest-margin uncertain phish if not enough FNs.
- 50 benign samples, preferring False Positives. Same filler rule.

This is a deliberately *hard* test set. A uniform-random 100 PP
samples would be dominated by easy cases the ensemble already
handles; that wouldn't tell us anything about the oracle's value
for Active Learning. The interesting question is: *of the cases
where the ensemble fails or hesitates, how many does the oracle
recover?* The chosen test set answers exactly that.

## Running

```bash
# 0. Make sure Step 1 + Step 1.5 outputs exist:
ls reboot/runs/osint/z_matrix_pp_osint.csv
ls reboot/runs/ensemble/osint/feature_manifest.json

# 1. Build the 100-sample test set (uses the Step-1 LogReg
#    predictions to pick FN/FP/uncertain).
python3 reboot/step2_oracle/sample_test_set.py

# 2. Set LLM credentials (one of):
export DEEPSEEK_API_KEY=sk-...
# OR
export AZURE_API_KEY=...
export AZURE_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1
export AZURE_MODEL=DeepSeek-V4-Flash   # or gpt-5.4, Kimi-K2.6, etc

# 3. Run the ablation — async, cached, resumable.
python3 reboot/step2_oracle/run_ablation.py

# Optional: smoke run on 5 samples first to validate the
# prompts and the LLM credentials.
python3 reboot/step2_oracle/run_ablation.py --limit 5

# Optional: one variant at a time
python3 reboot/step2_oracle/run_ablation.py --variants A_html_probs

# 4. Compute Precision/Recall/F1 per variant.
python3 reboot/step2_oracle/analyze.py
```

The ablation is fully cached — re-running is a no-op for samples
already done. To force re-fetch a specific (variant, sample), delete
the corresponding file under `cache/<variant>/<sample_id>.json.gz`.

## Outputs

```
reboot/runs/oracle_ablation/
├── test_set.csv                    100 selected PP samples
├── cache/<variant>/<id>.json.gz    raw LLM responses per call
├── results.csv                     long-form: one row per (sample, variant)
├── metrics.csv                     Precision/Recall/F1 per (variant, slice)
└── REPORT.md                       human-readable ablation report
```

The headline table the paper will quote lives in `REPORT.md` under
"Headline metrics". `metrics.csv` carries the FN/FP recovery rates
that demonstrate where each variant helps.

## Expected runtime

500 LLM calls (5 variants × 100 samples) at ~3-5 sec/call with
concurrency=8: roughly **3-5 minutes** wall-clock if the LLM provider
keeps up. Token cost ballpark: ~1.5M prompt tokens + ~0.4M completion
tokens (the OSINT dossier adds ~600 tokens per call vs the bare
PoC prompt).

If the run is interrupted (CTRL-C / timeout), just relaunch — every
already-completed call is in the cache.
