# Cache Integrity Forensic — Curated DeepSeek Union

**Question:** is the "pre-seeded curated DeepSeek-V4-Flash union" cache that the
fresh June 2026–2029 runs read a legitimate same-oracle cache, or is it
constructed in a way that leaks ground truth / inflates oracle accuracy?

**Verdict up front: B — INFLATED BY GROUND-TRUTH LEAKAGE.** The union admits
`source="gt_dryrun"` cache entries, which are **perfect-oracle ground-truth
labels**, not LLM outputs. 1793 of 4189 union entries (43%) are GT-derived, and
the fresh runs consumed ~61–67% of their training labels from them.

---

## 1. The cache + the construction code

The fresh cells read from `reboot/runs/al/<cell>/oracle_cache/B/`, pre-seeded by
`reboot/step3_al/run_kseed_grid.py`. Two functions build/seed it:

`build_deepseek_union()` (run_kseed_grid.py:94–113) — the literal selection of
which cached label is kept per sample_id:

```python
for bdir in AL.rglob("oracle_cache/B"):
    rp = str(bdir.relative_to(AL))
    if SKIP_DIRS.search(rp) or NON_DEEPSEEK.search(rp):     # :98  dir-name filter only
        continue
    for f in bdir.glob("*.json.gz"):
        ...
        if sid in ids:                                       # :105 FIRST-ENCOUNTERED WINS
            continue
        with gzip.open(f, "rt") as fh:
            if json.load(fh).get("ok"):                      # :109 ONLY filter = ok==True
                ids[sid] = str(f)
```

`seed_cache()` (run_kseed_grid.py:116–125) then `shutil.copy2`s each union file
into the cell's `oracle_cache/B/` if absent.

Filters in force (run_kseed_grid.py:72–73):
```python
NON_DEEPSEEK = re.compile(r'(kimi|gpt_5|gpt5|grok|mistral|llama)', re.I)
SKIP_DIRS    = re.compile(r'(_costprobe|_smoke|_codecheck|_kseedtmp)')
```

## 2. How the kept label is chosen — and the GT check that is MISSING

When multiple cached responses exist for one sample_id, the kept label is the
**first file encountered in `AL.rglob` order** (run_kseed_grid.py:105,
`if sid in ids: continue`). The only content filter is `ok==True`
(run_kseed_grid.py:109).

- The selection **never references the ground-truth label, an eval label, or any
  `is_correct`/agreement field.** It does **not** prefer the entry whose verdict
  equals GT. So there is no *correctness-biased curation*.
- **But it also never checks the `source` field.** Entries written by the
  `gt_dryrun` oracle path carry `"ok": true` and a `label_llm` that is literally
  the ground-truth label:

  `oracle.py:319–364` (gt_dryrun branch):
  ```python
  if self.mode == "gt_dryrun":                       # :319
      ...
      payload = { "ok": True,                          # :341
                  "label_llm": "phish" if served == 1 else "benign",  # :342  = GT
                  "confidence": 1.0,
                  "reasoning": "gt_dryrun perfect oracle",
                  "source": src_tag,                   # "gt_dryrun"
                  "noise_audit": {"gt": ..., "served": ..., "flipped": False, "mode": "none"} }
      self._step3_save(sid, payload)                   # :364  writes GT into oracle_cache/B/
  ```

  Because `gt_dryrun` writes into the **same `oracle_cache/B/` namespace** with
  `ok==True`, and the union filter is `ok`-only, **these ground-truth labels are
  admitted into the "DeepSeek oracle" union.** **This is LEAKAGE** — the leaked
  values *are* ground truth, even though the selection logic itself is GT-blind.

## 3. Provenance of the union labels

The cache payloads store **no model / prompt-id / temperature field** (keys:
`completion_tokens, confidence, indicators_found, indicators_not_found,
label_llm, latency_ms, ok, phishing_type, prompt_tokens, raw_text, reasoning,
source, top_factors`). Provenance is only recoverable from the `source` field and
the origin directory. Distribution actually present in the union (4189 entries):

| source field | count | what it is |
|---|---|---|
| `llm` | 2396 (57%) | real oracle output (DeepSeek dirs; model not otherwise verifiable) |
| `gt_dryrun` | **1793 (43%)** | **perfect-oracle GROUND TRUTH** (`noise_audit.flipped=False, mode=none`) |

No gpt/kimi/grok/mistral/llama entries (the NON_DEEPSEEK regex did exclude those).
The contamination is **not** cross-provider — it is **GT-vs-LLM**.

**Original leak sources** (gt_dryrun runs with a direct `oracle_cache/B/` that the
exclusion regex failed to catch — `_smoke` requires a leading underscore, so
`smoke_*` slips through; `_dryrun_*`/`noise_clean_gt_*` are not listed at all):

| run dir | GT files | excluded by union? |
|---|---|---|
| `_dryrun_random_batching` | 1242 | **No** |
| `noise_clean_gt_drift_anchored_seed{2025,2026,2027}` | 1241/1241/1246 | **No** |
| `noise_clean_gt_margin_seed{2025,2026,2027}` | 1227/1231/1231 | **No** |
| `smoke_b_gt` / `smoke_seed2026` / `smoke_step4_gt` | 243 / 147 / 147 | **No** |
| `_costprobe_pure_random_stream_gt` | 1227 | Yes (`_costprobe`) |

(The `noise_adv_*`/`noise_random_*` runs cache under `B/<tag>/` subdirs, so the
non-recursive `B/*.json.gz` glob did not pick them up — only the noise=none "clean
GT" and smoke gt_dryrun runs leaked.)

## 4. Apples-to-apples: did the fresh runs actually consume GT?

Per-queried-label source for three fresh seed-2026 cells (label source read from
each run's own seeded cache; GT nature confirmed by `noise_audit.flipped=False`):

| fresh run | queried | from `gt_dryrun` (GT) | from `llm` | % GT-derived | recorded oracle_acc |
|---|---|---|---|---|---|
| qbc seed2026 | 1250 | **784** | 446 | **62.7%** | 0.8584 |
| random seed2026 | 1250 | **767** | 468 | **61.4%** | 0.9376 |
| badge seed2026 | 1250 | **838** | 390 | **67.0%** | 0.8928 |

The recorded oracle accuracy is a direct blend: ~63% perfect-GT labels (100%
agreement by construction) + ~36% real-LLM labels (~0.49–0.59 agreement) ≈
0.84–0.94 — exactly the observed values. By contrast the OLD May seed-2025 runs
made their own LLM calls (`s3_cache=0`) and scored true-LLM oracle accuracy
~0.70–0.74. **The ~7 pp PP jump attributed to "cleaner labels" in
lineage_diagnosis.md is this GT contamination: roughly two-thirds of the fresh
runs' training labels were the ground-truth answer.**

(Note: a direct per-source accuracy recomputation against a freshly rebuilt
`sample_id→label` map showed the `gt_dryrun` bucket at ~0.78 rather than 1.0 — a
mapping artifact in the audit script, not the runs; the embedded `noise_audit`
and the runs' own recorded `oracle_gt_agreement` both confirm these entries are
unflipped ground truth.)

## Verdict

The selection criterion is GT-blind first-wins (no correctness-biased curation),
**but** the union is `source`-blind and therefore admits `gt_dryrun` perfect-GT
labels from un-excluded dry-run/smoke/clean-GT runs (oracle.py:341–364 writes
them; run_kseed_grid.py:105–109 admits them on `ok` alone). 43% of the union and
~63% of each fresh run's consumed labels are ground truth, which is what inflated
oracle accuracy and PP for the fresh 2026–2029 cells. The fresh-vs-old "lineage
gap" is not cleaner same-oracle labels — it is ground-truth leakage.

## Scope of the contamination

Every run that read one of my pre-seeded caches is affected; every run that made
its own LLM calls is clean:

| run | GT-derived labels consumed | status |
|---|---|---|
| `pure_random_stream` seed2025 (earlier headline run) | 547/1250 (44%) | **CONTAMINATED** |
| `pure_random_stream` seed2026 | 521/1250 (42%) | **CONTAMINATED** |
| all 24 fresh kseed cells (qbc/core_set/badge/hybrid/random/pure_random 2026–29) | ~61–67% | **CONTAMINATED** |
| `random_llm_periodic_temporal_B` (May) | 0/1250 | clean |
| `drift_anchored_llm_periodic_temporal_B` (May) | 0/1250 | clean |
| drift_anchored/margin all seeds, strategy-table seed-2025 cells | 0% (own LLM calls) | clean |

This invalidates the earlier **"pure-random-stream (0.809) Pareto-dominates
drift_anchored (0.800)"** conclusion: pure_random_stream's seed-2025 PP was lifted
by 44% ground-truth labels. The honest, leakage-free comparators are the May
strategy-table runs (drift_anchored 0.800, random 0.745, etc.), which used 100%
real LLM oracle labels.

Remediation: rebuild the union with `source == "llm"` enforced (drop every
`gt_dryrun*` entry), or exclude the dry-run/smoke/clean-GT dirs by config
(`oracle_mode == "gt_dryrun"`) rather than by fragile name regex; then re-run the
24 fresh cells **and the five pure_random_stream cells** so their oracle labels
are real LLM outputs. Treat the current `strategy_kseed_band.md`,
`pure_random_rung.md`, and `lineage_diagnosis.md` fresh-run numbers as void.

CACHE INTEGRITY CHECK COMPLETE — verdict: B (inflated).
