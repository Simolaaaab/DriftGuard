"""verifier_audit.py — Post-hoc audit of cached LLM oracle outputs.

For each cached oracle response in a run's `oracle_cache/B/` dir,
re-evaluates the LLM's claims (indicators_found + indicators_not_found)
against HTML/URL/OSINT using the same predicates as verifier.py.
Produces:

  1. Per-sample audit rows (sample_id, label_llm, gt, correct,
     confidence, T1/T2/T3, trust, n_verified/_unverifiable/_contradicted,
     n_indicators_total, hallucination_rate).
  2. Per-model summary: accuracy, mean trust, mean hallucination rate,
     trust × correctness cross-tab (does low trust ⇒ wrong label?).
  3. Cross-model diagnostic table.

This is the auditor the boss requested: "validates the LLM CoT, indicators
found and not found, to determine which model hallucinates more".

Usage:
  # one cache
  python3 reboot/step3_al/verifier_audit.py \\
      --cache reboot/runs/al/drift_anchored_llm_periodic_temporal_B/oracle_cache/B \\
      --model-name DeepSeek-V4-Flash

  # all model caches at once
  python3 reboot/step3_al/verifier_audit.py --multi
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step2_oracle"))
sys.path.insert(0, str(REBOOT / "step1_ensemble"))

from common import AL_RUNS_ROOT, build_splits, load_matrices  # noqa: E402
from html_loader import HtmlCacheIndex, clean_html, load_pp_html  # noqa
from verifier import compute_trust  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("verifier_audit")

OUT_ROOT = AL_RUNS_ROOT / "_verifier_audit"

# Provider display names — used for the audit table and downstream plots.
PROVIDER_DISPLAY_NAME = {
    "deepseek_v4":   "DeepSeek-V4-Flash",
    "gpt_5_4":       "GPT-5.4",
    "gpt_5_4_mini":  "GPT-5.4-mini",
    "gpt_5_1":       "GPT-5.1",
    "Llama":         "Llama-3.3-70B",
    "mistral_large": "Mistral-Large-3",
    "grok_4":        "Grok-4",
    "kimi_k2_6":     "Kimi-K2.6",
}

# Minimum cache size to include a provider in the bench audit. Drops
# providers that broke at health-monitor halt (e.g. grok_4 had 0 labels
# in the May 2026 bench; including it would give a degenerate row).
BENCH_MIN_CACHE = 50

# Default location of the multi-LLM bench output (created by
# step5_smallbench/run_multi_llm_bench.py).
DEFAULT_BENCH_DIR = AL_RUNS_ROOT / "multi_llm_bench"


def discover_bench_caches(
    bench_dir: Path = DEFAULT_BENCH_DIR,
    *,
    min_cache: int = BENCH_MIN_CACHE,
) -> dict[str, str]:
    """Walk a multi_llm_bench/ output directory and return
    {display_name → run_id (relative to AL_RUNS_ROOT)} for every cell
    that has a non-trivial oracle_cache/B/."""
    out: dict[str, str] = {}
    if not bench_dir.exists():
        log.warning(f"bench dir not found: {bench_dir}")
        return out
    for cell in sorted(bench_dir.iterdir()):
        if not cell.is_dir():
            continue
        cache_dir = cell / "oracle_cache" / "B"
        if not cache_dir.exists():
            continue
        n = sum(1 for _ in cache_dir.glob("*.json.gz"))
        if n < min_cache:
            log.info(f"  skip {cell.name}: only {n} cached labels "
                     f"(min={min_cache})")
            continue
        # Resolve display name from the cell folder pattern
        # "<provider>_drift_anchored_B_k20_seed2025[_<suffix>]".
        stem = cell.name
        provider_key = stem.split("_drift_anchored", 1)[0]
        display = PROVIDER_DISPLAY_NAME.get(provider_key, provider_key)
        # Suffix handling for ablation variants (e.g. _nomonitor).
        # Convention: cells ending in "_<suffix>" get " (<suffix>)" appended
        # to the display, so audit tables distinguish halted/nomonitor/etc.
        CANON = "_drift_anchored_B_k20_seed2025"
        tail = stem.split(CANON, 1)[1] if CANON in stem else ""
        if tail and tail.startswith("_"):
            display = f"{display} ({tail.lstrip('_')})"
        # Run id is the bench-relative path, kept relative to AL_RUNS_ROOT.
        run_id = str(cell.relative_to(AL_RUNS_ROOT))
        out[display] = run_id
        log.info(f"  found {display}: {run_id} (n={n})")
    return out


# Legacy K=50 catalogue kept for ablation comparisons. NOT the default
# anymore — the bench audit uses the new K=20 cells in multi_llm_bench/.
LEGACY_MODEL_CACHES: dict[str, str] = {
    "DeepSeek-V4-Flash":  "drift_anchored_llm_periodic_temporal_B",
    "Kimi-K2.6":          "drift_anchored_B_kimi_seed2025",
    "GPT-5.4":            "smallbench_drift_anchored_gpt_5_4_seed2025",
    "GPT-5.4-mini":       "smallbench_drift_anchored_gpt_5_4_mini_seed2025",
    "Mistral-Large-3":    "smallbench_drift_anchored_mistral_large_seed2025",
}

TRUST_BUCKETS = [(0.0, 0.30, "very_low"),
                 (0.30, 0.55, "low"),
                 (0.55, 0.75, "mid"),
                 (0.75, 0.90, "high"),
                 (0.90, 1.001, "very_high")]


def _label_str_to_int(s) -> int | None:
    if s == "phish":
        return 1
    if s == "benign":
        return 0
    return None


def _bucketize(trust: float) -> str:
    for lo, hi, name in TRUST_BUCKETS:
        if lo <= trust < hi:
            return name
    return "very_high"


def audit_cache(cache_dir: Path, *, model_name: str,
                pp_stream: pd.DataFrame,
                pp_test: pd.DataFrame,
                html_index: HtmlCacheIndex,
                trust_mode: str = "full") -> pd.DataFrame:
    """Walk every cached response in cache_dir and audit it.

    Returns a DataFrame with one row per sample.
    """
    # Pool PP-stream + PP-test rows so we can resolve GT and the
    # OSINT row regardless of which slice the sample landed in.
    pp_all = pd.concat([pp_stream, pp_test], ignore_index=True)
    pp_by_sid = pp_all.set_index("sample_id")

    rows: list[dict] = []
    cache_files = sorted(cache_dir.glob("*.json.gz"))
    log.info(f"[{model_name}] auditing {len(cache_files)} cached responses")

    n_skip_no_html = n_skip_uncertain = n_skip_no_osint = 0
    for cf in cache_files:
        # cf.name is "<sid>.json.gz"; cf.stem strips only the last
        # suffix → "<sid>.json". Take the first dot-segment.
        try:
            sid = int(cf.name.split(".", 1)[0])
        except ValueError:
            continue
        with gzip.open(cf, "rt") as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                continue

        llm_lab = _label_str_to_int(d.get("label_llm"))
        if llm_lab is None:
            n_skip_uncertain += 1
            continue
        conf = float(d.get("confidence") or 0.0)
        indicators_found = d.get("indicators_found") or []
        indicators_not_found = d.get("indicators_not_found") or []
        reasoning = (d.get("reasoning") or "")[:500]

        if sid not in pp_by_sid.index:
            n_skip_no_osint += 1
            continue
        row = pp_by_sid.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        osint_row = row.to_dict()
        gt = int(osint_row.get("label", -1))
        url = str(osint_row.get("url") or "")
        domain = str(osint_row.get("domain") or "").lower()
        sample_name = str(osint_row.get("sample_name") or "")

        raw_html = load_pp_html(sid, sample_name=sample_name, url=url,
                                index=html_index)
        if raw_html is None:
            n_skip_no_html += 1
            raw_html = ""
        html_lower = raw_html.lower()
        soup = BeautifulSoup(raw_html, "html.parser") if raw_html else None

        audit = compute_trust(
            label=llm_lab, confidence=conf,
            indicators_found=indicators_found,
            indicators_not_found=indicators_not_found,
            raw_html=raw_html, url=url, domain=domain,
            osint_row=osint_row, mode=trust_mode,
        )

        n_active = (len(indicators_found) if llm_lab == 1
                    else len(indicators_not_found))
        # Hallucination = the LLM's claim cannot be substantiated against
        # the page/OSINT. We count: not_found (positive claim absent from
        # HTML) + contradicted (negative claim contradicted) as
        # hallucinations. Unverifiable is genuine uncertainty (claim too
        # vague to mechanically check) — half-credit in trust, not counted
        # here as hallucination.
        n_hallucinated = audit.n_indicators_contradicted
        hallucination_rate = (n_hallucinated / n_active
                              if n_active > 0 else float("nan"))

        rows.append(dict(
            model=model_name,
            sample_id=sid,
            label_llm=llm_lab,
            gt_label=gt,
            correct=int(llm_lab == gt) if gt in (0, 1) else None,
            confidence=conf,
            t1=audit.t1_confidence,
            t2=audit.t2_evidence,
            t3=audit.t3_osint,
            trust=audit.trust,
            trust_bucket=_bucketize(audit.trust),
            n_indicators_found=len(indicators_found),
            n_indicators_not_found=len(indicators_not_found),
            n_indicators_total=audit.n_indicators_total,
            n_verified=audit.n_indicators_verified,
            n_unverifiable=audit.n_indicators_unverifiable,
            n_contradicted=audit.n_indicators_contradicted,
            hallucination_rate=hallucination_rate,
            html_available=int(bool(raw_html)),
            domain=domain,
            url=url[:200],
            reasoning_excerpt=reasoning,
        ))

    log.info(f"[{model_name}] audited n={len(rows)}  "
             f"skipped: no_html={n_skip_no_html} "
             f"uncertain={n_skip_uncertain} "
             f"no_osint={n_skip_no_osint}")
    return pd.DataFrame(rows)


def per_model_summary(df: pd.DataFrame) -> dict:
    """Aggregate stats per model — the diagnostic boss wants."""
    if df.empty:
        return {}
    sub = df.dropna(subset=["correct"])
    out = dict(
        model=str(df["model"].iloc[0]),
        n_total=int(len(df)),
        n_with_gt=int(len(sub)),
        accuracy=float(sub["correct"].mean()) if len(sub) else float("nan"),
        n_phish_llm=int((df["label_llm"] == 1).sum()),
        n_benign_llm=int((df["label_llm"] == 0).sum()),
        accuracy_phish=float(
            sub.loc[sub.label_llm == 1, "correct"].mean()
        ) if (sub.label_llm == 1).any() else float("nan"),
        accuracy_benign=float(
            sub.loc[sub.label_llm == 0, "correct"].mean()
        ) if (sub.label_llm == 0).any() else float("nan"),
        mean_confidence=float(df["confidence"].mean()),
        mean_trust=float(df["trust"].mean()),
        mean_t1=float(df["t1"].mean()),
        mean_t2=float(df["t2"].mean()),
        mean_t3=float(df["t3"].mean()),
        mean_indicators_found=float(df["n_indicators_found"].mean()),
        mean_indicators_not_found=float(df["n_indicators_not_found"].mean()),
        # Hallucination = fraction of claims that we could mechanically
        # disprove. Computed per-sample then averaged (NaN-aware: only
        # samples with ≥1 active claim contribute).
        mean_hallucination_rate=float(
            df["hallucination_rate"].dropna().mean()
        ),
        frac_samples_with_any_hallucination=float(
            (df["n_contradicted"] > 0).mean()
        ),
        # Verification quality: fraction of total claims VERIFIED.
        global_verify_rate=float(
            df["n_verified"].sum() /
            max(int(df["n_indicators_total"].sum()), 1)
        ),
        global_contradict_rate=float(
            df["n_contradicted"].sum() /
            max(int(df["n_indicators_total"].sum()), 1)
        ),
        global_unverifiable_rate=float(
            df["n_unverifiable"].sum() /
            max(int(df["n_indicators_total"].sum()), 1)
        ),
        # Trust × correctness: does low trust correlate with wrong labels?
        # This is the user's key question. Higher correlation = the
        # verifier is calibrated to label quality.
        trust_correctness_pearson=float(
            sub[["trust", "correct"]].corr().iloc[0, 1]
        ) if len(sub) >= 2 else float("nan"),
    )
    # Trust-bucket accuracy table (the cross-tab the user asked for):
    # "for low-trust samples, are the labels right or wrong?"
    for lo, hi, bn in TRUST_BUCKETS:
        m = sub[(sub.trust >= lo) & (sub.trust < hi)]
        out[f"acc_bucket_{bn}"] = (float(m["correct"].mean())
                                   if len(m) else float("nan"))
        out[f"n_bucket_{bn}"] = int(len(m))
    return out


def cross_model_run(models: dict[str, str], trust_mode: str = "full",
                    out_dir: Path = OUT_ROOT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading PP splits + HTML index (one-time setup)")
    dfs = load_matrices()
    splits = build_splits(dfs, stream_order="temporal", seed=2025)
    pp_stream = splits["pp_stream"].reset_index(drop=True)
    pp_test = splits["pp_test"].reset_index(drop=True)
    html_index = HtmlCacheIndex()

    per_sample_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for model_name, run_id in models.items():
        cache_dir = AL_RUNS_ROOT / run_id / "oracle_cache" / "B"
        if not cache_dir.exists():
            log.warning(f"missing cache for {model_name} @ {cache_dir}")
            continue
        df = audit_cache(cache_dir, model_name=model_name,
                         pp_stream=pp_stream, pp_test=pp_test,
                         html_index=html_index, trust_mode=trust_mode)
        if df.empty:
            log.warning(f"empty audit for {model_name}")
            continue
        # Per-model artefacts.
        slug = model_name.lower().replace(".", "").replace("-", "_")
        df.to_csv(out_dir / f"per_sample_{slug}.csv", index=False)
        per_sample_frames.append(df)
        s = per_model_summary(df)
        summaries.append(s)

    if not summaries:
        log.error("no models audited successfully")
        return

    pooled = pd.concat(per_sample_frames, ignore_index=True)
    pooled.to_csv(out_dir / "per_sample_pooled.csv", index=False)

    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(out_dir / "model_summary.csv", index=False)
    (out_dir / "model_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )

    # Headline diagnostic table.
    print()
    print("== Cross-model hallucination & trust audit ==")
    headline_cols = [
        "model", "n_total", "accuracy",
        "mean_trust", "mean_hallucination_rate",
        "global_verify_rate", "global_contradict_rate",
        "global_unverifiable_rate", "trust_correctness_pearson",
    ]
    print(sum_df[headline_cols].round(4).to_string(index=False))
    print()
    print("== Trust bucket × accuracy (the 'are low-trust labels wrong?' check) ==")
    bucket_cols = ["model"] + [
        f"{stat}_bucket_{bn}"
        for stat in ("acc", "n")
        for _, _, bn in TRUST_BUCKETS
    ]
    print(sum_df[bucket_cols].round(3).to_string(index=False))
    print()
    log.info(f"all artefacts in {out_dir}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path,
                   help="single oracle_cache/B dir to audit")
    p.add_argument("--model-name", default="custom",
                   help="display name when using --cache")
    p.add_argument("--multi", action="store_true",
                   help="audit all cells in the multi-LLM bench dir "
                        "(auto-discovered from --bench-dir)")
    p.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR,
                   help="root of multi-LLM bench output cells")
    p.add_argument("--legacy", action="store_true",
                   help="audit the legacy K=50 catalogue instead of the "
                        "new bench (for backward-compatibility comparisons)")
    p.add_argument("--min-cache", type=int, default=BENCH_MIN_CACHE,
                   help="skip cells with fewer than N cached labels "
                        "(drops broken providers like grok at R2)")
    p.add_argument("--trust-mode", default="full",
                   choices=("off", "t1_only", "t2_only", "t3_only",
                            "full", "hard_filter"))
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = p.parse_args()

    if args.multi:
        if args.legacy:
            models = LEGACY_MODEL_CACHES
            log.info(f"legacy mode: auditing {len(models)} canonical K=50 cells")
        else:
            log.info(f"auto-discovering cells under {args.bench_dir}")
            models = discover_bench_caches(args.bench_dir,
                                           min_cache=args.min_cache)
            if not models:
                raise SystemExit(
                    f"no usable cells found under {args.bench_dir} "
                    f"(min_cache={args.min_cache}). Pass --legacy to use "
                    f"the K=50 catalogue."
                )
        cross_model_run(models, trust_mode=args.trust_mode,
                        out_dir=args.out_dir)
        return

    if args.cache is None:
        raise SystemExit("pass --cache <dir> or --multi")

    cross_model_run({args.model_name: args.cache.parent.parent.name},
                    trust_mode=args.trust_mode, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
