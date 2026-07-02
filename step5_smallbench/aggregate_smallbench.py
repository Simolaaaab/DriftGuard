"""aggregate_smallbench.py — Comparative tables for the multi-LLM bench.

Reports Δ vs the in-bench DeepSeek baseline on the same small benchmark.
This is the supervisor's explicit reporting convention: "comparative not
absolute".

Outputs:
  reboot/runs/al/_aggregate/smallbench_matrix.csv     (raw cells)
  reboot/runs/al/_aggregate/smallbench_delta.csv      (Δ vs baseline)
  reboot/runs/al/_aggregate/smallbench.md             (paper-ready)

Also includes a sanity row: drift_anchored+B+DeepSeek_v4 on the small
benchmark vs the same config on the full PP — checks whether the
comparative ranking on the subset generalizes to the full dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
from common import AL_RUNS_ROOT          # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aggregate_smallbench")

OUT_DIR = AL_RUNS_ROOT / "_aggregate"
BASELINE_PROVIDER = "deepseek_v4"

# Reference full-PP results for the sanity row.
FULL_PP_REFERENCE = {
    "drift_anchored": {
        "pp_test_f1_weighted": 0.800,
        "kp_test_f1_weighted": 0.978,
        "dp_probe_f1_weighted": 0.627,
        "note": ("drift_anchored+B+temporal canonical seed 2025; "
                 "see Step-3 _aggregate/headline_table.csv"),
    },
    "margin": {
        "pp_test_f1_weighted": 0.777,
        "kp_test_f1_weighted": 0.978,
        "dp_probe_f1_weighted": 0.799,
        "note": "margin+B+temporal canonical seed 2025",
    },
}


def discover_smallbench_runs() -> list[dict]:
    """Walk reboot/runs/al/smallbench_* dirs and parse summary.json."""
    rows = []
    for d in sorted(AL_RUNS_ROOT.glob("smallbench_*")):
        sj = d / "summary.json"
        if not sj.exists():
            log.warning(f"skip {d.name}: no summary.json")
            continue
        try:
            s = json.loads(sj.read_text())
        except Exception as e:
            log.warning(f"skip {d.name}: parse error {e}")
            continue
        rid = d.name
        # Identify strategy + provider from run_id naming.
        # Convention: smallbench_<strategy>_<provider>_seed<N>
        parts = rid.split("_")
        if len(parts) < 4:
            log.warning(f"skip {d.name}: cannot parse run_id")
            continue
        # provider name may contain underscores (e.g. gpt_5_4); seed is last.
        seed_tok = parts[-1]
        if not seed_tok.startswith("seed"):
            log.warning(f"skip {d.name}: no seed token")
            continue
        seed = int(seed_tok[4:])
        # Two strategy naming schemes:
        # - "smallbench_drift_anchored_<provider>_seed..."  → 'drift_anchored'
        # - "smallbench_pure_random_<provider>_seed..."     → 'pure_random'
        # - "smallbench_margin_<provider>_seed..."
        # Strategy is parts[1:2-3]; provider is everything between.
        candidate_strategies = ["drift_anchored", "pure_random",
                                "margin", "qbc", "core_set", "badge",
                                "hybrid", "random"]
        strategy = None
        prov_start = None
        for cand in candidate_strategies:
            n = len(cand.split("_"))
            if "_".join(parts[1:1 + n]) == cand:
                strategy = cand
                prov_start = 1 + n
                break
        if strategy is None:
            log.warning(f"skip {d.name}: cannot identify strategy")
            continue
        provider = "_".join(parts[prov_start:-1])

        cfg = s.get("config") or {}
        fe = s.get("final_eval") or {}
        ie = s.get("initial_eval") or {}
        rows.append({
            "run_id": rid, "strategy": strategy, "provider": provider,
            "seed": seed,
            "manifest": s.get("manifest_name")
                or cfg.get("manifest_name") or "unknown",
            "rounds": s.get("n_rounds_completed"),
            "labels": s.get("cumulative_labels"),
            "llm_calls": s.get("cumulative_llm_calls"),
            "elapsed_s": s.get("elapsed_s"),
            "pp_test_f1w_init": (ie.get("pp_test", {}).get("f1_weighted")
                                 if isinstance(ie.get("pp_test"), dict)
                                 else ie.get("pp_test_f1_weighted")),
            "kp_test_f1w_init": (ie.get("kp_test", {}).get("f1_weighted")
                                 if isinstance(ie.get("kp_test"), dict)
                                 else ie.get("kp_test_f1_weighted")),
            "dp_probe_f1w_init": (ie.get("dp_probe", {}).get("f1_weighted")
                                  if isinstance(ie.get("dp_probe"), dict)
                                  else ie.get("dp_probe_f1_weighted")),
            "pp_test_f1w_final": fe.get("pp_test_f1_weighted"),
            "kp_test_f1w_final": fe.get("kp_test_f1_weighted"),
            "dp_probe_f1w_final": fe.get("dp_probe_f1_weighted"),
            "pp_test_recall_phish_final":
                fe.get("pp_test_recall_phish"),
        })
    return rows


def build_delta_table(matrix: pd.DataFrame) -> pd.DataFrame:
    """For each strategy, subtract the BASELINE_PROVIDER row from all others.
    Reports Δ on the final metrics."""
    rows = []
    metric_cols = ("pp_test_f1w_final", "kp_test_f1w_final",
                   "dp_probe_f1w_final", "pp_test_recall_phish_final")
    for strategy, sub in matrix.groupby("strategy"):
        base = sub[sub["provider"] == BASELINE_PROVIDER]
        if base.empty:
            log.warning(f"strategy={strategy}: no {BASELINE_PROVIDER} "
                        f"baseline row; skipping delta")
            continue
        base = base.iloc[0]
        for _, r in sub.iterrows():
            row = {
                "strategy": strategy,
                "provider": r["provider"],
                "seed": r["seed"],
            }
            for m in metric_cols:
                val = r[m]
                base_val = base[m]
                row[m] = val
                row[f"{m}_delta_vs_baseline"] = (
                    None if (val is None or base_val is None)
                    else float(val) - float(base_val)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def render_md(matrix: pd.DataFrame, delta: pd.DataFrame) -> str:
    md = ["# Small-Benchmark Comparative Bench (multi-LLM × strategy)", ""]
    md.append(f"Baseline provider: **`{BASELINE_PROVIDER}`** "
              "(absolute reference; Δ rows computed against this provider "
              "*per strategy* on the same small-benchmark manifest).")
    md.append("")
    md.append("## Absolute results")
    md.append("")
    md.append("| strategy | provider | seed | rounds | LLM calls "
              "| PP F1w | KP F1w | DP F1w | PP rec_phish |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in matrix.sort_values(
        ["strategy", "provider", "seed"]
    ).iterrows():
        fmt = lambda v: f"{v:.4f}" if v is not None else "—"
        md.append(
            f"| {r['strategy']} | `{r['provider']}` | {r['seed']} | "
            f"{r['rounds']} | {r['llm_calls']} | "
            f"{fmt(r['pp_test_f1w_final'])} | "
            f"{fmt(r['kp_test_f1w_final'])} | "
            f"{fmt(r['dp_probe_f1w_final'])} | "
            f"{fmt(r['pp_test_recall_phish_final'])} |"
        )
    md.append("")
    md.append(f"## Δ vs `{BASELINE_PROVIDER}` (same strategy, same bench)")
    md.append("")
    md.append("| strategy | provider | seed | ΔPP F1w | ΔKP F1w "
              "| ΔDP F1w | ΔR_phish |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in delta.sort_values(["strategy", "provider", "seed"]).iterrows():
        fmt = lambda v: (f"{v:+.4f}" if v is not None else "—")
        md.append(
            f"| {r['strategy']} | `{r['provider']}` | {r['seed']} | "
            f"{fmt(r['pp_test_f1w_final_delta_vs_baseline'])} | "
            f"{fmt(r['kp_test_f1w_final_delta_vs_baseline'])} | "
            f"{fmt(r['dp_probe_f1w_final_delta_vs_baseline'])} | "
            f"{fmt(r['pp_test_recall_phish_final_delta_vs_baseline'])} |"
        )

    # Anchor row: full-PP DeepSeek absolute numbers (NOT a sanity check).
    md.append("")
    md.append("## Full-PP reference (anchor, not a scale-generalization claim)")
    md.append("")
    md.append("The small benchmark uses |PP_stream|=500 / |PP_test|=250 to "
              "make multi-LLM bench affordable. Absolute F1w numbers on this "
              "subset are NOT comparable to the full-PP Main Results because: "
              "(i) PP_test n=250 has SE ≈ ±2.5pp on binomial F1w; (ii) the "
              "AL loop sees ~250 labels vs ~1250 on full → less information; "
              "(iii) the pool fills more slowly at small stream sizes, biasing "
              "drift-triggered selection. The small bench is for **relative** "
              "comparisons across LLM providers at matched budget, not for "
              "reproducing absolute Main Results numbers.")
    md.append("")
    md.append("Full-PP DeepSeek canonical (Step-3, K=5 seeds):")
    md.append("")
    md.append("| strategy | full-PP PP F1w | note |")
    md.append("|---|---:|---|")
    for strategy, ref in FULL_PP_REFERENCE.items():
        md.append(f"| {strategy} | {ref['pp_test_f1_weighted']:.3f} | "
                  f"{ref['note']} |")
    return "\n".join(md)


def main() -> None:
    global BASELINE_PROVIDER
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=BASELINE_PROVIDER,
                   help=f"provider used as Δ-reference (default "
                        f"{BASELINE_PROVIDER})")
    args = p.parse_args()
    BASELINE_PROVIDER = args.baseline

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = discover_smallbench_runs()
    if not rows:
        raise SystemExit(
            f"no runs found under {AL_RUNS_ROOT}/smallbench_*"
            " — launch run_multi_llm_bench.py first"
        )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUT_DIR / "smallbench_matrix.csv", index=False)
    log.info(f"matrix: {len(matrix)} runs across "
             f"{matrix['provider'].nunique()} providers × "
             f"{matrix['strategy'].nunique()} strategies")

    delta = build_delta_table(matrix)
    delta.to_csv(OUT_DIR / "smallbench_delta.csv", index=False)

    md = render_md(matrix, delta)
    (OUT_DIR / "smallbench.md").write_text(md)
    log.info(f"wrote {OUT_DIR}/smallbench_*.csv + smallbench.md")

    # Headline print.
    print("\n=== Headline (PP F1w) ===")
    for strategy in matrix["strategy"].unique():
        sub = matrix[matrix["strategy"] == strategy]
        for _, r in sub.iterrows():
            f1 = r["pp_test_f1w_final"]
            line = f"  {strategy:<18s}  {r['provider']:<15s}  "
            line += f"{f1:.4f}" if f1 is not None else "  —  "
            if r["provider"] != BASELINE_PROVIDER:
                d = delta[(delta["strategy"] == strategy)
                          & (delta["provider"] == r["provider"])]
                if not d.empty:
                    dv = d.iloc[0]["pp_test_f1w_final_delta_vs_baseline"]
                    if dv is not None:
                        line += f"   (Δ={dv:+.4f})"
            print(line)


if __name__ == "__main__":
    main()
