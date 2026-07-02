"""aggregate_kseed_band.py — Phase 2: K-seed band with CIs (READ-ONLY).

Recomputes, for every DONE (selector, seed) cell, the SAME metrics the existing
strategy table used, from eval_per_round.csv at the FINAL round:
  PP weighted F1; DP MACRO F1 = mean(f1_phish, f1_benign) recomputed from the
  per-class columns (NEVER dp_probe_f1w); KP weighted F1; #labels; oracle
  accuracy = sum(oracle_gt_agreement)/sum(n_queried) from rounds.jsonl.

Per selector across its seeds: mean, std, 95% normal-approx CI (mean±1.96·SE) and
a percentile bootstrap CI. Pairwise significance by 95%-CI overlap on PP and DP
macro. Writes reboot/runs/al/_aggregate/strategy_kseed_band.md when the full 8x5
grid is DONE, else strategy_kseed_band.PARTIAL.md.

Touches no existing run and no .tex.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from run_kseed_grid import AL, SELECTORS, SEEDS, target_dir, is_done


def _f1(p, r):
    p, r = float(p), float(r)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def cell_metrics(d: Path) -> dict:
    rows = list(csv.DictReader(open(d / "eval_per_round.csv")))
    last = rows[-1]
    dp_macro = (float(last["dp_probe_f1_phish"])
                + _f1(last["dp_probe_precision_benign"],
                      last["dp_probe_recall_benign"])) / 2.0
    rr = [json.loads(l) for l in (d / "rounds.jsonl").read_text().splitlines()
          if l.strip()]
    rr = [r for r in rr if "n_queried" in r]
    oacc = (sum(r["oracle_gt_agreement"] for r in rr)
            / sum(r["n_queried"] for r in rr)) if rr else float("nan")
    return {
        "pp": float(last["pp_test_f1_weighted"]),
        "dp_macro": dp_macro,
        "kp": float(last["kp_test_f1_weighted"]),
        "labels": rr[-1]["cumulative_labels"] if rr else 0,
        "oacc": oacc,
    }


def band(vals: list[float], n_boot: int = 10000, seed: int = 12345) -> dict:
    a = np.asarray(vals, float)
    n = len(a)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    lo_n, hi_n = mean - 1.96 * se, mean + 1.96 * se
    rng = np.random.default_rng(seed)
    boots = rng.choice(a, size=(n_boot, n), replace=True).mean(axis=1)
    lo_b, hi_b = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return dict(n=n, mean=mean, std=std, se=se,
                ci_normal=(lo_n, hi_n), ci_boot=(lo_b, hi_b))


def overlap(a: dict, b: dict) -> bool:
    """True if the two normal-approx 95% CIs overlap (= NOT significant)."""
    (alo, ahi), (blo, bhi) = a["ci_normal"], b["ci_normal"]
    return not (ahi < blo or bhi < alo)


def main() -> None:
    cells: dict[tuple[str, int], dict] = {}
    done_count = 0
    for sel in SELECTORS:
        for seed in SEEDS:
            d = target_dir(sel, seed)
            if is_done(d):
                cells[(sel, seed)] = {**cell_metrics(d), "dir": d.name}
                done_count += 1
    total = len(SELECTORS) * len(SEEDS)
    full = (done_count == total)

    # Sanity gate.
    da25 = cells.get(("drift_anchored", 2025))
    rd25 = cells.get(("random", 2025))
    if da25 and (abs(da25["pp"] - 0.8005) > 1e-3 or abs(da25["dp_macro"] - 0.5333) > 1e-3):
        raise SystemExit(f"SANITY FAIL: drift_anchored seed-2025 = PP {da25['pp']:.4f} "
                         f"/ DP {da25['dp_macro']:.4f}, expected 0.8005 / 0.5333.")
    if rd25 and abs(rd25["pp"] - 0.745) > 0.01:
        raise SystemExit(f"SANITY FAIL: random seed-2025 PP {rd25['pp']:.4f}, expected ~0.745.")

    bands = {}
    for sel in SELECTORS:
        pp = [cells[(sel, s)]["pp"] for s in SEEDS if (sel, s) in cells]
        dp = [cells[(sel, s)]["dp_macro"] for s in SEEDS if (sel, s) in cells]
        if pp:
            bands[sel] = {"pp": band(pp), "dp": band(dp),
                          "n": len(pp), "kp": [cells[(sel, s)]["kp"] for s in SEEDS if (sel, s) in cells]}

    # ----- write report -----
    out = AL / "_aggregate" / ("strategy_kseed_band.md" if full
                               else "strategy_kseed_band.PARTIAL.md")
    L = []
    L.append("# Selection-Strategy K-Seed Significance Band\n")
    L.append(f"Grid status: **{done_count}/{total} cells DONE**"
             + ("" if full else " — PARTIAL (remaining cells pending).") + "\n")
    L.append("Config (all cells): oracle=llm DeepSeek-V4-Flash, prompt B, periodic/150, "
             "K=50, ~25 rounds, full-union retrain, frozen scaler, verifier off, "
             "health monitor off. DP macro = mean(f1_phish, f1_benign) recomputed "
             "per-class. Oracle acc = Σ agree / Σ queried.\n")
    L.append("Lineage note: drift_anchored/margin seed-2025 use the strategy-table "
             "`*_llm_periodic_temporal_B` run; their 2026–2029 use the `*_B_seed*` "
             "bootstrap lineage; the other six selectors' 2026–2029 are fresh "
             "current-code runs. pure_random_stream is single-lineage (all current "
             "code).\n")

    L.append("\n## 1. Raw 8×5 matrix (PP F1w / DP macro per cell)\n")
    L.append("| selector | " + " | ".join(str(s) for s in SEEDS) + " |")
    L.append("|" + "---|" * (len(SEEDS) + 1))
    for sel in SELECTORS:
        cs = []
        for s in SEEDS:
            c = cells.get((sel, s))
            cs.append(f"{c['pp']:.4f} / {c['dp_macro']:.4f}" if c else "— / —")
        L.append(f"| {sel} | " + " | ".join(cs) + " |")

    L.append("\n## 2. Per-selector band (sorted by PP mean)\n")
    L.append("| selector | n | PP mean | PP 95% CI (norm) | PP 95% CI (boot) | DP mean | DP 95% CI (norm) | DP 95% CI (boot) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sel in sorted(bands, key=lambda s: -bands[s]["pp"]["mean"]):
        b = bands[sel]
        pp, dp = b["pp"], b["dp"]
        L.append(f"| {sel} | {b['n']} | {pp['mean']:.4f} | "
                 f"[{pp['ci_normal'][0]:.4f}, {pp['ci_normal'][1]:.4f}] | "
                 f"[{pp['ci_boot'][0]:.4f}, {pp['ci_boot'][1]:.4f}] | "
                 f"{dp['mean']:.4f} | "
                 f"[{dp['ci_normal'][0]:.4f}, {dp['ci_normal'][1]:.4f}] | "
                 f"[{dp['ci_boot'][0]:.4f}, {dp['ci_boot'][1]:.4f}] |")

    L.append("\n## 3. Pairwise significance (95% normal-approx CI overlap)\n")

    def report_pair(a, b, metric):
        if a not in bands or b not in bands:
            return f"- **{a} vs {b}** ({metric}): one side not yet complete."
        ba, bb = bands[a][metric], bands[b][metric]
        ov = overlap(ba, bb)
        d = ba["mean"] - bb["mean"]
        return (f"- **{a} vs {b}** ({metric}): {a} {ba['mean']:.4f} "
                f"[{ba['ci_normal'][0]:.4f},{ba['ci_normal'][1]:.4f}] vs "
                f"{b} {bb['mean']:.4f} [{bb['ci_normal'][0]:.4f},{bb['ci_normal'][1]:.4f}] "
                f"→ Δ={d:+.4f}, CIs {'OVERLAP → NOT significant' if ov else 'DISJOINT → significant'}.")

    informed = [s for s in bands if s not in ("random", "pure_random_stream")]
    best_inf = max(informed, key=lambda s: bands[s]["pp"]["mean"]) if informed else None
    for metric in ("pp", "dp"):
        L.append(f"\n**{ 'PP F1w' if metric=='pp' else 'DP macro' }:**")
        for a, b in [("pure_random_stream", "drift_anchored"),
                     ("pure_random_stream", "qbc"),
                     ("pure_random_stream", "hybrid")]:
            L.append(report_pair(a, b, metric))
        if best_inf:
            L.append(report_pair(best_inf, "random", metric)
                     + f"  _(best-informed = {best_inf})_")

    L.append("\n## 4. Verdict\n")
    if not full:
        L.append("_Verdict deferred until the full grid is complete._\n")
    else:
        # (i) any informed selector significantly beats pure-random on PP?
        prs = bands.get("pure_random_stream")
        beats_pp = []
        for s in informed:
            if prs and not overlap(bands[s]["pp"], prs["pp"]) and bands[s]["pp"]["mean"] > prs["pp"]["mean"]:
                beats_pp.append(s)
        L.append("**(i) Does any informed selector significantly beat pure-random on PP?** "
                 + (f"Yes: {', '.join(beats_pp)} (disjoint 95% CIs, higher mean)."
                    if beats_pp else
                    "No — no informed selector's PP 95% CI clears pure-random-stream's; "
                    "the PP differences sit within seed noise."))
        # (ii) real PP-DP tradeoff: do diffuse selectors exceed pure-random on DP macro?
        diffuse_better = []
        for s in ("qbc", "hybrid", "margin", "core_set"):
            if s in bands and prs and not overlap(bands[s]["dp"], prs["dp"]) and bands[s]["dp"]["mean"] > prs["dp"]["mean"]:
                diffuse_better.append(s)
        L.append("\n**(ii) Is there a real PP–DP tradeoff (diffuse selectors significantly "
                 "above pure-random on DP macro)?** "
                 + (f"Yes: {', '.join(diffuse_better)} exceed pure-random on DP macro with "
                    "disjoint CIs."
                    if diffuse_better else
                    "No — diffuse selectors do not significantly exceed pure-random on DP "
                    "macro; the tradeoff axis is within noise."))

    L.append("\nK-SEED BAND COMPLETE — strategy_kseed_band.md written; no existing runs modified."
             if full else
             f"\nPARTIAL — {done_count}/{total} cells done; run run_kseed_grid.py to fill the rest.")

    out.write_text("\n".join(L) + "\n")
    print(f"Wrote {out.relative_to(AL.parents[2])}  ({done_count}/{total} cells)")


if __name__ == "__main__":
    main()
