"""bootstrap_noise_ablation.py — Aggregate the 4-arm noise ablation
with stratified K-seed bootstrap CI.

Pipeline:
  1. For each (arm, seed):
       - Re-build splits with the seed.
       - Load the final model snapshot + frozen scaler.
       - Compute PP_test / KP_test / DP_probe predictions.
       - Stratified bootstrap (per class) × n_boot resamples.
       - Audit: effective noise rate served vs. nominal rate.
  2. Per-arm aggregation across seeds (stratified pooled CI).
  3. Cross-arm comparison table:
       - clean_gt        — ceiling (no noise)
       - random_flip     — symmetric noise control
       - adversarial_p2b — phish→benign worst-case
       - adversarial_b2p — benign→phish mirror
       - llm (optional)  — reference (different selection trajectory)
  4. Statistical test: pairwise bootstrap difference of means with
     two-sided p ≈ 2 · min(P(A>B), P(B>A)) on the pooled resamples.

The headline question for the paper:
  "Does the DP-OOD lift seen under LLM noise persist under random
   noise at matched rate? If yes, the lift is explained by generic
   label-noise regularisation; if no, the LLM noise is *structured*
   and the structure is the contribution."

Outputs:
  reboot/runs/al/_aggregate/noise_ablation_per_seed.csv
  reboot/runs/al/_aggregate/noise_ablation_per_arm.csv
  reboot/runs/al/_aggregate/noise_ablation_pairwise.csv
  reboot/runs/al/_aggregate/noise_ablation.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from common import (
    AL_RUNS_ROOT, build_splits, load_feature_cols, load_matrices,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("noise_ablation")

OUT_DIR = AL_RUNS_ROOT / "_aggregate"
METRICS = ("f1_weighted", "recall_phish", "recall_benign",
           "precision_phish", "precision_benign", "accuracy", "auc")
SLICES = ("pp_test", "kp_test", "dp_probe")


def _to_X(df, cols, means):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "step1_ensemble"))
    from ensemble_reboot import OSINT_NEG1_SENTINEL
    X = np.empty((len(df), len(cols)))
    for j, f in enumerate(cols):
        m = means.get(f, 0.0)
        if f in df.columns:
            col = pd.to_numeric(df[f], errors="coerce")
            if f in OSINT_NEG1_SENTINEL:
                col = col.where(col != -1, other=np.nan)
            X[:, j] = col.fillna(m).to_numpy()
        else:
            X[:, j] = m
    return X


def _metrics(y, pred, prob):
    n_classes = len(set(y.tolist()))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "recall_phish": float(recall_score(y, pred, pos_label=1,
                                           zero_division=0)),
        "recall_benign": float(recall_score(y, pred, pos_label=0,
                                            zero_division=0)),
        "precision_phish": float(precision_score(y, pred, pos_label=1,
                                                 zero_division=0)),
        "precision_benign": float(precision_score(y, pred, pos_label=0,
                                                  zero_division=0)),
        "auc": (float(roc_auc_score(y, prob))
                if n_classes > 1 else float("nan")),
    }


def _final_model_path(run_dir: Path) -> Path:
    snap = run_dir / "snapshots"
    cands = sorted(
        snap.glob("model_round_*.joblib"),
        key=lambda p: int(p.stem.replace("model_round_", "")),
    )
    if not cands:
        raise FileNotFoundError(f"no snapshots under {snap}")
    return cands[-1]


def _audit_noise_in_cache(run_dir: Path) -> dict:
    """Walk the noise cache and tally (gt, served, flipped) over all
    samples actually queried. Returns served-label class balance."""
    cache_glob = list((run_dir / "oracle_cache").rglob("*.json.gz"))
    n_total = n_flipped = 0
    n_phish_served = n_benign_served = 0
    n_phish_gt = n_benign_gt = 0
    for p in cache_glob:
        try:
            j = json.load(gzip.open(p, "rt"))
        except Exception:
            continue
        aud = j.get("noise_audit")
        if aud is None:
            # clean gt_dryrun cache (no noise_audit field) — treat as gt
            lbl = 1 if str(j.get("label_llm")) == "phish" else 0
            n_total += 1
            n_phish_served += int(lbl == 1)
            n_benign_served += int(lbl == 0)
            n_phish_gt += int(lbl == 1)
            n_benign_gt += int(lbl == 0)
            continue
        n_total += 1
        n_flipped += int(aud.get("flipped", False))
        served = int(aud["served"])
        gt = int(aud["gt"])
        n_phish_served += int(served == 1)
        n_benign_served += int(served == 0)
        n_phish_gt += int(gt == 1)
        n_benign_gt += int(gt == 0)
    return {
        "n_cached_total": n_total,
        "n_flipped": n_flipped,
        "effective_flip_rate": (n_flipped / n_total) if n_total else 0.0,
        "served_phish_frac": (n_phish_served / n_total) if n_total else 0.0,
        "served_benign_frac": (n_benign_served / n_total) if n_total else 0.0,
        "gt_phish_frac": (n_phish_gt / n_total) if n_total else 0.0,
        "gt_benign_frac": (n_benign_gt / n_total) if n_total else 0.0,
    }


def per_run_bootstrap(run_id: str, dfs, feat_cols, fill_means, n_boot):
    run_dir = AL_RUNS_ROOT / run_id
    summary = json.loads((run_dir / "summary.json").read_text())
    cfg = summary["config"]
    seed = int(cfg["seed"])
    arm = cfg.get("noise_mode", "none")
    if arm == "none":
        arm = "clean_gt" if cfg["oracle_mode"] == "gt_dryrun" else "llm"
    rate = float(cfg.get("noise_rate", 0.0))

    splits = build_splits(dfs, stream_order=cfg["stream_order"], seed=seed)
    model = joblib.load(_final_model_path(run_dir))
    scaler = joblib.load(run_dir / "snapshots" / "scaler.joblib")

    noise_audit = _audit_noise_in_cache(run_dir)

    slice_dfs = {
        "pp_test": splits["pp_test"],
        "kp_test": splits["kp_test"],
        "dp_probe": splits["dp_probe"],
    }
    boot: dict = {sl: {m: np.empty(n_boot) for m in METRICS} for sl in SLICES}
    point: dict = {}
    n_eval: dict = {}
    rng = np.random.default_rng(seed)
    for sl, df in slice_dfs.items():
        X = _to_X(df, feat_cols, fill_means)
        y = df["label"].values.astype(int)
        X_s = scaler.transform(X)
        pred = model.predict(X_s)
        prob = model.predict_proba(X_s)[:, 1]
        point[sl] = _metrics(y, pred, prob)
        n_eval[sl] = int(len(y))
        idx_p = np.where(y == 1)[0]
        idx_b = np.where(y == 0)[0]
        for b in range(n_boot):
            sp = rng.choice(idx_p, size=len(idx_p), replace=True) \
                if len(idx_p) else np.array([], dtype=int)
            sb = rng.choice(idx_b, size=len(idx_b), replace=True) \
                if len(idx_b) else np.array([], dtype=int)
            si = np.concatenate([sp, sb])
            mm = _metrics(y[si], pred[si], prob[si])
            for m in METRICS:
                boot[sl][m][b] = mm[m]
    log.info(
        f"[{run_id}] arm={arm} seed={seed} rate={rate} "
        f"flipped={noise_audit['n_flipped']}/{noise_audit['n_cached_total']} "
        f"(eff {noise_audit['effective_flip_rate']:.3f}) "
        f"PP F1w={point['pp_test']['f1_weighted']:.4f} "
        f"DP F1w={point['dp_probe']['f1_weighted']:.4f}"
    )
    return {
        "run_id": run_id, "strategy": cfg["strategy"],
        "arm": arm, "seed": seed,
        "noise_rate_nominal": rate,
        "n_eval": n_eval,
        "noise_audit": noise_audit,
        "point": point, "boot": boot,
    }


def aggregate_arms(per_run: list[dict]) -> tuple[pd.DataFrame,
                                                 pd.DataFrame,
                                                 pd.DataFrame,
                                                 str]:
    """Return (per_seed_df, per_arm_df, pairwise_df, markdown)."""
    rows = []
    for r in per_run:
        base = {
            "run_id": r["run_id"], "strategy": r["strategy"],
            "arm": r["arm"], "seed": r["seed"],
            "noise_rate_nominal": r["noise_rate_nominal"],
            "noise_n_flipped": r["noise_audit"]["n_flipped"],
            "noise_n_cached": r["noise_audit"]["n_cached_total"],
            "noise_eff_rate": r["noise_audit"]["effective_flip_rate"],
            "noise_served_phish_frac":
                r["noise_audit"]["served_phish_frac"],
            "noise_gt_phish_frac":
                r["noise_audit"]["gt_phish_frac"],
        }
        for sl in SLICES:
            for m in METRICS:
                base[f"{sl}_{m}_point"] = r["point"][sl][m]
                base[f"{sl}_{m}_ci_lo"] = float(np.nanpercentile(
                    r["boot"][sl][m], 2.5))
                base[f"{sl}_{m}_ci_hi"] = float(np.nanpercentile(
                    r["boot"][sl][m], 97.5))
        rows.append(base)
    per_seed = pd.DataFrame(rows)

    # Per-arm aggregate (pool bootstrap samples across seeds within arm).
    arm_rows = []
    arms_in_data = sorted({(r["strategy"], r["arm"]) for r in per_run})
    for (strategy, arm) in arms_in_data:
        rs = [r for r in per_run
              if r["strategy"] == strategy and r["arm"] == arm]
        if not rs:
            continue
        for sl in SLICES:
            for m in METRICS:
                pts = np.array([r["point"][sl][m] for r in rs], dtype=float)
                pooled = np.concatenate([r["boot"][sl][m] for r in rs])
                arm_rows.append({
                    "strategy": strategy, "arm": arm, "slice": sl,
                    "metric": m, "n_seeds": len(rs),
                    "n_bootstraps_total": int(len(pooled)),
                    "mean_of_points": float(np.nanmean(pts)),
                    "std_across_seeds":
                        float(np.nanstd(pts, ddof=1)) if len(pts) > 1
                        else float("nan"),
                    "pooled_mean": float(np.nanmean(pooled)),
                    "pooled_ci_lo": float(np.nanpercentile(pooled, 2.5)),
                    "pooled_ci_hi": float(np.nanpercentile(pooled, 97.5)),
                    "pooled_ci_half": 0.5 * (
                        float(np.nanpercentile(pooled, 97.5))
                        - float(np.nanpercentile(pooled, 2.5))
                    ),
                })
    per_arm = pd.DataFrame(arm_rows)

    # Pairwise bootstrap difference test (per-strategy, per-slice, per-metric).
    # H0: arm_A pooled = arm_B pooled. Two-sided p ≈ 2·min(p>, p<).
    pair_rows = []
    strategies = sorted({s for (s, _) in arms_in_data})
    for strategy in strategies:
        arms_here = [a for (s, a) in arms_in_data if s == strategy]
        for i, a in enumerate(arms_here):
            for b in arms_here[i + 1:]:
                runs_a = [r for r in per_run
                          if r["strategy"] == strategy and r["arm"] == a]
                runs_b = [r for r in per_run
                          if r["strategy"] == strategy and r["arm"] == b]
                if not runs_a or not runs_b:
                    continue
                for sl in SLICES:
                    for m in METRICS:
                        pa = np.concatenate(
                            [r["boot"][sl][m] for r in runs_a])
                        pb = np.concatenate(
                            [r["boot"][sl][m] for r in runs_b])
                        # Align lengths by truncating to min (different
                        # n_seeds across arms would not be comparable).
                        n = min(len(pa), len(pb))
                        d = pa[:n] - pb[:n]
                        pr_a_gt_b = float(np.mean(d > 0))
                        pr_b_gt_a = float(np.mean(d < 0))
                        p_two = 2.0 * min(pr_a_gt_b, pr_b_gt_a)
                        pair_rows.append({
                            "strategy": strategy,
                            "arm_a": a, "arm_b": b,
                            "slice": sl, "metric": m,
                            "n_paired_bootstraps": int(n),
                            "mean_diff_a_minus_b": float(np.mean(d)),
                            "p_two_sided": float(min(1.0, p_two)),
                            "ci95_lo": float(np.nanpercentile(d, 2.5)),
                            "ci95_hi": float(np.nanpercentile(d, 97.5)),
                        })
    pairwise = pd.DataFrame(pair_rows)

    # ── Markdown ─────────────────────────────────────────────────
    md = ["# Noise Ablation — Causal Origin of DP-OOD Robustness", ""]
    n_seeds_dom = per_seed.groupby(["strategy", "arm"])["seed"].nunique()
    md.append(f"Strategies: {sorted(per_seed['strategy'].unique())}")
    md.append(f"Arms: {sorted(per_seed['arm'].unique())}")
    md.append(f"K seeds per (strategy, arm): "
              f"min={int(n_seeds_dom.min())} max={int(n_seeds_dom.max())}")
    md.append("")
    md.append("## Effective noise rate (audit)")
    md.append("")
    md.append("| Strategy | Arm | seed | nominal_rate "
              "| n_flipped/n_cached | effective_rate "
              "| served_phish_frac | gt_phish_frac |")
    md.append("|---|---|---:|---:|---|---:|---:|---:|")
    for _, row in per_seed.sort_values(["strategy", "arm", "seed"]).iterrows():
        md.append(
            f"| {row['strategy']} | {row['arm']} | {row['seed']} | "
            f"{row['noise_rate_nominal']:.2f} | "
            f"{int(row['noise_n_flipped'])}/{int(row['noise_n_cached'])} | "
            f"{row['noise_eff_rate']:.3f} | "
            f"{row['noise_served_phish_frac']:.3f} | "
            f"{row['noise_gt_phish_frac']:.3f} |"
        )
    md.append("")
    md.append("> **Reading note**: the *nominal* rate is the Bernoulli "
              "parameter passed to the flipper. The *effective* rate "
              "is `n_flipped / n_cached`. For adversarial arms the "
              "effective rate is bounded above by the class share of "
              "the selected sub-population (adversarial_p2b only flips "
              "phish; if the selection is benign-dominated, the effective "
              "rate is much lower than the nominal).")
    md.append("")
    md.append("## Per-arm headline (pooled stratified bootstrap)")
    md.append("")
    md.append("| Strategy | Arm | Slice | F1w (pooled mean ± half_CI) "
              "| R_phish | R_benign |")
    md.append("|---|---|---|---|---|---|")
    for (s, a) in arms_in_data:
        for sl in SLICES:
            row_f1 = per_arm[(per_arm["strategy"] == s)
                             & (per_arm["arm"] == a)
                             & (per_arm["slice"] == sl)
                             & (per_arm["metric"] == "f1_weighted")]
            row_rp = per_arm[(per_arm["strategy"] == s)
                             & (per_arm["arm"] == a)
                             & (per_arm["slice"] == sl)
                             & (per_arm["metric"] == "recall_phish")]
            row_rb = per_arm[(per_arm["strategy"] == s)
                             & (per_arm["arm"] == a)
                             & (per_arm["slice"] == sl)
                             & (per_arm["metric"] == "recall_benign")]
            if row_f1.empty:
                continue
            f1 = row_f1.iloc[0]
            rp = row_rp.iloc[0]
            rb = row_rb.iloc[0]
            md.append(
                f"| {s} | {a} | {sl} | "
                f"{f1['pooled_mean']:.4f} ± {f1['pooled_ci_half']:.4f} | "
                f"{rp['pooled_mean']:.4f} ± {rp['pooled_ci_half']:.4f} | "
                f"{rb['pooled_mean']:.4f} ± {rb['pooled_ci_half']:.4f} |"
            )
    md.append("")
    md.append("## Headline pairwise tests (DP F1w)")
    md.append("")
    md.append("| Strategy | A vs B | mean_diff (A−B) | 95% CI | p (two-sided) |")
    md.append("|---|---|---:|---|---:|")
    for strategy in strategies:
        sub = pairwise[(pairwise["strategy"] == strategy)
                       & (pairwise["slice"] == "dp_probe")
                       & (pairwise["metric"] == "f1_weighted")]
        for _, r in sub.iterrows():
            md.append(
                f"| {strategy} | {r['arm_a']} vs {r['arm_b']} | "
                f"{r['mean_diff_a_minus_b']:+.4f} | "
                f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | "
                f"{r['p_two_sided']:.3f} |"
            )
    md.append("")
    md.append("## Interpretation guide for the paper")
    md.append("")
    md.append("- If **DP F1w (random_flip) ≈ DP F1w (llm)** with "
              "overlapping CIs ⇒ the DP-OOD lift is explained by "
              "generic label-noise regularisation; the LLM noise is "
              "*not* structurally privileged. The paper's Section 6 "
              "should be reframed accordingly.")
    md.append("- If **DP F1w (random_flip) < DP F1w (llm)** with "
              "p<0.05 ⇒ the LLM noise is *structured* and the "
              "structure is the contribution. Section 6 holds.")
    md.append("- If **DP F1w (adversarial_p2b) < DP F1w (random_flip)** ⇒ "
              "the noise *direction* matters; adversarial label-flipping "
              "is more destructive than symmetric noise, justifying the "
              "asymmetric verifier design.")
    md.append("- If **PP F1w (clean_gt) ≫ PP F1w (other arms)** ⇒ "
              "the AL machinery is correctly learning from labels; "
              "noise hurts in-distribution but helps OOD (classic "
              "bias-variance trade-off).")
    return per_seed, per_arm, pairwise, "\n".join(md)


def discover_runs(strategies, arms, seeds) -> list[str]:
    """Resolve arm/strategy/seed triples to actual run_id dirs on disk."""
    out = []
    for s in strategies:
        for a in arms:
            for sd in seeds:
                if a == "clean_gt":
                    rid = f"noise_clean_gt_{s}_seed{sd}"
                elif a == "llm":
                    # Canonical seeded LLM runs from Day 3-4.
                    rid = f"{s}_B_seed{sd}"
                elif a == "random_flip":
                    rid = f"noise_random_r0.30_{s}_seed{sd}"
                elif a == "adversarial_p2b":
                    rid = f"noise_adv_p2b_r0.30_{s}_seed{sd}"
                elif a == "adversarial_b2p":
                    rid = f"noise_adv_b2p_r0.30_{s}_seed{sd}"
                else:
                    continue
                if (AL_RUNS_ROOT / rid).exists():
                    out.append(rid)
                else:
                    log.warning(f"missing: {rid}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", action="store_true",
                   help="Aggregate the canonical bundle (4 noise arms + "
                        "llm reference) × K seeds × {drift_anchored, "
                        "margin}.")
    p.add_argument("--strategies", default="drift_anchored,margin")
    p.add_argument("--arms",
                   default="clean_gt,random_flip,adversarial_p2b,"
                           "adversarial_b2p,llm",
                   help="comma-separated subset of "
                        "{clean_gt, random_flip, adversarial_p2b, "
                        "adversarial_b2p, llm}")
    p.add_argument("--seeds", default="2025,2026,2027")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--allow-missing", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_ids = discover_runs(strategies, arms, seeds)
    if not run_ids:
        raise SystemExit("no runs found; have you launched run_noise_ablation.py?")
    log.info(f"Discovered {len(run_ids)} runs to aggregate")

    log.info("Loading matrices + feature set…")
    feat_cols, fill_means = load_feature_cols()
    dfs = load_matrices()

    per_run = []
    for rid in run_ids:
        per_run.append(per_run_bootstrap(
            rid, dfs, feat_cols, fill_means, n_boot=args.n_boot,
        ))

    per_seed, per_arm, pairwise, md = aggregate_arms(per_run)
    per_seed.to_csv(OUT_DIR / "noise_ablation_per_seed.csv", index=False)
    per_arm.to_csv(OUT_DIR / "noise_ablation_per_arm.csv", index=False)
    pairwise.to_csv(OUT_DIR / "noise_ablation_pairwise.csv", index=False)
    (OUT_DIR / "noise_ablation.md").write_text(md)
    log.info(f"\nWrote {OUT_DIR}/noise_ablation_*.csv + noise_ablation.md")

    # Headline print: DP F1w per arm, per strategy.
    for strategy in strategies:
        log.info(f"\n=== {strategy} — DP F1w headline ===")
        sub = per_arm[(per_arm["strategy"] == strategy)
                      & (per_arm["slice"] == "dp_probe")
                      & (per_arm["metric"] == "f1_weighted")]
        for _, r in sub.sort_values("arm").iterrows():
            log.info(
                f"  {r['arm']:<20s}  "
                f"{r['pooled_mean']:.4f} ± {r['pooled_ci_half']:.4f} "
                f"(K={int(r['n_seeds'])})"
            )


if __name__ == "__main__":
    main()
