"""bootstrap_ci.py — K-seed bootstrap 95% CI for canonical AL runs.

For each seed run:
  1. Re-build splits with the run's seed (matching loop.py construction).
  2. Load the final snapshot model + frozen scaler from snapshots/.
  3. Compute per-sample PP_test predictions.
  4. Bootstrap-resample sample indices `n_boot` times (default 2000) →
     bootstrap distribution of the metric (F1_weighted, recall_phish, etc.).
  5. Per-seed point estimate + 95% CI (percentile method).

Across seeds:
  6. Mean of point estimates.
  7. Half-width of the *pooled* (stratified) 95% CI: union of
     2000 × K bootstrap samples, take 2.5th / 97.5th percentiles.

Output:
  reboot/runs/al/_aggregate/bootstrap_<group>.csv     per-seed rows
  reboot/runs/al/_aggregate/bootstrap_<group>.md       headline numbers

Usage:
  python3 reboot/step3_al/bootstrap_ci.py \\
      --group drift_anchored_B \\
      --runs drift_anchored_B_seed2025,drift_anchored_B_seed2026,...

  # or all matching a glob:
  python3 reboot/step3_al/bootstrap_ci.py \\
      --group drift_anchored_B \\
      --glob 'drift_anchored_B_seed20*'

  # report both canonical groups in one go:
  python3 reboot/step3_al/bootstrap_ci.py --canonical
"""

from __future__ import annotations

import argparse
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
log = logging.getLogger("bootstrap_ci")

OUT_DIR = AL_RUNS_ROOT / "_aggregate"

METRICS = (
    "f1_weighted", "f1_phish",
    "precision_phish", "recall_phish",
    "precision_benign", "recall_benign",
    "accuracy", "auc",
)


# ── Helpers ─────────────────────────────────────────────────────


def _to_X(df: pd.DataFrame, cols: list[str],
          means: dict[str, float]) -> np.ndarray:
    """Same imputation as loop._to_X (OSINT_NEG1_SENTINEL → NaN → mean)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step1_ensemble"))
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


def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: np.ndarray) -> dict:
    """Compute the metric set on a resampled subset."""
    n_classes_truth = len(set(y_true.tolist()))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_phish": float(f1_score(y_true, y_pred, pos_label=1,
                                   zero_division=0)),
        "precision_phish": float(precision_score(y_true, y_pred, pos_label=1,
                                                 zero_division=0)),
        "recall_phish": float(recall_score(y_true, y_pred, pos_label=1,
                                           zero_division=0)),
        "precision_benign": float(precision_score(y_true, y_pred, pos_label=0,
                                                  zero_division=0)),
        "recall_benign": float(recall_score(y_true, y_pred, pos_label=0,
                                            zero_division=0)),
        "auc": (float(roc_auc_score(y_true, y_prob))
                if n_classes_truth > 1 else float("nan")),
    }
    return out


def _final_model_path(run_dir: Path) -> Path:
    """Pick the highest-round snapshot in <run_dir>/snapshots/."""
    snap_dir = run_dir / "snapshots"
    cands = sorted(
        snap_dir.glob("model_round_*.joblib"),
        key=lambda p: int(p.stem.replace("model_round_", "")),
    )
    if not cands:
        raise FileNotFoundError(f"no model snapshots under {snap_dir}")
    return cands[-1]


def _load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def per_seed_bootstrap(
    run_id: str, dfs: dict, feature_cols: list[str],
    fill_means: dict[str, float], n_boot: int = 2000,
) -> dict:
    """Run bootstrap CI on one seed run. Returns per-seed dict with:
       - point: dict[metric] point estimate (on full PP_test)
       - boot:  dict[metric] np.ndarray of length n_boot (resampled)
       - n_pp_test, seed, final_round
    """
    run_dir = AL_RUNS_ROOT / run_id
    summary = _load_summary(run_dir)
    seed = int(summary["config"]["seed"])
    stream_order = summary["config"]["stream_order"]
    log.info(f"[{run_id}] seed={seed}, stream={stream_order}, "
             f"re-building splits to extract PP_test…")

    splits = build_splits(dfs, stream_order=stream_order, seed=seed)
    pp_test = splits["pp_test"]
    X_pp = _to_X(pp_test, feature_cols, fill_means)
    y_pp = pp_test["label"].values.astype(int)

    model = joblib.load(_final_model_path(run_dir))
    scaler = joblib.load(run_dir / "snapshots" / "scaler.joblib")
    X_pp_s = scaler.transform(X_pp)
    y_pred = model.predict(X_pp_s)
    y_prob = model.predict_proba(X_pp_s)[:, 1]

    point = _metrics_from_preds(y_pp, y_pred, y_prob)
    log.info(f"[{run_id}] PP_test n={len(y_pp)}  "
             f"point F1w={point['f1_weighted']:.4f}  "
             f"recall_phish={point['recall_phish']:.4f}")

    # Stratified bootstrap by class to keep both classes present in every
    # resample (avoid degenerate AUC / f1 when a tiny minority class can
    # vanish).
    rng = np.random.default_rng(seed)
    idx_phish = np.where(y_pp == 1)[0]
    idx_benign = np.where(y_pp == 0)[0]
    n_phish = len(idx_phish)
    n_benign = len(idx_benign)

    boot: dict[str, np.ndarray] = {m: np.empty(n_boot) for m in METRICS}
    for b in range(n_boot):
        s_phish = rng.choice(idx_phish, size=n_phish, replace=True)
        s_benign = rng.choice(idx_benign, size=n_benign, replace=True)
        s_idx = np.concatenate([s_phish, s_benign])
        m = _metrics_from_preds(y_pp[s_idx], y_pred[s_idx], y_prob[s_idx])
        for k in METRICS:
            boot[k][b] = m[k]

    return {
        "run_id": run_id, "seed": seed,
        "stream_order": stream_order,
        "n_pp_test": int(len(y_pp)),
        "n_pp_test_phish": int(n_phish),
        "n_pp_test_benign": int(n_benign),
        "final_round": int(summary.get("n_rounds_completed", -1)),
        "point": point, "boot": boot,
    }


def aggregate_group(
    group_name: str, per_seed: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Build per-seed table + across-seed pooled table + markdown."""
    # Per-seed table.
    per_rows = []
    for r in per_seed:
        row = {"run_id": r["run_id"], "seed": r["seed"],
               "n": r["n_pp_test"],
               "n_phish": r["n_pp_test_phish"],
               "n_benign": r["n_pp_test_benign"],
               "final_round": r["final_round"]}
        for m in METRICS:
            row[f"{m}_point"] = r["point"][m]
            row[f"{m}_ci_lo"] = float(np.percentile(r["boot"][m], 2.5))
            row[f"{m}_ci_hi"] = float(np.percentile(r["boot"][m], 97.5))
            row[f"{m}_ci_half"] = 0.5 * (row[f"{m}_ci_hi"] - row[f"{m}_ci_lo"])
        per_rows.append(row)
    df_seed = pd.DataFrame(per_rows)

    # Pooled / aggregate.
    agg_rows = []
    for m in METRICS:
        pts = np.array([r["point"][m] for r in per_seed], dtype=float)
        pooled = np.concatenate([r["boot"][m] for r in per_seed])
        agg_rows.append({
            "metric": m,
            "mean_of_points": float(np.nanmean(pts)),
            "std_across_seeds": float(np.nanstd(pts, ddof=1))
                if len(pts) > 1 else float("nan"),
            "pooled_mean": float(np.nanmean(pooled)),
            "pooled_ci_lo": float(np.nanpercentile(pooled, 2.5)),
            "pooled_ci_hi": float(np.nanpercentile(pooled, 97.5)),
            "pooled_ci_half": 0.5 * (
                float(np.nanpercentile(pooled, 97.5))
                - float(np.nanpercentile(pooled, 2.5))
            ),
            "n_seeds": len(per_seed),
            "n_bootstraps_total": int(len(pooled)),
        })
    df_agg = pd.DataFrame(agg_rows)

    # Markdown report.
    md = [f"# Bootstrap CI — {group_name}", ""]
    md.append(f"K = {len(per_seed)} seeds × n_bootstrap = "
              f"{len(per_seed[0]['boot'][METRICS[0]])} per seed "
              f"= {df_agg.iloc[0]['n_bootstraps_total']:,.0f} "
              f"pooled bootstrap resamples")
    md.append("")
    md.append("## Per-seed point estimates")
    md.append("")
    md.append("| run_id | seed | n | F1w | R_phish | R_benign |")
    md.append("|---|---:|---:|---|---|---|")
    for _, row in df_seed.iterrows():
        md.append(
            f"| `{row['run_id']}` | {row['seed']} | {row['n']} | "
            f"{row['f1_weighted_point']:.4f} "
            f"[{row['f1_weighted_ci_lo']:.4f}, "
            f"{row['f1_weighted_ci_hi']:.4f}] | "
            f"{row['recall_phish_point']:.4f} "
            f"[{row['recall_phish_ci_lo']:.4f}, "
            f"{row['recall_phish_ci_hi']:.4f}] | "
            f"{row['recall_benign_point']:.4f} "
            f"[{row['recall_benign_ci_lo']:.4f}, "
            f"{row['recall_benign_ci_hi']:.4f}] |"
        )
    md.append("")
    md.append("## Across-seed aggregate (pooled stratified bootstrap)")
    md.append("")
    md.append("| Metric | mean of points | std (n-1, K seeds) "
              "| pooled mean ± half_CI |")
    md.append("|---|---:|---:|---:|")
    for _, row in df_agg.iterrows():
        md.append(
            f"| {row['metric']} | {row['mean_of_points']:.4f} | "
            f"{row['std_across_seeds']:.4f} | "
            f"{row['pooled_mean']:.4f} ± {row['pooled_ci_half']:.4f} "
            f"[{row['pooled_ci_lo']:.4f}, {row['pooled_ci_hi']:.4f}] |"
        )
    md.append("")
    md.append("**Reporting convention**: paper headline uses "
              "`pooled_mean ± pooled_ci_half` "
              "(stratified bootstrap across K seeds × 2000 resamples each). "
              "`std_across_seeds` is the seed-to-seed variability of the "
              "point estimates (sanity).")
    return df_seed, df_agg, "\n".join(md)


# ── CLI ─────────────────────────────────────────────────────────


CANONICAL_GROUPS = {
    "drift_anchored_B": [
        "drift_anchored_B_seed2025", "drift_anchored_B_seed2026",
        "drift_anchored_B_seed2027", "drift_anchored_B_seed2028",
        "drift_anchored_B_seed2029",
    ],
    "margin_B": [
        "margin_B_seed2025", "margin_B_seed2026", "margin_B_seed2027",
        "margin_B_seed2028", "margin_B_seed2029",
    ],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default=None,
                   help="output prefix; required unless --canonical")
    p.add_argument("--runs", default=None,
                   help="comma-separated run_ids under reboot/runs/al/")
    p.add_argument("--glob", default=None,
                   help="glob to find runs under reboot/runs/al/")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--canonical", action="store_true",
                   help="run both drift_anchored_B and margin_B groups "
                        "with the canonical K=5 seed list (2025..2029)")
    p.add_argument("--allow-missing", action="store_true",
                   help="silently skip run_ids whose directory is missing "
                        "(useful while seeds are still running)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.canonical:
        groups = CANONICAL_GROUPS
    else:
        if args.group is None:
            raise SystemExit("--group required unless --canonical")
        if args.runs:
            run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
        elif args.glob:
            run_ids = sorted(p.name for p in AL_RUNS_ROOT.glob(args.glob)
                             if p.is_dir())
        else:
            raise SystemExit("--runs or --glob required (or use --canonical)")
        groups = {args.group: run_ids}

    log.info("Loading Z matrices + Step-1 feature set…")
    feature_cols, fill_means = load_feature_cols()
    dfs = load_matrices()

    for group, run_ids in groups.items():
        log.info(f"\n=== Group: {group} ({len(run_ids)} runs) ===")
        per_seed = []
        for rid in run_ids:
            run_dir = AL_RUNS_ROOT / rid
            if not run_dir.exists():
                msg = f"missing run dir: {run_dir}"
                if args.allow_missing:
                    log.warning(f"skip {rid}: {msg}")
                    continue
                else:
                    raise FileNotFoundError(msg)
            per_seed.append(per_seed_bootstrap(
                rid, dfs, feature_cols, fill_means, n_boot=args.n_boot,
            ))

        if not per_seed:
            log.warning(f"group {group}: no valid runs found, skipping")
            continue

        df_seed, df_agg, md = aggregate_group(group, per_seed)
        df_seed.to_csv(OUT_DIR / f"bootstrap_{group}_per_seed.csv",
                       index=False)
        df_agg.to_csv(OUT_DIR / f"bootstrap_{group}_aggregate.csv",
                      index=False)
        (OUT_DIR / f"bootstrap_{group}.md").write_text(md)
        log.info(f"[{group}] wrote {OUT_DIR}/bootstrap_{group}_*.csv "
                 f"+ bootstrap_{group}.md")

        # Headline print.
        row = df_agg[df_agg["metric"] == "f1_weighted"].iloc[0]
        log.info(
            f"[{group}] PP F1w pooled = "
            f"{row['pooled_mean']:.4f} ± {row['pooled_ci_half']:.4f}  "
            f"(95% CI [{row['pooled_ci_lo']:.4f}, "
            f"{row['pooled_ci_hi']:.4f}], "
            f"K={len(per_seed)} seeds)"
        )


if __name__ == "__main__":
    main()
