"""cursor_experiment.py — LLM→GT correction cursor at {0,25,50,75,100}%.

Causal probe of the "imperfect LLM labels regularise the meta-learner"
hypothesis (paper §6.7).

Design — PATH-FIXED CURSOR (vs. REPLAY cursor):
  We hold the *selected sample subset* constant — the exact 1250 (sid)
  picked by the canonical LLM run (drift_anchored_B_seed2025) — and
  vary ONLY the label-quality variable. ε% of the LLM mistakes (LLM≠GT)
  are flipped back to GT; everything else stays at the LLM verdict.

  For each ε ∈ {0, 25, 50, 75, 100} × K=3 correction_seeds, we retrain
  the meta-learner from scratch on KP_train ∪ accepted_labels and
  evaluate on KP_test, PP_test, DP_probe.

  Expected pattern (the causal claim):
    PP F1w  monotonically INCREASES with ε    ← more correct = better PP
    DP F1w  monotonically DECREASES with ε    ← noise was the regulariser

  ε = 0   reproduces the LLM-trained model on the held subset.
  ε = 100 == GT-trained model on the same held subset (matched-budget GT).

Outputs (under reboot/runs/al/_cursor/):
  cursor_results.csv     one row per (epsilon, seed) with per-slice metrics
  cursor_summary.json    aggregate means/CIs per epsilon
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step1_ensemble"))

from common import (AL_RUNS_ROOT, build_feature_vector, build_splits,  # noqa
                    load_feature_cols, load_matrices)
from retrain import retrain  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cursor")

CANONICAL_LLM_RUN = "drift_anchored_llm_periodic_temporal_B"
OUT_DIR = AL_RUNS_ROOT / "_cursor"
EPSILONS = (0.0, 0.25, 0.50, 0.75, 1.0)
CORRECTION_SEEDS = (2025, 2026, 2027)


def _llm_label_str_to_int(s) -> int | None:
    if s == "phish":
        return 1
    if s == "benign":
        return 0
    return None  # uncertain / parse_error / etc.


def _eval_slice(model, scaler, X_raw_df: pd.DataFrame, y: np.ndarray,
                feature_cols, fill_means) -> dict[str, float]:
    """Build X via the shared feature_vector pipeline, then evaluate."""
    feat = X_raw_df.apply(
        lambda row: build_feature_vector(row, feature_cols, fill_means),
        axis=1,
    )
    X = np.array([[fv[f] for f in feature_cols] for fv in feat])
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    y_prob = model.predict_proba(Xs)[:, 1]
    has_both = len(set(y)) > 1
    return dict(
        n=int(len(y)),
        phish_frac=float((y == 1).mean()),
        accuracy=float(accuracy_score(y, y_pred)),
        f1_weighted=float(f1_score(y, y_pred, average="weighted",
                                    zero_division=0)),
        f1_macro=float(f1_score(y, y_pred, average="macro",
                                zero_division=0)),
        f1_phish=float(f1_score(y, y_pred, pos_label=1, zero_division=0)),
        precision_phish=float(precision_score(y, y_pred, pos_label=1,
                                              zero_division=0)),
        recall_phish=float(recall_score(y, y_pred, pos_label=1,
                                        zero_division=0)),
        auc_roc=float(roc_auc_score(y, y_prob)) if has_both else 0.0,
        pr_auc=float(average_precision_score(y, y_prob))
                if has_both else 0.0,
    )


def _load_accepted_from_canonical(run_dir: Path,
                                  pp_stream: pd.DataFrame,
                                  ) -> pd.DataFrame:
    """Reconstruct one row per (sid, llm_label, gt_label) for the queried
    samples that have a usable LLM label in the cache."""
    rounds = [json.loads(l) for l in
              (run_dir / "rounds.jsonl").read_text().splitlines()
              if l.strip()]
    queried = [sid for r in rounds for sid in r["queried_sample_ids"]]
    gt_by_sid = pp_stream.set_index("sample_id")["label"].to_dict()
    cache_dir = run_dir / "oracle_cache" / "B"

    rows = []
    n_miss = n_uncertain = n_no_gt = 0
    for sid in queried:
        cp = cache_dir / f"{sid}.json.gz"
        if not cp.exists():
            n_miss += 1
            continue
        with gzip.open(cp, "rt") as f:
            d = json.load(f)
        llm = _llm_label_str_to_int(d.get("label_llm"))
        if llm is None:
            n_uncertain += 1
            continue
        gt = gt_by_sid.get(int(sid))
        if gt is None:
            n_no_gt += 1
            continue
        rows.append(dict(
            sample_id=int(sid),
            llm_label=int(llm),
            gt_label=int(gt),
            is_error=int(llm != gt),
        ))
    df = pd.DataFrame(rows)
    log.info(f"canonical run: queried={len(queried)} "
             f"cache_miss={n_miss} uncertain={n_uncertain} "
             f"no_gt={n_no_gt} -> usable={len(df)}")
    log.info(f"LLM accuracy on usable subset: "
             f"{1.0 - df['is_error'].mean():.4f}  "
             f"({int((df['gt_label']==1).sum())} phish in subset, "
             f"{int(((df['llm_label']==0)&(df['gt_label']==1)).sum())} "
             f"phish→benign errors, "
             f"{int(((df['llm_label']==1)&(df['gt_label']==0)).sum())} "
             f"benign→phish errors)")
    return df


def _build_seed_X(kp_train: pd.DataFrame, feature_cols, fill_means,
                  scaler) -> tuple[np.ndarray, np.ndarray]:
    feat = kp_train.apply(
        lambda r: build_feature_vector(r, feature_cols, fill_means),
        axis=1,
    )
    X_raw = np.array([[fv[f] for f in feature_cols] for fv in feat])
    Xs = scaler.transform(X_raw)
    y = kp_train["label"].astype(int).to_numpy()
    return Xs, y


def run_cursor(canonical_run: str = CANONICAL_LLM_RUN) -> None:
    # Per-run sub-directory so cursors over different canonical runs
    # (drift_anchored vs margin etc.) don't overwrite each other.
    # Slashes in nested run ids (e.g. "multi_llm_bench/...") become "_".
    out_subdir = canonical_run.replace("/", "_")
    out_dir = OUT_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"output dir: {out_dir}")
    run_dir = AL_RUNS_ROOT / canonical_run
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} not found")

    # Re-derive the same splits used by the canonical run.
    summary = json.loads((run_dir / "summary.json").read_text())
    cfg = summary["config"]
    seed = int(cfg.get("seed", 2025))
    stream_order = str(cfg.get("stream_order", "temporal"))

    feature_cols, fill_means = load_feature_cols()
    dfs = load_matrices()
    splits = build_splits(dfs, stream_order=stream_order, seed=seed)
    pp_stream = splits["pp_stream"].reset_index(drop=True)
    pp_test = splits["pp_test"].reset_index(drop=True)
    kp_train = splits["kp_train"].reset_index(drop=True)
    kp_test = splits["kp_test"].reset_index(drop=True)
    dp = splits["dp_probe"].reset_index(drop=True)

    # Frozen scaler: prefer the one persisted by the canonical run; if
    # absent, rebuild it deterministically (StandardScaler.fit on the
    # same KP_train slice — this matches loop.py:130 exactly).
    from sklearn.preprocessing import StandardScaler
    scaler_path = run_dir / "snapshots" / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        log.info(f"loaded frozen scaler from {scaler_path}")
    else:
        feat_kp = kp_train.apply(
            lambda r: build_feature_vector(r, feature_cols, fill_means),
            axis=1,
        )
        X_kp = np.array([[fv[f] for f in feature_cols] for fv in feat_kp])
        scaler = StandardScaler().fit(X_kp)
        log.info("re-fit frozen scaler from KP_train (canonical run had "
                 "no scaler.joblib on disk)")

    # Build accepted-set: (sid, llm_label, gt_label, is_error).
    acc = _load_accepted_from_canonical(run_dir, pp_stream)

    # Index pp_stream rows by sample_id once.
    pp_by_sid = pp_stream.set_index("sample_id")
    if not set(acc["sample_id"]).issubset(set(pp_by_sid.index)):
        missing = set(acc["sample_id"]) - set(pp_by_sid.index)
        raise RuntimeError(
            f"{len(missing)} accepted sample_ids not in pp_stream split — "
            f"the canonical run was built with different splits."
        )

    # Pre-build feature vectors for accepted samples (they don't change
    # across ε; only labels do).
    feat_by_sid: dict[int, dict[str, float]] = {}
    for sid in acc["sample_id"]:
        feat_by_sid[int(sid)] = build_feature_vector(
            pp_by_sid.loc[sid], feature_cols, fill_means,
        )

    # KP seed matrix.
    seed_X, seed_y = _build_seed_X(kp_train, feature_cols, fill_means, scaler)
    log.info(f"seed (KP_train): n={len(seed_y)}, "
             f"phish_frac={float((seed_y==1).mean()):.3f}")

    # Index of error sids → those we can flip.
    err_sids = acc.loc[acc.is_error == 1, "sample_id"].astype(int).tolist()
    n_errors = len(err_sids)
    log.info(f"errors available to flip: {n_errors}")

    results: list[dict] = []

    for eps in EPSILONS:
        n_to_flip = int(round(eps * n_errors))
        for c_seed in CORRECTION_SEEDS:
            rng = np.random.RandomState(c_seed)
            flip_idx = rng.choice(n_errors, size=n_to_flip, replace=False) \
                if n_to_flip > 0 else np.array([], dtype=int)
            flipped_sids = {err_sids[i] for i in flip_idx}

            # Build labels: flipped → GT, others → LLM.
            sid_list = acc["sample_id"].astype(int).tolist()
            llm_lab = dict(zip(sid_list,
                               acc["llm_label"].astype(int).tolist()))
            gt_lab = dict(zip(sid_list,
                              acc["gt_label"].astype(int).tolist()))
            corrected_labels = [
                gt_lab[sid] if sid in flipped_sids else llm_lab[sid]
                for sid in sid_list
            ]
            accepted_fv = [feat_by_sid[int(sid)] for sid in sid_list]

            model, rr = retrain(
                seed_X=seed_X, seed_y=seed_y,
                accepted_feature_vecs=accepted_fv,
                accepted_labels=corrected_labels,
                feature_cols=feature_cols, fill_means=fill_means,
                frozen_scaler=scaler, seed=seed,
            )

            # Evaluate.
            kp_m = _eval_slice(model, scaler,
                               kp_test.drop(columns=["label"]),
                               kp_test["label"].astype(int).to_numpy(),
                               feature_cols, fill_means)
            pp_m = _eval_slice(model, scaler,
                               pp_test.drop(columns=["label"]),
                               pp_test["label"].astype(int).to_numpy(),
                               feature_cols, fill_means)
            dp_m = _eval_slice(model, scaler,
                               dp.drop(columns=["label"]),
                               dp["label"].astype(int).to_numpy(),
                               feature_cols, fill_means)

            row = dict(
                epsilon=eps,
                correction_seed=c_seed,
                n_accepted=len(corrected_labels),
                n_errors_total=n_errors,
                n_errors_flipped=n_to_flip,
                n_train_total=rr.n_train_total,
            )
            for slc_name, m in (("kp", kp_m), ("pp", pp_m), ("dp", dp_m)):
                for k, v in m.items():
                    row[f"{slc_name}_{k}"] = v
            results.append(row)
            log.info(
                f"ε={eps:.2f} seed={c_seed}  "
                f"PP F1w={pp_m['f1_weighted']:.4f}  "
                f"DP F1w={dp_m['f1_weighted']:.4f}  "
                f"KP F1w={kp_m['f1_weighted']:.4f}  "
                f"PP rec_p={pp_m['recall_phish']:.3f}"
            )

    df = pd.DataFrame(results)
    out_csv = out_dir / "cursor_results.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}")

    # Aggregate per epsilon: mean ± std across K seeds.
    agg = df.groupby("epsilon").agg(
        pp_f1w_mean=("pp_f1_weighted", "mean"),
        pp_f1w_std=("pp_f1_weighted", "std"),
        pp_f1macro_mean=("pp_f1_macro", "mean"),
        pp_f1phish_mean=("pp_f1_phish", "mean"),
        pp_recp_mean=("pp_recall_phish", "mean"),
        pp_prauc_mean=("pp_pr_auc", "mean"),
        dp_f1w_mean=("dp_f1_weighted", "mean"),
        dp_f1w_std=("dp_f1_weighted", "std"),
        dp_f1macro_mean=("dp_f1_macro", "mean"),
        dp_f1macro_std=("dp_f1_macro", "std"),
        dp_f1phish_mean=("dp_f1_phish", "mean"),
        dp_f1phish_std=("dp_f1_phish", "std"),
        dp_recp_mean=("dp_recall_phish", "mean"),
        dp_prauc_mean=("dp_pr_auc", "mean"),
        dp_prauc_std=("dp_pr_auc", "std"),
        kp_f1w_mean=("kp_f1_weighted", "mean"),
        kp_f1w_std=("kp_f1_weighted", "std"),
        n_flipped=("n_errors_flipped", "first"),
    ).reset_index()
    summary_out = dict(
        canonical_run=canonical_run,
        n_accepted_subset=int(len(acc)),
        n_errors_total=int(n_errors),
        epsilons=list(EPSILONS),
        correction_seeds=list(CORRECTION_SEEDS),
        per_epsilon=agg.round(6).to_dict(orient="records"),
    )
    out_json = out_dir / "cursor_summary.json"
    out_json.write_text(json.dumps(summary_out, indent=2))
    log.info(f"wrote {out_json}")

    # Headline print: PP (balanced → F1w fine) and DP (imbalanced → use
    # F1_phish + PR-AUC + Macro F1 in addition to F1w).
    print()
    print("PP_test (balanced 48.8/51.2 — F1w is fine):")
    print(f"{'ε':>6s}  {'F1w':>14s}  {'F1macro':>10s}  {'F1phish':>10s}"
          f"  {'rec_p':>8s}  {'PR-AUC':>8s}")
    for r in agg.to_dict(orient="records"):
        print(f"{r['epsilon']:>6.2f}  "
              f"{r['pp_f1w_mean']:>6.4f}±{r['pp_f1w_std'] or 0:.3f}  "
              f"{r['pp_f1macro_mean']:>10.4f}  "
              f"{r['pp_f1phish_mean']:>10.4f}  "
              f"{r['pp_recp_mean']:>8.4f}  "
              f"{r['pp_prauc_mean']:>8.4f}")
    print()
    print("DP_probe (IMBALANCED 15/85 — prefer F1_phish, PR-AUC, Macro F1):")
    print(f"{'ε':>6s}  {'F1w':>14s}  {'F1macro':>14s}  {'F1phish':>14s}"
          f"  {'rec_p':>8s}  {'PR-AUC':>14s}")
    for r in agg.to_dict(orient="records"):
        print(f"{r['epsilon']:>6.2f}  "
              f"{r['dp_f1w_mean']:>6.4f}±{r['dp_f1w_std'] or 0:.3f}  "
              f"{r['dp_f1macro_mean']:>6.4f}±{r['dp_f1macro_std'] or 0:.3f}  "
              f"{r['dp_f1phish_mean']:>6.4f}±{r['dp_f1phish_std'] or 0:.3f}  "
              f"{r['dp_recp_mean']:>8.4f}  "
              f"{r['dp_prauc_mean']:>6.4f}±{r['dp_prauc_std'] or 0:.3f}")
    print()
    print(f"KP stability: F1w {agg['kp_f1w_mean'].min():.4f}–"
          f"{agg['kp_f1w_mean'].max():.4f} across all ε.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-run", default=CANONICAL_LLM_RUN,
                   help="run id of the LLM-trained AL run to use as the "
                        "fixed-path baseline")
    args = p.parse_args()
    run_cursor(args.canonical_run)


if __name__ == "__main__":
    main()
