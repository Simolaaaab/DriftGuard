"""verifier_controlled_ablation.py — Fixed-trajectory verifier ablation.

Every verifier mode replays the EXACT per-round queried-sample sequence of the
verifier-OFF canonical run (drift_anchored, DeepSeek, K=50, temporal, seed 2025),
so the ONLY variable is the per-label trust weight. This removes the
trajectory-divergence confound of the original tab:verifier, where each mode
retrains differently -> selects different samples -> the "labels" column conflated
trust filtering with a diverged sample path.

Pure cache replay: NO new LLM calls. Reuses the loop's exact init
(load_feature_cols / build_splits / frozen scaler / round-0 retrain),
build_feature_vector, retrain() and evaluate().

Outputs (runs/al/_verifier_controlled/):
  eval_per_round_<mode>.csv   per-round KP/PP/DP
  per_label_trust.csv         sid, mode, label_llm, gt, correct, weight
  summary.csv                 final metrics + effective label count per mode
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
for p in ("step3_al", "step2_oracle", "step1_ensemble"):
    sys.path.insert(0, str(REBOOT / p))

from common import (  # noqa: E402
    AL_RUNS_ROOT, build_feature_vector, build_splits, load_feature_cols,
    load_matrices,
)
from evaluator import evaluate  # noqa: E402
from html_loader import HtmlCacheIndex, load_pp_html  # noqa: E402
from oracle import STEP2_CACHE  # noqa: E402
from retrain import retrain  # noqa: E402
from verifier import compute_trust  # noqa: E402
from ensemble_reboot import OSINT_NEG1_SENTINEL  # noqa: E402

OFF_RUN = "drift_anchored_llm_periodic_temporal_B_v4_off"
CACHE_DIR = AL_RUNS_ROOT / OFF_RUN / "oracle_cache" / "B"
OUT = AL_RUNS_ROOT / "_verifier_controlled"
MODES = ["off", "t1_only", "t2_only", "t3_only", "full", "hard_filter"]
SEED = 2025
EVAL_SLICES = ("kp_test", "pp_test", "dp_probe")


def _to_X(df, cols, means):
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


def _lab_to_int(s):
    s = str(s).strip().lower()
    if s in ("phish", "phishing", "1", "true"):
        return 1
    if s in ("benign", "legitimate", "0", "false"):
        return 0
    return None


def _eval_all(model, scaler, mats):
    rec = {}
    for slc, (Xv, yv) in mats.items():
        for k, v in evaluate(model, scaler, Xv, yv).items():
            rec[f"{slc}_{k}"] = v
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feature_cols, fill_means = load_feature_cols()
    splits = build_splits(load_matrices(), stream_order="temporal", seed=SEED)

    X_kp_train = _to_X(splits["kp_train"], feature_cols, fill_means)
    y_kp_train = splits["kp_train"]["label"].values
    frozen_scaler = StandardScaler().fit(X_kp_train)
    X_kp_train_s = frozen_scaler.transform(X_kp_train)
    mats = {
        "kp_test": (_to_X(splits["kp_test"], feature_cols, fill_means),
                    splits["kp_test"]["label"].values),
        "pp_test": (_to_X(splits["pp_test"], feature_cols, fill_means),
                    splits["pp_test"]["label"].values),
        "dp_probe": (_to_X(splits["dp_probe"], feature_cols, fill_means),
                     splits["dp_probe"]["label"].values),
    }

    row_by_id = {int(r["sample_id"]): r
                 for _, r in splits["pp_stream"].iterrows()}
    off_rounds = [json.loads(l) for l in
                  (AL_RUNS_ROOT / OFF_RUN / "rounds.jsonl").read_text()
                  .splitlines() if l.strip()]
    trajectory = [r["queried_sample_ids"] for r in off_rounds]
    n_traj = sum(len(s) for s in trajectory)
    print(f"OFF trajectory: {len(trajectory)} rounds, {n_traj} queried samples")

    html_index = HtmlCacheIndex()
    cache = {}

    def get_cache(sid):
        # Mirror CachedOracle's lookup order: step-2 cache (D_html_osint)
        # first, then the run's own variant-B cache.
        if sid not in cache:
            p2 = STEP2_CACHE / f"{sid}.json.gz"
            f = CACHE_DIR / f"{sid}.json.gz"
            if p2.exists():
                cache[sid] = json.loads(gzip.open(p2).read())
            elif f.exists():
                cache[sid] = json.loads(gzip.open(f).read())
            else:
                cache[sid] = None
        return cache[sid]

    summary, per_label_rows = [], []
    for mode in MODES:
        accepted_fv, accepted_y, accepted_w = [], [], []
        model, _ = retrain(
            seed_X=X_kp_train_s, seed_y=y_kp_train,
            accepted_feature_vecs=[], accepted_labels=[],
            feature_cols=feature_cols, fill_means=fill_means,
            frozen_scaler=frozen_scaler, seed=SEED)
        eval_rows = [{"round": 0, **_eval_all(model, frozen_scaler, mats)}]

        for rnd, sids in enumerate(trajectory, 1):
            for sid in sids:
                sid = int(sid)
                j = get_cache(sid)
                if j is None:
                    continue
                lab = _lab_to_int(j.get("label_llm"))
                if lab is None:        # uncertain -> skip (as the loop does)
                    continue
                raw = row_by_id.get(sid)
                if raw is None:
                    continue
                gt = int(raw.get("label", -1))
                if mode == "off":
                    w = 1.0
                else:
                    url = str(raw.get("url") or "")
                    html = load_pp_html(
                        sid, sample_name=str(raw.get("sample_name") or ""),
                        url=url, index=html_index)
                    audit = compute_trust(
                        label=lab, confidence=float(j.get("confidence") or 0.0),
                        indicators_found=j.get("indicators_found") or [],
                        indicators_not_found=j.get("indicators_not_found") or [],
                        raw_html=html, url=url,
                        domain=str(raw.get("domain") or ""),
                        osint_row=raw.to_dict(), mode=mode)
                    w = audit.trust
                per_label_rows.append({
                    "mode": mode, "sid": sid, "round": rnd, "label_llm": lab,
                    "gt": gt, "correct": int(lab == gt), "weight": round(w, 4)})
                if mode == "hard_filter" and w == 0.0:
                    continue           # the only mode that drops outright
                accepted_fv.append(
                    build_feature_vector(raw, feature_cols, fill_means))
                accepted_y.append(lab)
                accepted_w.append(w)

            model, _ = retrain(
                seed_X=X_kp_train_s, seed_y=y_kp_train,
                accepted_feature_vecs=accepted_fv, accepted_labels=accepted_y,
                feature_cols=feature_cols, fill_means=fill_means,
                frozen_scaler=frozen_scaler, seed=SEED,
                accepted_trust_weights=(accepted_w if mode != "off" else None))
            eval_rows.append({"round": rnd,
                              **_eval_all(model, frozen_scaler, mats)})

        pd.DataFrame(eval_rows).to_csv(
            OUT / f"eval_per_round_{mode}.csv", index=False)
        fin = eval_rows[-1]
        # effective label count: weight>0 entering retrain
        n_eff = sum(1 for w in accepted_w if w > 0)
        n_lowhalf = sum(1 for w in accepted_w if w < 0.5)
        summary.append({
            "mode": mode, "n_accepted": len(accepted_fv),
            "n_weight_gt0": n_eff, "n_weight_lt0.5": n_lowhalf,
            "PP_f1w": round(fin["pp_test_f1_weighted"], 4),
            "PP_recall_phish": round(fin["pp_test_recall_phish"], 4),
            "DP_f1w": round(fin["dp_probe_f1_weighted"], 4),
            "KP_f1w": round(fin["kp_test_f1_weighted"], 4)})
        print(f"  {mode:12} n_acc={len(accepted_fv):4} "
              f"w<0.5={n_lowhalf:4} PP={fin['pp_test_f1_weighted']:.4f} "
              f"DP={fin['dp_probe_f1_weighted']:.4f} "
              f"KP={fin['kp_test_f1_weighted']:.4f}")

    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_label_rows).to_csv(OUT / "per_label_trust.csv", index=False)
    print(f"\nwrote {OUT}/summary.csv")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
