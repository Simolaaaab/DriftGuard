"""sample_test_set.py — Pick 100 hard PP samples for the oracle lab.

Strategy
────────
Train the leakage-clean meta-learner (Step 1, osint+gate) on KP,
predict on PP, and select the 100 most diagnostic samples:

  PHISH side (50 samples):
    - All False Negatives (true=1, pred=0). These are phishing the
      ensemble missed — the oracle is being tested on the same
      content the LogReg couldn't classify.
    - Filler: lowest-margin |prob - 0.5| with true=1, until 50.

  BENIGN side (50 samples):
    - All False Positives (true=0, pred=1). The ensemble cried wolf
      on these benigns — checking if the oracle has the same bias.
    - Filler: lowest-margin |prob - 0.5| with true=0, until 50.

Why this matters
────────────────
A uniformly random 100-sample test set would mostly catch easy cases
the ensemble already handles. By concentrating on the disagreement
boundary, we measure the oracle's lift exactly where it matters for
Active Learning in Step 3.

Output
──────
  reboot/runs/oracle_ablation/test_set.csv
    columns: sample_id, sample_name, domain, url, label,
             ensemble_prob, ensemble_pred, error_type,
             + all original Z-matrix + OSINT columns

`url` is resolved from the PP JSON dataset for prompt building.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]

# Reuse the gate logic from Step 1 by importing its constants and helpers.
sys.path.insert(0, str(REBOOT / "step1_ensemble"))
from ensemble_reboot import (  # noqa: E402
    CORE_FEATURES, OSINT_FEATURES, KNOWN_LEAKY_CANDIDATES,
    OSINT_NEG1_SENTINEL, LEAKAGE_AUC_THRESHOLD,
    TEMPORAL_KP_FLOOR, CROSS_DATASET_GAP_THRESHOLD,
    SEED, TEST_RATIO,
    feature_auc_alone, apply_leakage_gates, prepare_X,
)

Z_OSINT = {
    "kp": REBOOT / "runs" / "osint" / "z_matrix_kp_osint.csv",
    "dp": REBOOT / "runs" / "osint" / "z_matrix_dp_osint.csv",
    "pp": REBOOT / "runs" / "osint" / "z_matrix_pp_osint.csv",
}
PP_JSON = REBOOT / "datasets" / "phreshphish_data.json"
OUT = REBOOT / "runs" / "oracle_ablation" / "test_set.csv"

N_PER_CLASS = 50


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load KP/DP/PP enriched matrices.
    df_kp = pd.read_csv(Z_OSINT["kp"])
    df_dp = pd.read_csv(Z_OSINT["dp"])
    df_pp = pd.read_csv(Z_OSINT["pp"])
    print(f"KP={df_kp.shape}  DP={df_dp.shape}  PP={df_pp.shape}")

    # 2. Reproduce the Step 1 feature selection exactly.
    candidates = CORE_FEATURES + KNOWN_LEAKY_CANDIDATES + OSINT_FEATURES
    in_kp = {c for c in candidates if c in df_kp.columns}
    in_dp = {c for c in candidates if c in df_dp.columns}
    in_pp = {c for c in candidates if c in df_pp.columns}
    intersection = sorted(in_kp & in_dp & in_pp,
                          key=lambda c: candidates.index(c))
    aucs_kp = feature_auc_alone(df_kp, intersection)
    aucs_dp = feature_auc_alone(df_dp, intersection)
    aucs_pp = feature_auc_alone(df_pp, intersection)
    kept, rejected = apply_leakage_gates(intersection, aucs_kp, aucs_dp, aucs_pp)
    print(f"Final feature set ({len(kept)}): {kept}")

    # 3. Train LogReg on the *full* KP (no train/test split — we want
    #    the strongest possible model, since we're using its PP
    #    predictions only for sample selection, not for evaluation).
    fill_means = {
        f: (float(pd.to_numeric(df_kp[f], errors="coerce").mean())
            if f in df_kp.columns and not np.isnan(
                pd.to_numeric(df_kp[f], errors="coerce").mean())
            else 0.0)
        for f in kept
    }
    X_kp = prepare_X(df_kp, kept, fill_means)
    X_pp = prepare_X(df_pp, kept, fill_means)
    y_kp = df_kp.label.values
    y_pp = df_pp.label.values

    scaler = StandardScaler().fit(X_kp)
    model = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    model.fit(scaler.transform(X_kp), y_kp)

    prob_pp = model.predict_proba(scaler.transform(X_pp))[:, 1]
    pred_pp = (prob_pp > 0.5).astype(int)

    cm = pd.crosstab(pd.Series(y_pp, name="truth"),
                     pd.Series(pred_pp, name="pred"))
    print("\nPP confusion matrix:")
    print(cm)

    # 4. Build error/uncertain pools.
    margin = np.abs(prob_pp - 0.5)
    is_fn = (y_pp == 1) & (pred_pp == 0)
    is_fp = (y_pp == 0) & (pred_pp == 1)
    print(f"\nFN (true phish, missed): {is_fn.sum()}")
    print(f"FP (false alarm benign): {is_fp.sum()}")

    # PHISH side.
    fn_idx = np.where(is_fn)[0]
    if len(fn_idx) >= N_PER_CLASS:
        # Top N_PER_CLASS FNs by *lowest* model prob — the ones the
        # ensemble was most confident were benign and still got wrong.
        order = np.argsort(prob_pp[fn_idx])
        phish_idx = fn_idx[order[:N_PER_CLASS]]
        phish_kind = ["FN"] * N_PER_CLASS
    else:
        # FNs not enough → fill with low-margin true phishing.
        phish_idx = list(fn_idx)
        phish_kind = ["FN"] * len(fn_idx)
        remaining = N_PER_CLASS - len(phish_idx)
        candidates_phish = np.where(y_pp == 1)[0]
        candidates_phish = [i for i in candidates_phish if i not in fn_idx]
        # Sort by smallest margin (most uncertain).
        candidates_phish.sort(key=lambda i: margin[i])
        phish_idx = list(phish_idx) + candidates_phish[:remaining]
        phish_kind += ["uncertain_phish"] * remaining

    # BENIGN side.
    fp_idx = np.where(is_fp)[0]
    if len(fp_idx) >= N_PER_CLASS:
        # Top N_PER_CLASS FPs by *highest* prob.
        order = np.argsort(-prob_pp[fp_idx])
        benign_idx = fp_idx[order[:N_PER_CLASS]]
        benign_kind = ["FP"] * N_PER_CLASS
    else:
        benign_idx = list(fp_idx)
        benign_kind = ["FP"] * len(fp_idx)
        remaining = N_PER_CLASS - len(benign_idx)
        candidates_benign = np.where(y_pp == 0)[0]
        candidates_benign = [i for i in candidates_benign if i not in fp_idx]
        candidates_benign.sort(key=lambda i: margin[i])
        benign_idx = list(benign_idx) + candidates_benign[:remaining]
        benign_kind += ["uncertain_benign"] * remaining

    selected_idx = np.concatenate([np.asarray(phish_idx), np.asarray(benign_idx)])
    selected_kind = phish_kind + benign_kind

    out = df_pp.iloc[selected_idx].copy()
    out["ensemble_prob"] = prob_pp[selected_idx]
    out["ensemble_pred"] = pred_pp[selected_idx]
    out["error_type"] = selected_kind

    # 5. Attach URL from the PP JSON.
    with PP_JSON.open() as f:
        entries = json.load(f)
    url_by_id = {int(e["id"]): e.get("url", "") for e in entries}
    domain_by_id = {int(e["id"]): (e.get("domain_name") or "").lower()
                    for e in entries}
    out["url"] = out["sample_id"].astype(int).map(url_by_id)
    # 'domain' may already exist from enrichment; refresh from JSON
    # for safety so it matches the URL.
    out["domain"] = out["sample_id"].astype(int).map(domain_by_id)

    out.to_csv(OUT, index=False)
    print(f"\nSelected {len(out)} samples → {OUT.relative_to(REBOOT)}")
    print(out["error_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
