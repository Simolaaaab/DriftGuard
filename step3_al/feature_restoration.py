"""feature_restoration.py — Test ③: is the student blind to the teacher?

Hypothesis H2 says the meta-learner cannot reproduce the LLM oracle because the
leakage gate removed the very features (Tranco/Wayback/DNS/ASN) the oracle
reasons on. We test it by RESTORING those 11 hidden features to the student and
measuring whether it can now (A) express the oracle's decision function and
(B) change the PP/DP recovery story.

Fully offline: labels come from the cached oracle (no API).

PART A — Teacher expressivity
  Train a LogReg to predict the *LLM label* (not GT) on PP-stream samples,
  gated (26 feats) vs restored (26+11). If restored ≫ gated, the model was
  structurally blind to the oracle's decision function. A GT-target control
  shows whether the hidden features specifically help match the ORACLE or help
  in general. We also report the restored model's feature importances — do the
  hidden features rank high when fitting the oracle?

PART B — System effect
  Reconstruct the canonical AL accepted set (drift_anchored_B_seed2025) and
  retrain KP_train ∪ accepted on the SAME sample path under three arms:
    gated            — deployed features, LLM labels        (baseline)
    restored         — + 11 gated-out features, LLM labels  (leakage test)
    gated_gt_ceiling — deployed features, GROUND-TRUTH labels (perfect-label
                       ceiling, computed in-pipeline)
  Evaluate PP_test / DP_probe / KP_test on GROUND TRUTH and agreement vs the LLM
  on held-out rows. If restored (LLM) PP exceeds the in-pipeline GT ceiling, the
  restored gain is leakage, proven without mixing pipelines.

Outputs → runs/feature_restoration/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step1_ensemble"))

from common import AL_RUNS_ROOT, build_splits, load_matrices  # noqa: E402
from ensemble_reboot import OSINT_NEG1_SENTINEL                # noqa: E402
from lar_labels import load_llm_label_map                      # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feature_restoration")

OUT_DIR = REBOOT / "runs" / "feature_restoration"
SEED = 2025
CANONICAL_RUN = "drift_anchored_B_seed2025"

MANIFEST = REBOOT / "runs" / "ensemble" / "osint" / "feature_manifest.json"
DROPPED = REBOOT / "runs" / "ensemble" / "osint" / "dropped_for_oracle.json"


def _build_X(df: pd.DataFrame, cols: list[str],
             fill_means: dict[str, float]) -> np.ndarray:
    X = np.empty((len(df), len(cols)))
    for j, f in enumerate(cols):
        m = fill_means.get(f, 0.0)
        if f in df.columns:
            col = pd.to_numeric(df[f], errors="coerce")
            if f in OSINT_NEG1_SENTINEL:
                col = col.where(col != -1, other=np.nan)
            X[:, j] = col.fillna(m).to_numpy()
        else:
            X[:, j] = m
    return X


def _means(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out = {}
    for f in cols:
        if f in df.columns:
            c = pd.to_numeric(df[f], errors="coerce")
            if f in OSINT_NEG1_SENTINEL:
                c = c.where(c != -1, other=np.nan)
            v = c.mean()
            out[f] = float(v) if not np.isnan(v) else 0.0
        else:
            out[f] = 0.0
    return out


def _fit_eval(Xtr, ytr, Xte, yte) -> dict:
    m = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    return {
        "acc": float(accuracy_score(yte, pred)),
        "f1w": float(f1_score(yte, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "model": m,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gated = json.loads(MANIFEST.read_text())["final_feature_set"]
    hidden = [f["feature"] for f in json.loads(DROPPED.read_text())["features"]]
    restored = gated + hidden
    log.info(f"gated={len(gated)}  hidden={len(hidden)}  restored={len(restored)}")

    dfs = load_matrices()
    splits = build_splits(dfs, stream_order="temporal", seed=SEED)
    pp_stream = splits["pp_stream"].reset_index(drop=True)
    pp_test = splits["pp_test"].reset_index(drop=True)
    kp_train = splits["kp_train"].reset_index(drop=True)
    kp_test = splits["kp_test"].reset_index(drop=True)
    dp = splits["dp_probe"].reset_index(drop=True)

    llm = load_llm_label_map()                      # {sample_id: int}

    # ───────────────────────── PART A ─────────────────────────
    # PP-stream rows that carry an LLM label.
    s = pp_stream.copy()
    s["llm"] = s["sample_id"].astype(int).map(llm)
    s = s[s["llm"].notna()].reset_index(drop=True)
    s["llm"] = s["llm"].astype(int)
    log.info(f"[A] PP-stream rows with LLM label: {len(s)}")

    tr, te = train_test_split(s, test_size=0.30, random_state=SEED,
                              stratify=s["llm"].values)
    partA = []
    importances_restored = None
    for target in ("llm", "gt"):
        ytr = tr["llm"].values if target == "llm" else tr["label"].values
        yte = te["llm"].values if target == "llm" else te["label"].values
        for fset_name, cols in (("gated", gated), ("restored", restored)):
            fm = _means(tr, cols)
            sc = StandardScaler().fit(_build_X(tr, cols, fm))
            Xtr = sc.transform(_build_X(tr, cols, fm))
            Xte = sc.transform(_build_X(te, cols, fm))
            r = _fit_eval(Xtr, ytr, Xte, yte)
            partA.append({"target": target, "features": fset_name,
                          "n_feats": len(cols),
                          "agreement_acc": r["acc"], "f1w": r["f1w"]})
            log.info(f"[A] target={target:3s} feats={fset_name:8s} "
                     f"acc={r['acc']:.4f} f1w={r['f1w']:.4f}")
            if target == "llm" and fset_name == "restored":
                coefs = np.abs(r["model"].coef_[0])
                tot = coefs.sum() or 1.0
                importances_restored = sorted(
                    [{"feature": f, "abs_coef": float(c),
                      "pct": float(100 * c / tot),
                      "hidden": f in hidden}
                     for f, c in zip(cols, coefs)],
                    key=lambda d: -d["abs_coef"])
    pd.DataFrame(partA).to_csv(OUT_DIR / "partA_teacher_expressivity.csv",
                               index=False)
    pd.DataFrame(importances_restored).to_csv(
        OUT_DIR / "partA_restored_importance.csv", index=False)

    # ───────────────────────── PART B ─────────────────────────
    # Reconstruct accepted set from the canonical run.
    rounds = [json.loads(l) for l in
              (AL_RUNS_ROOT / CANONICAL_RUN / "rounds.jsonl")
              .read_text().splitlines() if l.strip()]
    queried_ids = []
    for r in rounds:
        if r.get("halted"):
            continue
        queried_ids += [int(x) for x in r.get("queried_sample_ids", [])]
    by_id = {int(r["sample_id"]): r for _, r in pp_stream.iterrows()}
    acc_rows, acc_y = [], []
    for sid in queried_ids:
        lab = llm.get(sid)
        if lab is None or sid not in by_id:
            continue                                # uncertain / missing → skip
        acc_rows.append(by_id[sid])
        acc_y.append(int(lab))
    acc_df = pd.DataFrame(acc_rows)
    log.info(f"[B] accepted labels reconstructed: {len(acc_df)} "
             f"({sum(acc_y)} phish / {len(acc_y)-sum(acc_y)} benign)")

    # Held-out stream rows (LLM-labeled, NOT in accepted) for LLM-agreement.
    acc_ids = set(int(r["sample_id"]) for r in acc_rows)
    held = s[~s["sample_id"].astype(int).isin(acc_ids)].reset_index(drop=True)

    acc_y_arr = np.asarray(acc_y)                       # LLM labels of accepted
    acc_gt = acc_df["label"].values.astype(int)         # ground-truth of accepted
    partB = []
    # Three arms, all on the SAME reconstructed sample path so the comparison is
    # self-contained (absolute values are NOT meant to match the main loop):
    #   gated      — deployed feature space, LLM labels      → baseline
    #   restored   — + 11 gated-out OSINT features, LLM labels → leakage test
    #   gated_gt_ceiling — deployed features, GROUND-TRUTH labels → in-pipeline
    #                      perfect-label ceiling. If restored (LLM) > this, the
    #                      restored gain can only be leakage, proven within-pipeline.
    ARMS = (("gated", gated, "llm"),
            ("restored", restored, "llm"),
            ("gated_gt_ceiling", gated, "gt"))
    for arm_name, cols, label_src in ARMS:
        fm = _means(kp_train, cols)
        sc = StandardScaler().fit(_build_X(kp_train, cols, fm))
        X_kp = sc.transform(_build_X(kp_train, cols, fm))
        X_acc = sc.transform(_build_X(acc_df, cols, fm))
        Xtr = np.vstack([X_kp, X_acc])
        acc_labels = acc_gt if label_src == "gt" else acc_y_arr
        ytr = np.concatenate([kp_train["label"].values, acc_labels])
        model = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        model.fit(Xtr, ytr)

        def ev(df_eval, ycol="label"):
            Xe = sc.transform(_build_X(df_eval, cols, fm))
            pred = model.predict(Xe)
            yt = df_eval[ycol].values
            return (float(f1_score(yt, pred, average="weighted", zero_division=0)),
                    float(f1_score(yt, pred, average="macro", zero_division=0)))

        pp_f1w, _ = ev(pp_test)
        dp_f1w, dp_mac = ev(dp)
        kp_f1w, _ = ev(kp_test)
        # LLM agreement on held-out stream rows.
        Xh = sc.transform(_build_X(held, cols, fm))
        agree = float(accuracy_score(held["llm"].values, model.predict(Xh)))
        partB.append({"features": arm_name, "label_src": label_src,
                      "n_feats": len(cols),
                      "PP_test_f1w": pp_f1w, "DP_macro": dp_mac,
                      "KP_test_f1w": kp_f1w, "LLM_agreement_heldout": agree})
        log.info(f"[B] arm={arm_name:16s} PP={pp_f1w:.4f} DP_macro={dp_mac:.4f} "
                 f"KP={kp_f1w:.4f} LLM_agree={agree:.4f}")
    pd.DataFrame(partB).to_csv(OUT_DIR / "partB_system.csv", index=False)

    # ───────────────────────── Report ─────────────────────────
    A = pd.DataFrame(partA)
    B = pd.DataFrame(partB)
    print("\n" + "=" * 66)
    print("TEST ③ — FEATURE RESTORATION  (gated 26  vs  restored 37)")
    print("=" * 66)
    print("\nPART A — can the student express the teacher? (predict LLM label)")
    print(f"  {'target':6s} {'gated_acc':>10s} {'restored_acc':>13s} {'Δ(pp)':>8s}")
    for target in ("llm", "gt"):
        g = A[(A.target == target) & (A.features == "gated")].iloc[0]
        r = A[(A.target == target) & (A.features == "restored")].iloc[0]
        d = 100 * (r.agreement_acc - g.agreement_acc)
        tag = "← oracle" if target == "llm" else "← ground truth (control)"
        print(f"  {target:6s} {g.agreement_acc:>10.4f} {r.agreement_acc:>13.4f} "
              f"{d:>+7.1f}  {tag}")

    print("\n  Restored model — where the 11 hidden features rank (predict LLM):")
    imp = pd.DataFrame(importances_restored)
    for i, row in imp.head(10).iterrows():
        mark = "  ★HIDDEN" if row.hidden else ""
        print(f"    {i+1:2d}. {row.feature:<26s} {row.pct:5.1f}%{mark}")
    hid_share = imp[imp.hidden].pct.sum()
    hid_in_top10 = int(imp.head(10).hidden.sum())
    print(f"  → hidden features carry {hid_share:.1f}% of |coef| mass, "
          f"{hid_in_top10}/10 of the top-10 when fitting the oracle.")

    print("\nPART B — system effect (KP ∪ accepted, GT-evaluated)")
    print(f"  {'arm':16s} {'PP_f1w':>7s} {'DP_macro':>9s} {'KP_f1w':>7s} "
          f"{'LLM_agree':>10s}")
    for _, r in B.iterrows():
        print(f"  {r.features:16s} {r.PP_test_f1w:>7.4f} {r.DP_macro:>9.4f} "
              f"{r.KP_test_f1w:>7.4f} {r.LLM_agreement_heldout:>10.4f}")
    # Leakage verdict, computed within the same pipeline.
    pp_restored = B[B.features == "restored"].iloc[0].PP_test_f1w
    pp_ceiling = B[B.features == "gated_gt_ceiling"].iloc[0].PP_test_f1w
    pp_gated = B[B.features == "gated"].iloc[0].PP_test_f1w
    verdict = ("LEAKAGE confirmed: restored (LLM labels) exceeds the in-pipeline "
               "perfect-label ceiling" if pp_restored > pp_ceiling else
               "restored does NOT exceed the in-pipeline ceiling")
    print(f"\n  → gated={pp_gated:.4f}  restored={pp_restored:.4f}  "
          f"GT-ceiling={pp_ceiling:.4f}")
    print(f"  → {verdict} "
          f"({pp_restored:.4f} {'>' if pp_restored > pp_ceiling else '<='} "
          f"{pp_ceiling:.4f}).")

    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"partA": partA, "partB": partB,
         "restored_importance_top": importances_restored[:15],
         "hidden_coef_share_pct": float(hid_share),
         "gated_feats": gated, "hidden_feats": hidden}, indent=2))
    print(f"\nartifacts: {OUT_DIR.relative_to(REBOOT)}/")


if __name__ == "__main__":
    main()
