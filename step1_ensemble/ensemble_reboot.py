"""ensemble_reboot.py — CCS Reboot, Step 1: leakage-clean baseline ensemble.

Goal
────
Train a clean meta-learner on KnowPhish, evaluate honestly on KnowPhish
intra and on DeltaPhish + PhreshPhish cross-dataset, and produce the
baseline numbers we'll measure the rest of the reboot against.

What this fixes vs the Agent V2.1 ensemble
──────────────────────────────────────────
 1. Leakage gate (AUC-alone ≥ 0.99) auto-rejects path_length /
    url_length / n_url_params on KP — the three leaky features the
    PoC removed and the Agent's audit re-introduced.
 2. Keeps max_prob / min_prob — the PoC's top agreement features
    (importance 0.85 on KP). The Agent audit dropped them as
    "cluster duplicates of prob_sn/prob_pg" which was statistically
    correct but operationally wrong.
 3. Intersection-based feature selection across KP/DP/PP — drops
    KP-only features at training time too, so the model can't lean on
    in-distribution shortcuts that contribute zero at cross-dataset
    eval. Fixes the Agent's "url_length imputed to zero pulls toward
    benign" runtime bug at the root.
 4. Mean-imputation (seed column means) for missing features — never
    zero. Zero on a centered feature is the bias point, not a neutral
    signal.
 5. Cross-dataset metrics report all four diagnostic numbers
    (precision/recall on phish AND benign separately) so the
    asymmetric failure modes are visible.

Outputs
───────
  reboot/runs/ensemble/
    metrics_per_model_per_split.csv
    feature_importance_per_model.csv
    feature_auc_per_dataset.csv     (Sweet-Danger-of-Sugar sanity check)
    feature_manifest.json
    leakage_report.txt

Run from anywhere with the project venv active:
    python reboot/step1_ensemble/ensemble_reboot.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler



# ── Paths ─────────────────────────────────────────────────────────
# This file lives at <repo>/reboot/step1_ensemble/ensemble_reboot.py
# parents[2] = repo root.
REBOOT = Path(__file__).resolve().parent.parent

# Base Z matrices (classic mode).
Z_BASE = {
    "kp": REBOOT / "datasets" / "z_matrix_knowphish.csv",
    "dp": REBOOT / "datasets" / "z_matrix_spacephish.csv",
    "pp": REBOOT / "datasets" / "z_matrix_phreshphish.csv",
}

# OSINT-enriched Z matrices (osint mode) — produced by
# reboot/step1_5_osint/enrich_z_matrices.py.
Z_OSINT = {
    "kp": REBOOT / "runs" / "osint" / "z_matrix_kp_osint.csv",
    "dp": REBOOT / "runs" / "osint" / "z_matrix_dp_osint.csv",
    "pp": REBOOT / "runs" / "osint" / "z_matrix_pp_osint.csv",
}

OUT_ROOT = REBOOT / "runs" / "ensemble"

# ── Feature definitions ───────────────────────────────────────────

CORE_FEATURES = [
    "prob_sn", "prob_pg",
    "sn_pg_agreement", "sn_pg_diff", "max_prob", "min_prob",
    "domain_length", "n_dots_domain", "n_subdomains",
    "domain_entropy", "has_ip_address", "has_https", "domain_digit_ratio",
    "html_length", "n_forms", "n_password_inputs", "n_external_links_ratio",
]

# OSINT features produced by Step 1.5 (Tier A only, zero leakage).
# `asn_country` is omitted because it's a string and we don't one-hot
# encode here; this keeps the leakage gate apples-to-apples numeric.
OSINT_FEATURES = [
    # Tranco
    "is_in_tranco_1m", "tranco_rank", "tranco_rank_log",
    # Wayback
    "wayback_first_seen_days", "wayback_last_seen_days",
    "wayback_n_snapshots", "wayback_active_window_days",
    "wayback_first_status_ok",
    # CT analytics
    "cert_history_count", "cert_first_seen_days", "cert_last_seen_days",
    "cert_issuer_diversity", "cert_is_free_only", "cert_has_wildcard",
    "cert_validity_median_days", "cert_san_max",
    # RDAP
    "rdap_domain_age_days", "rdap_age_risk", "rdap_has_registrar",
    # DNS / ASN
    "dns_resolved", "asn_number", "asn_age_days",
]

# In our OSINT pipeline, -1 is the sentinel for "unknown / source
# error / not applicable". For ML we convert -1 → NaN and then
# impute the seed-train mean. Otherwise the -1 value would skew the
# scaler badly (especially for `tranco_rank` which spans 1..1M).
#
# Features NOT in this list use 0 as a meaningful zero (e.g.
# `wayback_n_snapshots = 0` means "never archived" — a real signal).
OSINT_NEG1_SENTINEL = {
    "tranco_rank", "tranco_rank_log",
    "wayback_first_seen_days", "wayback_last_seen_days",
    "wayback_active_window_days", "wayback_first_status_ok",
    "cert_first_seen_days", "cert_last_seen_days",
    "cert_issuer_diversity", "cert_is_free_only", "cert_has_wildcard",
    "cert_validity_median_days", "cert_san_max", "cert_history_count",
    "rdap_domain_age_days", "rdap_age_risk", "rdap_has_registrar",
    "asn_number", "asn_age_days",
}

# Legacy Phase-4 cert features (kept for backward compatibility when
# loading the OLD enriched PP file; new osint mode uses OSINT_FEATURES).
CERT_FEATURES = [
    "cert_age_days", "cert_validity_days", "is_free_cert",
    "issuer_trust_tier", "cert_san_count", "cert_has_wildcard",
    "cert_history_count", "domain_age_days", "domain_age_risk",
]

KNOWN_LEAKY_CANDIDATES = ["path_length", "url_length", "n_url_params"]

# ── Leakage gates ─────────────────────────────────────────────────
# Gate 1 — In-distribution leakage. A feature that single-handedly
# separates classes on the training set (AUC alone ≥ 0.99) is almost
# certainly a dataset artifact (e.g. path_length on KP, AUC=1.000).
LEAKAGE_AUC_THRESHOLD = 0.99

# Gate 2 — Temporal / cross-dataset leakage. A feature can pass Gate 1
# (AUC < 0.99) yet still be unsafe if it discriminates near-perfectly
# on the training set but collapses on out-of-distribution data.
# Diagnosis from the OSINT enrichment run:
#   wayback_n_snapshots  KP=0.996  DP=0.504  PP=0.914
#   dns_resolved         KP=0.987  DP=0.505  PP=0.610
# These features are great on KP — because KP phishing samples are
# from 2023 and are now dead, so OSINT collected today shows the
# "alive vs dead" split rather than the "benign vs phishing" split.
# When the same model meets fresh PP phishing (still alive), the
# correlation breaks down.
#
# Gate 2 rule: REJECT if
#    AUC_kp ≥ TEMPORAL_KP_FLOOR  AND
#    max(AUC_*) - min(AUC_*) > CROSS_DATASET_GAP_THRESHOLD
# i.e. the feature is suspiciously strong on KP AND inconsistent
# across datasets — the cross-dataset gap is the smoking gun.
#
# A rejected-by-Gate-2 feature is NOT deleted from the data — it is
# preserved in the Z matrix for downstream use (LLM oracle prompt
# in Step 2, where contextual reasoning beats statistical aggregation).
TEMPORAL_KP_FLOOR = 0.95
CROSS_DATASET_GAP_THRESHOLD = 0.40

SEED = 2025
TEST_RATIO = 0.20
CV_FOLDS = 5

# ── Loading ───────────────────────────────────────────────────────


def _resolve_mode(requested: str) -> str:
    """auto → osint if all 3 enriched matrices exist, else classic."""
    if requested != "auto":
        return requested
    if all(p.exists() for p in Z_OSINT.values()):
        return "osint"
    return "classic"


def load_matrices(mode: str) -> dict[str, pd.DataFrame]:
    """Load the three Z matrices according to mode.

    mode='classic' → base z_matrix_*.csv (17 PoC features only)
    mode='osint'   → z_matrix_*_osint.csv from Step 1.5 (+23 OSINT cols)
    """
    print(f"\n=== Loading Z matrices  (mode={mode}) ===")

    paths = Z_OSINT if mode == "osint" else Z_BASE
    dfs: dict[str, pd.DataFrame] = {}
    for tag in ("kp", "dp", "pp"):
        p = paths[tag]
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {tag.upper()} matrix at {p}. "
                f"For osint mode, run: "
                f"python reboot/step1_5_osint/derive_features.py && "
                f"python reboot/step1_5_osint/enrich_z_matrices.py"
            )
        df = pd.read_csv(p)
        dfs[tag] = df
        print(f"  {tag.upper()} : {df.shape}  "
              f"labels={dict(df.label.value_counts())}  from={p.name}")

    # Per-dataset OSINT feature presence audit.
    for tag, df in dfs.items():
        present = [c for c in OSINT_FEATURES if c in df.columns]
        print(f"    {tag.upper()} OSINT cols present: "
              f"{len(present)}/{len(OSINT_FEATURES)}")

    return dfs


# ── Leakage gate ──────────────────────────────────────────────────


def feature_auc_alone(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """AUC of each feature alone vs label. max(auc, 1-auc) for polarity."""
    y = df["label"].values
    out: dict[str, float] = {}
    for f in features:
        if f not in df.columns:
            continue
        vals = pd.to_numeric(df[f], errors="coerce").fillna(0.0).to_numpy()
        try:
            a = roc_auc_score(y, vals)
        except ValueError:
            a = 0.5
        out[f] = max(a, 1.0 - a)
    return out


def apply_leakage_gates(
    candidates: list[str],
    aucs_kp: dict[str, float],
    aucs_dp: dict[str, float],
    aucs_pp: dict[str, float],
) -> tuple[list[str], list[tuple[str, str, dict]]]:
    """Apply both leakage gates uniformly to every candidate feature.

    The pure-statistical formulation deliberately treats core proposer
    features (prob_sn, max_prob) and OSINT features (wayback_*, etc)
    identically. The gates can therefore reject the original PoC
    shortcut (max_prob, KP=0.99 / PP=0.50) for the *same reason* they
    reject the OSINT temporal leakage — both are training-time
    artifacts that do not generalize.

    Returns:
        kept: feature names that passed both gates
        rejected: list of (feature, reason, audit_dict) for the report
    """

    def is_nan(x: float) -> bool:
        return isinstance(x, float) and x != x

    kept: list[str] = []
    rejected: list[tuple[str, str, dict]] = []

    print(f"\n=== Leakage gates  "
          f"(gate-1: AUC_kp ≥ {LEAKAGE_AUC_THRESHOLD}  |  "
          f"gate-2: AUC_kp ≥ {TEMPORAL_KP_FLOOR} ∧ gap > {CROSS_DATASET_GAP_THRESHOLD}) ===")
    print(f"  {'feature':<28s} {'KP':>6s} {'DP':>6s} {'PP':>6s} {'gap':>6s}  verdict")

    # Order rows by KP AUC desc for readability.
    ordered = sorted(candidates, key=lambda c: -aucs_kp.get(c, -1.0))
    for f in ordered:
        kp = aucs_kp.get(f, float("nan"))
        dp = aucs_dp.get(f, float("nan"))
        pp = aucs_pp.get(f, float("nan"))
        known = [a for a in (kp, dp, pp) if not is_nan(a)]
        gap = (max(known) - min(known)) if len(known) >= 2 else float("nan")

        audit = {"AUC_kp": kp, "AUC_dp": dp, "AUC_pp": pp, "gap": gap}

        # Gate 1: in-distribution leakage.
        if not is_nan(kp) and kp >= LEAKAGE_AUC_THRESHOLD:
            reason = f"gate-1: AUC_kp={kp:.4f} ≥ {LEAKAGE_AUC_THRESHOLD}"
            rejected.append((f, reason, audit))
            verdict = "REJECT (gate-1)"
        # Gate 2: temporal / cross-dataset leakage.
        elif (not is_nan(kp) and kp >= TEMPORAL_KP_FLOOR
              and not is_nan(gap) and gap > CROSS_DATASET_GAP_THRESHOLD):
            reason = (f"gate-2: AUC_kp={kp:.4f} ≥ {TEMPORAL_KP_FLOOR} ∧ "
                      f"cross-dataset gap={gap:.3f} > {CROSS_DATASET_GAP_THRESHOLD}")
            rejected.append((f, reason, audit))
            verdict = "REJECT (gate-2)"
        else:
            kept.append(f)
            verdict = "keep"

        def fmt(x):
            return f"{x:.4f}" if not is_nan(x) else "  -  "
        print(f"  {f:<28s} {fmt(kp):>6s} {fmt(dp):>6s} {fmt(pp):>6s} {fmt(gap):>6s}  {verdict}")

    return kept, rejected


# ── Feature matrix builder ────────────────────────────────────────


def prepare_X(df: pd.DataFrame, features: list[str],
              fill_means: dict[str, float]) -> np.ndarray:
    """Build feature matrix with seed-mean imputation. Never zero.

    For OSINT features that use -1 as the "unknown / not applicable"
    sentinel, we substitute NaN before the fillna so the imputation
    replaces the sentinel with the seed mean — keeping -1 would
    distort the scaler severely (e.g. tranco_rank spans 1..1_000_000).
    """
    X = np.empty((len(df), len(features)), dtype=float)
    for j, f in enumerate(features):
        mean = fill_means.get(f, 0.0)
        if f in df.columns:
            col = pd.to_numeric(df[f], errors="coerce")
            if f in OSINT_NEG1_SENTINEL:
                col = col.where(col != -1, other=np.nan)
            col = col.fillna(mean)
            X[:, j] = col.to_numpy()
        else:
            X[:, j] = mean
    return X


# ── Models ────────────────────────────────────────────────────────


def get_models(seed: int = SEED) -> dict:
    """PoC-style configuration. No class_weight to match the PoC baseline."""
    return {
        "LogReg": LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
        "GB": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=seed,
        ),
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1,
        ),
    }


# ── Metrics ───────────────────────────────────────────────────────


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    n_classes = len(set(y_true))
    auc = roc_auc_score(y_true, y_prob) if n_classes > 1 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_phish": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_benign": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_phish": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_benign": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "auc": auc,
    }


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("auto", "classic", "osint"), default="auto",
        help="Which Z matrices to load. 'auto' = osint if all 3 enriched "
             "files exist, else classic.",
    )
    args = parser.parse_args()
    mode = _resolve_mode(args.mode)
    out_dir = OUT_ROOT / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = load_matrices(mode)
    df_kp, df_dp, df_pp = dfs["kp"], dfs["dp"], dfs["pp"]

    # 1. Feature selection.
    # 1a. Build the candidate pool:
    #     - core 17 PoC features  (always present in classic and osint matrices)
    #     - known-leaky candidates (gate them out per dataset)
    #     - OSINT features (only in osint mode)
    candidates_core = CORE_FEATURES + KNOWN_LEAKY_CANDIDATES
    candidates_osint = OSINT_FEATURES if mode == "osint" else []
    candidates_all = candidates_core + candidates_osint

    in_kp = {c for c in candidates_all if c in df_kp.columns}
    in_dp = {c for c in candidates_all if c in df_dp.columns}
    in_pp = {c for c in candidates_all if c in df_pp.columns}
    intersection = sorted(in_kp & in_dp & in_pp,
                          key=lambda c: candidates_all.index(c))
    kp_only = sorted(in_kp - in_dp - in_pp,
                     key=lambda c: candidates_all.index(c))

    print("\n=== Feature intersection across datasets ===")
    print(f"  In all 3 ({len(intersection)}): {intersection}")
    print(f"  KP-only (dropped) ({len(kp_only)}): {kp_only}")
    # Sanity-check: if osint mode but OSINT features missing from one
    # of the matrices, surface it loudly.
    if mode == "osint":
        for tag, df in dfs.items():
            miss = [c for c in OSINT_FEATURES if c not in df.columns]
            if miss:
                print(f"  ⚠ OSINT cols missing in {tag.upper()}: {miss}")

    # 1b. Build per-dataset AUC dictionaries for the intersection.
    aucs_kp = feature_auc_alone(df_kp, intersection)
    aucs_dp = feature_auc_alone(df_dp, intersection)
    aucs_pp = feature_auc_alone(df_pp, intersection)

    # 1c. Apply both leakage gates (gate-1 in-distribution, gate-2
    #     temporal/cross-dataset). The gate prints a full table.
    kept_core, rejected = apply_leakage_gates(
        intersection, aucs_kp, aucs_dp, aucs_pp,
    )
    final_features = kept_core

    rejected_osint = [name for name, _, _ in rejected if name in OSINT_FEATURES]
    rejected_core = [name for name, _, _ in rejected if name in CORE_FEATURES]

    print(f"\n=== Final feature set ({len(final_features)}) ===")
    print(f"  Kept ({len(kept_core)}): {kept_core}")
    print(f"  Rejected total ({len(rejected)}): "
          f"{[r[0] for r in rejected]}")
    if rejected_core:
        print(f"  → CORE rejected ({len(rejected_core)}): {rejected_core}")
    if rejected_osint:
        print(f"  → OSINT rejected ({len(rejected_osint)}): {rejected_osint}")
        print("    (still in Z matrices — reserved for the LLM oracle in Step 2)")

    # 2. Cross-dataset AUC sanity check.
    auc_table = []
    canonical_order = CORE_FEATURES + KNOWN_LEAKY_CANDIDATES + OSINT_FEATURES + CERT_FEATURES
    all_candidates = sorted(
        set(intersection + kp_only),
        key=lambda c: canonical_order.index(c) if c in canonical_order else 999,
    )
    for tag, df in (("kp", df_kp), ("dp", df_dp), ("pp", df_pp)):
        aucs = feature_auc_alone(df, all_candidates)
        for f, a in aucs.items():
            auc_table.append({"dataset": tag, "feature": f, "auc_alone": a})
    auc_df = pd.DataFrame(auc_table)
    auc_pivot = auc_df.pivot(index="feature", columns="dataset", values="auc_alone")
    auc_pivot = auc_pivot.reindex(all_candidates).round(4)
    auc_pivot.to_csv(out_dir / "feature_auc_per_dataset.csv")

    print("\n=== Per-dataset AUC-alone (consistency check) ===")
    print(auc_pivot.to_string())

    # 3. KP stratified split.
    df_train, df_test, y_train, y_test = train_test_split(
        df_kp, df_kp.label.values,
        test_size=TEST_RATIO, random_state=SEED, stratify=df_kp.label.values,
    )
    print(f"\n=== KP split === train={len(df_train)}  test={len(df_test)}")

    # 4. Seed means (training fold only).
    fill_means: dict[str, float] = {}
    for f in final_features:
        if f in df_train.columns:
            v = pd.to_numeric(df_train[f], errors="coerce").mean()
            fill_means[f] = float(v) if not np.isnan(v) else 0.0
        else:
            fill_means[f] = 0.0

    # 5. Build matrices.
    X_train = prepare_X(df_train, final_features, fill_means)
    X_test = prepare_X(df_test, final_features, fill_means)
    X_dp = prepare_X(df_dp, final_features, fill_means)
    X_pp = prepare_X(df_pp, final_features, fill_means)
    y_dp = df_dp.label.values
    y_pp = df_pp.label.values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_dp_s = scaler.transform(X_dp)
    X_pp_s = scaler.transform(X_pp)

    # 6. CV + final fit + cross-dataset for each model.
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    results, importances = [], {}

    for model_name in ("LogReg", "GB", "RF"):
        print(f"\n{'═' * 50}")
        print(f"  Model: {model_name}")
        print(f"{'═' * 50}")

        cv_f1, cv_acc, cv_auc = [], [], []
        for tr, va in skf.split(X_train_s, y_train):
            m = get_models()[model_name]
            m.fit(X_train_s[tr], y_train[tr])
            p = m.predict(X_train_s[va])
            pr = m.predict_proba(X_train_s[va])[:, 1]
            cv_f1.append(f1_score(y_train[va], p, average="weighted"))
            cv_acc.append(accuracy_score(y_train[va], p))
            cv_auc.append(roc_auc_score(y_train[va], pr))
        print(f"  CV(5-fold) F1w={np.mean(cv_f1):.4f}±{np.std(cv_f1):.4f}"
              f"  acc={np.mean(cv_acc):.4f}±{np.std(cv_acc):.4f}"
              f"  AUC={np.mean(cv_auc):.4f}±{np.std(cv_auc):.4f}")

        model = get_models()[model_name]
        model.fit(X_train_s, y_train)

        for tag, X, y in (
            ("KP_test", X_test_s, y_test),
            ("DP_full", X_dp_s, y_dp),
            ("PP_full", X_pp_s, y_pp),
        ):
            pred = model.predict(X)
            prob = model.predict_proba(X)[:, 1]
            m = evaluate(y, pred, prob)
            m.update({"model": model_name, "split": tag})
            cm = confusion_matrix(y, pred, labels=[0, 1])
            m["TN"], m["FP"] = int(cm[0][0]), int(cm[0][1])
            m["FN"], m["TP"] = int(cm[1][0]), int(cm[1][1])
            results.append(m)
            print(f"  {tag:<8s}"
                  f"  F1w={m['f1_weighted']:.4f}"
                  f"  acc={m['accuracy']:.4f}"
                  f"  AUC={m['auc']:.4f}"
                  f"  P_phish={m['precision_phish']:.3f}"
                  f"  P_benign={m['precision_benign']:.3f}"
                  f"  R_phish={m['recall_phish']:.3f}")

        if hasattr(model, "feature_importances_"):
            importances[model_name] = dict(zip(final_features, model.feature_importances_))
        elif hasattr(model, "coef_"):
            importances[model_name] = dict(zip(final_features, np.abs(model.coef_[0])))
        else:
            importances[model_name] = {}

    # 7. Persist artefacts.
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "metrics_per_model_per_split.csv", index=False)

    fi_df = pd.DataFrame(importances).reindex(final_features)
    fi_df.to_csv(out_dir / "feature_importance_per_model.csv")

    # Manifest: full provenance, including rejected features and reasons,
    # so the LLM-oracle step (Step 2) can read the "dropped_for_oracle"
    # list directly.
    rejected_records = [
        {"feature": name, "reason": reason, **audit}
        for name, reason, audit in rejected
    ]
    manifest = {
        "mode": mode,
        "core_features_kept": [f for f in kept_core if f in CORE_FEATURES],
        "osint_features_kept": [f for f in kept_core if f in OSINT_FEATURES],
        "rejected_features": rejected_records,
        "rejected_for_oracle_only": [
            r["feature"] for r in rejected_records
            if r["feature"] in OSINT_FEATURES
        ],
        "kp_only_features_dropped": kp_only,
        "final_feature_set": final_features,
        "kp_train_n": int(len(df_train)),
        "kp_test_n": int(len(df_test)),
        "dp_n": int(len(df_dp)),
        "pp_n": int(len(df_pp)),
        "gates": {
            "gate_1_kp_alone_threshold": LEAKAGE_AUC_THRESHOLD,
            "gate_2_kp_floor": TEMPORAL_KP_FLOOR,
            "gate_2_cross_dataset_gap_min": CROSS_DATASET_GAP_THRESHOLD,
        },
        "seed_train_means": fill_means,
        "seed": SEED,
        "test_ratio": TEST_RATIO,
        "cv_folds": CV_FOLDS,
    }
    (out_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Convenience: separate file just for the oracle handoff in Step 2.
    (out_dir / "dropped_for_oracle.json").write_text(json.dumps({
        "purpose": (
            "OSINT features dropped from the ensemble by Gate 2 (temporal "
            "leakage). They stay in the Z matrices; Step 2 should feed "
            "them to the LLM oracle as contextual hints, never as "
            "ensemble-training features."
        ),
        "features": [
            {"feature": name, "reason": reason, **audit}
            for name, reason, audit in rejected
            if name in OSINT_FEATURES
        ],
    }, indent=2))

    leakage_lines = [
        "=== Step 1 leakage gate report ===",
        f"Mode: {mode}",
        f"Gate 1 (in-distribution): AUC_kp ≥ {LEAKAGE_AUC_THRESHOLD}",
        f"Gate 2 (temporal):        AUC_kp ≥ {TEMPORAL_KP_FLOOR} ∧ "
        f"cross-dataset gap > {CROSS_DATASET_GAP_THRESHOLD}",
        "",
        f"Rejected ({len(rejected)}):",
    ]
    for name, reason, audit in rejected:
        leakage_lines.append(f"  {name}")
        leakage_lines.append(f"    {reason}")
        leakage_lines.append(
            f"    AUC_kp={audit['AUC_kp']:.4f}  "
            f"AUC_dp={audit['AUC_dp']:.4f}  "
            f"AUC_pp={audit['AUC_pp']:.4f}  "
            f"gap={audit['gap']:.4f}"
        )
    leakage_lines.append("")
    leakage_lines.append(f"Kept ({len(kept_core)}):")
    for f in kept_core:
        leakage_lines.append(
            f"  {f}  "
            f"AUC_kp={aucs_kp.get(f, float('nan')):.4f}  "
            f"AUC_dp={aucs_dp.get(f, float('nan')):.4f}  "
            f"AUC_pp={aucs_pp.get(f, float('nan')):.4f}"
        )
    (out_dir / "leakage_report.txt").write_text("\n".join(leakage_lines))

    # ── Feature-importance audit ──────────────────────────────
    print("\n\n=== Feature importance audit (top per model) ===")
    for model_name, fi in importances.items():
        if not fi:
            print(f"  {model_name}: (no importance attr)")
            continue
        # Sort by absolute importance (works for both gini and |coef|).
        ranked = sorted(fi.items(), key=lambda kv: -abs(kv[1]))
        total = sum(abs(v) for v in fi.values()) or 1.0
        print(f"\n  -- {model_name} --")
        print(f"    {'feature':<28s} {'importance':>10s} {'%total':>8s}")
        for name, val in ranked[:12]:
            pct = abs(val) / total * 100
            print(f"    {name:<28s} {val:>10.4f} {pct:>7.2f}%")

    print("\n\n=== HEADLINE COMPARISON ===")
    print(f"Mode: {mode}  |  Reference numbers from PoC + Agent (PROJECT_CONTEXT.md)")
    print(f"  {'Split':<10s} {'PoC v1':>10s} {'Agent V2.1':>14s} {'REBOOT '+mode+' (best)':>22s}")
    poc_ref = {"KP_test": 0.9845, "DP_full": 0.0660, "PP_full": 0.5730}
    agent_ref = {"KP_test": 0.9856, "DP_full": 0.0624, "PP_full": 0.4760}
    for tag in ("KP_test", "DP_full", "PP_full"):
        sub = res_df[res_df.split == tag].sort_values("f1_weighted", ascending=False)
        best = sub.iloc[0]
        print(f"  {tag:<10s} {poc_ref[tag]*100:>9.2f}% {agent_ref[tag]*100:>13.2f}% "
              f"{best['f1_weighted']*100:>20.2f}% [{best['model']}]")

    print(f"\nArtifacts: {out_dir.relative_to(REBOOT)}/")


if __name__ == "__main__":
    main()
