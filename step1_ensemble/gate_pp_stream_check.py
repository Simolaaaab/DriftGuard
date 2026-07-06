"""gate_ppstream_check.py — Feature-selection leak sanity check.

Reviewer concern
────────────────
Step-1's leakage gate (ensemble_reboot.py) computes the single-feature PP AUC
(`aucs_pp`) on the FULL PhreshPhish matrix (df_pp). Downstream, PP is split
80/20 (seed=2025, stratified) into PP_stream (train/adapt) and PP_test (the
held-out drift-recovery metric). Because the gate sees the full PP matrix, the
20% PP_test rows participate in the feature-admission decision — a weak
feature-selection leak (not a training leak: the model is fit only on KP_train).

This script proves whether it matters. It recomputes the gate's PP AUC using
PP_stream ONLY (the exact same 80% rows used in the real AL runs), re-runs the
two gates with identical thresholds, and diffs the kept/rejected sets against
the original full-PP run.

It reuses ensemble_reboot.feature_auc_alone / apply_leakage_gates verbatim
(imported, not reimplemented) and the exact PP split from
step3_al/common.build_splits — so the comparison is apples-to-apples.

    python3 reboot/step1_ensemble/gate_ppstream_check.py

No model is trained. Runs in seconds. Read-only w.r.t. everything on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

STEP1 = Path(__file__).resolve().parent
REBOOT = STEP1.parent

# Make the two source modules importable.
sys.path.insert(0, str(STEP1))
sys.path.insert(0, str(REBOOT / "step3_al"))

# ── Reuse the EXACT gate logic from Step 1 (do not reimplement) ──────
from ensemble_reboot import (  # noqa: E402
    CORE_FEATURES,
    KNOWN_LEAKY_CANDIDATES,
    OSINT_FEATURES,
    apply_leakage_gates,
    feature_auc_alone,
    load_matrices,
)

# ── Reuse the EXACT PP 80/20 split from the real AL runs ─────────────
# common.build_splits does:
#   train_test_split(df_pp, test_size=0.20, random_state=2025,
#                    stratify=df_pp["label"].values)
# → first output (80%) is PP_stream, second (20%) is PP_test.
from common import SEED, build_splits  # noqa: E402


def _verdict_of(feature: str, kept: list[str],
                rejected: list[tuple[str, str, dict]]) -> str:
    """Collapse the gate output into a coarse verdict for diffing."""
    if feature in kept:
        return "keep"
    for name, reason, _audit in rejected:
        if name == feature:
            if reason.startswith("gate-1"):
                return "REJECT(gate-1)"
            if reason.startswith("gate-2"):
                return "REJECT(gate-2)"
            return f"REJECT({reason[:12]})"
    return "??"


def main() -> None:
    # 1. Load the SAME three OSINT matrices the gate uses.
    dfs = load_matrices("osint")
    df_kp, df_dp, df_pp = dfs["kp"], dfs["dp"], dfs["pp"]

    # 2. Reproduce Step-1's candidate pool + intersection EXACTLY
    #    (ensemble_reboot.main lines 392-400).
    candidates_all = CORE_FEATURES + KNOWN_LEAKY_CANDIDATES + OSINT_FEATURES
    in_kp = {c for c in candidates_all if c in df_kp.columns}
    in_dp = {c for c in candidates_all if c in df_dp.columns}
    in_pp = {c for c in candidates_all if c in df_pp.columns}
    intersection = sorted(in_kp & in_dp & in_pp,
                          key=lambda c: candidates_all.index(c))

    # 3. Reproduce the exact PP_stream (80%) via the real split path.
    #    Ordering (temporal/shuffle) is irrelevant to a per-feature AUC —
    #    it's the same SET of rows — so we take the default 'temporal'.
    splits = build_splits(dfs, stream_order="temporal", seed=SEED)
    pp_stream = splits["pp_stream"]
    pp_test = splits["pp_test"]

    print("=" * 74)
    print("Feature-selection leak check — gate PP AUC: full PP vs PP_stream")
    print("=" * 74)
    print(f"  PP full   : n={len(df_pp):5d}  "
          f"labels={dict(df_pp.label.value_counts())}")
    print(f"  PP_stream : n={len(pp_stream):5d}  "
          f"labels={dict(pp_stream.label.value_counts())}  (80%, seed={SEED})")
    print(f"  PP_test   : n={len(pp_test):5d}  "
          f"labels={dict(pp_test.label.value_counts())}  (20%, excluded)")
    print(f"  Intersection candidates gated: {len(intersection)}")

    # 4. KP and DP AUCs are IDENTICAL in both runs (unchanged, per spec).
    aucs_kp = feature_auc_alone(df_kp, intersection)
    aucs_dp = feature_auc_alone(df_dp, intersection)

    # 5a. PP AUC on the FULL matrix (the original gate).
    aucs_pp_full = feature_auc_alone(df_pp, intersection)
    # 5b. PP AUC on PP_stream ONLY (the leak-free recompute).
    aucs_pp_stream = feature_auc_alone(pp_stream, intersection)

    # 6. Run the SAME two gates twice, same thresholds, only aucs_pp differs.
    print("\n" + "#" * 74)
    print("#  ORIGINAL RUN — gate sees FULL PP")
    print("#" * 74)
    kept_full, rejected_full = apply_leakage_gates(
        intersection, aucs_kp, aucs_dp, aucs_pp_full,
    )

    print("\n" + "#" * 74)
    print("#  RECOMPUTE — gate sees PP_stream ONLY (PP_test rows excluded)")
    print("#" * 74)
    kept_stream, rejected_stream = apply_leakage_gates(
        intersection, aucs_kp, aucs_dp, aucs_pp_stream,
    )

    # 7. Diff.
    set_kept_full, set_kept_stream = set(kept_full), set(kept_stream)
    set_rej_full = {n for n, _, _ in rejected_full}
    set_rej_stream = {n for n, _, _ in rejected_stream}

    kept_identical = set_kept_full == set_kept_stream
    rej_identical = set_rej_full == set_rej_stream

    reasons_full = {_verdict_of(f, kept_full, rejected_full)
                    for f in intersection}  # noqa: F841 (kept for clarity)
    verdict_full = {f: _verdict_of(f, kept_full, rejected_full)
                    for f in intersection}
    verdict_stream = {f: _verdict_of(f, kept_stream, rejected_stream)
                      for f in intersection}
    reasons_identical = all(
        verdict_full[f] == verdict_stream[f] for f in intersection
    )
    flips = [f for f in intersection if verdict_full[f] != verdict_stream[f]]

    # 8. Per-feature table (ordered by KP AUC desc, as the gate prints).
    ordered = sorted(intersection, key=lambda c: -aucs_kp.get(c, -1.0))
    print("\n" + "=" * 90)
    print("PER-FEATURE COMPARISON")
    print("=" * 90)
    print(f"  {'feature':<26s} {'AUC_pp(full)':>12s} {'AUC_pp(stream)':>14s} "
          f"{'Δ':>8s}  {'verdict(full)':>15s} -> {'verdict(stream)':<15s}")
    print("  " + "-" * 86)
    for f in ordered:
        pf = aucs_pp_full.get(f, float("nan"))
        ps = aucs_pp_stream.get(f, float("nan"))
        d = ps - pf
        flag = "  <== FLIP" if verdict_full[f] != verdict_stream[f] else ""
        print(f"  {f:<26s} {pf:>12.4f} {ps:>14.4f} {d:>+8.4f}  "
              f"{verdict_full[f]:>15s} -> {verdict_stream[f]:<15s}{flag}")

    # 9. Verdict summary.
    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  Kept feature set IDENTICAL?      "
          f"{'YES' if kept_identical else 'NO'}   "
          f"(full={len(set_kept_full)}, stream={len(set_kept_stream)})")
    print(f"  Rejected feature set IDENTICAL?  "
          f"{'YES' if rej_identical else 'NO'}   "
          f"(full={len(set_rej_full)}, stream={len(set_rej_stream)})")
    print(f"  Gate-1/Gate-2 verdicts unchanged? "
          f"{'YES' if reasons_identical else 'NO'}")

    if flips:
        print(f"\n  ⚠ {len(flips)} feature(s) FLIPPED verdict:")
        for f in flips:
            print(f"      {f}: {verdict_full[f]} -> {verdict_stream[f]}  "
                  f"(AUC_pp {aucs_pp_full[f]:.4f} -> {aucs_pp_stream[f]:.4f})")
    else:
        print("\n  ✓ No feature flipped verdict. The feature-admission decision "
              "is\n    invariant to whether PP_test rows are included. The "
              "feature-selection\n    leak has zero effect on the kept/rejected "
              "sets.")

    # Also surface set-level diffs explicitly if any.
    if not kept_identical:
        print(f"\n  kept only in FULL   : {sorted(set_kept_full - set_kept_stream)}")
        print(f"  kept only in STREAM : {sorted(set_kept_stream - set_kept_full)}")

    print()


if __name__ == "__main__":
    main()
