"""analyze.py — Compute Precision/Recall/F1 per oracle variant.

Reads `results.csv` + `test_set.csv`, joins on `sample_id`, and
produces the ablation table the paper will quote.

Outputs
───────
  reboot/runs/oracle_ablation/metrics.csv
    one row per (variant, slice). Slice in
      {overall, FN_recovery, FP_recovery, phish, benign}.
  reboot/runs/oracle_ablation/REPORT.md
    human-readable summary with the headline table and per-variant
    discussion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
LAB_DIR = REBOOT / "runs" / "oracle_ablation"
TEST_SET = LAB_DIR / "test_set.csv"
RESULTS = LAB_DIR / "results.csv"
METRICS = LAB_DIR / "metrics.csv"
REPORT = LAB_DIR / "REPORT.md"


def _llm_to_int(lbl) -> int | None:
    """Coerce variable LLM label forms into 0/1, return None for uncertain."""
    if lbl is None:
        return None
    s = str(lbl).strip().lower()
    if s in ("phish", "phishing", "1", "true"):
        return 1
    if s in ("benign", "legitimate", "0", "false"):
        return 0
    return None  # uncertain / unparseable


def per_slice_metrics(y_true, y_pred, *, label_pos: int = 1) -> dict:
    """Skips rows where y_pred is None (uncertain). Reports n_used."""
    pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp is not None]
    n_total = len(y_true)
    n_used = len(pairs)
    if not pairs:
        return {"n_total": n_total, "n_used": 0, "n_uncertain": n_total,
                "accuracy": float("nan"),
                "precision_phish": float("nan"),
                "recall_phish": float("nan"),
                "precision_benign": float("nan"),
                "recall_benign": float("nan"),
                "f1_weighted": float("nan"),
                "f1_phish": float("nan")}
    yt = [p[0] for p in pairs]
    yp = [p[1] for p in pairs]
    return {
        "n_total": n_total,
        "n_used": n_used,
        "n_uncertain": n_total - n_used,
        "accuracy": accuracy_score(yt, yp),
        "precision_phish": precision_score(yt, yp, pos_label=1, zero_division=0),
        "recall_phish": recall_score(yt, yp, pos_label=1, zero_division=0),
        "precision_benign": precision_score(yt, yp, pos_label=0, zero_division=0),
        "recall_benign": recall_score(yt, yp, pos_label=0, zero_division=0),
        "f1_weighted": f1_score(yt, yp, average="weighted"),
        "f1_phish": f1_score(yt, yp, pos_label=1, zero_division=0),
    }


def main() -> None:
    if not RESULTS.exists() or not TEST_SET.exists():
        raise FileNotFoundError(
            "Missing results.csv or test_set.csv — run sample_test_set.py "
            "and run_ablation.py first."
        )
    df_t = pd.read_csv(TEST_SET)
    df_r = pd.read_csv(RESULTS)
    df = df_r.merge(
        df_t[["sample_id", "label", "ensemble_pred", "error_type"]],
        on="sample_id", how="left",
    )
    df["label_llm_int"] = df["label_llm"].map(_llm_to_int)

    rows = []
    for variant, sub in df.groupby("variant", sort=False):
        # Overall
        m = per_slice_metrics(sub["label"].values,
                              sub["label_llm_int"].tolist())
        m.update({"variant": variant, "slice": "overall"})
        rows.append(m)
        # Phish-only slice (50 hard-phish samples)
        sub_p = sub[sub["label"] == 1]
        m = per_slice_metrics(sub_p["label"].values,
                              sub_p["label_llm_int"].tolist())
        m.update({"variant": variant, "slice": "phish_subset"})
        rows.append(m)
        # Benign-only slice
        sub_b = sub[sub["label"] == 0]
        m = per_slice_metrics(sub_b["label"].values,
                              sub_b["label_llm_int"].tolist())
        m.update({"variant": variant, "slice": "benign_subset"})
        rows.append(m)
        # FN recovery: how many "true-phish-the-ensemble-missed" did
        # the LLM correctly relabel as phish?
        sub_fn = sub[sub["error_type"] == "FN"]
        if len(sub_fn):
            n_recovered = int((sub_fn["label_llm_int"] == 1).sum())
            m = {"variant": variant, "slice": "FN_recovery",
                 "n_total": len(sub_fn), "n_used": int(sub_fn["label_llm_int"].notna().sum()),
                 "n_uncertain": int(sub_fn["label_llm_int"].isna().sum()),
                 "recall_phish": n_recovered / len(sub_fn)}
            rows.append(m)
        # FP recovery: same for benign-the-ensemble-falsely-flagged.
        sub_fp = sub[sub["error_type"] == "FP"]
        if len(sub_fp):
            n_recovered = int((sub_fp["label_llm_int"] == 0).sum())
            m = {"variant": variant, "slice": "FP_recovery",
                 "n_total": len(sub_fp), "n_used": int(sub_fp["label_llm_int"].notna().sum()),
                 "n_uncertain": int(sub_fp["label_llm_int"].isna().sum()),
                 "recall_benign": n_recovered / len(sub_fp)}
            rows.append(m)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS, index=False)

    # Build a readable Markdown report.
    lines: list[str] = [
        "# Step 2 — Oracle Ablation Results",
        "",
        f"Test set: {len(df_t)} samples "
        f"({(df_t.label == 1).sum()} phish, "
        f"{(df_t.label == 0).sum()} benign)",
        f"  - FN seeds: {(df_t.error_type == 'FN').sum()}",
        f"  - FP seeds: {(df_t.error_type == 'FP').sum()}",
        f"  - Uncertain phish fillers: "
        f"{(df_t.error_type == 'uncertain_phish').sum()}",
        f"  - Uncertain benign fillers: "
        f"{(df_t.error_type == 'uncertain_benign').sum()}",
        "",
        "## Headline metrics (overall, n=test-set size)",
        "",
        "| Variant | Acc | F1w | P_phish | R_phish | P_benign | R_benign | uncertain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in df["variant"].unique():
        ov = metrics[(metrics.variant == variant) & (metrics.slice == "overall")]
        if not len(ov):
            continue
        ov = ov.iloc[0]
        lines.append(
            f"| {variant} | {ov.accuracy:.3f} | {ov.f1_weighted:.3f} | "
            f"{ov.precision_phish:.3f} | {ov.recall_phish:.3f} | "
            f"{ov.precision_benign:.3f} | {ov.recall_benign:.3f} | "
            f"{int(ov.n_uncertain)} |"
        )
    lines.extend([
        "",
        "## FN recovery — how many of the ensemble's missed phishings"
        " the oracle catches",
        "",
        "| Variant | FN recovery rate | uncertain |",
        "|---|---:|---:|",
    ])
    for variant in df["variant"].unique():
        fn = metrics[(metrics.variant == variant) & (metrics.slice == "FN_recovery")]
        if not len(fn):
            continue
        fn = fn.iloc[0]
        rate = fn.get("recall_phish", float("nan"))
        lines.append(
            f"| {variant} | {rate:.3f} "
            f"({int(rate * fn.n_total)} / {int(fn.n_total)}) | "
            f"{int(fn.n_uncertain)} |"
        )
    lines.extend([
        "",
        "## FP recovery — how many of the ensemble's false alarms"
        " the oracle correctly downgrades to benign",
        "",
        "| Variant | FP recovery rate | uncertain |",
        "|---|---:|---:|",
    ])
    for variant in df["variant"].unique():
        fp = metrics[(metrics.variant == variant) & (metrics.slice == "FP_recovery")]
        if not len(fp):
            continue
        fp = fp.iloc[0]
        rate = fp.get("recall_benign", float("nan"))
        lines.append(
            f"| {variant} | {rate:.3f} "
            f"({int(rate * fp.n_total)} / {int(fp.n_total)}) | "
            f"{int(fp.n_uncertain)} |"
        )

    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nArtifacts: {METRICS.relative_to(REBOOT)}, {REPORT.relative_to(REBOOT)}")


if __name__ == "__main__":
    main()
