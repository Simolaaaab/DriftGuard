"""halluc_correlation.py — Cross-model hallucination ↔ correctness analysis.

Consumes the per-sample CSVs produced by verifier_audit.py --multi and emits:

  1. halluc_per_model.csv
       provider, n, accuracy, mean_halluc, mean_trust,
       pearson_trust_correct, spearman_trust_correct,
       point_biserial_halluc_correct.

  2. contingency_per_model.csv
       Per provider, the 2×2 (hallucinated_yes/no × correct_yes/no)
       with Fisher exact test (odds ratio + p-value).
       Threshold for "hallucinated": >=1 contradicted indicator
       (i.e. the LLM made a positive claim that the verifier
       could refute on HTML/URL/OSINT).

  3. trust_quartile_acc.csv
       Per provider × trust quartile (Q1..Q4), accuracy + count.
       Documents the "very_high trust paradox" we found in the
       cross-model bucket table.

  4. halluc_correlation.md
       Paper-ready table + interpretation.

  5. fig_halluc_provider.{png,pdf}
       Single-panel: hallucination rate vs accuracy per provider,
       size = n, color = trust↔correct Pearson. The "is hallucinating
       bad for the model?" plot.

Usage:
    python3 reboot/step3_al/halluc_correlation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact, pearsonr, pointbiserialr, spearmanr  # noqa: E402

REBOOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REBOOT / "runs" / "al" / "_verifier_audit"
OUT_DIR = REBOOT / "runs" / "al" / "_halluc"

PROVIDER_ORDER = [
    "DeepSeek-V4-Flash", "GPT-5.4", "GPT-5.4-mini", "GPT-5.1",
    "Llama-3.3-70B", "Mistral-Large-3",
]
PROVIDER_COLOR = {
    "DeepSeek-V4-Flash": "#1f77b4",
    "GPT-5.4":           "#ff7f0e",
    "GPT-5.4-mini":      "#2ca02c",
    "GPT-5.1":           "#9467bd",
    "Llama-3.3-70B":     "#d62728",
    "Mistral-Large-3":   "#8c564b",
}


def _slug(name: str) -> str:
    return name.lower().replace(".", "").replace("-", "_")


def _load_per_sample() -> dict[str, pd.DataFrame]:
    """Load every per_sample_<provider>.csv emitted by verifier_audit.py."""
    out: dict[str, pd.DataFrame] = {}
    for name in PROVIDER_ORDER:
        p = AUDIT_DIR / f"per_sample_{_slug(name)}.csv"
        if not p.exists():
            print(f"  [skip] {name}: {p} not found")
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        # Sanity columns we rely on.
        for col in ("correct", "trust", "n_contradicted",
                    "n_indicators_total", "label_llm", "gt_label"):
            if col not in df.columns:
                print(f"  [skip] {name}: missing column {col}")
                df = pd.DataFrame()
                break
        if df.empty:
            continue
        out[name] = df
    return out


def per_model_metrics(per_sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute correlation metrics per provider."""
    rows: list[dict] = []
    for name, df in per_sample.items():
        n = len(df)
        acc = float(df["correct"].mean())
        # Hallucination = at least one contradicted positive claim.
        halluc = (df["n_contradicted"] >= 1).astype(int).to_numpy()
        correct = df["correct"].astype(int).to_numpy()
        trust = df["trust"].astype(float).to_numpy()

        # Continuous metrics. Pearson on trust × correctness.
        pr, pr_p = pearsonr(trust, correct) if len(set(trust)) > 1 else (np.nan, np.nan)
        sp, sp_p = spearmanr(trust, correct) if len(set(trust)) > 1 else (np.nan, np.nan)
        # Point-biserial: binary halluc × continuous? We want binary×binary
        # actually, but pointbiserial is the natural form when one is binary.
        # Here halluc is binary and correct is binary; equivalent to phi
        # coefficient via Pearson on binary arrays.
        pb, pb_p = pearsonr(halluc, correct) if len(set(halluc)) > 1 else (np.nan, np.nan)

        # Fraction of samples flagged as hallucinated.
        halluc_frac = float(halluc.mean())
        # Fraction of indicators that are contradicted (the granular metric).
        mean_halluc_rate = float(df["n_contradicted"].sum() /
                                 max(int(df["n_indicators_total"].sum()), 1))
        mean_trust = float(df["trust"].mean())

        rows.append(dict(
            provider=name, n=n,
            accuracy=acc,
            mean_trust=mean_trust,
            halluc_sample_frac=halluc_frac,
            halluc_indicator_rate=mean_halluc_rate,
            pearson_trust_correct=float(pr),
            pearson_trust_correct_p=float(pr_p),
            spearman_trust_correct=float(sp),
            spearman_trust_correct_p=float(sp_p),
            phi_halluc_correct=float(pb),
            phi_halluc_correct_p=float(pb_p),
        ))
    return pd.DataFrame(rows)


def contingency_per_model(per_sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """2×2 (hallucinated yes/no × correct yes/no) + Fisher exact per provider.

    Hallucinated = n_contradicted >= 1 (at least one claim refuted).
    """
    rows: list[dict] = []
    for name, df in per_sample.items():
        halluc = (df["n_contradicted"] >= 1).astype(int)
        correct = df["correct"].astype(int)
        a = int(((halluc == 1) & (correct == 1)).sum())   # hallucinated + correct
        b = int(((halluc == 1) & (correct == 0)).sum())   # hallucinated + wrong
        c = int(((halluc == 0) & (correct == 1)).sum())   # clean + correct
        d = int(((halluc == 0) & (correct == 0)).sum())   # clean + wrong

        # Fisher exact: H1 = hallucination ⇒ wrong (OR < 1 favours wrong).
        # Table: [[clean+correct, clean+wrong], [halluc+correct, halluc+wrong]]
        # OR > 1 means halluc samples are MORE likely to be wrong.
        try:
            odds_ratio, p_two = fisher_exact([[a, b], [c, d]],
                                             alternative="two-sided")
        except Exception:
            odds_ratio, p_two = (np.nan, np.nan)

        # Conditional accuracies.
        acc_when_halluc = a / max(a + b, 1)
        acc_when_clean = c / max(c + d, 1)
        acc_drop = acc_when_clean - acc_when_halluc

        rows.append(dict(
            provider=name,
            n_halluc_correct=a, n_halluc_wrong=b,
            n_clean_correct=c, n_clean_wrong=d,
            acc_when_halluc=acc_when_halluc,
            acc_when_clean=acc_when_clean,
            acc_drop_due_to_halluc=acc_drop,
            fisher_odds_ratio=float(odds_ratio),
            fisher_p_two_sided=float(p_two),
        ))
    return pd.DataFrame(rows)


def trust_quartile_acc(per_sample: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per provider × trust quartile, the accuracy.

    Documents the universal "very_high trust paradox" we found:
    in 5/6 providers, accuracy DROPS in the top-trust bucket.
    """
    rows: list[dict] = []
    qs = [0.0, 0.25, 0.50, 0.75, 1.001]
    qnames = ["Q1", "Q2", "Q3", "Q4"]
    for name, df in per_sample.items():
        # Per-provider quartile thresholds (sample-relative).
        try:
            cuts = df["trust"].quantile([0.25, 0.50, 0.75]).values
        except Exception:
            continue
        edges = [0.0] + list(cuts) + [1.001]
        # Dedupe edges (in case the trust distribution is degenerate).
        edges = sorted(set(edges))
        if len(edges) < 5:
            # Fall back to fixed bins.
            edges = qs[:]
        labels = qnames[: len(edges) - 1]
        df = df.copy()
        df["bucket"] = pd.cut(df["trust"], bins=edges, labels=labels,
                              include_lowest=True)
        for lab in labels:
            sub = df[df["bucket"] == lab]
            if len(sub) == 0:
                continue
            rows.append(dict(
                provider=name, bucket=str(lab),
                bucket_lo=float(edges[labels.index(lab)]),
                bucket_hi=float(edges[labels.index(lab) + 1]),
                n=int(len(sub)),
                accuracy=float(sub["correct"].mean()),
                mean_trust=float(sub["trust"].mean()),
            ))
    return pd.DataFrame(rows)


def plot_scatter(metrics: pd.DataFrame, out_path: Path) -> None:
    """Hallucination rate vs accuracy, one point per provider.
    Size = n, color = trust-correctness Pearson."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = metrics["halluc_indicator_rate"] * 100
    y = metrics["accuracy"]
    s = metrics["n"] * 0.6 + 30
    c = metrics["pearson_trust_correct"]
    sc = ax.scatter(x, y, s=s, c=c, cmap="viridis",
                    vmin=0.0, vmax=0.5,
                    edgecolors="black", linewidths=0.7, zorder=3)
    for _, r in metrics.iterrows():
        ax.annotate(r["provider"],
                    (r["halluc_indicator_rate"] * 100, r["accuracy"]),
                    xytext=(7, 4), textcoords="offset points",
                    fontsize=8)
    ax.set_xlabel("Hallucination rate (% of LLM-claimed indicators refuted by verifier)")
    ax.set_ylabel("Accuracy on AL boundary samples")
    ax.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Trust ↔ correctness (Pearson)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_path.with_suffix("." + ext)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal pure-Python markdown table writer (avoids the tabulate dep).
    Numeric columns get 4-decimal formatting; ints stay ints."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.4f}" if not pd.isna(v) else "—")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_markdown_report(metrics: pd.DataFrame,
                          conting: pd.DataFrame,
                          quartiles: pd.DataFrame,
                          out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Cross-LLM hallucination ↔ correctness analysis\n")
    lines.append("All providers: drift_anchored + Prompt B + K=20 + "
                 "periodic_audit + temporal PP_stream, seed 2025.\n")

    lines.append("## Per-model headline\n")
    cols = ["provider", "n", "accuracy", "mean_trust",
            "halluc_indicator_rate",
            "pearson_trust_correct", "phi_halluc_correct"]
    lines.append(_df_to_md(metrics[cols]))
    lines.append("\n")

    lines.append("## 2×2 contingency  (hallucinated × correct)\n")
    lines.append("Hallucinated = ≥1 indicator refuted by the verifier "
                 "(`n_contradicted ≥ 1`). Higher Fisher odds ratio means "
                 "hallucinated samples are MORE likely to be wrong.\n")
    ccols = ["provider", "n_halluc_correct", "n_halluc_wrong",
             "n_clean_correct", "n_clean_wrong",
             "acc_when_halluc", "acc_when_clean",
             "acc_drop_due_to_halluc",
             "fisher_odds_ratio", "fisher_p_two_sided"]
    lines.append(_df_to_md(conting[ccols]))
    lines.append("\n")

    lines.append("## Trust-quartile accuracy table\n")
    lines.append("Per provider × trust quartile (sample-relative). The "
                 "headline finding is the *very-high trust paradox*: "
                 "in most providers Q4 accuracy is LOWER than Q3, because "
                 "the verifier confirms evidence-grounding but cannot "
                 "discriminate sample-level ambiguity.\n")
    lines.append(_df_to_md(quartiles))
    lines.append("\n")

    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", type=Path, default=AUDIT_DIR,
                   help="dir produced by verifier_audit.py --multi")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading per-sample CSVs from {args.audit_dir}")
    per_sample = _load_per_sample()
    if not per_sample:
        raise SystemExit("no per-sample CSVs found; run verifier_audit.py "
                         "--multi first.")
    print(f"providers found: {list(per_sample)}")

    metrics = per_model_metrics(per_sample)
    metrics.to_csv(args.out_dir / "halluc_per_model.csv", index=False)

    conting = contingency_per_model(per_sample)
    conting.to_csv(args.out_dir / "contingency_per_model.csv", index=False)

    quartiles = trust_quartile_acc(per_sample)
    quartiles.to_csv(args.out_dir / "trust_quartile_acc.csv", index=False)

    write_markdown_report(
        metrics, conting, quartiles,
        out_md=args.out_dir / "halluc_correlation.md",
    )
    plot_scatter(metrics, out_path=args.out_dir / "fig_halluc_provider.png")

    # Console headline.
    print()
    print("== Provider-level metrics ==")
    print(metrics[["provider", "n", "accuracy", "mean_trust",
                   "halluc_indicator_rate",
                   "pearson_trust_correct", "phi_halluc_correct"]]
          .round(4).to_string(index=False))
    print()
    print("== Fisher 2×2 ==")
    print(conting[["provider", "acc_when_halluc", "acc_when_clean",
                   "acc_drop_due_to_halluc",
                   "fisher_odds_ratio", "fisher_p_two_sided"]]
          .round(4).to_string(index=False))
    print()
    print(f"all artefacts in {args.out_dir}/")


if __name__ == "__main__":
    main()
