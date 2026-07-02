"""plot_safety_figures.py — figures for the Safety & Oracle sections.

Generates three column-friendly PDFs into runs/al/_aggregate/:
  fig_verifier_quality.pdf   — oracle label quality ALL vs KEPT vs DROPPED
  fig_llama_halt.pdf         — health-monitor timeline of the Llama halt
  fig_injection_live.pdf     — live prompt-injection telemetry (naive/smart)

Style matches plot_figure1_sawtooth.py: no title, legend small, grid 0.25,
save pdf+png at dpi 160.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "runs" / "al"
OUT = ROOT / "_aggregate"


def _save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT / f"{stem}.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


# ───────────────────────── verifier figures (controlled ablation) ────────
# Fixed-trajectory controlled ablation (all modes replay the OFF run's exact
# 1247-label sequence; only the per-label weight differs). Validated: the
# OFF replay reproduces the on-disk v4_off run exactly.
_DOWNSTREAM = {  # mode: (PP F1w, DP macro-F1) — controlled
    "off": (0.8005, 0.6268), "t1": (0.7970, 0.6184), "t2": (0.7820, 0.6005),
    "t3": (0.7811, 0.5380), "full": (0.7988, 0.5790),
    "hard_filter": (0.8038, 0.5238),
}
# map figure tier-column name -> controlled per_label_trust mode
_TIER_TO_MODE = {"t1": "t1_only", "t2": "t2_only", "t3": "t3_only",
                 "trust": "full"}


def _load_canon(mode: str = "full"):
    """Per-label rows for one verifier mode from the controlled ablation.
    Exposes a uniform interface: every row has a 'trust' key = that mode's
    sample-weight, plus label_llm/gt/correct."""
    rows = []
    for r in csv.DictReader(open(
            ROOT / "_verifier_controlled" / "per_label_trust.csv")):
        if r["mode"] != mode:
            continue
        rows.append({"sample_id": r["sid"], "label_llm": r["label_llm"],
                     "gt_label": r["gt"], "correct": r["correct"],
                     "trust": r["weight"], "t1": r["weight"],
                     "t2": r["weight"], "t3": r["weight"]})
    return rows


def _subset_stat(rs):
    n = len(rs)
    if n == 0:
        return n, float("nan"), float("nan")
    acc = sum(int(r["correct"]) for r in rs) / n
    gtph = [r for r in rs if r["gt_label"] == "1"]
    rec = (sum(1 for r in gtph if r["label_llm"] == "1") / len(gtph)
           if gtph else float("nan"))
    return n, acc, rec


def verifier_tier(tier_col: str, mode: str, pretty: str, stem: str,
                  show_downstream: bool = True) -> None:
    """One figure per verifier technique: how well its score separates
    correct from incorrect oracle labels (All / High-trust / Low-trust)."""
    rows = _load_canon(_TIER_TO_MODE[tier_col])
    allr = rows
    hi = [r for r in rows if float(r["trust"]) >= 0.5]
    lo = [r for r in rows if float(r["trust"]) < 0.5]
    groups = ["All\n(verifier off)", r"High-trust" + "\n" + r"($\geq0.5$, kept)",
              r"Low-trust" + "\n" + r"($<0.5$, dropped)"]
    accs, recs, ns = [], [], []
    for rs in (allr, hi, lo):
        n, a, r = _subset_stat(rs)
        ns.append(n); accs.append(a); recs.append(r)

    x = np.arange(3)
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    b1 = ax.bar(x - w / 2, [0 if np.isnan(v) else v for v in accs], w,
                color="#1f77b4", label="Oracle accuracy")
    b2 = ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in recs], w,
                color="#d62728", label="Oracle recall on phishing")
    for b, v in zip(b1, accs):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=8)
    for b, v in zip(b2, recs):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=8)
    for xi, n in zip(x, ns):
        ax.text(xi, 0.04, f"n={n}", ha="center", fontsize=7.5,
                color="white" if n else "gray", fontweight="bold")
    if show_downstream:
        pp, dp = _DOWNSTREAM[mode]
        ppoff, dpoff = _DOWNSTREAM["off"]
        ax.text(0.5, 0.92,
                f"downstream: PP F1$_w$ {pp:.3f} ({(pp-ppoff)*100:+.1f}pp)   "
                f"DP {dp:.3f} ({(dp-dpoff)*100:+.1f}pp)",
                transform=ax.transAxes, ha="center", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="gray",
                          lw=0.5))
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("Score on AL-queried labels")
    ax.set_title(f"Verifier mode: {pretty}", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2,
              fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, stem)


def verifier_quality() -> None:
    # the four single-tier / full views
    verifier_tier("t1", "t1", "T1 (LLM confidence)", "fig_verifier_t1")
    verifier_tier("t2", "t2", "T2 (evidence verification)",
                  "fig_verifier_t2")
    verifier_tier("t3", "t3", "T3 (OSINT coherence)", "fig_verifier_t3")
    verifier_tier("trust", "full", "FULL (T1+T2+T3, asymmetric)",
                  "fig_verifier_full", show_downstream=False)
    # keep the old name as an alias of the full view (used in the paper)
    verifier_tier("trust", "full", "FULL (T1+T2+T3, asymmetric)",
                  "fig_verifier_quality", show_downstream=False)
    _verifier_overview()


def _verifier_overview() -> None:
    """All six modes at a glance: kept-subset label quality (bars) vs the
    downstream PP / DP F1w (markers) — the disconnect."""
    modes = [("off", "off"), ("t1_only", "t1"), ("t2_only", "t2"),
             ("t3_only", "t3"), ("full", "full"),
             ("hard_filter", "hard\nfilter")]
    dmap = {"off": "off", "t1_only": "t1", "t2_only": "t2", "t3_only": "t3",
            "full": "full", "hard_filter": "hard_filter"}
    accs, recs, pps, dps, labels = [], [], [], [], []
    for mode, lab in modes:
        rows = _load_canon(mode)
        sub = [r for r in rows if float(r["trust"]) >= 0.5]
        _, a, r = _subset_stat(sub)
        mode_key = dmap[mode]
        accs.append(a); recs.append(r)
        pps.append(_DOWNSTREAM[mode_key][0]); dps.append(_DOWNSTREAM[mode_key][1])
        labels.append(lab)

    x = np.arange(len(modes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.bar(x - w / 2, accs, w, color="#1f77b4", alpha=0.85,
           label="kept-subset oracle accuracy")
    ax.bar(x + w / 2, recs, w, color="#d62728", alpha=0.85,
           label="kept-subset phishing recall")
    ax2 = ax.twinx()
    ax2.plot(x, pps, "o-", color="black", lw=1.6, ms=5,
             label=r"PP-test F1$_w$ (downstream)")
    ax2.plot(x, dps, "s--", color="gray", lw=1.4, ms=5,
             label=r"DP macro-F1 (downstream, OOD)")
    ax2.axhline(_DOWNSTREAM["off"][0], color="black", ls=":", lw=0.7,
                alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("kept-subset label quality")
    ax2.set_ylabel(r"downstream F1$_w$")
    ax.set_ylim(0, 1.05); ax2.set_ylim(0.40, 0.92)
    ax.grid(True, axis="y", alpha=0.2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=7.4,
              frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _save(fig, "fig_verifier_overview")


# ───────────────────────── Fig 2: Llama halt timeline ────────────────────
def llama_halt() -> None:
    p = ROOT / "multi_llm_bench" / "Llama_drift_anchored_B_k20_seed2025"
    rounds = [json.loads(l) for l in (p / "rounds.jsonl").read_text()
              .splitlines() if l.strip()]
    rounds = [r for r in rounds if "health_parse_ok_rate" in r
              and "pp_test_f1w" in r]
    R = [r["round"] for r in rounds]
    parse = [r["health_parse_ok_rate"] for r in rounds]
    ppf1 = [r["pp_test_f1w"] for r in rounds]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # threshold bands
    ax.axhspan(0.0, 0.80, color="#d62728", alpha=0.06)
    ax.axhspan(0.80, 0.95, color="#ff7f0e", alpha=0.08)
    ax.axhline(0.95, color="#ff7f0e", ls="--", lw=0.8,
               label="parse warn (0.95)")
    ax.axhline(0.80, color="#d62728", ls="--", lw=0.8,
               label="parse critical (0.80)")
    ax.plot(R, parse, color="#1f77b4", marker="o", ms=4, lw=1.8,
            label="parse-ok rate (monitor sees this)")
    # consecutive warn rounds R16,R17 then halt at R18
    for r, v in zip(R, parse):
        if v < 0.95 and r >= 16:
            ax.scatter([r], [v], s=90, facecolors="none",
                       edgecolors="#d62728", linewidths=1.6, zorder=5)
    ax.axvline(18, color="black", lw=1.2)
    ax.text(18.05, 0.50, "HALT @ R18\n(3rd consecutive\nparse<0.95)",
            fontsize=7.5, va="center")
    # PP F1w on twin axis — invisible to monitor
    ax2 = ax.twinx()
    ax2.plot(R, ppf1, color="gray", ls=":", lw=1.6,
             label=r"PP-test F1$_w$ (monitor is blind to this)")
    ax2.set_ylabel(r"PP-test F1$_w$", color="gray", fontsize=9)
    ax2.set_ylim(0.60, 0.66)
    ax2.tick_params(axis="y", colors="gray")
    ax.set_xlabel("AL round")
    ax.set_ylabel("parse-ok / yield rate")
    ax.set_ylim(0.4, 1.04)
    ax.set_xticks(range(1, 19))
    ax.grid(True, alpha=0.2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=7.2,
              frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig_llama_halt")


# ───────────────────────── Fig 3: live injection ─────────────────────────
def injection_live() -> None:
    base = ROOT / "_injection_study_live"

    def load(variant):
        f = base / f"{variant}_f1.0" / "telemetry.csv"
        rr = list(csv.DictReader(open(f)))
        return ([int(r["round"]) for r in rr],
                [float(r["parse_ok_rate"]) for r in rr],
                [float(r["class_skew"]) for r in rr])

    nR, nP, nS = load("naive")
    sR, sP, sS = load("smart")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, (R, P, S), title in (
            (axl, (nR, nP, nS), r"\textsc{naive} (format attack)"),
            (axr, (sR, sP, sS), r"\textsc{smart} (semantic attack)")):
        ax.axhspan(0.99, 1.001, color="#d62728", alpha=0.06)
        ax.axhline(0.99, color="#d62728", ls="--", lw=0.8)
        ax.axhline(0.90, color="#ff7f0e", ls="--", lw=0.8)
        ax.plot(R, P, color="#1f77b4", marker="o", ms=4, lw=1.8,
                label="parse-ok")
        ax.plot(R, S, color="#2ca02c", marker="s", ms=4, lw=1.8,
                label="class skew")
        ax.set_title(title.replace(r"\textsc{", "").replace("}", ""),
                     fontsize=9)
        ax.set_xlabel("injected round")
        ax.set_ylim(0.3, 1.05)
        ax.set_xticks(R)
        ax.grid(True, alpha=0.2)
    axl.set_ylabel("rate")
    axl.text(0.5, 0.42, "never halts", fontsize=8, ha="center",
             transform=axl.transAxes, color="gray")
    axr.text(0.5, 0.42, "never halts", fontsize=8, ha="center",
             transform=axr.transAxes, color="gray")
    h, l = axr.get_legend_handles_labels()
    # add threshold legend entries
    from matplotlib.lines import Line2D
    h += [Line2D([0], [0], color="#d62728", ls="--", lw=0.8),
          Line2D([0], [0], color="#ff7f0e", ls="--", lw=0.8)]
    l += ["skew critical (0.99)", "skew warn (0.90)"]
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=7.2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig_injection_live")


if __name__ == "__main__":
    verifier_quality()
    llama_halt()
    injection_live()
