"""fig7_verifier_ablation.py — Step-4 verifier ablation grouped bars.

Two side-by-side bar groups (PP F1w and DP F1w) for the 6 verifier
modes + baseline. Shows the negative result: every verifier mode
yields neutral PP and consistently degrades DP (implicit-regularisation
finding).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import COLORS, ONE_COL, save

AL_RUNS = Path(__file__).resolve().parents[1] / "runs" / "al"

MODES = [
    ("baseline\n(no verifier)",      "drift_anchored_llm_periodic_temporal_B"),
    ("t1_only\n(confidence)",         "drift_anchored_llm_periodic_temporal_B_v4_t1_only"),
    ("t2_only\n(evidence)",           "drift_anchored_llm_periodic_temporal_B_v4_t2_only"),
    ("t3_only\n(OSINT)",              "drift_anchored_llm_periodic_temporal_B_v4_t3_only"),
    ("full\n(T1+T2+T3 soft)",         "drift_anchored_llm_periodic_temporal_B_v4"),
    ("hard_filter\n(drop trust<0.5)", "drift_anchored_llm_periodic_temporal_B_v4_hard_filter"),
]


def _final(run_id):
    p = AL_RUNS / run_id / "summary.json"
    if not p.exists():
        return float("nan"), float("nan")
    d = json.load(open(p))
    fe = d["final_eval"]
    return fe["pp_test_f1_weighted"], fe["dp_probe_f1_weighted"]


def main() -> None:
    labels = [m[0] for m in MODES]
    pps = [_final(m[1])[0] for m in MODES]
    dps = [_final(m[1])[1] for m in MODES]

    baseline_pp = pps[0]
    baseline_dp = dps[0]

    fig, ax = plt.subplots(figsize=(ONE_COL * 1.9, 3.0))
    x = np.arange(len(labels))
    w = 0.38

    ax.bar(x - w/2, pps, w, color=COLORS["pp"], alpha=0.85,
           edgecolor="white", linewidth=0.6, label="PP-test F1w")
    ax.bar(x + w/2, dps, w, color=COLORS["dp"], alpha=0.85,
           edgecolor="white", linewidth=0.6, label="DP-probe F1w")

    # baseline reference lines
    ax.axhline(baseline_pp, color=COLORS["pp"], lw=0.6, ls=":", alpha=0.6)
    ax.axhline(baseline_dp, color=COLORS["dp"], lw=0.6, ls=":", alpha=0.6)

    # numerical delta annotation per bar
    for i, (pp, dp) in enumerate(zip(pps, dps)):
        if i == 0:
            ax.text(i - w/2, pp + 0.005, f"{pp:.3f}",
                    ha="center", fontsize=7, color=COLORS["pp"])
            ax.text(i + w/2, dp + 0.005, f"{dp:.3f}",
                    ha="center", fontsize=7, color=COLORS["dp"])
        else:
            dpp = (pp - baseline_pp) * 100
            ddp = (dp - baseline_dp) * 100
            ax.text(i - w/2, pp + 0.005,
                    f"{pp:.3f}\n({dpp:+.1f}pp)",
                    ha="center", fontsize=6.5, color=COLORS["pp"])
            ax.text(i + w/2, dp + 0.005,
                    f"{dp:.3f}\n({ddp:+.1f}pp)",
                    ha="center", fontsize=6.5, color=COLORS["dp"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("F1w")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.set_title("Step-4 verifier ablation — drift_anchored+B, seed=2025\n"
                 "Negative result: every verifier mode neutral on PP, degrades DP",
                 fontsize=9, pad=8)
    fig.tight_layout()
    save(fig, "fig7_verifier_ablation")
    plt.close(fig)


if __name__ == "__main__":
    main()
