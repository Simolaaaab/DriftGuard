"""fig5_pp_dp_tradeoff.py — Final PP F1w vs Final DP F1w per strategy.

Shows the trade-off: drift_anchored maximises PP but sacrifices DP,
while margin/qbc/hybrid favour balanced PP+DP. Colour-coded by strategy
with D vs B oracle as marker shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from _style import COLORS, ONE_COL, save

AL_RUNS = Path(__file__).resolve().parents[1] / "runs" / "al"

POINTS = [
    # (label, run_id, oracle_variant_marker)
    ("drift_anchored+B", "drift_anchored_llm_periodic_temporal_B",  "o"),
    ("margin+B",         "margin_llm_periodic_temporal_B",          "o"),
    ("qbc+B",            "qbc_llm_periodic_temporal_B",             "o"),
    ("hybrid+B",         "hybrid_llm_periodic_temporal_B",          "o"),
    ("core_set+B",       "core_set_llm_periodic_temporal_B",        "o"),
    ("badge+B",          "badge_llm_periodic_temporal_B",           "o"),
    ("random+B",         "random_llm_periodic_temporal_B",          "o"),
    ("drift_anchored+D", "drift_anchored_llm_periodic_temporal",    "^"),
    ("margin+D",         "margin_llm_periodic_temporal",            "^"),
    ("qbc+D",            "qbc_llm_periodic_temporal",               "^"),
    ("hybrid+D",         "hybrid_llm_periodic_temporal",            "^"),
    ("random+D",         "random_llm_periodic_temporal",            "^"),
]
GT_RUN = "smoke_periodic_gt"


def _final(run_id: str) -> tuple[float, float]:
    p = AL_RUNS / run_id / "summary.json"
    if not p.exists(): return float("nan"), float("nan")
    d = json.load(open(p))
    fe = d["final_eval"]
    return fe["pp_test_f1_weighted"], fe["dp_probe_f1_weighted"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(ONE_COL * 1.5, 3.0))

    # Baseline marker (round-0 KP-only)
    ax.scatter([0.6234], [0.5141], s=70, marker="s", color="grey",
               edgecolor="black", linewidth=0.8, zorder=5,
               label="KP-only baseline (round 0)")

    # GT ceiling marker
    pp_gt, dp_gt = _final(GT_RUN)
    ax.scatter([pp_gt], [dp_gt], s=110, marker="*", color="black",
               edgecolor="white", linewidth=0.8, zorder=6,
               label="GT oracle (perfect, 25R)")

    for label, run_id, marker in POINTS:
        strategy = label.split("+")[0]
        variant = label.split("+")[1]
        color = COLORS.get(strategy, "#888")
        pp, dp = _final(run_id)
        if pp != pp: continue
        ax.scatter([pp], [dp], s=55, color=color, marker=marker,
                   edgecolor="white", linewidth=0.6, alpha=0.95,
                   zorder=4)
        # Label only the B variant to keep clean
        if variant == "B":
            ax.annotate(strategy.replace("_", " "),
                        xy=(pp, dp), xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=7.5, color=color, weight="bold")

    # Iso-balance line (PP + DP = constant): show 1.4, 1.5, 1.6
    for c in (1.4, 1.5, 1.6):
        ax.plot([0.55, 0.90], [c - 0.55, c - 0.90],
                color="#bbb", lw=0.4, ls=":")
        ax.text(0.90, c - 0.90 - 0.005, f"PP+DP={c:.1f}",
                fontsize=6.5, color="#999", ha="right")

    # Custom legend for shape (B vs D)
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markersize=8, label="Variant B (canonical)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#555",
               markersize=8, label="Variant D (Step-2 winner)"),
        Line2D([0], [0], marker="s", color="grey", markerfacecolor="grey",
               markersize=7, label="KP-only baseline"),
        Line2D([0], [0], marker="*", color="black", markerfacecolor="black",
               markersize=11, label="GT oracle"),
    ]
    ax.legend(handles=leg_handles, loc="lower left",
              frameon=True, framealpha=0.95, fontsize=7.5)

    ax.set_xlabel("Final PP-test F1w  (natural drift recovery)")
    ax.set_ylabel("Final DP-probe F1w  (adversarial OOD)")
    ax.set_title("Selection-strategy trade-off  —  PP vs DP at round 25",
                 fontsize=10)
    ax.set_xlim(0.59, 0.88)
    ax.set_ylim(0.35, 0.85)
    fig.tight_layout()
    save(fig, "fig5_pp_dp_tradeoff")
    plt.close(fig)


if __name__ == "__main__":
    main()
