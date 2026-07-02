"""fig3_pareto.py — Cumulative labels vs PP F1w (Pareto-style).

Per-round trajectory for each strategy + final-point markers. Shows
that drift_anchored dominates the Pareto frontier on PP recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import COLORS, ONE_COL, save

AL_RUNS = Path(__file__).resolve().parents[1] / "runs" / "al"

# canonical strategy runs (B variant, seed=2025, periodic, temporal)
STRATEGIES = {
    "drift_anchored": "drift_anchored_llm_periodic_temporal_B",
    "margin":         "margin_llm_periodic_temporal_B",
    "qbc":            "qbc_llm_periodic_temporal_B",
    "hybrid":         "hybrid_llm_periodic_temporal_B",
    "core_set":       "core_set_llm_periodic_temporal_B",
    "badge":          "badge_llm_periodic_temporal_B",
    "random":         "random_llm_periodic_temporal_B",
}

GT_RUN = "smoke_periodic_gt"


def _curve(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    p_rounds = AL_RUNS / run_id / "rounds.jsonl"
    if not p_rounds.exists():
        return np.array([]), np.array([])
    rounds = [json.loads(l) for l in p_rounds.read_text().splitlines() if l.strip()]
    labels = np.array([r.get("cumulative_labels", 0) for r in rounds])
    f1 = np.array([r.get("pp_test_f1w", 0.0) for r in rounds])
    return labels, f1


def main() -> None:
    fig, ax = plt.subplots(figsize=(ONE_COL * 1.6, 3.0))

    # plot each strategy
    for name, run_id in STRATEGIES.items():
        x, y = _curve(run_id)
        if len(x) == 0:
            continue
        color = COLORS.get(name, "#888")
        ax.plot(x, y, color=color, lw=1.4, marker="o", markersize=2.5,
                alpha=0.9, label=name)
        # mark final
        ax.scatter([x[-1]], [y[-1]], s=40, color=color,
                   edgecolor="white", zorder=5, linewidth=1.0)

    # GT ceiling
    x, y = _curve(GT_RUN)
    if len(x):
        ax.plot(x, y, color="black", lw=1.0, ls="--", marker="x", markersize=3,
                label="GT oracle (perfect)", alpha=0.7)

    # KP-only baseline (round 0)
    ax.axhline(0.6234, color="#888", lw=0.6, ls=":")
    ax.text(20, 0.628, "KP-only baseline", fontsize=7.5,
            color="#666", style="italic")

    ax.set_xlabel("Cumulative oracle labels acquired")
    ax.set_ylabel("PP-test F1w (drift recovery)")
    ax.set_title("Label-efficiency Pareto across selection strategies",
                 fontsize=10)
    ax.set_xlim(0, 1300)
    ax.set_ylim(0.60, 0.90)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "fig3_pareto")
    plt.close(fig)


if __name__ == "__main__":
    main()
