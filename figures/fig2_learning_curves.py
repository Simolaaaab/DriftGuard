"""fig2_learning_curves.py — Per-round PP F1w with K=3 std bands.

Three panels: PP_test, KP_test, DP_probe.
For each: drift_anchored+B and margin+B with K=3 mean ± std bands,
plus GT-oracle ceiling and KP-only baseline reference lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import COLORS, TWO_COL, save

AL_RUNS = Path(__file__).resolve().parents[1] / "runs" / "al"

STRATEGIES = {
    "drift_anchored": ["drift_anchored_B_seed2025",
                       "drift_anchored_B_seed2026",
                       "drift_anchored_B_seed2027"],
    "margin":         ["margin_B_seed2025",
                       "margin_B_seed2026",
                       "margin_B_seed2027"],
}

GT_RUN = "smoke_periodic_gt"


def load_curve(run_id: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    p = AL_RUNS / run_id / "eval_per_round.csv"
    if not p.exists():
        return np.array([]), np.array([])
    df = pd.read_csv(p)
    return df["round"].values, df[col].values


def kseed_band(seeds: list[str], col: str
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounds_ref = None
    arrs = []
    for s in seeds:
        r, v = load_curve(s, col)
        if rounds_ref is None:
            rounds_ref = r
        # Pad/truncate to length of first seed for clean stacking.
        if len(v) < len(rounds_ref):
            v = np.pad(v, (0, len(rounds_ref) - len(v)), constant_values=v[-1])
        elif len(v) > len(rounds_ref):
            v = v[:len(rounds_ref)]
        arrs.append(v)
    if not arrs:
        return np.array([]), np.array([]), np.array([])
    mat = np.vstack(arrs)
    return rounds_ref, mat.mean(axis=0), mat.std(axis=0)


def panel(ax, col: str, ylabel: str, title: str,
          show_legend: bool = False) -> None:
    # K-seed bands for our two canonical strategies
    for name, color in [("drift_anchored", COLORS["drift_anchored"]),
                        ("margin", COLORS["margin"])]:
        r, m, s = kseed_band(STRATEGIES[name], col)
        if len(r) == 0:
            continue
        ax.fill_between(r, m - s, m + s, color=color, alpha=0.18)
        ax.plot(r, m, color=color, lw=1.8, label=f"{name}+B (K=3)")

    # GT ceiling for PP only
    if col == "pp_test_f1_weighted":
        r, v = load_curve(GT_RUN, col)
        if len(r):
            ax.plot(r, v, color=COLORS["gt"], lw=1.2, ls="--",
                    label="GT oracle (perfect)")

    # Round-0 baseline horizontal line
    if col == "pp_test_f1_weighted":
        ax.axhline(0.6234, color="#888", lw=0.6, ls=":")
        ax.text(0.5, 0.628, "KP-only baseline = 0.623",
                fontsize=7.5, color="#666")

    ax.set_xlabel("AL round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 26)
    if show_legend:
        ax.legend(loc="lower right", frameon=True, framealpha=0.95)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL * 1.5, 3.0))
    panel(axes[0], "pp_test_f1_weighted", "F1w (PhreshPhish-test)",
          "(a) Natural drift recovery — PP", show_legend=True)
    panel(axes[1], "kp_test_f1_weighted", "F1w (KnowPhish-test)",
          "(b) In-distribution — KP (forgetting check)")
    panel(axes[2], "dp_probe_f1_weighted", "F1w (DeltaPhish-probe)",
          "(c) Adversarial OOD — DP")
    axes[0].set_ylim(0.58, 0.90)
    axes[1].set_ylim(0.96, 0.99)
    axes[2].set_ylim(0.40, 0.85)
    fig.suptitle("Active Learning trajectory across 25 rounds — "
                 "K=3 seeds, mean ± std",
                 fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig2_learning_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
