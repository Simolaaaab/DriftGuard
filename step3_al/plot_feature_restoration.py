"""plot_feature_restoration.py — figure for the teacher--student gap (Test ③).

Reads runs/feature_restoration/partB_system.csv (produced by
feature_restoration.py) and draws the leakage fingerprint: handed the eleven
gated-out OSINT features back, the statistical detector reaches a PP score
*above* the in-pipeline perfect-label ceiling — a level only temporal leakage
can buy. Lollipop on a zoomed axis (no truncated bars).

Output → runs/feature_restoration/fig_restoration.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "runs" / "feature_restoration"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,   # TrueType, editor-friendly
    "ps.fonttype": 42,
})


def main() -> None:
    B = pd.read_csv(OUT / "partB_system.csv").set_index("features")
    gated = float(B.loc["gated", "PP_test_f1w"])
    restored = float(B.loc["restored", "PP_test_f1w"])
    ceil = float(B.loc["gated_gt_ceiling", "PP_test_f1w"])
    kp_g = float(B.loc["gated", "KP_test_f1w"])
    kp_r = float(B.loc["restored", "KP_test_f1w"])

    lo, hi = 0.83, 0.915
    fig, ax = plt.subplots(figsize=(3.35, 1.95))

    # perfect-label ceiling (in-pipeline, GT labels on the gated features)
    ax.axvline(ceil, ls="--", lw=1.1, color="0.45", zorder=1)
    ax.text(ceil, 2.62, "perfect-label\nceiling (GT)", ha="center", va="bottom",
            fontsize=6.8, color="0.35", linespacing=0.95)

    rows = [("gated (LLM labels)", gated, 1, "0.55"),
            ("restored (LLM labels)", restored, 2, "#c1121f")]
    for label, val, y, col in rows:
        ax.hlines(y, lo, val, color=col, lw=1.4, zorder=2)
        ax.plot(val, y, "o", color=col, ms=8, zorder=3)
        dx = -0.004 if val > ceil else 0.004
        ha = "right" if val > ceil else "left"
        ax.text(val + dx, y + 0.22, f"{val:.3f}", ha=ha, va="bottom",
                fontsize=7.2, color=col, fontweight="bold")

    # the "+features" move and the leakage callout
    ax.annotate("", xy=(restored, 1.78), xytext=(gated, 1.22),
                arrowprops=dict(arrowstyle="-|>", color="#c1121f", lw=1.1,
                                connectionstyle="arc3,rad=-0.25"), zorder=2)
    ax.text((gated + restored) / 2 + 0.012, 1.5, "+11 OSINT\nfeatures back",
            fontsize=6.6, color="#c1121f", ha="left", va="center",
            linespacing=0.95)
    ax.text(restored, 2.0, "  leakage", fontsize=7.0, color="#c1121f",
            va="center", ha="left", style="italic")

    ax.set_xlim(lo, hi)
    ax.set_ylim(0.4, 2.9)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["gated", "restored"])
    ax.set_xlabel(r"PP weighted $F_1$")
    ax.set_title(f"Restoring the gated-out features  "
                 f"(KP {kp_g:.3f}$\\to${kp_r:.3f})", fontsize=7.4, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)

    fig.tight_layout(pad=0.4)
    pdf = OUT / "fig_restoration.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {pdf}")
    print(f"  gated={gated:.4f}  ceiling(GT)={ceil:.4f}  restored={restored:.4f}"
          f"  -> restored {'>' if restored > ceil else '<='} ceiling")


if __name__ == "__main__":
    main()
