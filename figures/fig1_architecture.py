"""fig1_architecture.py — System architecture block diagram.

Cleaner two-row layout, no text overlap.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _style import save, TWO_COL


def _box(ax, x, y, w, h, text, color="#2E86AB", text_color="white",
         fontsize=8):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.88,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight="bold")


def _arrow(ax, x1, y1, x2, y2, label=None, color="#3a3a3a", rad=0.0,
           label_offset_y=0.25):
    cs = f"arc3,rad={rad}" if rad else "arc3"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                        mutation_scale=12, connectionstyle=cs),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + label_offset_y
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=6.5, color=color, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(TWO_COL * 1.45, 4.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.5)
    ax.axis("off")

    # ── Row 1 (top): stream → ensemble → pool → drift → selection ──
    y1 = 5.2
    _box(ax, 0.2, y1, 1.9, 0.9, "PP stream\n(temporal)", color="#7B5EA7")
    _box(ax, 2.6, y1, 2.0, 0.9, "Leakage-clean\nEnsemble (LR)", color="#2E86AB")
    _box(ax, 5.1, y1, 2.0, 0.9, "Uncertainty\nPool (cap 2000)", color="#A23B72")
    _box(ax, 7.6, y1, 2.0, 0.9, "Drift detector\n(periodic 150)", color="#F2A65A")
    _box(ax, 10.1, y1, 2.4, 0.9, "Selection\n(drift_anchored…)", color="#118C5C")

    _arrow(ax, 2.1, y1+0.45, 2.6, y1+0.45)
    _arrow(ax, 4.6, y1+0.45, 5.1, y1+0.45, label="band ∨ disagree")
    _arrow(ax, 7.1, y1+0.45, 7.6, y1+0.45)
    _arrow(ax, 9.6, y1+0.45, 10.1, y1+0.45, label="trigger")

    # ── Row 2 (middle): LLM Oracle ←→ Health Monitor ──
    y2 = 3.2
    _box(ax, 4.5, y2, 2.5, 1.0,
         "Verifier\nT1 conf · T2 evidence\nT3 OSINT coherence",
         color="#7B5EA7", fontsize=7.5)
    _box(ax, 7.4, y2, 2.5, 1.0,
         "Health Monitor\nparse_ok · class_skew\nyield · trust",
         color="#5B5B5B", fontsize=7.5)
    _box(ax, 10.3, y2, 2.6, 1.0,
         "LLM Oracle (B variant)\nHTML + probs + OSINT",
         color="#E84855", fontsize=7.5)

    _arrow(ax, 11.3, y1, 11.3, y2+1.0, label="K=50 samples")
    _arrow(ax, 10.3, y2+0.5, 9.9, y2+0.5, label=None, color="#666")
    ax.text(10.1, y2+1.05, "audit", fontsize=6.5, color="#666",
            ha="center", style="italic")
    _arrow(ax, 7.4, y2+0.5, 7.0, y2+0.5, color="#666")

    # ── Row 3 (bottom): Retrain → Eval, with HALT branch ──
    y3 = 1.3
    _box(ax, 0.4, y3, 2.4, 1.0, "Retrain LogReg\nKP ∪ accepted\nsample_weight",
         color="#2E86AB", fontsize=7.5)
    _box(ax, 8.4, y3, 1.8, 0.6, "HALT\n(if broken)",
         color="#E84855", fontsize=7.5)
    _arrow(ax, 8.7, y2, 9.0, y3+0.6)

    _box(ax, 3.5, 0.1, 9.4, 0.9,
         "Per-round eval: KP_test · PP_test · DP_probe",
         color="#118C5C", fontsize=9)

    _arrow(ax, 4.5, y2+0.5, 2.8, y3+0.7, label="weights", color="#3a3a3a")
    _arrow(ax, 1.5, y3, 1.5, 1.0)

    # Feedback loop arrow
    ax.annotate(
        "", xy=(3.6, y1), xytext=(1.5, y3+1.0),
        arrowprops=dict(arrowstyle="-|>", color="#118C5C", lw=1.2,
                        mutation_scale=12, connectionstyle="arc3,rad=-0.30"),
    )
    ax.text(0.55, 3.0, "next\nround", fontsize=8, color="#118C5C",
            style="italic", ha="center")

    ax.set_title("PhishHook AL Architecture — pool-based, drift-triggered, "
                 "class-asymmetric verifier, label-free oracle health monitor",
                 fontsize=10, pad=8)
    save(fig, "fig1_architecture")
    plt.close(fig)


if __name__ == "__main__":
    main()
