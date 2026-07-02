"""plot_figure1_sawtooth.py — Hero figure for the paper (two files).

Generates TWO separate step-jump figures, one per trigger mode:
  figure1a_periodic.{png,pdf}
  figure1b_kl_only.{png,pdf}

What is plotted
───────────────
  - X-axis: stream row index (temporal order on PP_stream, 2024-25)
  - Y-axis: PP_test F1_weighted — evaluated on the held-out 20% test set
            after every retrain event. This is the same metric the paper
            quotes as the headline (R0=0.623 → R25≈0.798).
  - Step lines: horizontal between trigger fires (model is unchanged),
                vertical at each trigger (retrain → new F1w).
  - Vertical thin dashed lines: trigger fire positions on the stream.

History note (audit 2026-05)
────────────────────────────
The previous version of this script computed a rolling F1w of the live
model directly on PP_stream rows. That metric is *composition-confounded*:
PP_stream segments have phish_frac swinging from 3% to 100%, so the
visual "decay" between triggers was not the model losing skill but the
traffic mix changing. Segment 0 (rows 0-148) is 8.7% phish, so a
KP-only-trained classifier (benign-specialist) trivially scored F1w=0.88
— and the figure looked like it started at ~0.88, contradicting the
paper's R0=0.623 baseline. We now evaluate every snapshot on the FIXED
PP_test held-out set, which is what the paper cites everywhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "runs" / "al"
OUT = ROOT / "_aggregate"

# (mode_tag, run_id, line_color, legend_label, out_file_stem)
TRIGGERS = [
    ("periodic", "drift_anchored_periodic_B_seed2025_full_snap",
     "#1f77b4", "Periodic trigger (every 150 rows)",
     "figure1a_periodic"),
    ("kl_only", "drift_anchored_kl_only_B_seed2025_full_snap",
     "#d62728", r"KL-only trigger (input-based, $\mathrm{KL}\geq 0.30$)",
     "figure1b_kl_only"),
]


def render_one(trigger_name: str, run_id: str, color: str,
               legend_label: str, out_stem: str) -> None:
    run_dir = ROOT / run_id

    # PP_test F1w per round — the metric the paper quotes.
    eval_df = pd.read_csv(run_dir / "eval_per_round.csv").sort_values("round")
    rounds = eval_df["round"].astype(int).to_numpy()
    pp_f1w = eval_df["pp_test_f1_weighted"].astype(float).to_numpy()
    # Trigger rows: tells us where on the stream each retrain happened.
    fires = pd.read_csv(run_dir / "stream_fires.csv").sort_values("round")
    fire_rows = fires["trigger_row"].astype(int).to_numpy()
    # PP_stream length (for the x-axis extent).
    per_row = pd.read_csv(run_dir / "stream_per_row.csv",
                          usecols=["row_idx"])
    stream_end = int(per_row["row_idx"].max())

    summary = json.loads((run_dir / "summary.json").read_text())
    pp0 = float(summary["initial_eval"]["pp_test"]["f1_weighted"])
    pp_final = float(summary["final_eval"]["pp_test_f1_weighted"])

    # Align: rounds[0]=0 is the round-0 (KP-only) eval. rounds[1..K] are
    # post-trigger evals, one per fire. fire_rows[k-1] is where the
    # k-th retrain (rounds[k]) actually happened on the stream.
    # Step semantics:
    #   - between [stream_start=0,           fire_rows[0])  → F1w = pp_f1w[0] (R0)
    #   - between [fire_rows[k-1], fire_rows[k])             → F1w = pp_f1w[k]
    #   - between [fire_rows[-1], stream_end+1)              → F1w = pp_f1w[-1]
    if len(rounds) != 1 + len(fire_rows):
        # Robust fallback if there is a missing round / extra fire.
        n_steps = min(len(pp_f1w) - 1, len(fire_rows))
        pp_f1w = pp_f1w[: n_steps + 1]
        fire_rows = fire_rows[: n_steps]

    # Build the step polyline manually so the verticals are crisp.
    seg_xs = [0] + list(fire_rows) + [stream_end + 1]
    seg_ys = list(pp_f1w)
    poly_x: list[float] = []
    poly_y: list[float] = []
    for i, y in enumerate(seg_ys):
        x_a, x_b = seg_xs[i], seg_xs[i + 1]
        poly_x.extend([x_a, x_b])
        poly_y.extend([y, y])
        # Vertical connector to the next segment (the retrain jump).
        if i + 1 < len(seg_ys):
            poly_x.append(x_b)
            poly_y.append(seg_ys[i + 1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.2))

    # Vertical thin dashed lines = retrain (oracle fire) events.
    for tr in fire_rows:
        ax.axvline(tr, color="black", ls="--", lw=0.6, alpha=0.35, zorder=1)

    # The step curve itself.
    ax.plot(poly_x, poly_y, color=color, lw=2.2, alpha=0.95, zorder=3)

    # Mark each round point: R=0 at x=0, R=k at fire_rows[k-1].
    dot_xs = [0] + list(fire_rows)
    dot_ys = list(pp_f1w)
    ax.scatter(dot_xs, dot_ys, color=color, s=22, zorder=4,
               edgecolors="white", linewidths=0.6)

    # Round-0 baseline reference line.
    ax.axhline(pp0, color="gray", ls=":", lw=0.8, alpha=0.7, zorder=2)
    ax.text(stream_end + 8, pp0,
            f" R0 baseline {pp0:.3f}",
            color="gray", fontsize=8, va="center", ha="left")
    # Final F1w marker.
    ax.text(stream_end + 8, pp_final,
            f" R{len(fire_rows)} final {pp_final:.3f}",
            color=color, fontsize=8, va="center", ha="left",
            fontweight="bold")

    ax.set_xlabel("PhreshPhish stream row index "
                  "(temporal order)")
    ax.set_ylabel(r"PhreshPhish-test F1$_w$ (held-out, $n=956$)")
    ax.set_xlim(-30, stream_end + 220)
    ax.set_ylim(0.55, 0.85)
    ax.grid(True, alpha=0.25)

    # Per supervisor feedback:
    #  - NO main title (Overleaf caption will name the figure)
    #  - legend ABOVE the plot area, small font, clean labels
    from matplotlib.lines import Line2D
    leg_elems = [
        Line2D([0], [0], color=color, lw=2.0, marker="o", ms=5,
               label=f"PhreshPhish-test F1$_w$ per round  ({legend_label})"),
        Line2D([0], [0], color="black", ls="--", lw=0.7,
               label=f"Retrain event ({len(fire_rows)} total)"),
        Line2D([0], [0], color="gray", ls=":", lw=0.8,
               label="Round-0 baseline"),
    ]
    ax.legend(
        handles=leg_elems,
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
        ncol=3, fontsize=8, frameon=False,
        handlelength=2.4, handletextpad=0.6, columnspacing=1.4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        out = OUT / f"{out_stem}.{ext}"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for trig in TRIGGERS:
        render_one(*trig)


if __name__ == "__main__":
    main()
