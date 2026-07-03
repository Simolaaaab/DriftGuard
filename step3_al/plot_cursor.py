"""plot_cursor.py — render the LLM→GT cursor figure (single panel).

Reads reboot/runs/al/_cursor/<canonical_run>/cursor_results.csv and writes:
  cursor_pp_vs_dp.png       single-panel PP↑/DP↓ curves
  cursor_pp_vs_dp.pdf       same for camera-ready

Default canonical_run = drift_anchored_llm_periodic_temporal_B.
Use --run margin_B_seed2025 (etc.) to plot a different strategy's cursor.

Per supervisor feedback (2026-05):
  - drop the right-side false-alarm panel (will be referenced in text)
  - legend ABOVE the plot, smaller font, dataset names only
  - no plot title (caption goes in Overleaf)
  - x-axis ticks show BOTH the % corrected and the absolute support
    N (number of LLM errors actually flipped at that ε)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

CURSOR_ROOT = (
    Path(__file__).resolve().parents[1] / "runs" / "al" / "_cursor"
)
DEFAULT_RUN = "drift_anchored_llm_periodic_temporal_B"


def _band(ax, x, mean, std, *, color, label):
    ax.plot(x, mean, "-o", color=color, label=label, lw=2, ms=6)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


def render(canonical_run: str) -> None:
    in_dir = CURSOR_ROOT / canonical_run.replace("/", "_")
    csv_path = in_dir / "cursor_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run cursor_experiment.py "
            f"--canonical-run {canonical_run} first."
        )

    df = pd.read_csv(csv_path)

    g = df.groupby("epsilon").agg(
        pp_m=("pp_f1_weighted", "mean"),
        pp_s=("pp_f1_weighted", "std"),
        dp_macro_m=("dp_f1_macro", "mean"),
        dp_macro_s=("dp_f1_macro", "std"),
        n_flip=("n_errors_flipped", "first"),
    ).reset_index().fillna(0.0)
    x = (g["epsilon"].values * 100).astype(int)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _band(ax, x, g["pp_m"].values, g["pp_s"].values,
          color="#1f77b4", label="PhreshPhish")
    _band(ax, x, g["dp_macro_m"].values, g["dp_macro_s"].values,
          color="#d62728", label="DeltaPhish")

    ax.set_xlabel("% LLM errors corrected toward GT")
    ax.set_ylabel("F1 score  (PP weighted  /  DP macro)")

    tick_labels = [f"{int(p)}%\n(N={int(n)})"
                   for p, n in zip(x, g["n_flip"].values)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)

    ax.set_ylim(0.30, 0.90)
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
        ncol=2, fontsize=8, frameon=False,
        handlelength=2.0, handletextpad=0.6, columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = in_dir / "cursor_pp_vs_dp.png"
    out_pdf = in_dir / "cursor_pp_vs_dp.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=DEFAULT_RUN,
                   help="canonical AL run id whose cursor results to plot")
    args = p.parse_args()
    render(args.run)


if __name__ == "__main__":
    main()
