"""plot_cursor_compare.py — overlay LLM→GT cursor across selection strategies.

Stacks PP and DP curves for two (or more) canonical AL runs in a single
single-panel figure, line-style distinguishing the *strategy* and color
distinguishing the *dataset*. Used to make the causal claim about
oracle-noise regularisation visually argue itself: the slope and end-point
of the DP curve replicate across strategies, while PP improves
monotonically in both — i.e. the trade-off is not strategy-specific.

Layout per supervisor feedback:
  - one panel only
  - legend ABOVE the plot, small font, dataset name + strategy
  - no main title (Overleaf caption)
  - x-axis: % corrected + absolute support N (same for both — the
    epsilons grid is identical, only N varies per run since the LLM
    error count is strategy-dependent)

Default comparison: drift_anchored_llm_periodic_temporal_B  vs
                    margin_B_seed2025
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

# Display labels (legend) and line styles per strategy.
DEFAULT_RUNS = [
    ("drift_anchored_llm_periodic_temporal_B", "drift-anchored",  "-"),
    ("margin_B_seed2025",                       "margin",          "--"),
]

# Per-dataset colors.
COLOR_PP = "#1f77b4"
COLOR_DP = "#d62728"


def _band(ax, x, mean, std, *, color, ls, label):
    ax.plot(x, mean, ls + "o", color=color, label=label, lw=2, ms=5)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)


def _load_agg(canonical_run: str) -> pd.DataFrame:
    sub = canonical_run.replace("/", "_")
    csv_path = CURSOR_ROOT / sub / "cursor_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run cursor_experiment.py "
            f"--canonical-run {canonical_run} first."
        )
    df = pd.read_csv(csv_path)
    return df.groupby("epsilon").agg(
        pp_m=("pp_f1_weighted", "mean"),
        pp_s=("pp_f1_weighted", "std"),
        dp_macro_m=("dp_f1_macro", "mean"),
        dp_macro_s=("dp_f1_macro", "std"),
        n_flip=("n_errors_flipped", "first"),
    ).reset_index().fillna(0.0)


def render(runs: list[tuple[str, str, str]], out_name: str) -> None:
    aggs = [(run, label, ls, _load_agg(run)) for run, label, ls in runs]

    # Use the first run's epsilon grid as x. All runs share the
    # canonical {0, 25, 50, 75, 100}.
    x = (aggs[0][3]["epsilon"].values * 100).astype(int)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # Plot in order: all PP curves, then all DP curves — keeps the
    # legend grouped by dataset.
    for run, label, ls, g in aggs:
        _band(ax, x, g["pp_m"].values, g["pp_s"].values,
              color=COLOR_PP, ls=ls,
              label=f"PhreshPhish — {label}")
    for run, label, ls, g in aggs:
        _band(ax, x, g["dp_macro_m"].values, g["dp_macro_s"].values,
              color=COLOR_DP, ls=ls,
              label=f"DeltaPhish — {label}")

    ax.set_xlabel("% LLM errors corrected toward GT")
    ax.set_ylabel("F1 score  (PP weighted  /  DP macro)")

    # X-ticks: %  +  N for each run on its own line. With 2 strategies,
    # show both Ns. If only 1 strategy, falls back to single N.
    tick_labels = []
    for i, p in enumerate(x):
        ns = " / ".join(f"N={int(g.iloc[i]['n_flip'])}"
                        for _, _, _, g in aggs)
        tick_labels.append(f"{int(p)}%\n({ns})")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=7)

    ax.set_ylim(0.30, 0.90)
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
        ncol=2, fontsize=8, frameon=False,
        handlelength=2.4, handletextpad=0.6, columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir = CURSOR_ROOT
    out_png = out_dir / f"{out_name}.png"
    out_pdf = out_dir / f"{out_name}.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")

    # Also print a tiny side-by-side numerical comparison.
    print("\nFinal-point summary (ε=1.0):")
    for run, label, _ls, g in aggs:
        end = g[g["epsilon"] == 1.0].iloc[0]
        start = g[g["epsilon"] == 0.0].iloc[0]
        print(f"  {label:>15s} "
              f"| PP {start['pp_m']:.3f}→{end['pp_m']:.3f} "
              f"(+{(end['pp_m']-start['pp_m'])*100:+.1f}pp)  "
              f"| DPmacro {start['dp_macro_m']:.3f}→{end['dp_macro_m']:.3f} "
              f"({(end['dp_macro_m']-start['dp_macro_m'])*100:+.1f}pp)  "
              f"| N_errors={int(g[g['epsilon']==1.0]['n_flip'].iloc[0])}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=None,
                   help="comma-sep list of canonical_run ids "
                        "(default: drift_anchored, margin_B_seed2025)")
    p.add_argument("--labels", default=None,
                   help="comma-sep display labels matching --runs")
    p.add_argument("--out-name", default="cursor_compare_drift_vs_margin")
    args = p.parse_args()

    if args.runs:
        runs_list = [r.strip() for r in args.runs.split(",") if r.strip()]
        labels = (
            [s.strip() for s in args.labels.split(",")]
            if args.labels else runs_list
        )
        if len(labels) != len(runs_list):
            raise SystemExit("--labels count must match --runs count")
        styles = ["-", "--", ":", "-."][: len(runs_list)]
        runs = list(zip(runs_list, labels, styles))
    else:
        runs = DEFAULT_RUNS

    render(runs, args.out_name)


if __name__ == "__main__":
    main()
