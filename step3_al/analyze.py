"""analyze.py — Cross-strategy comparison + Pareto.

Reads every `eval_per_round.csv` under `reboot/runs/al/{run_id}/`,
aggregates by run_id, and produces:

  reboot/runs/al/_aggregate/
    learning_curves_pp.csv           (per-round PP F1w per strategy)
    learning_curves_kp.csv           (per-round KP F1w per strategy)
    learning_curves_dp.csv           (per-round DP F1w per strategy)
    pareto_labels_vs_f1.csv          (PP F1w vs cumulative labels)
    headline_table.csv               (final-round metrics per strategy)
    REPORT.md                        (human-readable Markdown)

No plots here — keeps dependencies (matplotlib etc.) optional. Plots
go in a separate notebook if needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import AL_RUNS_ROOT

AGG_DIR = AL_RUNS_ROOT / "_aggregate"


def _collect_run_dirs() -> list[Path]:
    out = []
    for p in AL_RUNS_ROOT.iterdir():
        if p.is_dir() and p.name != "_aggregate":
            if (p / "eval_per_round.csv").exists():
                out.append(p)
    return sorted(out, key=lambda d: d.name)


def _load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    evals = pd.read_csv(run_dir / "eval_per_round.csv")
    rounds = []
    rl = run_dir / "rounds.jsonl"
    if rl.exists():
        for line in rl.read_text().splitlines():
            if line.strip():
                rounds.append(json.loads(line))
    return summary, evals, rounds


def main() -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = _collect_run_dirs()
    if not run_dirs:
        print("No AL runs found under reboot/runs/al/.")
        return

    rows_pp: list[dict] = []
    rows_kp: list[dict] = []
    rows_dp: list[dict] = []
    rows_pareto: list[dict] = []
    rows_headline: list[dict] = []

    for rd in run_dirs:
        summary, evals, rounds = _load_run(rd)
        cfg = summary.get("config") or {}
        run_id = rd.name
        strategy = cfg.get("strategy", run_id)
        oracle = cfg.get("oracle_mode", "")
        # Per-round curves.
        for _, r in evals.iterrows():
            rows_pp.append({
                "run_id": run_id, "strategy": strategy, "oracle": oracle,
                "round": int(r["round"]),
                "n_accepted": int(r["n_accepted"]),
                "pp_f1w": r["pp_test_f1_weighted"],
                "pp_recall_phish": r["pp_test_recall_phish"],
                "pp_ece": r["pp_test_ece"],
            })
            rows_kp.append({
                "run_id": run_id, "strategy": strategy, "oracle": oracle,
                "round": int(r["round"]),
                "kp_f1w": r["kp_test_f1_weighted"],
                "kp_recall_phish": r["kp_test_recall_phish"],
            })
            rows_dp.append({
                "run_id": run_id, "strategy": strategy, "oracle": oracle,
                "round": int(r["round"]),
                "dp_f1w": r["dp_probe_f1_weighted"],
                "dp_recall_phish": r["dp_probe_recall_phish"],
            })
        # Pareto points.
        cum_calls = 0
        for rec in rounds:
            cum_calls = rec.get("cumulative_llm_calls", cum_calls)
            rows_pareto.append({
                "run_id": run_id, "strategy": strategy, "oracle": oracle,
                "cumulative_llm_calls": cum_calls,
                "cumulative_labels": rec.get("cumulative_labels", 0),
                "pp_f1w": rec.get("pp_test_f1w"),
            })
        # Headline (final round).
        if len(evals):
            final = evals.iloc[-1]
            init = evals.iloc[0]
            rows_headline.append({
                "run_id": run_id, "strategy": strategy, "oracle": oracle,
                "n_rounds": int(final["round"]),
                "n_accepted": int(final["n_accepted"]),
                "n_llm_calls": summary.get("cumulative_llm_calls", 0),
                "pp_f1w_initial": init["pp_test_f1_weighted"],
                "pp_f1w_final": final["pp_test_f1_weighted"],
                "pp_delta_f1w": final["pp_test_f1_weighted"]
                                - init["pp_test_f1_weighted"],
                "kp_f1w_initial": init["kp_test_f1_weighted"],
                "kp_f1w_final": final["kp_test_f1_weighted"],
                "kp_delta_f1w": final["kp_test_f1_weighted"]
                                - init["kp_test_f1_weighted"],
                "dp_f1w_initial": init["dp_probe_f1_weighted"],
                "dp_f1w_final": final["dp_probe_f1_weighted"],
                "dp_delta_f1w": final["dp_probe_f1_weighted"]
                                - init["dp_probe_f1_weighted"],
                "elapsed_s": summary.get("elapsed_s", 0),
            })

    pd.DataFrame(rows_pp).to_csv(AGG_DIR / "learning_curves_pp.csv", index=False)
    pd.DataFrame(rows_kp).to_csv(AGG_DIR / "learning_curves_kp.csv", index=False)
    pd.DataFrame(rows_dp).to_csv(AGG_DIR / "learning_curves_dp.csv", index=False)
    pd.DataFrame(rows_pareto).to_csv(AGG_DIR / "pareto_labels_vs_f1.csv",
                                     index=False)
    head_df = pd.DataFrame(rows_headline).sort_values("pp_f1w_final",
                                                       ascending=False)
    head_df.to_csv(AGG_DIR / "headline_table.csv", index=False)

    # Markdown report.
    lines: list[str] = [
        "# Step 3 — Active Learning Headline Comparison",
        "",
        "## Final-round results per run",
        "",
        "| run_id | strategy | oracle | rounds | labels | LLM calls | "
        "PP F1w init→final | ΔPP | ΔKP | ΔDP |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for _, r in head_df.iterrows():
        lines.append(
            f"| {r['run_id']} | {r['strategy']} | {r['oracle']} | "
            f"{r['n_rounds']} | {r['n_accepted']} | {r['n_llm_calls']} | "
            f"{r['pp_f1w_initial']:.3f}→{r['pp_f1w_final']:.3f} | "
            f"{r['pp_delta_f1w']:+.3f} | "
            f"{r['kp_delta_f1w']:+.3f} | "
            f"{r['dp_delta_f1w']:+.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- **ΔKP near zero** confirms full-retrain prevents catastrophic forgetting.",
        "- **ΔPP positive** = AL recovery on natural drift.",
        "- **ΔDP negative** = adversarial regression — boundary of self-healing approach.",
        "",
        "## Files",
        "- `learning_curves_pp.csv` — per-round PP F1w per run.",
        "- `pareto_labels_vs_f1.csv` — labels-vs-F1w Pareto data.",
        "- `headline_table.csv` — this table.",
    ]
    (AGG_DIR / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nArtifacts: {AGG_DIR.relative_to(AL_RUNS_ROOT.parents[2])}/")


if __name__ == "__main__":
    main()
