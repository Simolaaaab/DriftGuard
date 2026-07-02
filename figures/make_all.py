"""make_all.py — Regenerate all 7 paper figures.

Run from anywhere:
    python3 reboot/figures/make_all.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCRIPTS = [
    "fig1_architecture",
    "fig2_learning_curves",
    "fig3_pareto",
    "fig4_oracle_health",
    "fig5_pp_dp_tradeoff",
    "fig6_leakage_heatmap",
    "fig7_verifier_ablation",
]


def main() -> None:
    for name in SCRIPTS:
        print(f"\n=== {name} ===")
        spec = importlib.util.spec_from_file_location(
            name, HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    print(f"\nAll figures in: {HERE / 'output'}")


if __name__ == "__main__":
    main()
