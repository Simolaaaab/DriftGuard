"""run_noise_ablation.py — Orchestrate the Noise Ablation arms.

The Noise Ablation isolates the *causal origin* of the DP OOD-robustness
effect observed in the canonical LLM-supervised run. Four arms are
compared under identical AL machinery:

    Arm                    Oracle source     Label transformation
    ─────────────────────────────────────────────────────────────────
    clean_gt               gt_dryrun         identity (control)
    random_flip            gt_dryrun         flip GT with prob=rate
    adversarial_p2b        gt_dryrun         flip phish→benign at rate
    adversarial_b2p        gt_dryrun         flip benign→phish at rate
    llm  (reference)       LLM (Variant B)   real LLM labels

The first four are CHEAP (no LLM calls; identical to the canonical
gt_dryrun smoke run plus a deterministic flip). `llm` is taken from
the canonical seed runs already on disk for the comparison.

Determinism:
  - `--noise-seed` controls which samples get flipped. Same
    (sample_id, noise_seed) → same flip across runs.
  - For K-seed variance bands set --noise-seed equal to the run seed,
    so seed 2025/2026/... each have a different noise realisation.

Usage:
  # Canonical: 3 noise arms × K=3 seeds (clean is the existing
  # smoke_periodic_gt; we re-run it as part of the bundle for cleanliness).
  python3 reboot/step3_al/run_noise_ablation.py --canonical

  # Single arm:
  python3 reboot/step3_al/run_noise_ablation.py \\
      --arm random_flip --rate 0.30 --seeds 2025,2026,2027

The script is a thin orchestrator: it shells out to run_al.py with the
right flags, and prints a summary at the end. No new code paths beyond
what's already in run_al.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("noise_ablation")

THIS = Path(__file__).resolve()
RUN_AL = THIS.parent / "run_al.py"
REBOOT = THIS.parents[1]
AL_RUNS = REBOOT / "runs" / "al"


ARMS = {
    "clean_gt":         ("none",            0.0),
    "random_flip":      ("random_flip",     0.30),
    "adversarial_p2b":  ("adversarial_p2b", 0.30),
    "adversarial_b2p":  ("adversarial_b2p", 0.30),
}


def run_id_for(strategy: str, arm: str, rate: float, seed: int) -> str:
    if arm == "clean_gt":
        return f"noise_clean_gt_{strategy}_seed{seed}"
    tag = arm.replace("_flip", "").replace("adversarial_", "adv_")
    return f"noise_{tag}_r{rate:.2f}_{strategy}_seed{seed}"


def launch_arm(
    *, strategy: str, arm: str, rate: float, seed: int,
    max_rounds: int, dry_run: bool, force: bool,
) -> tuple[str, bool]:
    """Return (run_id, succeeded)."""
    mode, default_rate = ARMS[arm]
    rate = rate if rate is not None else default_rate
    run_id = run_id_for(strategy, arm, rate, seed)
    out_dir = AL_RUNS / run_id
    summary = out_dir / "summary.json"
    if summary.exists() and not force:
        log.info(f"[skip] {run_id} already exists at {summary}; "
                 f"--force to re-run")
        return run_id, True

    cmd = [
        sys.executable, str(RUN_AL),
        "--strategy", strategy,
        "--oracle", "gt_dryrun",
        "--threshold-preset", "periodic_audit",
        "--oracle-prompt", "B",
        "--verifier-mode", "off",
        "--max-rounds", str(max_rounds),
        "--seed", str(seed),
        "--run-id", run_id,
        "--noise-mode", mode,
        "--noise-rate", f"{rate}",
        "--noise-seed", str(seed),
    ]
    log.info(f"\n=== [{strategy}|{arm}|rate={rate}|seed={seed}] ===")
    log.info(f"$ {' '.join(cmd)}")
    if dry_run:
        return run_id, True
    rc = subprocess.run(cmd).returncode
    ok = (rc == 0) and summary.exists()
    if not ok:
        log.error(f"[FAIL] {run_id} (rc={rc})")
    return run_id, ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical", action="store_true",
                   help="Run the full canonical bundle: all 4 arms × "
                        "K seeds × {drift_anchored, margin}. ~30 min total "
                        "(no LLM calls).")
    p.add_argument("--arm", choices=list(ARMS), default=None)
    p.add_argument("--strategies", default="drift_anchored,margin",
                   help="comma-separated; default: drift_anchored,margin")
    p.add_argument("--seeds", default="2025,2026,2027",
                   help="comma-separated; default: 2025,2026,2027")
    p.add_argument("--rate", type=float, default=None,
                   help="override the arm's default rate (0.30 for noisy "
                        "arms, 0.0 for clean_gt)")
    p.add_argument("--max-rounds", type=int, default=30)
    p.add_argument("--force", action="store_true",
                   help="re-run arms that already have a summary.json")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands, don't execute")
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.canonical:
        arms = list(ARMS)
    elif args.arm:
        arms = [args.arm]
    else:
        raise SystemExit("--canonical OR --arm required")

    log.info(f"Strategies: {strategies}")
    log.info(f"Arms:       {arms}")
    log.info(f"Seeds:      {seeds}")
    log.info(f"max_rounds: {args.max_rounds}")
    log.info(f"dry_run:    {args.dry_run}")

    plan = [(s, a, args.rate, sd)
            for s in strategies for a in arms for sd in seeds]
    log.info(f"\nTotal runs: {len(plan)}")

    results: list[tuple[str, bool]] = []
    for s, a, r, sd in plan:
        results.append(launch_arm(
            strategy=s, arm=a, rate=r, seed=sd,
            max_rounds=args.max_rounds,
            dry_run=args.dry_run, force=args.force,
        ))

    log.info("\n=== Summary ===")
    n_ok = sum(1 for _, ok in results if ok)
    log.info(f"OK: {n_ok}/{len(results)}")
    if n_ok < len(results):
        log.warning("Failed runs:")
        for rid, ok in results:
            if not ok:
                log.warning(f"  - {rid}")
    log.info("\nNext step:")
    log.info("  python3 reboot/step3_al/bootstrap_noise_ablation.py "
             "--canonical --n-boot 2000")


if __name__ == "__main__":
    main()
