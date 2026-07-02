"""run_multi_llm_bench.py — Full-PP multi-LLM benchmark orchestrator.

Launches the **canonical Step-3 AL pipeline** (drift_anchored + Prompt B +
periodic_audit + verifier off + health monitor on) on the *full* PhreshPhish
stream for each registered LLM provider, with a budget-friendly K=20 so a
6-provider bench fits in a single ~$15-30 session.

Goal: isolate **provider behavior** (label quality, hallucination, parse
failures, CoT richness) under one fixed selection strategy and prompt. The
selection-strategy ablation is already covered by Step 3's headline_table;
the strategy-vs-random comparison is a separate experiment.

Design choices (locked with supervisor):
  • K = 20         budget bilanciato; ~500 LLM call/provider su PP_stream
                   (=3821 rows / periodic_every=150 → ~25 round)
  • Strategia      drift_anchored only (canonical PP-optimal)
  • Prompt         B only (AL-canonical, probs+OSINT)
  • Seed           single 2025 (varianza che ci interessa è cross-model)
  • Exclude        kimi_k2_6 by default (già documentato come broken da
                   health monitor — non ri-pagarlo)
  • Verifier       off in-loop (audit retrospettivo via verifier_audit.py)
  • Health mon.    on (per halt automatico se un altro provider è broken)

Outputs: each cell → `reboot/runs/al/multi_llm_bench/<provider>_drift_anchored_B_k20_seed2025/`
                     (full oracle_cache, rounds.jsonl, eval, summary.json)

After the bench completes, downstream pipeline:
  step3_al/verifier_audit.py --multi    →  per-model audit CSVs
  step3_al/halluc_correlation.py        →  hallucination↔correctness tables (TODO)
  step3_al/extract_anomalies.py         →  CoT extraction per paper (TODO)

Usage:
  # all available providers, excluding kimi
  python3 reboot/step5_smallbench/run_multi_llm_bench.py

  # explicit subset
  python3 reboot/step5_smallbench/run_multi_llm_bench.py \\
      --models deepseek_v4,gpt_5_4,mistral_large

  # dry-run to see commands without launching
  python3 reboot/step5_smallbench/run_multi_llm_bench.py --dry-run

  # force re-run even if summary.json exists
  python3 reboot/step5_smallbench/run_multi_llm_bench.py --force
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(THIS.parent))
from llm_providers import PROVIDERS, available_providers  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multi_llm_bench")

# Canonical Step-3 driver — already supports --provider after the patch.
RUN_AL = REBOOT / "step3_al" / "run_al.py"

# Default exclude list. Calibrated against the K2.6 ablation: health monitor
# halted Kimi at R2 with parse_ok=32% + mono-class 100%. Re-running is waste.
DEFAULT_EXCLUDE = {"kimi_k2_6"}

# Where each cell's artifacts land. Nested under "multi_llm_bench/" so they
# don't collide with canonical seed runs.
BENCH_TAG = "multi_llm_bench"


def run_id_for(*, provider: str, k: int, seed: int) -> str:
    """Run id => bench_tag/provider_strategy_prompt_k<K>_seed<S>.
    Slash makes the cell artifacts live under a shared parent dir."""
    return f"{BENCH_TAG}/{provider}_drift_anchored_B_k{k}_seed{seed}"


def launch_cell(
    *, provider: str, k: int, max_rounds: int, seed: int,
    dry_run: bool, force: bool,
) -> tuple[str, bool, float]:
    rid = run_id_for(provider=provider, k=k, seed=seed)
    run_dir = REBOOT / "runs" / "al" / rid
    summary = run_dir / "summary.json"
    if summary.exists() and not force:
        log.info(f"[skip] {rid} (summary.json exists; --force to re-run)")
        return rid, True, 0.0

    cmd = [
        sys.executable, str(RUN_AL),
        "--provider", provider,
        "--strategy", "drift_anchored",
        "--oracle", "llm",
        "--oracle-prompt", "B",
        "--threshold-preset", "periodic_audit",
        "--verifier-mode", "off",
        "--stream-order", "temporal",
        "--k", str(k),
        "--max-rounds", str(max_rounds),
        "--min-pool-for-trigger", str(k),
        "--seed", str(seed),
        "--run-id", rid,
    ]
    log.info(f"\n=== cell: {provider}  (k={k}, max_rounds={max_rounds}, "
             f"seed={seed}) ===")
    log.info(f"$ {' '.join(cmd)}")
    if dry_run:
        return rid, True, 0.0

    t0 = time.monotonic()
    rc = subprocess.run(cmd).returncode
    elapsed = time.monotonic() - t0
    ok = (rc == 0) and summary.exists()
    if not ok:
        log.error(f"[FAIL] {rid} (rc={rc}, elapsed={elapsed/60:.1f} min)")
    else:
        log.info(f"[OK]   {rid}  elapsed={elapsed/60:.1f} min")
    return rid, ok, elapsed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Full-PP multi-LLM benchmark — canonical Step-3 pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", default="auto",
                   help="comma-separated providers, or 'auto' for all "
                        "registered providers whose creds are present, "
                        "minus --exclude")
    p.add_argument("--exclude", default=",".join(sorted(DEFAULT_EXCLUDE)),
                   help="comma-separated providers to skip (auto only)")
    p.add_argument("--k", type=int, default=20,
                   help="labels per AL round (pool size)")
    p.add_argument("--max-rounds", type=int, default=25,
                   help="max AL rounds; 25 matches Step-3 canonical "
                        "on periodic_every=150 over 3821-row PP_stream")
    p.add_argument("--seed", type=int, default=2025,
                   help="single seed; multi-seed is a separate experiment")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands without launching")
    p.add_argument("--force", action="store_true",
                   help="re-run cells even if their summary.json exists")
    args = p.parse_args()

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    if args.models == "auto":
        providers = [m for m in available_providers() if m not in exclude]
    else:
        providers = [s.strip() for s in args.models.split(",") if s.strip()]
        unknown = [m for m in providers if m not in PROVIDERS]
        if unknown:
            raise SystemExit(f"unknown providers: {unknown}; "
                             f"registry={list(PROVIDERS)}")

    if not providers:
        raise SystemExit(
            "no providers selected. Available with creds: "
            f"{available_providers()}; after exclude={exclude}: none. "
            "Run `python3 reboot/step5_smallbench/llm_providers.py` to diagnose."
        )

    log.info("=" * 60)
    log.info("Multi-LLM bench plan")
    log.info("=" * 60)
    log.info(f"  providers (n={len(providers)}): {providers}")
    log.info(f"  excluded:                 {sorted(exclude)}")
    log.info(f"  strategy:                 drift_anchored")
    log.info(f"  prompt:                   B")
    log.info(f"  threshold preset:         periodic_audit (every 150 rows)")
    log.info(f"  verifier mode:            off (audit post-hoc)")
    log.info(f"  health monitor:           ON")
    log.info(f"  K per round:              {args.k}")
    log.info(f"  max rounds:               {args.max_rounds}")
    log.info(f"  seed:                     {args.seed}")
    log.info(f"  expected calls/provider:  ~{args.k * args.max_rounds}")
    log.info(f"  expected total calls:     ~{args.k * args.max_rounds * len(providers)}")
    log.info(f"  output root:              runs/al/{BENCH_TAG}/")
    log.info("=" * 60)

    results: list[tuple[str, bool, float]] = []
    for provider in providers:
        results.append(launch_cell(
            provider=provider, k=args.k, max_rounds=args.max_rounds,
            seed=args.seed, dry_run=args.dry_run, force=args.force,
        ))

    print()
    log.info("=" * 60)
    log.info("Bench summary")
    log.info("=" * 60)
    n_ok = sum(1 for _, ok, _ in results if ok)
    total_min = sum(e for _, _, e in results) / 60.0
    log.info(f"  cells OK: {n_ok}/{len(results)}  "
             f"total wall-clock: {total_min:.1f} min")
    for rid, ok, elapsed in results:
        mark = "✓" if ok else "✗"
        log.info(f"  {mark} {rid}  ({elapsed/60:.1f} min)")

    if n_ok < len(results):
        log.warning("Some cells failed — see logs above. Re-run with --force "
                    "after fixing the underlying issue.")

    log.info("")
    log.info("Next steps (post-hoc analysis, no LLM calls):")
    log.info("  1. python3 reboot/step3_al/verifier_audit.py --multi  "
             "# per-model hallucination + trust audit")
    log.info("  2. python3 reboot/step3_al/halluc_correlation.py     "
             "# cross-model correlation tables    (TODO)")
    log.info("  3. python3 reboot/step3_al/extract_anomalies.py      "
             "# CoT extraction per paper           (TODO)")


if __name__ == "__main__":
    main()
