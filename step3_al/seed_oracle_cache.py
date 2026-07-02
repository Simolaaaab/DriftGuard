"""seed_oracle_cache.py — Pre-populate a run's oracle cache from the union.

CachedOracle (oracle.py) only resolves labels from (1) the Step-2 cache and
(2) *this run's own* oracle_cache/<variant>/ dir. It deliberately does NOT read
other runs' caches. So a fresh run re-queries the LLM for every selected sample
even though an identical variant-B label already exists in some sibling run —
correct, but slow and budget-wasteful.

This helper hardlinks the union of every existing variant-B cache file into a
target run's oracle_cache/B/ dir *before* the run starts. The run then resolves
those samples from its Step-3 cache (no API call). Hardlinks share inodes, cost
nothing, and are never mutated (CachedOracle only writes on a miss), so this is
safe and reversible (delete the run dir to undo).

Usage:
    # seed one or more run dirs (created if absent):
    python3 reboot/step3_al/seed_oracle_cache.py --runs A,B,C
    # or all four LAR sweep runs at once (default):
    python3 reboot/step3_al/seed_oracle_cache.py
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
AL_RUNS_ROOT = REBOOT / "runs" / "al"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("seed_oracle_cache")

DEFAULT_RUNS = [
    "drift_anchored_random_lar_w100_B_seed2025_full_snap",
    "drift_anchored_random_lar_w200_B_seed2025_full_snap",
    "drift_anchored_kl_guided_lar_w100_B_seed2025_full_snap",
    "drift_anchored_kl_guided_lar_w200_B_seed2025_full_snap",
]


def build_union(variant: str = "B") -> dict[int, Path]:
    """Map sample_id → a source cache file, unioned over all runs.

    Skips the target runs themselves implicitly (they may be empty). On
    collision keeps the first seen (all B labels are interchangeable)."""
    union: dict[int, Path] = {}
    pat = str(AL_RUNS_ROOT / "*" / "oracle_cache" / variant / "*.json.gz")
    for p in glob.glob(pat):
        name = Path(p).name
        try:
            sid = int(name.split(".")[0])
        except ValueError:
            continue
        # Also pick up the backfill payloads (full indicators, needed by the
        # verifier) produced by lar_labels.py --backfill.
        union.setdefault(sid, Path(p))
    # Include the backfill run explicitly in case its dir name doesn't match
    # the glob depth (it does, but be safe).
    bf = AL_RUNS_ROOT / "_stream_llm_labels" / "_backfill_run" / "oracle_cache" / variant
    if bf.exists():
        for p in bf.glob("*.json.gz"):
            try:
                sid = int(p.name.split(".")[0])
            except ValueError:
                continue
            union.setdefault(sid, p)
    return union


def seed_run(run_id: str, union: dict[int, Path], variant: str = "B") -> tuple[int, int]:
    """Hardlink union files into <run_id>/oracle_cache/<variant>/. Returns
    (n_linked, n_already_present)."""
    dst_dir = AL_RUNS_ROOT / run_id / "oracle_cache" / variant
    dst_dir.mkdir(parents=True, exist_ok=True)
    linked = present = 0
    for sid, src in union.items():
        dst = dst_dir / f"{sid}.json.gz"
        if dst.exists():
            present += 1
            continue
        try:
            os.link(src, dst)          # hardlink (instant, shared inode)
            linked += 1
        except OSError:
            # Cross-device or race → fall back to copy.
            import shutil
            shutil.copy2(src, dst)
            linked += 1
    return linked, present


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default=None,
                   help="comma-separated run ids to seed "
                        "(default: the 4 LAR sweep runs)")
    p.add_argument("--variant", default="B")
    args = p.parse_args()

    runs = (args.runs.split(",") if args.runs else DEFAULT_RUNS)
    union = build_union(args.variant)
    log.info(f"union of variant-{args.variant} labels: {len(union)} samples")
    for rid in runs:
        rid = rid.strip()
        if not rid:
            continue
        if (AL_RUNS_ROOT / rid / "summary.json").exists():
            log.info(f"[{rid}] already finished (summary.json present) — "
                     f"seeding skipped, results stand")
            continue
        linked, present = seed_run(rid, union, args.variant)
        log.info(f"[{rid}] hardlinked {linked} labels "
                 f"({present} already present) into oracle_cache/{args.variant}/")
    log.info("done. Re-run launch_lar_sweep.sh — pre-seeded runs hit cache, no API.")


if __name__ == "__main__":
    main()
