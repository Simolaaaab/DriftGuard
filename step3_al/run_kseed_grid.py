"""run_kseed_grid.py — Fill the 8x5 selection-strategy K-seed grid (ADDITIVE).

Produces the missing canonical-config cells for the strategy significance band:
selectors x seeds {2025..2029}, each at the exact canonical config of the
existing strategy-table runs (oracle=llm DeepSeek-V4-Flash, prompt B, periodic
trigger every 150 rows, K=50, ~25 rounds, full-union retrain, frozen scaler,
--verifier-mode off --disable-health-monitor).

Idempotency (HARD RULE 1): a cell is DONE if its canonical run dir exists with
summary.json + eval_per_round.csv + a completed final round. DONE cells are
NEVER re-run, overwritten, or deleted. The seed-2025 cells and the
drift_anchored/margin 2026-2029 seeds already exist under their historical names
(see CANON_MAP) and are skipped — this protects the paper's existing bootstrap.

Cache-first (HARD RULE 2): before running a cell, its oracle_cache/B is seeded
from the union of all existing DeepSeek-V4-Flash variant-B caches (non-DeepSeek
provider caches excluded), so only genuinely novel samples hit the API. Seeding
only writes into the NEW cell's own dir.

Resumable: re-running the script skips DONE cells and continues. run_al.py wipes
only rounds.jsonl/eval on (re)start, never oracle_cache, so an interrupted cell
re-runs cheaply against its accumulated + seeded cache.

Usage:
    # coverage + cache pre-seed only (no API, safe without credentials):
    python3 reboot/step3_al/run_kseed_grid.py --plan-only
    # full fill (needs AZURE_*/DEEPSEEK_* credentials):
    python3 reboot/step3_al/run_kseed_grid.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
AL = REBOOT / "runs" / "al"

SELECTORS = ["drift_anchored", "margin", "qbc", "core_set", "badge",
             "hybrid", "random", "pure_random_stream"]
SEEDS = [2025, 2026, 2027, 2028, 2029]

# Existing canonical-config runs under historical names → treated as DONE and
# reused verbatim. Everything not listed here uses the target grid name
# "<selector>_llm_periodic_temporal_B_seed<seed>".
CANON_MAP = {
    ("drift_anchored", 2025): "drift_anchored_llm_periodic_temporal_B",
    ("drift_anchored", 2026): "drift_anchored_B_seed2026",
    ("drift_anchored", 2027): "drift_anchored_B_seed2027",
    ("drift_anchored", 2028): "drift_anchored_B_seed2028",
    ("drift_anchored", 2029): "drift_anchored_B_seed2029",
    ("margin", 2025): "margin_llm_periodic_temporal_B",
    ("margin", 2026): "margin_B_seed2026",
    ("margin", 2027): "margin_B_seed2027",
    ("margin", 2028): "margin_B_seed2028",
    ("margin", 2029): "margin_B_seed2029",
    ("qbc", 2025): "qbc_llm_periodic_temporal_B",
    ("core_set", 2025): "core_set_llm_periodic_temporal_B",
    ("badge", 2025): "badge_llm_periodic_temporal_B",
    ("hybrid", 2025): "hybrid_llm_periodic_temporal_B",
    ("random", 2025): "random_llm_periodic_temporal_B",
    ("pure_random_stream", 2025): "pure_random_stream_llm_periodic_temporal_B_seed2025",
}

NON_DEEPSEEK = re.compile(r'(kimi|gpt_5|gpt5|grok|mistral|llama)', re.I)
SKIP_DIRS = re.compile(r'(_costprobe|_smoke|_codecheck|_kseedtmp)')


def target_dir(sel: str, seed: int) -> Path:
    """Canonical run dir for a cell: historical name if it exists, else the
    grid target name."""
    if (sel, seed) in CANON_MAP:
        return AL / CANON_MAP[(sel, seed)]
    return AL / f"{sel}_llm_periodic_temporal_B_seed{seed}"


def is_done(d: Path) -> bool:
    if not (d / "summary.json").exists() or not (d / "eval_per_round.csv").exists():
        return False
    try:
        sm = json.loads((d / "summary.json").read_text())
        return sm.get("n_rounds_completed", 0) >= 1 and sm.get("final_eval") is not None
    except Exception:
        return False


def build_deepseek_union() -> dict[int, str]:
    ids: dict[int, str] = {}
    for bdir in AL.rglob("oracle_cache/B"):
        rp = str(bdir.relative_to(AL))
        if SKIP_DIRS.search(rp) or NON_DEEPSEEK.search(rp):
            continue
        for f in bdir.glob("*.json.gz"):
            try:
                sid = int(f.name.split('.')[0])
            except ValueError:
                continue
            if sid in ids:
                continue
            try:
                with gzip.open(f, "rt") as fh:
                    if json.load(fh).get("ok"):
                        ids[sid] = str(f)
            except Exception:
                pass
    return ids


def seed_cache(cell_dir: Path, union: dict[int, str]) -> int:
    dst = cell_dir / "oracle_cache" / "B"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for sid, src in union.items():
        tgt = dst / f"{sid}.json.gz"
        if not tgt.exists():
            shutil.copy2(src, tgt)
            n += 1
    return n


def have_credentials() -> bool:
    return bool(
        (os.environ.get("AZURE_API_KEY") and os.environ.get("AZURE_BASE_URL"))
        or os.environ.get("DEEPSEEK_API_KEY")
    )


def run_cell(sel: str, seed: int) -> int:
    run_id = f"{sel}_llm_periodic_temporal_B_seed{seed}"
    cmd = [
        sys.executable, str(REBOOT / "step3_al" / "run_al.py"),
        "--strategy", sel,
        "--oracle", "llm",
        "--threshold-preset", "periodic_audit",
        "--oracle-prompt", "B",
        "--max-rounds", "30",
        "--seed", str(seed),
        "--verifier-mode", "off",
        "--disable-health-monitor",
        "--run-id", run_id,
    ]
    print(f"  RUN {run_id}: {' '.join(cmd[3:])}")
    return subprocess.call(cmd, cwd=str(REBOOT.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true",
                    help="print coverage + pre-seed caches; do not call the API")
    args = ap.parse_args()

    union = build_deepseek_union()
    print(f"DeepSeek-V4-Flash variant-B cache union: {len(union)} usable sample_ids\n")

    print(f"{'selector':<20}" + "".join(f"{s:>7}" for s in SEEDS))
    missing: list[tuple[str, int]] = []
    for sel in SELECTORS:
        marks = []
        for seed in SEEDS:
            d = target_dir(sel, seed)
            if is_done(d):
                marks.append("  DONE")
            else:
                marks.append("  ----")
                missing.append((sel, seed))
        print(f"{sel:<20}" + "".join(f"{m:>7}" for m in marks))
    print(f"\nMISSING cells: {len(missing)}")
    for sel, seed in missing:
        print(f"  {sel}_llm_periodic_temporal_B_seed{seed}")

    if not missing:
        print("\nGrid is FULL — nothing to run.")
        return

    # Pre-seed caches for every missing cell (safe without credentials).
    print("\nPre-seeding caches for missing cells:")
    for sel, seed in missing:
        cell = target_dir(sel, seed)
        n = seed_cache(cell, union)
        print(f"  {cell.name}: +{n} cache files (total "
              f"{len(list((cell/'oracle_cache'/'B').glob('*.json.gz')))})")

    if args.plan_only:
        print("\n--plan-only: caches seeded; no API calls made.")
        return

    if not have_credentials():
        print("\nNO LLM CREDENTIALS — refusing to run cells (would produce "
              "degraded runs with dropped labels on cache misses). Export "
              "AZURE_API_KEY+AZURE_BASE_URL+AZURE_MODEL=DeepSeek-V4-Flash (or "
              "DEEPSEEK_API_KEY) and re-run. Caches are already seeded.")
        sys.exit(2)

    print("\nFilling missing cells (skipping DONE):")
    for sel, seed in missing:
        d = target_dir(sel, seed)
        if is_done(d):     # double-check (resumability across crashes)
            print(f"  SKIP {d.name} (became DONE)")
            continue
        rc = run_cell(sel, seed)
        if rc != 0:
            print(f"  !! {sel} seed{seed} exited {rc}; continuing")
            continue
        sm = json.loads((target_dir(sel, seed) / "summary.json").read_text())
        tel = sm.get("oracle_telemetry", {})
        print(f"  done {sel} seed{seed}: fresh_calls={tel.get('n_llm_calls')} "
              f"step3_hits={tel.get('n_step3_hits')} step2_hits={tel.get('n_step2_hits')}")

    print("\nGrid fill pass complete.")


if __name__ == "__main__":
    main()
