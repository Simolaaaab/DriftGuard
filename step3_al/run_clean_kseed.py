"""run_clean_kseed.py — Leakage-free K-seed band rebuild (REAL DeepSeek labels only).

Integrity model (vs the voided rebuild):
  * ONE clean SHARED cache, built by admitting an existing cache entry ONLY if its
    in-file `source == "llm"` (and ok==True, and no `noise_audit`/gt marker).
    Filtered by the IN-FILE source field, never by directory name. Every
    `gt_dryrun` / noise / ambiguous entry is rejected.
  * Every fresh cell's `oracle_cache/B` is a SYMLINK to the shared clean cache, so
    real LLM labels are reused across runs (valid — a real label is valid whoever
    requested it) and fresh `source=="llm"` calls are written back for later runs.
  * No run in this batch uses oracle_mode==gt_dryrun, so nothing can write a GT
    label into the shared cache. A hard guard re-scans the shared cache and STOPS
    if any non-llm entry is ever present.
  * Per-cell verification: 0 consumed labels may be gt_dryrun/noise; else FAIL+STOP.

Canonical config (verbatim from drift_anchored_llm_periodic_temporal_B): oracle=llm,
prompt B, periodic/150, K=50, max-rounds 30, verifier off, health monitor off,
use_step2_cache default (the clean May lineage used it). Only selector+seed vary.

Reuse policy: the seven confirmed-clean May strategy-table seed-2025 runs
(<sel>_llm_periodic_temporal_B) are referenced as the seed-2025 cell and NOT re-run.
Fresh cells: seeds 2026-2029 for those seven selectors + all five seeds for
pure_random_stream. New dirs: <sel>_llm_periodic_temporal_CLEAN_seed<seed>/ — existing
dirs are never touched.

Usage:
  python3 reboot/step3_al/run_clean_kseed.py --plan           # coverage + commands, no writes
  python3 reboot/step3_al/run_clean_kseed.py --build-cache-only  # build+guard clean cache, NO LLM
  python3 reboot/step3_al/run_clean_kseed.py                  # full batch (needs creds) + aggregate
"""
from __future__ import annotations

import argparse, csv, gzip, json, math, os, subprocess, sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
AL = REBOOT / "runs" / "al"
SHARED = AL / "_clean_shared_cache" / "oracle_cache" / "B"   # the live clean cache

SELECTORS = ["drift_anchored","margin","qbc","core_set","badge","hybrid","random","pure_random_stream"]
SEEDS = [2025,2026,2027,2028,2029]
ANCHOR = {s: f"{s}_llm_periodic_temporal_B" for s in SELECTORS if s != "pure_random_stream"}  # seed-2025 reuse


def clean_dir(sel, seed): return AL / f"{sel}_llm_periodic_temporal_CLEAN_seed{seed}"

def cell_dir(sel, seed):
    """Resolved dir used by aggregation: May anchor for the 7 selectors' seed-2025,
    else the fresh CLEAN dir."""
    if seed == 2025 and sel in ANCHOR:
        return AL / ANCHOR[sel]
    return clean_dir(sel, seed)

def is_fresh_cell(sel, seed):
    return not (seed == 2025 and sel in ANCHOR)

def is_done(d):
    if not (d/"summary.json").exists() or not (d/"eval_per_round.csv").exists(): return False
    try:
        sm = json.loads((d/"summary.json").read_text())
        return sm.get("n_rounds_completed",0) >= 1 and sm.get("final_eval") is not None
    except Exception:
        return False

def _is_gt(p):
    return ("gt_dryrun" in str(p.get("source",""))) or ("noise_audit" in p)

# ---------------------------------------------------------------- clean cache
def build_clean_cache():
    SHARED.mkdir(parents=True, exist_ok=True)
    admit = reject_gt = reject_other = 0
    origin = {}
    for f in AL.rglob("oracle_cache/B/**/*.json.gz"):
        if SHARED in f.parents: continue          # never read our own shared cache
        try:
            with gzip.open(f,"rt") as fh: p = json.load(fh)
        except Exception:
            reject_other += 1; continue
        if not p.get("ok"): reject_other += 1; continue
        if _is_gt(p): reject_gt += 1; continue
        if str(p.get("source","")) != "llm": reject_other += 1; continue
        try: sid = int(f.name.split('.')[0])
        except ValueError: reject_other += 1; continue
        if sid in origin: continue                # first clean-llm wins
        tgt = SHARED / f"{sid}.json.gz"
        if not tgt.exists():
            import shutil; shutil.copy2(f, tgt)
        origin[sid] = str(f); admit += 1
    print(f"[clean-cache] admitted(source==llm): {admit}")
    print(f"[clean-cache] rejected as gt_dryrun/noise: {reject_gt}")
    print(f"[clean-cache] rejected other (not-ok / source!=llm / unreadable): {reject_other}")
    # HARD GUARD (rule 1d): shared cache must be 100% source==llm
    bad = 0
    for f in SHARED.glob("*.json.gz"):
        with gzip.open(f,"rt") as fh: p = json.load(fh)
        if str(p.get("source","")) != "llm" or _is_gt(p):
            print(f"[GUARD-FAIL] non-llm entry in shared cache: sid={f.name} "
                  f"source={p.get('source')!r} origin={origin.get(int(f.name.split('.')[0]),'?')}")
            bad += 1
    if bad:
        sys.exit(f"ABORT: {bad} non-llm entries in shared cache (rule 1d).")
    print(f"[clean-cache] GUARD OK — {len(list(SHARED.glob('*.json.gz')))} entries, all source==llm")

def link_shared(run_dir):
    (run_dir/"oracle_cache").mkdir(parents=True, exist_ok=True)
    b = run_dir/"oracle_cache"/"B"
    if b.is_symlink():
        if b.resolve() != SHARED.resolve(): sys.exit(f"ABORT: {b} points elsewhere.")
        return
    if b.exists(): sys.exit(f"ABORT: {b} exists and is not a symlink — refuse to clobber.")
    b.symlink_to(SHARED.resolve())

# ---------------------------------------------------------------- verify
def verify_clean(sel, seed):
    d = clean_dir(sel, seed)
    rr = [json.loads(l) for l in (d/"rounds.jsonl").read_text().splitlines() if l.strip()]
    rr = [r for r in rr if "n_queried" in r]
    q = [sid for r in rr for sid in r["queried_sample_ids"]]
    n_gt = 0
    for sid in q:
        f = SHARED / f"{sid}.json.gz"
        if not f.exists(): continue
        with gzip.open(f,"rt") as fh: p = json.load(fh)
        if _is_gt(p): n_gt += 1
    oacc = sum(r["oracle_gt_agreement"] for r in rr)/sum(r["n_queried"] for r in rr)
    return n_gt, oacc

def run_cell(sel, seed):
    rid = clean_dir(sel, seed).name
    d = clean_dir(sel, seed)
    if is_done(d):
        n_gt, oacc = verify_clean(sel, seed)
        if n_gt == 0:
            print(f"[skip] {rid} already done & clean (oacc={oacc:.4f})"); return True
        sys.exit(f"ABORT: {rid} previously done but consumed {n_gt} GT labels.")
    link_shared(d)
    cmd = [sys.executable, str(REBOOT/"step3_al"/"run_al.py"),
           "--strategy", sel, "--oracle", "llm",
           "--threshold-preset", "periodic_audit", "--oracle-prompt", "B",
           "--max-rounds", "30", "--seed", str(seed),
           "--verifier-mode", "off", "--disable-health-monitor",
           "--run-id", rid]
    print(f"[run] {rid}: {' '.join(cmd[3:])}")
    rc = subprocess.call(cmd, cwd=str(REBOOT.parent))
    if rc != 0: sys.exit(f"ABORT: {rid} exited {rc}.")
    n_gt, oacc = verify_clean(sel, seed)
    last = list(csv.DictReader(open(d/"eval_per_round.csv")))[-1]
    pp = float(last["pp_test_f1_weighted"]); dpm = _dp_macro(last)
    tel = json.loads((d/"summary.json").read_text())["oracle_telemetry"]
    flag = ""
    if n_gt > 0: sys.exit(f"ABORT: {rid} consumed {n_gt} GT labels (rule 4).")
    if oacc > 0.90: flag = "  <<WARN: oacc>0.90 — verify (gt=0 confirmed clean)"
    print(f"  -> dir={rid} fresh_calls={tel.get('n_llm_calls')} PP={pp:.4f} "
          f"DPmac={dpm:.4f} oracle_acc={oacc:.4f} GT_consumed={n_gt} [CLEAN]{flag}")
    return True

# ---------------------------------------------------------------- aggregate
def _dp_macro(last):
    def f1(p,r): p,r=float(p),float(r); return 0.0 if p+r==0 else 2*p*r/(p+r)
    return (float(last["dp_probe_f1_phish"]) + f1(last["dp_probe_precision_benign"], last["dp_probe_recall_benign"]))/2

def _metrics(d):
    last = list(csv.DictReader(open(d/"eval_per_round.csv")))[-1]
    return float(last["pp_test_f1_weighted"]), _dp_macro(last)

def _band(vals, nb=10000, seed=12345):
    a=np.asarray(vals,float); n=len(a); m=float(a.mean())
    sd=float(a.std(ddof=1)) if n>1 else 0.0; se=sd/math.sqrt(n) if n>1 else 0.0
    rng=np.random.default_rng(seed); bs=rng.choice(a,size=(nb,n),replace=True).mean(1)
    return m,(m-1.96*se,m+1.96*se),(float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))

def aggregate():
    cells={}
    for sel in SELECTORS:
        for seed in SEEDS:
            d=cell_dir(sel,seed)
            if is_done(d): cells[(sel,seed)]=_metrics(d)
    print("\n=== RAW 8x5 (PP / DP macro) — CLEAN lineage ===")
    print("selector            " + " ".join(f"{s:>15}" for s in SEEDS))
    for sel in SELECTORS:
        row=" ".join((f"{cells[(sel,s)][0]:.4f}/{cells[(sel,s)][1]:.4f}" if (sel,s) in cells else "      —/—      ") for s in SEEDS)
        print(f"{sel:<20}{row}")
    bands={}
    for sel in SELECTORS:
        pp=[cells[(sel,s)][0] for s in SEEDS if (sel,s) in cells]
        dp=[cells[(sel,s)][1] for s in SEEDS if (sel,s) in cells]
        if len(pp)==5: bands[sel]=(_band(pp),_band(dp),pp,dp)
    print("\n=== PER-SELECTOR BAND (sorted by PP mean) ===")
    print(f"{'selector':<20}{'PP mean':<9}{'PP 95%CI norm':<22}{'PP 95%CI boot':<22}{'DP mean':<9}{'DP 95%CI norm'}")
    for sel in sorted(bands,key=lambda s:-bands[s][0][0]):
        (ppm,ppn,ppb),(dpm,dpn,dpb),_,_=bands[sel]
        print(f"{sel:<20}{ppm:<9.4f}[{ppn[0]:.4f},{ppn[1]:.4f}]      [{ppb[0]:.4f},{ppb[1]:.4f}]      {dpm:<9.4f}[{dpn[0]:.4f},{dpn[1]:.4f}]")
    def ov(a,b,i): (alo,ahi)=bands[a][i][1]; (blo,bhi)=bands[b][i][1]; return not (ahi<blo or bhi<alo)
    inf=[s for s in bands if s not in ("random","pure_random_stream")]
    best=max(inf,key=lambda s:bands[s][0][0]) if inf else None
    print("\n=== PAIRWISE (95% normal-CI overlap) ===")
    for i,lab in ((0,"PP"),(1,"DP macro")):
        print(f" [{lab}]")
        for a,b in [("pure_random_stream","drift_anchored"),("pure_random_stream","qbc"),("pure_random_stream","hybrid")]+([(best,"random")] if best else []):
            if a in bands and b in bands:
                d=bands[a][i][0]-bands[b][i][0]
                print(f"   {a} vs {b}: Δ={d:+.4f}  {'OVERLAP→ns' if ov(a,b,i) else 'DISJOINT→sig'}")

# ---------------------------------------------------------------- main
def print_plan():
    print("REUSE-CLEAN-ANCHOR (seed-2025, NOT re-run):")
    for s in ANCHOR: print(f"  ({s},2025) -> {ANCHOR[s]}")
    print("FRESH-RUN (clean shared cache):")
    for sel in SELECTORS:
        for seed in SEEDS:
            if is_fresh_cell(sel,seed): print(f"  ({sel},{seed}) -> {clean_dir(sel,seed).name}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",action="store_true")
    ap.add_argument("--build-cache-only",action="store_true")
    a=ap.parse_args()
    if a.plan: print_plan(); return
    build_clean_cache()
    if a.build_cache_only:
        print("[done] clean cache built + guarded; no runs executed."); return
    if not (os.environ.get("AZURE_API_KEY") and os.environ.get("AZURE_BASE_URL")) and not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("NO LLM CREDENTIALS — export AZURE_API_KEY+AZURE_BASE_URL+AZURE_MODEL=DeepSeek-V4-Flash (or DEEPSEEK_API_KEY).")
    fresh=[(sel,seed) for sel in SELECTORS for seed in SEEDS if is_fresh_cell(sel,seed)]
    print(f"\n[batch] {len(fresh)} fresh cells (sequential, shared clean cache)")
    for sel,seed in fresh: run_cell(sel,seed)
    aggregate()
    print("\nCLEAN K-SEED REBUILD DONE.")

if __name__=="__main__":
    main()
