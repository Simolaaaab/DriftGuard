"""run_alpha_sweep.py — drift_anchored alpha sensitivity sweep (clean labels only).

Sweeps alpha in {0.0, 0.25, 0.5, 1.0, 2.0} x seeds {2025..2029}, reusing existing
clean columns and running only the missing ones:
  * alpha=0.0  == margin selector (proven: the drift term collapses to 1, so
    s(x)=1/margin ranks identically to pure margin). REUSE the clean margin cells.
  * alpha=0.5  == canonical drift_anchored. REUSE the clean drift_anchored cells.
  * alpha in {0.25,1.0,2.0} -> 15 fresh runs to drift_anchored_alpha<val>_CLEAN_seed<seed>/.

Integrity (post-leakage):
  * Each fresh cell's oracle_cache/B is a SYMLINK to _clean_shared_cache (source=="llm"
    only). Fresh source=="llm" calls are written back and reused. No gt_dryrun dir is
    ever read; no oracle_mode==gt_dryrun run is launched.
  * After each run: verify 0 consumed labels are gt_dryrun and oracle accuracy is in
    the real-LLM band (0.65-0.86). Any GT label or oacc in 0.86-0.94 -> STOP & flag.
  * Hard guard: if the shared cache ever contains a non-llm entry, abort.

Canonical config verbatim from drift_anchored_llm_periodic_temporal_B; only alpha+seed
vary. Idempotent/resumable (skips done+clean cells).

Usage:
  python3 reboot/step3_al/run_alpha_sweep.py --plan          # coverage only
  python3 reboot/step3_al/run_alpha_sweep.py                 # run missing + aggregate (needs creds)
  python3 reboot/step3_al/run_alpha_sweep.py --aggregate-only# print bands from finished cells
"""
from __future__ import annotations

import argparse, csv, gzip, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve(); REBOOT = THIS.parents[1]; AL = REBOOT/"runs"/"al"
SHARED = AL/"_clean_shared_cache"/"oracle_cache"/"B"
SEEDS = [2025,2026,2027,2028,2029]
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]

# Reused clean columns (no re-run).
REUSE = {
    0.0: {2025:"margin_llm_periodic_temporal_B",
          **{s:f"margin_llm_periodic_temporal_CLEAN_seed{s}" for s in (2026,2027,2028,2029)}},
    0.5: {2025:"drift_anchored_llm_periodic_temporal_B",
          **{s:f"drift_anchored_llm_periodic_temporal_CLEAN_seed{s}" for s in (2026,2027,2028,2029)}},
}

def fresh_dir(a, seed): return AL/f"drift_anchored_alpha{a}_CLEAN_seed{seed}"
def cell_dir(a, seed):
    if a in REUSE: return AL/REUSE[a][seed]
    return fresh_dir(a, seed)

def is_done(d):
    if not (d/"summary.json").exists() or not (d/"eval_per_round.csv").exists(): return False
    try:
        sm=json.loads((d/"summary.json").read_text())
        return sm.get("n_rounds_completed",0)>=1 and sm.get("final_eval") is not None
    except Exception: return False

def _src(sid):
    f=SHARED/f"{sid}.json.gz"
    if not f.exists(): return None
    with gzip.open(f,"rt") as fh: return str(json.load(fh).get("source",""))

def guard_shared():
    bad=0
    for f in SHARED.glob("*.json.gz"):
        with gzip.open(f,"rt") as fh: p=json.load(fh)
        if str(p.get("source",""))!="llm" or "noise_audit" in p:
            print(f"[GUARD-FAIL] non-llm entry sid={f.name} source={p.get('source')!r}"); bad+=1
    if bad: sys.exit(f"ABORT: shared cache has {bad} non-llm entries.")

def verify(d):
    rr=[json.loads(l) for l in (d/"rounds.jsonl").read_text().splitlines() if l.strip()]
    rr=[r for r in rr if "n_queried" in r]
    q=[s for r in rr for s in r["queried_sample_ids"]]
    gt=sum(1 for s in q if (_src(s) or "").find("gt_dryrun")>=0)
    oacc=sum(r["oracle_gt_agreement"] for r in rr)/sum(r["n_queried"] for r in rr)
    return gt, oacc

def link_shared(d):
    (d/"oracle_cache").mkdir(parents=True, exist_ok=True)
    b=d/"oracle_cache"/"B"
    if b.is_symlink():
        if b.resolve()!=SHARED.resolve(): sys.exit(f"ABORT: {b} points elsewhere.")
        return
    if b.exists(): sys.exit(f"ABORT: {b} exists, not a symlink.")
    b.symlink_to(SHARED.resolve())

def run_cell(a, seed):
    d=fresh_dir(a,seed); rid=d.name
    if is_done(d):
        gt,oacc=verify(d)
        if gt==0 and oacc<=0.86: print(f"[skip] {rid} done & clean (oacc={oacc:.4f})"); return
        sys.exit(f"ABORT: {rid} flagged (gt={gt}, oacc={oacc:.4f}).")
    link_shared(d)
    cmd=[sys.executable,str(REBOOT/"step3_al"/"run_al.py"),
         "--strategy","drift_anchored","--oracle","llm",
         "--threshold-preset","periodic_audit","--oracle-prompt","B",
         "--max-rounds","30","--seed",str(seed),
         "--drift-alpha",str(a),
         "--verifier-mode","off","--disable-health-monitor","--run-id",rid]
    print(f"[run] {rid}: alpha={a} seed={seed}")
    if subprocess.call(cmd, cwd=str(REBOOT.parent))!=0: sys.exit(f"ABORT: {rid} failed.")
    guard_shared()
    gt,oacc=verify(d)
    if gt>0: sys.exit(f"ABORT: {rid} consumed {gt} GT labels.")
    if oacc>0.86: sys.exit(f"ABORT: {rid} oacc={oacc:.4f} in contaminated range (>0.86).")
    last=list(csv.DictReader(open(d/"eval_per_round.csv")))[-1]
    print(f"  -> {rid} PP={float(last['pp_test_f1_weighted']):.4f} oacc={oacc:.4f} GT=0 [CLEAN]")

def _metrics(d):
    last=list(csv.DictReader(open(d/"eval_per_round.csv")))[-1]
    def f1(p,r): p,r=float(p),float(r); return 0.0 if p+r==0 else 2*p*r/(p+r)
    dpm=(float(last["dp_probe_f1_phish"])+f1(last["dp_probe_precision_benign"],last["dp_probe_recall_benign"]))/2
    return float(last["pp_test_f1_weighted"]), dpm

def _band(v,nb=10000):
    a=np.array(v,float); m=float(a.mean()); se=float(a.std(ddof=1))/math.sqrt(len(a))
    rng=np.random.default_rng(123); bs=rng.choice(a,size=(nb,len(a)),replace=True).mean(1)
    return m,(m-1.96*se,m+1.96*se),(float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))

def aggregate():
    print("\n=== ALPHA SWEEP — per-alpha band (5 seeds) ===")
    print(f"{'alpha':<8}{'n':<4}{'PP mean':<9}{'PP 95%CI norm':<22}{'DP mean':<9}{'DP 95%CI norm'}")
    rows={}
    for a in ALPHAS:
        pps=[]; dps=[]
        for s in SEEDS:
            d=cell_dir(a,s)
            if is_done(d): pp,dp=_metrics(d); pps.append(pp); dps.append(dp)
        if len(pps)==5:
            (pm,pn,_),(dm,dn,_)=_band(pps),_band(dps)
            rows[a]=(pm,dm)
            tag=" (==margin)" if a==0.0 else (" (canonical)" if a==0.5 else "")
            print(f"{a:<8}{len(pps):<4}{pm:<9.4f}[{pn[0]:.4f},{pn[1]:.4f}]      {dm:<9.4f}[{dn[0]:.4f},{dn[1]:.4f}]{tag}")
        else:
            print(f"{a:<8}{len(pps):<4}-- incomplete ({len(pps)}/5 seeds) --")
    if len(rows)>=3:
        pp_series=[rows[a][0] for a in ALPHAS if a in rows]
        dp_series=[rows[a][1] for a in ALPHAS if a in rows]
        pp_rng=max(pp_series)-min(pp_series); dp_rng=max(dp_series)-min(dp_series)
        print(f"\nRead: PP varies {pp_rng*100:.1f}pp across alpha "
              f"({'flat/robust' if pp_rng<0.02 else 'has structure'}); "
              f"DP varies {dp_rng*100:.1f}pp "
              f"({'flat' if dp_rng<0.02 else 'monotone/has optimum — inspect'}).")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",action="store_true")
    ap.add_argument("--aggregate-only",action="store_true")
    a=ap.parse_args()
    missing=[(al,s) for al in ALPHAS if al not in REUSE for s in SEEDS if not is_done(fresh_dir(al,s))]
    if a.plan:
        print("REUSE columns: alpha=0.0->margin clean, alpha=0.5->drift_anchored clean")
        print(f"FRESH cells ({len(missing)}):")
        for al,s in missing: print(f"  {fresh_dir(al,s).name}")
        return
    if a.aggregate_only: aggregate(); print("\nALPHA SWEEP DONE."); return
    guard_shared()
    if not (os.environ.get("AZURE_API_KEY") and os.environ.get("AZURE_BASE_URL")) and not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("NO LLM CREDENTIALS — export AZURE_*/DEEPSEEK_* and re-run. Reuse columns + plan are credential-free.")
    print(f"[batch] {len(missing)} fresh alpha cells (sequential, shared clean cache)")
    for al,s in missing: run_cell(al,s)
    aggregate(); print("\nALPHA SWEEP DONE.")

if __name__=="__main__": main()
