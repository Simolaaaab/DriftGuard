"""export_oracle_metrics.py — Dump per-sample DeepSeek metrics for the
best AL run (drift_anchored + prompt B) to a single CSV.

For each of the ~1250 samples that DeepSeek labelled during the run, we
record:
  round                AL round in which the sample was queried
  sample_id            PP sample id
  gt                   ground-truth label (0=benign, 1=phish)
  llm_label_raw        the string returned by the LLM ("phish"/"benign"/...)
  llm_label            0/1 normalisation (None if unparseable)
  confidence           LLM-reported confidence
  parse_ok             1 if the JSON parsed and label is in {0,1}
  agreement            1 if gt == llm_label, 0 otherwise (NaN if unparseable)
  error_type           "" | "FP" | "FN" (relative to gt)
  prompt_tokens        token count of the prompt
  completion_tokens    token count of the response
  latency_ms           per-call latency in ms
  phishing_type        LLM-reported category (legitimate / credential / ...)
  source               where the label came from in the cache tier

Usage:
    python3 reboot/step3_al/export_oracle_metrics.py \
        --run drift_anchored_llm_periodic_temporal_B \
        --variant B \
        --out reboot/runs/al/drift_anchored_llm_periodic_temporal_B/oracle_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REBOOT = HERE.parent
sys.path.insert(0, str(HERE))

from common import load_matrices  # noqa: E402


def _llm_to_int(s):
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("phish", "phishing", "1"):
        return 1
    if s in ("benign", "legitimate", "legit", "0"):
        return 0
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="drift_anchored_llm_periodic_temporal_B")
    ap.add_argument("--variant", default="B",
                    help="prompt variant subfolder under oracle_cache/")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: <run>/oracle_metrics.csv)")
    args = ap.parse_args()

    run_dir = REBOOT / "runs" / "al" / args.run
    rounds_path = run_dir / "rounds.jsonl"
    cache_dir = run_dir / "oracle_cache" / args.variant
    out_path = Path(args.out) if args.out else run_dir / "oracle_metrics.csv"

    if not rounds_path.exists():
        sys.exit(f"missing {rounds_path}")
    if not cache_dir.exists():
        sys.exit(f"missing {cache_dir}")

    pp = load_matrices()["pp"][["sample_id", "label"]]
    gt_by_id = dict(zip(pp["sample_id"].astype(int), pp["label"].astype(int)))

    rows_out = []
    n_missing = 0
    for line in rounds_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        round_num = rec["round"]
        for sid in rec.get("queried_sample_ids", []):
            sid = int(sid)
            p = cache_dir / f"{sid}.json.gz"
            if not p.exists():
                # Fall back to step-2 ablation cache (variant D), used for
                # samples seen first by step-2.
                p2 = (REBOOT / "runs" / "oracle_ablation" / "cache"
                      / "D_html_osint" / f"{sid}.json.gz")
                if p2.exists():
                    p = p2
                else:
                    n_missing += 1
                    rows_out.append({
                        "round": round_num, "sample_id": sid,
                        "gt": gt_by_id.get(sid),
                        "llm_label_raw": "", "llm_label": "",
                        "confidence": "", "parse_ok": 0, "agreement": "",
                        "error_type": "MISSING",
                        "prompt_tokens": "", "completion_tokens": "",
                        "latency_ms": "", "phishing_type": "",
                        "source": "missing",
                    })
                    continue
            try:
                with gzip.open(p, "rt") as f:
                    d = json.load(f)
            except Exception:
                rows_out.append({
                    "round": round_num, "sample_id": sid,
                    "gt": gt_by_id.get(sid),
                    "llm_label_raw": "", "llm_label": "",
                    "confidence": "", "parse_ok": 0, "agreement": "",
                    "error_type": "PARSE_ERR",
                    "prompt_tokens": "", "completion_tokens": "",
                    "latency_ms": "", "phishing_type": "",
                    "source": "parse_err",
                })
                continue

            ok = bool(d.get("ok"))
            raw_label = d.get("label_llm")
            label_int = _llm_to_int(raw_label) if ok else None
            gt = gt_by_id.get(sid)
            agreement = ""
            err_type = ""
            parse_ok = 1 if (ok and label_int is not None) else 0
            if parse_ok and gt is not None:
                agreement = 1 if gt == label_int else 0
                if not agreement:
                    err_type = "FP" if label_int == 1 else "FN"
            rows_out.append({
                "round": round_num,
                "sample_id": sid,
                "gt": gt,
                "llm_label_raw": raw_label or "",
                "llm_label": label_int if label_int is not None else "",
                "confidence": d.get("confidence") or "",
                "parse_ok": parse_ok,
                "agreement": agreement,
                "error_type": err_type,
                "prompt_tokens": d.get("prompt_tokens") or "",
                "completion_tokens": d.get("completion_tokens") or "",
                "latency_ms": d.get("latency_ms") or "",
                "phishing_type": d.get("phishing_type") or "",
                "source": d.get("source") or "",
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # Quick summary on stdout
    df = pd.DataFrame(rows_out)
    n = len(df)
    parsed = df[df["parse_ok"] == 1]
    n_p = len(parsed)
    if n_p > 0:
        acc = parsed["agreement"].astype(int).mean()
        n_fp = (parsed["error_type"] == "FP").sum()
        n_fn = (parsed["error_type"] == "FN").sum()
        # Per-class precision (precision on what the LLM CALLED that class)
        pred_phish = parsed[parsed["llm_label"] == 1]
        pred_benign = parsed[parsed["llm_label"] == 0]
        prec_phish = (pred_phish["agreement"].astype(int).mean()
                      if len(pred_phish) else float("nan"))
        prec_benign = (pred_benign["agreement"].astype(int).mean()
                       if len(pred_benign) else float("nan"))
    else:
        acc = prec_phish = prec_benign = float("nan")
        n_fp = n_fn = 0

    print(f"wrote {out_path}")
    print(f"  total queries:     {n}")
    print(f"  parsed OK:         {n_p}")
    print(f"  missing cache:     {n_missing}")
    print(f"  overall accuracy:  {acc:.3f}")
    print(f"  precision phish:   {prec_phish:.3f}  "
          f"(LLM said phish:  {len(pred_phish) if n_p else 0})")
    print(f"  precision benign:  {prec_benign:.3f}  "
          f"(LLM said benign: {len(pred_benign) if n_p else 0})")
    print(f"  FP (LLM=phish,  gt=benign): {n_fp}")
    print(f"  FN (LLM=benign, gt=phish):  {n_fn}")


if __name__ == "__main__":
    main()
