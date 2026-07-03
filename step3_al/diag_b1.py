"""diag_b1.py — Test Variant B (probs + OSINT) on AL-stream boundary samples.

Hypothesis under test:
  On the Step 2 hard-FN/FP test set, Variant B (probs + OSINT) was
  WORSE than D (OSINT only) because the proposer probabilities were
  systematically WRONG on those cherry-picked samples — anchoring
  the LLM to the wrong answer.

  On the AL-stream boundary samples, the proposer probabilities are
  ~0.5 (genuinely uncertain). It's possible the LLM uses them as
  "look harder" signal rather than as misleading anchor.

Test:
  Pick 50 samples where D got it WRONG, 50 where D got it RIGHT,
  re-query each with Variant B, compare per-sample.

  Wins for B:        D wrong → B right
  Losses for B:      D right → B wrong
  Net lift:          wins - losses
"""

from __future__ import annotations

import asyncio
import gzip
import json
import random
import sys
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step2_oracle"))

from llm_client import AsyncLLM, parse_json_object       # noqa: E402
from html_loader import HtmlCacheIndex, clean_html, load_pp_html  # noqa: E402
from prompts import build_variant_B                       # noqa: E402

PP_JSON = REBOOT / "datasets" / "phreshphish_data.json"
PP_Z = REBOOT / "runs" / "osint" / "z_matrix_pp_osint.csv"
SOURCE_RUN = REBOOT / "runs" / "al" / "margin_llm_periodic_temporal"
SOURCE_CACHE = SOURCE_RUN / "oracle_cache"

OUT_DIR = REBOOT / "runs" / "al" / "_diag_b1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONCURRENCY = 4
N_PER_BUCKET = 50


def _to_int(s):
    s = str(s).strip().lower()
    if s in ("phish", "phishing", "1"):
        return 1
    if s in ("benign", "legitimate", "0"):
        return 0
    return None


def load_d_answers() -> dict[int, tuple[int, int]]:
    """{sample_id: (D_label, gt)} for the margin run."""
    pp_z = pd.read_csv(PP_Z)
    gt_by_id = dict(zip(pp_z.sample_id.astype(int), pp_z.label.astype(int)))
    out: dict[int, tuple[int, int]] = {}
    for p in SOURCE_CACHE.glob("*.json.gz"):
        sid = int(p.stem.replace(".json", ""))
        if sid not in gt_by_id:
            continue
        with gzip.open(p, "rt") as f:
            d = json.load(f)
        if not d.get("ok"):
            continue
        lbl = _to_int(d.get("label_llm"))
        if lbl is None:
            continue
        out[sid] = (lbl, gt_by_id[sid])
    return out


async def query_b(llm: AsyncLLM, sem: asyncio.Semaphore,
                  sid: int, row: dict, html_idx: HtmlCacheIndex) -> dict:
    sn = str(row.get("sample_name") or "")
    url = str(row.get("url") or "")
    raw_html = load_pp_html(sid, sample_name=sn, url=url, index=html_idx)
    cleaned = (clean_html(raw_html, max_chars=8000)
               if raw_html else "(HTML not available)")
    system, user = build_variant_B(row, cleaned)
    async with sem:
        try:
            resp = await llm.chat(system=system, user=user,
                                  temperature=0.1, max_tokens=2000)
        except Exception as e:
            return {"sample_id": sid, "label_b": None,
                    "confidence": 0.0, "reasoning": "",
                    "error": str(e)}
    parsed = parse_json_object(resp.text) or {}
    # Confidence may arrive as 'high'/'medium'/'low'; coerce via oracle helper.
    sys.path.insert(0, str(REBOOT / "step3_al"))
    from oracle import _coerce_confidence  # noqa: E402
    return {
        "sample_id": sid,
        "label_b": _to_int(parsed.get("label")),
        "confidence": _coerce_confidence(parsed.get("confidence")),
        "reasoning": str(parsed.get("reasoning") or "")[:300],
        "error": None,
    }


async def main_async() -> None:
    print("Loading D-oracle answers + GT...")
    d_answers = load_d_answers()
    print(f"  margin oracle cache: {len(d_answers)} D-labeled samples")

    wrong = [(s, l, g) for s, (l, g) in d_answers.items() if l != g]
    right = [(s, l, g) for s, (l, g) in d_answers.items() if l == g]
    print(f"  D-wrong: {len(wrong)}  D-right: {len(right)}")

    random.seed(2025)
    wrong_pick = random.sample(wrong, min(N_PER_BUCKET, len(wrong)))
    right_pick = random.sample(right, min(N_PER_BUCKET, len(right)))
    targets = wrong_pick + right_pick
    print(f"\nSelected: {len(wrong_pick)} D-wrong + {len(right_pick)} D-right")
    phish_n = sum(1 for _, _, g in targets if g == 1)
    benign_n = sum(1 for _, _, g in targets if g == 0)
    print(f"Class breakdown: phish={phish_n} benign={benign_n}")
    sample_ids = [s for s, _, _ in targets]

    # Load PP Z matrix rows for prompt building.
    pp_z = pd.read_csv(PP_Z)
    row_by_id: dict[int, dict] = {int(r.sample_id): r.to_dict()
                                  for _, r in pp_z.iterrows()}
    with PP_JSON.open() as f:
        entries = json.load(f)
    url_by_id = {int(e["id"]): str(e.get("url") or "") for e in entries}
    domain_by_id = {int(e["id"]): str(e.get("domain_name") or "").lower()
                    for e in entries}
    for sid in sample_ids:
        row_by_id[sid]["url"] = url_by_id.get(sid, "")
        row_by_id[sid]["domain"] = domain_by_id.get(sid, "")

    html_idx = HtmlCacheIndex()
    sem = asyncio.Semaphore(CONCURRENCY)
    print(f"\nQuerying LLM with Variant B (concurrency={CONCURRENCY})...")
    async with AsyncLLM.from_env() as llm:
        print(f"  backend={llm.backend} model={llm.model}")
        results = await asyncio.gather(*[
            query_b(llm, sem, sid, row_by_id[sid], html_idx)
            for sid in sample_ids
        ])

    # Join with D answers + GT.
    rows = []
    for (sid, d_lbl, gt), b in zip(targets, results):
        rows.append({
            "sample_id": sid, "gt": gt,
            "d_label": d_lbl, "b_label": b["label_b"],
            "d_correct": d_lbl == gt,
            "b_correct": (b["label_b"] == gt) if b["label_b"] is not None else None,
            "b_confidence": b["confidence"],
            "b_reasoning_preview": b["reasoning"],
            "b_error": b["error"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "diag_b1.csv", index=False)

    # Headline diagnostic.
    print()
    print("=" * 70)
    print("HEADLINE: Variant B vs Variant D on AL boundary samples")
    print("=" * 70)
    n_total = len(df)
    n_b_correct = int(df["b_correct"].sum(skipna=True))
    n_d_correct = int(df["d_correct"].sum())
    n_b_uncertain = int(df["b_label"].isna().sum())
    print(f"  D accuracy: {n_d_correct}/{n_total} = {n_d_correct/n_total:.1%}  "
          f"(by construction: {len(wrong_pick)} wrong + {len(right_pick)} right)")
    print(f"  B accuracy: {n_b_correct}/{n_total} = {n_b_correct/n_total:.1%}"
          f"  (+{n_b_uncertain} uncertain)")
    delta = n_b_correct - n_d_correct
    print(f"  Δ (B - D): {delta:+d} ({delta/n_total:+.1%})")

    # Per-bucket: how many D-wrong did B fix? How many D-right did B break?
    sub_w = df[df["sample_id"].isin([s for s, _, _ in wrong_pick])]
    fixed = int((sub_w["b_correct"] == True).sum())
    print()
    print(f"On the 50 D-WRONG samples:")
    print(f"  B fixed:        {fixed}/{len(sub_w)}  ({fixed/len(sub_w):.0%})")
    print(f"  B also wrong:   {int((sub_w['b_correct'] == False).sum())}/{len(sub_w)}")
    print(f"  B uncertain:    {int(sub_w['b_label'].isna().sum())}/{len(sub_w)}")

    sub_r = df[df["sample_id"].isin([s for s, _, _ in right_pick])]
    broken = int((sub_r["b_correct"] == False).sum())
    print()
    print(f"On the 50 D-RIGHT samples:")
    print(f"  B kept right:   {int((sub_r['b_correct'] == True).sum())}/{len(sub_r)}")
    print(f"  B broke:        {broken}/{len(sub_r)}")
    print(f"  B uncertain:    {int(sub_r['b_label'].isna().sum())}/{len(sub_r)}")

    # Per-class accuracy.
    def _acc(sub, col):
        ok = sub[sub[col].notna()]
        return (ok[col] == True).sum() / max(len(ok), 1)
    df_p = df[df.gt == 1]
    df_b = df[df.gt == 0]
    print()
    print(f"Per-class:")
    print(f"  D phish accuracy:  {(df_p['d_correct'] == True).sum()}/{len(df_p)} = "
          f"{(df_p['d_correct'] == True).sum()/max(len(df_p),1):.1%}")
    print(f"  B phish accuracy:  {(df_p['b_correct'] == True).sum()}/{len(df_p)} = "
          f"{(df_p['b_correct'] == True).sum()/max(len(df_p),1):.1%}")
    print(f"  D benign accuracy: {(df_b['d_correct'] == True).sum()}/{len(df_b)} = "
          f"{(df_b['d_correct'] == True).sum()/max(len(df_b),1):.1%}")
    print(f"  B benign accuracy: {(df_b['b_correct'] == True).sum()}/{len(df_b)} = "
          f"{(df_b['b_correct'] == True).sum()/max(len(df_b),1):.1%}")

    print(f"\nArtifact: {(OUT_DIR / 'diag_b1.csv').relative_to(REBOOT)}")


if __name__ == "__main__":
    asyncio.run(main_async())
