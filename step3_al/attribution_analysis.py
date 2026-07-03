"""attribution_analysis.py — What evidence does the oracle decide on?

Tests hypothesis H2 (teacher-student feature-space mismatch): the LLM oracle
reasons over signals the gated meta-learner cannot see, so its labels carry a
component the student is structurally blind to.

For every cached variant-B verdict we classify its *stated decisive factors*
(`top_factors`, the LLM's own list) into evidence buckets, split by whether the
model can represent that evidence:

  MODEL-BLIND
    HIDDEN   : tranco / wayback / dns / asn / reputation  → gated OUT of the
               model by the leakage gate, but GIVEN to the LLM in the dossier.
               The model provably lacks these inputs.
    SEMANTIC : brand impersonation / "content matches legitimate X" / scam-lure
               / typosquat → meaning of the page. The model has only counts
               (n_forms, n_password_inputs, html_length), not content meaning.

  MODEL-VISIBLE
    SHARED_OSINT : domain age / registration / certificate / CT  → kept features
    HTML_STRUCT  : login form / password field / iframe / redirect / obfuscation
    URL_LEXICAL  : ip / https / tld / subdomain / domain length
    PROPOSER     : static classifier / phishgraph (prob_pg is a model input)

Headline metrics:
  - % of decisions that cite ≥1 HIDDEN factor (the cleanest H2 signal)
  - % that cite ≥1 model-blind factor (HIDDEN ∪ SEMANTIC)
  - % that cite ONLY model-blind factors (no visible anchor) → strongest case
  - the same, split by LLM==GT vs LLM≠GT  (the cursor-relevant cut: is the
    *noise* the model can't fix exactly the hidden/semantic-driven decisions?)

Outputs (under runs/al/_attribution/):
  attribution_per_sample.csv   one row per oracle verdict + bucket flags
  attribution_summary.json     the headline percentages
  prints a human report with examples.

Run:  python3 reboot/step3_al/attribution_analysis.py
"""

from __future__ import annotations

import glob
import gzip
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step1_ensemble"))

from common import AL_RUNS_ROOT, Z_PP  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("attribution")

OUT_DIR = AL_RUNS_ROOT / "_attribution"

# ── Evidence taxonomy (regex, grounded in the real top_factors vocabulary) ──
# Each bucket is a list of patterns; a decision is tagged with a bucket if ANY
# of its top_factors matches ANY pattern.
BUCKETS: dict[str, list[str]] = {
    # MODEL-BLIND ───────────────────────────────────────────────
    "HIDDEN": [
        r"tranco", r"wayback", r"archiv", r"\bdns\b", r"\basn\b",
        r"reputation", r"popularit", r"\bpagerank\b",
    ],
    "SEMANTIC": [
        r"impersonat", r"\bbrand\b", r"\bfake\b", r"\bscam\b", r"\blure\b",
        r"typosquat", r"deceptive", r"genuine",
        r"content (?:match|consistent|is|type|matches)",
        r"legitimate[\w\s-]*?(?:content|html|page|site|structure|app|stack|"
        r"front-end|template)",
        r"standard[\w\s-]*?(?:template|blog|page|site|article|cms)",
        r"\barticle\b", r"looks? legitimate", r"expected[\w\s-]*?(?:page|site)",
    ],
    # MODEL-VISIBLE ──────────────────────────────────────────────
    "SHARED_OSINT": [
        r"domain age", r"\bage\b", r"established", r"\d+\s*[- ]?year",
        r"old domain", r"long history", r"registr", r"\brdap\b",
        r"\bcert\b", r"certificate", r"\bct\b", r"\bissuer\b",
        r"wildcard", r"validity",
    ],
    "HTML_STRUCT": [
        r"login form", r"password field", r"\bform\b", r"credential[- ]harvest",
        r"\biframe\b", r"redirect", r"obfuscat", r"external link",
        r"hidden", r"\bscript",
    ],
    "URL_LEXICAL": [
        r"\bip address\b", r"\bhttps?\b", r"\btld\b", r"\.gov", r"\.xyz",
        r"\.shop", r"subdomain", r"domain length", r"url (?:length|structure)",
    ],
    "PROPOSER": [
        r"classifier", r"phishgraph", r"specularnet", r"static (?:classifier|model)",
    ],
}
_COMPILED = {b: [re.compile(p) for p in pats] for b, pats in BUCKETS.items()}

MODEL_BLIND = ("HIDDEN", "SEMANTIC")
MODEL_VISIBLE = ("SHARED_OSINT", "HTML_STRUCT", "URL_LEXICAL", "PROPOSER")


def tag_factors(top_factors: list) -> dict[str, bool]:
    text = " ; ".join(str(x) for x in (top_factors or [])).lower()
    return {b: any(rx.search(text) for rx in rxs) for b, rxs in _COMPILED.items()}


def _cid(name: str) -> int:
    return int(name.split(".")[0])


def load_union_payloads(variant: str = "B") -> dict[int, dict]:
    """sample_id → richest payload (one with top_factors) over all runs."""
    out: dict[int, dict] = {}
    pat = str(AL_RUNS_ROOT / "*" / "oracle_cache" / variant / "*.json.gz")
    for p in glob.glob(pat):
        try:
            sid = _cid(Path(p).name)
        except ValueError:
            continue
        try:
            d = json.load(gzip.open(p))
        except (OSError, json.JSONDecodeError):
            continue
        if not d.get("ok"):
            continue
        # Prefer a payload that actually has top_factors.
        if sid not in out or (d.get("top_factors") and not out[sid].get("top_factors")):
            out[sid] = d
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = load_union_payloads("B")
    log.info(f"loaded {len(payloads)} unique variant-B verdicts")

    # GT join (PP only — that's what the shared cache covers).
    gt_by_id = {}
    if Z_PP.exists():
        zpp = pd.read_csv(Z_PP)
        gt_by_id = {int(r.sample_id): int(r.label) for r in zpp.itertuples()}
    log.info(f"GT available for {sum(1 for s in payloads if s in gt_by_id)} "
             f"of {len(payloads)} samples")

    rows = []
    for sid, d in payloads.items():
        tf = d.get("top_factors") or []
        tags = tag_factors(tf)
        llm = d.get("label_llm")
        llm_int = 1 if str(llm).lower() in ("phish", "phishing", "1") else (
            0 if str(llm).lower() in ("benign", "legitimate", "0") else None)
        gt = gt_by_id.get(sid)
        rows.append({
            "sample_id": sid,
            "llm_label": llm,
            "llm_int": llm_int,
            "gt": gt,
            "llm_correct": (None if (gt is None or llm_int is None)
                            else int(llm_int == gt)),
            "phishing_type": d.get("phishing_type"),
            "confidence": d.get("confidence"),
            "n_factors": len(tf),
            **{f"is_{b}": tags[b] for b in BUCKETS},
            "any_blind": any(tags[b] for b in MODEL_BLIND),
            "any_visible": any(tags[b] for b in MODEL_VISIBLE),
            "only_blind": (any(tags[b] for b in MODEL_BLIND)
                           and not any(tags[b] for b in MODEL_VISIBLE)),
            "top_factors": " | ".join(str(x) for x in tf),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "attribution_per_sample.csv", index=False)

    n = len(df)
    has_tf = df[df.n_factors > 0]
    m = len(has_tf)

    def pct(mask) -> float:
        return 100.0 * mask.sum() / max(m, 1)

    summary = {
        "n_verdicts": n,
        "n_with_top_factors": m,
        "pct_cite_HIDDEN": pct(has_tf.is_HIDDEN),
        "pct_cite_SEMANTIC": pct(has_tf.is_SEMANTIC),
        "pct_cite_model_blind": pct(has_tf.any_blind),
        "pct_only_model_blind": pct(has_tf.only_blind),
        "pct_cite_SHARED_OSINT": pct(has_tf.is_SHARED_OSINT),
        "pct_cite_HTML_STRUCT": pct(has_tf.is_HTML_STRUCT),
        "pct_cite_PROPOSER": pct(has_tf.is_PROPOSER),
    }

    # Cursor-relevant cut: among samples with GT, split LLM==GT vs LLM≠GT.
    g = has_tf[has_tf.llm_correct.notna()]
    cut = {}
    if len(g):
        for name, sub in (("llm_correct", g[g.llm_correct == 1]),
                          ("llm_wrong", g[g.llm_correct == 0])):
            cut[name] = {
                "n": int(len(sub)),
                "pct_HIDDEN": 100.0 * sub.is_HIDDEN.mean(),
                "pct_SEMANTIC": 100.0 * sub.is_SEMANTIC.mean(),
                "pct_model_blind": 100.0 * sub.any_blind.mean(),
                "pct_only_blind": 100.0 * sub.only_blind.mean(),
            }
    summary["by_correctness"] = cut
    (OUT_DIR / "attribution_summary.json").write_text(json.dumps(summary, indent=2))

    # ── Report ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("ORACLE EVIDENCE ATTRIBUTION  (variant B, top_factors)")
    print("=" * 64)
    print(f"verdicts: {n}   with top_factors: {m}")
    print("\n— Share of decisions citing each evidence bucket —")
    for b in ("HIDDEN", "SEMANTIC", "SHARED_OSINT", "HTML_STRUCT",
              "URL_LEXICAL", "PROPOSER"):
        blindmark = "  ← MODEL-BLIND" if b in MODEL_BLIND else ""
        print(f"  {b:<13s} {pct(has_tf[f'is_{b}']):5.1f}%{blindmark}")
    print(f"\n  ≥1 model-blind factor (HIDDEN∪SEMANTIC): "
          f"{summary['pct_cite_model_blind']:.1f}%")
    print(f"  ONLY model-blind (no visible anchor):     "
          f"{summary['pct_only_model_blind']:.1f}%")

    if cut:
        print("\n— Split by oracle correctness vs GT (the cursor's noise) —")
        print(f"  {'':<14s} {'n':>5s} {'HIDDEN':>8s} {'SEMANTIC':>9s} "
              f"{'blind':>7s} {'only-blind':>11s}")
        for name in ("llm_correct", "llm_wrong"):
            c = cut[name]
            print(f"  {name:<14s} {c['n']:>5d} {c['pct_HIDDEN']:>7.1f}% "
                  f"{c['pct_SEMANTIC']:>8.1f}% {c['pct_model_blind']:>6.1f}% "
                  f"{c['pct_only_blind']:>10.1f}%")
        dh = cut["llm_wrong"]["pct_model_blind"] - cut["llm_correct"]["pct_model_blind"]
        print(f"\n  → on the labels the LLM gets WRONG (the 'noise' the cursor "
              f"corrects),\n    model-blind evidence is {dh:+.1f} pp "
              f"{'MORE' if dh>0 else 'less'} cited than on correct ones.")

    # phishing_type breakdown (Mode-2 scam should be more OSINT/hidden-driven).
    print("\n— model-blind share by phishing_type (top 6) —")
    for pt, sub in sorted(has_tf.groupby("phishing_type"),
                          key=lambda kv: -len(kv[1]))[:6]:
        print(f"  {str(pt):<22s} n={len(sub):<5d} "
              f"blind={100*sub.any_blind.mean():5.1f}%  "
              f"HIDDEN={100*sub.is_HIDDEN.mean():5.1f}%")

    # Examples of ONLY-blind decisions (the smoking gun).
    print("\n— Examples: decisions resting ONLY on model-blind evidence —")
    for _, r in has_tf[has_tf.only_blind].head(6).iterrows():
        gv = r["gt"]
        gtxt = "" if pd.isna(gv) else f" gt={int(gv)}"
        print(f"  [{r['llm_label']}{gtxt}] {r['top_factors']}")

    print(f"\nartifacts: {OUT_DIR.relative_to(REBOOT)}/")


if __name__ == "__main__":
    main()
