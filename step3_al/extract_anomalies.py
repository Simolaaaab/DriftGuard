"""extract_anomalies.py — Cross-LLM anomaly extractor for manual paper inspection.

Six anomaly classes (per supervisor brief + the 'format_failure' addition):

  1. confident_wrong   confidence ≥ 0.85  AND  label_llm ≠ gt
       (overconfident failures — the most damaging to the AL loop)

  2. cross_model_split ≥ N_AGREE providers concordi  AND  all wrong vs gt
       (sample where every healthy provider lands on the same wrong label
        — likely an oracle blind spot; worth a paper appendix entry)

  3. lucky_halluc      n_contradicted ≥ 2  AND  label_llm == gt
       (LLM refuted by verifier on multiple claims but still got the
        verdict right — the "right answer for fabricated reasons" case)

  4. groundless_phish  n_indicators_found == 0  AND  label_llm == "phish"
       (assertive phish call with zero positive indicators — the LLM
        committed to a label without naming evidence)

  5. inverse_conservatism  n_indicators_found ≥ 3  AND  label_llm == "benign"
       (LLM listed multiple phishing indicators yet labelled benign —
        the structural opposite of Conservatism Bug)

  6. format_failure    no parseable JSON OR (label_llm not in {"phish",
       "benign"} AND raw_text length > 100)
       (provider produced free Markdown / non-JSON output despite the
        explicit schema in the system prompt — surfaces non-DeepSeek
        models that struggle with the prompt structure)

Inputs:
  - reboot/runs/al/_verifier_audit/per_sample_<provider>.csv  (audit outputs)
  - reboot/runs/al/multi_llm_bench/<cell>/oracle_cache/B/<sid>.json.gz
    (raw cached LLM responses, for full reasoning + raw_text)
  - reboot/runs/osint/z_matrix_pp_osint.csv  (URL + GT lookup)
  - HtmlCacheIndex                          (HTML excerpt)

Outputs (under reboot/runs/al/_anomalies/):
  anomalies.jsonl          one row per anomaly instance, with reasoning,
                           indicators, html excerpt, OSINT signals
  anomalies_for_paper.md   curated 6-12 cases per class (cherry-picked
                           by anomaly severity) ready to cite in
                           Section "Qualitative analysis" of the paper

Usage:
    python3 reboot/step3_al/extract_anomalies.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step2_oracle"))
sys.path.insert(0, str(REBOOT / "step1_ensemble"))

from common import AL_RUNS_ROOT  # noqa: E402
from html_loader import HtmlCacheIndex, clean_html, load_pp_html  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("extract_anomalies")

AUDIT_DIR = AL_RUNS_ROOT / "_verifier_audit"
BENCH_DIR = AL_RUNS_ROOT / "multi_llm_bench"
OUT_DIR = AL_RUNS_ROOT / "_anomalies"

PROVIDER_RUN = {
    "DeepSeek-V4-Flash":
        "deepseek_v4_drift_anchored_B_k20_seed2025",
    "GPT-5.4":
        "gpt_5_4_drift_anchored_B_k20_seed2025",
    "GPT-5.4-mini":
        "gpt_5_4_mini_drift_anchored_B_k20_seed2025",
    "GPT-5.1":
        "gpt_5_1_drift_anchored_B_k20_seed2025",
    "Llama-3.3-70B":
        "Llama_drift_anchored_B_k20_seed2025",
    "Mistral-Large-3":
        "mistral_large_drift_anchored_B_k20_seed2025",
}

# Thresholds — tweakable via CLI.
CONFIDENT_TAU = 0.85
LUCKY_HALLUC_MIN_CONTRADICTED = 2
# Use n_VERIFIED (predicates confirmed it's a phishing signal), NOT
# n_indicators_found — the latter includes benign-positive bullets like
# "valid HTTPS" / "established domain", which inflate the count.
INVERSE_CONS_MIN_VERIFIED = 3
CROSS_MODEL_MIN_AGREE = 4         # need this many providers concurring
HTML_EXCERPT_CHARS = 1500


def _slug(name: str) -> str:
    return name.lower().replace(".", "").replace("-", "_")


def _load_raw_cache(provider: str, sid: int) -> dict | None:
    rid = PROVIDER_RUN.get(provider)
    if rid is None:
        return None
    p = AL_RUNS_ROOT / rid / "oracle_cache" / "B" / f"{sid}.json.gz"
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _load_audits() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for prov in PROVIDER_RUN:
        p = AUDIT_DIR / f"per_sample_{_slug(prov)}.csv"
        if p.exists():
            out[prov] = pd.read_csv(p)
    return out


def _all_cached_sids(provider: str) -> set[int]:
    rid = PROVIDER_RUN.get(provider)
    if rid is None:
        return set()
    d = AL_RUNS_ROOT / rid / "oracle_cache" / "B"
    out: set[int] = set()
    for p in d.glob("*.json.gz"):
        try:
            out.add(int(p.name.split(".", 1)[0]))
        except ValueError:
            pass
    return out


@dataclass
class Anomaly:
    klass: str
    provider: str
    sample_id: int
    domain: str
    url: str
    label_llm: str
    label_llm_int: int | None
    label_gt: int
    correct: int
    confidence: float
    trust: float
    n_indicators_found: int
    n_indicators_not_found: int
    n_contradicted: int
    indicators_found: list
    indicators_not_found: list
    reasoning: str
    html_excerpt: str
    raw_text_excerpt: str
    extra: dict


def _audit_row_dict(row) -> dict:
    return {k: row[k] for k in row.index}


def _build_anomaly(
    klass: str, provider: str, audit_row, raw: dict | None,
    html_excerpt: str, extra: dict | None = None,
) -> Anomaly:
    raw = raw or {}
    inds_f = raw.get("indicators_found") or []
    inds_nf = raw.get("indicators_not_found") or []
    reasoning = str(raw.get("reasoning") or audit_row.get("reasoning_excerpt") or "")
    raw_text = str(raw.get("raw_text") or "")
    return Anomaly(
        klass=klass,
        provider=provider,
        sample_id=int(audit_row["sample_id"]),
        domain=str(audit_row.get("domain") or ""),
        url=str(audit_row.get("url") or ""),
        label_llm=str(raw.get("label_llm") or audit_row.get("label_llm") or ""),
        label_llm_int=(int(audit_row["label_llm"])
                       if not pd.isna(audit_row.get("label_llm")) else None),
        label_gt=int(audit_row["gt_label"]),
        correct=int(audit_row["correct"]),
        confidence=float(audit_row.get("confidence", 0.0) or 0.0),
        trust=float(audit_row.get("trust", 0.0) or 0.0),
        n_indicators_found=int(audit_row.get("n_indicators_found", 0) or 0),
        n_indicators_not_found=int(audit_row.get("n_indicators_not_found", 0) or 0),
        n_contradicted=int(audit_row.get("n_contradicted", 0) or 0),
        indicators_found=inds_f[:8],
        indicators_not_found=inds_nf[:6],
        reasoning=reasoning[:600],
        html_excerpt=html_excerpt[:HTML_EXCERPT_CHARS],
        raw_text_excerpt=raw_text[:600],
        extra=extra or {},
    )


def extract_anomalies(
    audits: dict[str, pd.DataFrame],
    html_index: HtmlCacheIndex,
    pp_lookup: pd.DataFrame,
) -> list[Anomaly]:
    """Walk every per_sample CSV row, tag the anomaly classes that match."""
    out: list[Anomaly] = []
    pp_by_sid = pp_lookup.set_index("sample_id")

    # ── Per-provider local classes (1, 3, 4, 5) ───────────────────────
    for provider, df in audits.items():
        for _, row in df.iterrows():
            sid = int(row["sample_id"])
            raw = _load_raw_cache(provider, sid)
            html_excerpt = ""
            if sid in pp_by_sid.index:
                pp_row = pp_by_sid.loc[sid]
                if isinstance(pp_row, pd.DataFrame):
                    pp_row = pp_row.iloc[0]
                raw_html = load_pp_html(
                    sid,
                    sample_name=str(pp_row.get("sample_name") or ""),
                    url=str(pp_row.get("url") or ""),
                    index=html_index,
                )
                if raw_html:
                    html_excerpt = clean_html(raw_html, max_chars=HTML_EXCERPT_CHARS)

            conf = float(row["confidence"]) if not pd.isna(row.get("confidence")) else 0.0
            label_llm_int = (int(row["label_llm"])
                             if not pd.isna(row.get("label_llm")) else None)
            n_found = int(row["n_indicators_found"]) if not pd.isna(row.get("n_indicators_found")) else 0
            n_contr = int(row["n_contradicted"]) if not pd.isna(row.get("n_contradicted")) else 0
            n_verif = int(row["n_verified"]) if not pd.isna(row.get("n_verified")) else 0
            # Critical fix: `0 or -1` evaluates to -1 in Python — would
            # eat every `correct==0` row, which is exactly the set we want.
            correct = int(row["correct"]) if not pd.isna(row.get("correct")) else -1

            # 1. confident_wrong
            if conf >= CONFIDENT_TAU and correct == 0:
                out.append(_build_anomaly(
                    "confident_wrong", provider, row, raw, html_excerpt,
                    extra={"severity": conf * (1 - correct)},
                ))

            # 3. lucky_halluc — LLM said "phish", listed ≥2 indicators
            # FOUND that the verifier could not corroborate, yet got the
            # final label right anyway. "Right answer, fabricated reasons."
            if (n_contr >= LUCKY_HALLUC_MIN_CONTRADICTED
                    and correct == 1 and label_llm_int == 1):
                out.append(_build_anomaly(
                    "lucky_halluc", provider, row, raw, html_excerpt,
                    extra={"severity": n_contr},
                ))

            # 4. groundless_phish — assertive "phish" with zero positive
            # indicators cited. Note: empirically zero across all 6 healthy
            # providers — they always justify a phish call with ≥1 claim.
            if n_found == 0 and label_llm_int == 1:
                out.append(_build_anomaly(
                    "groundless_phish", provider, row, raw, html_excerpt,
                    extra={"severity": conf},
                ))

            # 5. inverse_conservatism — LLM said "benign" but the verifier
            # contradicted ≥2 of its "not_found" claims (i.e. the LLM
            # asserted "no login form / no typosquat / no domain-mismatch"
            # while the page actually has them). The structural opposite
            # of the PoC's Conservatism Bug.
            if n_contr >= 2 and label_llm_int == 0:
                out.append(_build_anomaly(
                    "inverse_conservatism", provider, row, raw, html_excerpt,
                    extra={"severity": n_contr,
                           "n_contradicted_absent_claims": n_contr},
                ))

    # ── Cross-model class (2) ────────────────────────────────────────
    # For every sample audited by ≥ CROSS_MODEL_MIN_AGREE providers,
    # check if they all converge on the SAME wrong label.
    by_sid: dict[int, list[tuple[str, pd.Series]]] = defaultdict(list)
    for provider, df in audits.items():
        for _, row in df.iterrows():
            by_sid[int(row["sample_id"])].append((provider, row))

    for sid, entries in by_sid.items():
        if len(entries) < CROSS_MODEL_MIN_AGREE:
            continue
        labs = [int(r["label_llm"])
                for _, r in entries
                if not pd.isna(r.get("label_llm"))]
        if len(labs) < CROSS_MODEL_MIN_AGREE:
            continue
        if len(set(labs)) != 1:
            continue                # not unanimous
        gt = int(entries[0][1]["gt_label"])
        if labs[0] == gt:
            continue                # unanimous AND right → not anomalous
        # Unanimous AND wrong: emit as one anomaly per provider.
        if sid in pp_by_sid.index:
            pp_row = pp_by_sid.loc[sid]
            if isinstance(pp_row, pd.DataFrame):
                pp_row = pp_row.iloc[0]
            raw_html = load_pp_html(
                sid,
                sample_name=str(pp_row.get("sample_name") or ""),
                url=str(pp_row.get("url") or ""),
                index=html_index,
            )
            html_excerpt = (clean_html(raw_html, max_chars=HTML_EXCERPT_CHARS)
                            if raw_html else "")
        else:
            html_excerpt = ""

        for provider, row in entries:
            raw = _load_raw_cache(provider, sid)
            out.append(_build_anomaly(
                "cross_model_split", provider, row, raw, html_excerpt,
                extra={"severity": len(entries),
                       "n_providers_concurring": len(entries),
                       "unanimous_label": int(labs[0])},
            ))

    # ── Format-failure class (6) ─────────────────────────────────────
    # Walk every raw cache file across all providers — including those
    # the audit dropped (uncertain / unparseable). If raw_text is long
    # but label_llm is None or non-canonical → format failure.
    audited_sids = {
        (provider, int(row["sample_id"]))
        for provider, df in audits.items()
        for _, row in df.iterrows()
    }
    for provider in PROVIDER_RUN:
        for sid in _all_cached_sids(provider):
            raw = _load_raw_cache(provider, sid)
            if raw is None:
                continue
            # Two signals of format failure:
            #   (a) parser flagged the response unparseable (ok=False);
            #   (b) parser succeeded but the label is neither phish/benign
            #       (e.g. "uncertain", null, missing).
            # Either way, the LLM produced text that the AL loop CANNOT use.
            ok = bool(raw.get("ok", True))
            label_llm = raw.get("label_llm")
            raw_text = str(raw.get("raw_text") or "")
            if ok and label_llm in ("phish", "benign"):
                continue                # parsed cleanly to a usable label
            if len(raw_text) < 100 and ok:
                continue                # silent / blank response — not a format issue
            # Build a minimal audit-like row from pp_lookup.
            if sid not in pp_by_sid.index:
                continue
            pp_row = pp_by_sid.loc[sid]
            if isinstance(pp_row, pd.DataFrame):
                pp_row = pp_row.iloc[0]
            fake_audit = pd.Series({
                "sample_id": sid,
                "domain": str(pp_row.get("domain") or ""),
                "url": str(pp_row.get("url") or ""),
                "label_llm": float("nan"),
                "gt_label": int(pp_row.get("label", -1)),
                "correct": 0,
                "confidence": float(raw.get("confidence") or 0.0),
                "trust": 0.0,
                "n_indicators_found": 0,
                "n_indicators_not_found": 0,
                "n_contradicted": 0,
                "reasoning_excerpt": "",
            })
            raw_html = load_pp_html(
                sid,
                sample_name=str(pp_row.get("sample_name") or ""),
                url=str(pp_row.get("url") or ""),
                index=html_index,
            )
            html_excerpt = (clean_html(raw_html, max_chars=HTML_EXCERPT_CHARS)
                            if raw_html else "")
            out.append(_build_anomaly(
                "format_failure", provider, fake_audit, raw, html_excerpt,
                extra={"severity": min(len(raw_text), 1000) / 1000.0,
                       "raw_label_llm": str(label_llm)},
            ))
    return out


def _write_jsonl(anomalies: list[Anomaly], out: Path) -> None:
    with out.open("w", encoding="utf-8") as f:
        for a in anomalies:
            d = {
                "class": a.klass, "provider": a.provider,
                "sample_id": a.sample_id,
                "domain": a.domain, "url": a.url,
                "label_llm": a.label_llm, "label_gt": a.label_gt,
                "correct": a.correct,
                "confidence": a.confidence, "trust": a.trust,
                "n_indicators_found": a.n_indicators_found,
                "n_indicators_not_found": a.n_indicators_not_found,
                "n_contradicted": a.n_contradicted,
                "indicators_found": a.indicators_found,
                "indicators_not_found": a.indicators_not_found,
                "reasoning": a.reasoning,
                "html_excerpt": a.html_excerpt,
                "raw_text_excerpt": a.raw_text_excerpt,
                "extra": a.extra,
            }
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


def _write_markdown(anomalies: list[Anomaly], out: Path,
                    per_class_cap: int = 6) -> None:
    """Cherry-pick top-`per_class_cap` cases per class by 'severity'.
    Output a paper-ready markdown the user can drop into Overleaf."""
    by_class: dict[str, list[Anomaly]] = defaultdict(list)
    for a in anomalies:
        by_class[a.klass].append(a)

    lines: list[str] = []
    lines.append("# Cross-LLM anomaly catalogue (paper appendix)\n")
    lines.append("Six anomaly classes extracted from the multi-LLM bench "
                 "(drift_anchored + B + K=20, seed 2025). "
                 f"Top {per_class_cap} cases per class, ranked by severity.\n")

    counts = {k: len(v) for k, v in by_class.items()}
    lines.append("## Counts per class (all providers pooled)\n")
    for k in ("confident_wrong", "cross_model_split", "lucky_halluc",
              "groundless_phish", "inverse_conservatism", "format_failure"):
        lines.append(f"- **{k}**: {counts.get(k, 0)} instances")
    lines.append("\n---\n")

    for klass, items in by_class.items():
        items_sorted = sorted(items,
                              key=lambda a: a.extra.get("severity", 0),
                              reverse=True)[:per_class_cap]
        lines.append(f"## {klass}  ({counts.get(klass, 0)} total)\n")
        for i, a in enumerate(items_sorted, 1):
            lines.append(f"### {i}. {a.provider} — sample {a.sample_id}\n")
            lines.append(f"- **URL**: `{a.url}`")
            lines.append(f"- **Domain**: `{a.domain}`")
            lines.append(f"- **Label LLM**: `{a.label_llm}`  |  "
                         f"**GT**: {a.label_gt}  |  "
                         f"**correct**: {a.correct}")
            lines.append(f"- **Confidence**: {a.confidence:.2f}  |  "
                         f"**Trust**: {a.trust:.2f}  |  "
                         f"**n_found/n_not/n_contr**: "
                         f"{a.n_indicators_found}/"
                         f"{a.n_indicators_not_found}/{a.n_contradicted}")
            if a.indicators_found:
                bullets = "\n".join(f"    * {ind}" for ind in a.indicators_found)
                lines.append(f"- **Indicators FOUND** (truncated):\n{bullets}")
            if a.indicators_not_found:
                bullets = "\n".join(f"    * {ind}" for ind in a.indicators_not_found)
                lines.append(f"- **Indicators NOT FOUND** (truncated):\n{bullets}")
            if a.reasoning:
                lines.append(f"- **Reasoning**: {a.reasoning}")
            if klass == "format_failure" and a.raw_text_excerpt:
                lines.append(f"- **Raw text excerpt** (no JSON):\n  > "
                             + a.raw_text_excerpt.replace("\n", "\n  > "))
            lines.append("\n")
        lines.append("---\n")

    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--per-class-cap", type=int, default=6,
                   help="how many cases per class in the markdown")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading audits + PP lookup + HTML index")
    audits = _load_audits()
    pp_lookup = pd.read_csv(REBOOT / "runs" / "osint" / "z_matrix_pp_osint.csv")
    html_index = HtmlCacheIndex()

    log.info(f"audits loaded for {list(audits)}")
    anomalies = extract_anomalies(audits, html_index, pp_lookup)

    counts: dict[str, int] = defaultdict(int)
    for a in anomalies:
        counts[a.klass] += 1
    log.info("anomaly counts:")
    for k in ("confident_wrong", "cross_model_split", "lucky_halluc",
              "groundless_phish", "inverse_conservatism", "format_failure"):
        log.info(f"  {k:>22s}: {counts.get(k, 0)}")
    log.info(f"  TOTAL                : {sum(counts.values())}")

    _write_jsonl(anomalies, args.out_dir / "anomalies.jsonl")
    _write_markdown(anomalies, args.out_dir / "anomalies_for_paper.md",
                    per_class_cap=args.per_class_cap)

    log.info(f"artefacts in {args.out_dir}/")


if __name__ == "__main__":
    main()
