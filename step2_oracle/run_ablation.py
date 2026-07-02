"""run_ablation.py — Async LLM oracle ablation lab.

Runs every prompt variant against every sample in the test set, with
on-disk caching so re-launches pick up where they left off.

Result layout
─────────────
  reboot/runs/oracle_ablation/
    test_set.csv                            (from sample_test_set.py)
    results.csv                             (long form: one row per (sample, variant))
    cache/<variant>/<sample_id>.json        (raw LLM payload, gzipped)

Each `cache/.../<sample_id>.json` is the parsed JSON object returned
by the LLM plus the request metadata (latency, tokens, raw_text).
Existence-as-cache: if the file exists we skip the API call.

Usage
─────
    # First time / resume
    AZURE_API_KEY=... AZURE_BASE_URL=... \\
    python3 reboot/step2_oracle/run_ablation.py

    # Smaller smoke run
    python3 reboot/step2_oracle/run_ablation.py --limit 5

    # One specific variant
    python3 reboot/step2_oracle/run_ablation.py --variants A_html_probs

    # Force re-fetch (delete cache first)
    rm -rf reboot/runs/oracle_ablation/cache && python3 ... run_ablation.py
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
from html_loader import HtmlCacheIndex, clean_html, load_pp_html  # noqa: E402
from llm_client import AsyncLLM, LLMError, parse_json_object       # noqa: E402
from prompts import VARIANTS                                       # noqa: E402

REBOOT = THIS.parents[1]
LAB_DIR = REBOOT / "runs" / "oracle_ablation"
TEST_SET = LAB_DIR / "test_set.csv"
RESULTS = LAB_DIR / "results.csv"
CACHE_DIR = LAB_DIR / "cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("oracle_ablation")

# Concurrency: LLM providers tolerate ~8 parallel chat completions.
DEFAULT_CONCURRENCY = 8
HTML_8K = 8000
HTML_2K = 2000


# ── Cache helpers ─────────────────────────────────────────────────


def _cache_path(variant: str, sample_id: int) -> Path:
    return CACHE_DIR / variant / f"{sample_id}.json.gz"


def _cache_load(variant: str, sample_id: int) -> dict | None:
    p = _cache_path(variant, sample_id)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _cache_save(variant: str, sample_id: int, payload: dict) -> None:
    p = _cache_path(variant, sample_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


# ── Per-call worker ──────────────────────────────────────────────


async def one_call(
    llm: AsyncLLM,
    semaphore: asyncio.Semaphore,
    variant_key: str,
    row: dict,
    html_8k: str,
    html_2k: str,
) -> dict[str, Any]:
    """Call one (variant, sample). Returns the cache record."""
    sample_id = int(row["sample_id"])
    hit = _cache_load(variant_key, sample_id)
    if hit is not None:
        return hit

    build_fn = VARIANTS[variant_key]
    if variant_key == "E_poc_2k":
        system_prompt, user_prompt = build_fn(row, html_2k)
    elif variant_key in ("C_html_only", "A_html_probs", "B_html_probs_osint", "D_html_osint"):
        system_prompt, user_prompt = build_fn(row, html_8k)
    else:
        raise ValueError(f"unknown variant: {variant_key}")

    async with semaphore:
        t0 = time.monotonic()
        try:
            resp = await llm.chat(
                system=system_prompt, user=user_prompt,
                temperature=0.1, max_tokens=2000,
            )
            parsed = parse_json_object(resp.text)
            record = {
                "variant": variant_key,
                "sample_id": sample_id,
                "ok": parsed is not None,
                "label_llm": (parsed or {}).get("label"),
                "confidence": (parsed or {}).get("confidence"),
                "reasoning": (parsed or {}).get("reasoning"),
                "phishing_type": (parsed or {}).get("phishing_type"),
                "indicators_found": (parsed or {}).get("indicators_found"),
                "indicators_not_found": (parsed or {}).get("indicators_not_found"),
                "top_factors": (parsed or {}).get("top_factors"),
                "raw_text": resp.text,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "latency_ms": resp.latency_ms,
                "error": None if parsed is not None else "json_parse_failed",
            }
        except LLMError as e:
            record = {
                "variant": variant_key,
                "sample_id": sample_id,
                "ok": False,
                "label_llm": None,
                "confidence": None,
                "reasoning": None,
                "phishing_type": None,
                "indicators_found": None,
                "indicators_not_found": None,
                "top_factors": None,
                "raw_text": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": (time.monotonic() - t0) * 1000.0,
                "error": str(e),
            }

    _cache_save(variant_key, sample_id, record)
    return record


# ── Main async loop ──────────────────────────────────────────────


async def main_async(args) -> None:
    if not TEST_SET.exists():
        raise FileNotFoundError(
            f"Missing {TEST_SET}. Run sample_test_set.py first."
        )
    df = pd.read_csv(TEST_SET)
    if args.limit:
        df = df.head(args.limit)
    log.info(f"Test set: {len(df)} samples")

    variants = ([v for v in args.variants.split(",") if v]
                if args.variants else list(VARIANTS.keys()))
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"Unknown variant {v!r}; "
                             f"known: {list(VARIANTS)}")
    log.info(f"Variants: {variants}")

    # Pre-load HTML for every sample (cheap, all local).
    index = HtmlCacheIndex()
    html_8k: dict[int, str] = {}
    html_2k: dict[int, str] = {}
    missing_html = 0
    for _, row in df.iterrows():
        sid = int(row["sample_id"])
        sn = str(row.get("sample_name") or "")
        url = str(row.get("url") or "")
        raw = load_pp_html(sid, sample_name=sn, url=url, index=index)
        if raw is None:
            missing_html += 1
            html_8k[sid] = "(HTML not available)"
            html_2k[sid] = "(HTML not available)"
        else:
            html_8k[sid] = clean_html(raw, max_chars=HTML_8K)
            # For PoC v1 reproduction we send the RAW first-2000-chars
            # (no clean), matching the original prompt's behaviour.
            html_2k[sid] = raw[:HTML_2K]
    if missing_html:
        log.warning(f"{missing_html}/{len(df)} samples have no HTML in the cache")

    sem = asyncio.Semaphore(args.concurrency)
    backend_hint = args.backend or None

    async with AsyncLLM.from_env(backend=backend_hint, model=args.model) as llm:
        log.info(f"LLM backend={llm.backend}  model={llm.model}")
        tasks = []
        for v in variants:
            for _, r in df.iterrows():
                tasks.append(one_call(
                    llm, sem, v, r.to_dict(),
                    html_8k[int(r["sample_id"])],
                    html_2k[int(r["sample_id"])],
                ))
        log.info(f"Scheduling {len(tasks)} calls "
                 f"({len(variants)} variants × {len(df)} samples)  "
                 f"concurrency={args.concurrency}")
        t0 = time.monotonic()
        done = 0
        results: list[dict] = []
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            results.append(rec)
            done += 1
            if done % 25 == 0 or done == len(tasks):
                el = time.monotonic() - t0
                eta = el * (len(tasks) - done) / done if done else 0
                log.info(f"  {done}/{len(tasks)}  elapsed={el/60:.1f}m  "
                         f"eta={eta/60:.1f}m")

    # Save the long-form results CSV.
    cols = ["variant", "sample_id", "ok", "label_llm", "confidence",
            "phishing_type", "reasoning",
            "indicators_found", "indicators_not_found", "top_factors",
            "prompt_tokens", "completion_tokens", "latency_ms", "error"]
    out_df = pd.DataFrame(results)[cols]
    out_df.to_csv(RESULTS, index=False)
    log.info(f"Wrote {RESULTS.relative_to(REBOOT)}  shape={out_df.shape}")

    # Quick per-variant sanity print.
    print()
    for v in variants:
        sub = out_df[out_df.variant == v]
        ok = int(sub.ok.sum())
        n = len(sub)
        avg_tok = int(sub.completion_tokens.mean()) if n else 0
        med_latency = int(sub.latency_ms.median()) if n else 0
        print(f"  {v:<24s} ok={ok}/{n}  avg_completion_tokens={avg_tok}  "
              f"median_latency_ms={med_latency}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="",
                   help="comma-separated subset of "
                        + ",".join(VARIANTS.keys()))
    p.add_argument("--limit", type=int, default=0,
                   help="limit number of samples (0 = all)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--backend", choices=("azure", "deepseek"), default=None,
                   help="override auto-detect")
    p.add_argument("--model", default=None,
                   help="override the default model for the backend")
    args = p.parse_args()
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
