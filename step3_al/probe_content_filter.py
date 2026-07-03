"""probe_content_filter.py — Send 5 known-failing samples to each provider.

The 5 sids 4924, 4712, 2957, 3815, 4592 all triggered Azure RAI content
filter on DeepSeek-V4-Flash with the B prompt. This script sends each
of them to every available provider and reports which provider(s)
return a 200 vs which 400-content_filter, so we can pick a replacement
oracle that survives the filter.

Output: a small table per provider, ok/blocked/error counts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step2_oracle"))
sys.path.insert(0, str(REBOOT / "step5_smallbench"))

from common import build_splits, load_feature_cols, load_matrices  # noqa: E402
from html_loader import HtmlCacheIndex, clean_html, load_pp_html   # noqa: E402
from llm_client import LLMError                                     # noqa: E402
from llm_providers import PROVIDERS, available_providers, get_llm_client  # noqa: E402
from prompts import build_variant_B                                 # noqa: E402

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("probe_cf")

FAILED_SIDS = [4924, 4712, 2957, 3815, 4592]


async def probe_one(provider_name: str, samples: list[dict]) -> dict:
    counts = {"ok": 0, "content_filter": 0, "other_400": 0,
              "other_error": 0, "details": []}
    try:
        llm_ctx = get_llm_client(provider_name)
    except Exception as e:
        counts["details"].append(f"client construction failed: {e}")
        return counts
    async with llm_ctx as llm:
        for s in samples:
            try:
                resp = await llm.chat(
                    system=s["system"], user=s["user"],
                    temperature=0.1, max_tokens=600,
                )
                counts["ok"] += 1
                counts["details"].append(
                    f"sid={s['sid']:<5d} OK  "
                    f"tok={resp.prompt_tokens}/{resp.completion_tokens}  "
                    f"text[:60]={(resp.text or '')[:60].strip()!r}"
                )
            except LLMError as e:
                msg = str(e)
                if "content_filter" in msg or "ResponsibleAI" in msg:
                    counts["content_filter"] += 1
                    counts["details"].append(
                        f"sid={s['sid']:<5d} BLOCKED (content_filter)"
                    )
                elif "HTTP 4" in msg:
                    counts["other_400"] += 1
                    counts["details"].append(
                        f"sid={s['sid']:<5d} 4xx: {msg[:120]}"
                    )
                else:
                    counts["other_error"] += 1
                    counts["details"].append(
                        f"sid={s['sid']:<5d} ERR: {msg[:120]}"
                    )
            except Exception as e:
                counts["other_error"] += 1
                counts["details"].append(
                    f"sid={s['sid']:<5d} EXCEPTION: {type(e).__name__}: {str(e)[:100]}"
                )
    return counts


async def main_async() -> None:
    feature_cols, fill_means = load_feature_cols()
    dfs = load_matrices()
    splits = build_splits(dfs, stream_order="temporal", seed=2025)
    pp_stream = splits["pp_stream"]
    pp_by_sid = {int(r["sample_id"]): r for _, r in pp_stream.iterrows()}

    # Build prompts for the 5 sids using the SAME variant B builder used
    # at runtime in oracle.py.
    html_index = HtmlCacheIndex()
    samples = []
    for sid in FAILED_SIDS:
        if sid not in pp_by_sid:
            log.warning(f"sid={sid} not in pp_stream (seed=2025) — skipping")
            continue
        row = pp_by_sid[sid]
        sample_name = str(row.get("sample_name") or "")
        url = str(row.get("url") or "")
        raw_html = load_pp_html(int(sid), sample_name=sample_name,
                                url=url, index=html_index)
        cleaned = clean_html(raw_html, max_chars=8000)
        prompt_row = dict(row)
        prompt_row["url"] = url
        prompt_row["domain"] = row.get("domain") or ""
        system_prompt, user_prompt = build_variant_B(prompt_row, cleaned)
        samples.append({"sid": int(sid),
                        "system": system_prompt, "user": user_prompt})

    print(f"Built {len(samples)} probe prompts. "
          f"Average user-prompt size: "
          f"{sum(len(s['user']) for s in samples)/len(samples):.0f} chars")

    avail = available_providers()
    print(f"\nAvailable providers in env: {avail}\n")

    results: dict[str, dict] = {}
    for prov in avail:
        print(f"=== {prov} ===")
        r = await probe_one(prov, samples)
        results[prov] = r
        print(f"  ok={r['ok']}  content_filter={r['content_filter']}  "
              f"other_400={r['other_400']}  other_error={r['other_error']}")
        for d in r["details"]:
            print(f"    {d}")
        print()

    print("=" * 60)
    print("FINAL SUMMARY (OK count out of 5)")
    print("=" * 60)
    print(f"{'provider':<20s} {'ok':>4s} {'blocked':>8s} {'4xx':>5s} {'err':>5s}")
    for prov, r in results.items():
        print(f"{prov:<20s} {r['ok']:>4d} "
              f"{r['content_filter']:>8d} "
              f"{r['other_400']:>5d} {r['other_error']:>5d}")
    print()
    winners = [p for p, r in results.items() if r["ok"] == len(samples)]
    if winners:
        print(f"✓ Provider(s) that survived all 5: {winners}")
        print(f"  → Use --provider {winners[0]} (or AZURE_MODEL accordingly)")
    else:
        partial = sorted(results.items(),
                         key=lambda kv: kv[1]["ok"], reverse=True)
        print(f"✗ No provider survived all 5.")
        print(f"  Best-effort ranking: "
              f"{[(p, r['ok']) for p, r in partial[:3]]}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
