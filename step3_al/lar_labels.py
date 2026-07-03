"""lar_labels.py — Complete per-stream LLM-label map for the LAR triggers.

The supervisor's LAR (LLM-Agreement Rate) triggers need, at every stream
row, the *dirty oracle*'s verdict so they can compare it against the live
model's prediction. We already have ~89% of the PP_stream labelled by the
variant-B oracle, scattered across the per-run `oracle_cache/B/<sid>.json.gz`
files of every previous AL run. This module:

  1. Unions all of those caches into one dict  {sample_id: (label_int, conf)}
     — `load_llm_label_map()`.
  2. Materialises that union (plus an optional API backfill of the still-
     missing rows) to a canonical, run-independent JSONL so the whole LAR
     experiment is offline, deterministic, and reproducible from artifact —
     `build_stream_label_table()` / the CLI below.

`label_int` follows the project convention: 1 = phish, 0 = benign. The raw
cache stores `label_llm` as the string "phish"/"benign"; "uncertain" (rare
on variant B) maps to None and such rows are treated as *unlabelled* — the
LAR detector simply does not probe them (it resamples within the window).

Coverage today (PP_stream = 3821 rows):
    union of all B caches  → 3413/3821 (89.3%)
    408 rows have no cached B label → backfill with `--backfill`.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step1_ensemble"))

from common import AL_RUNS_ROOT, build_splits, load_matrices  # noqa: E402

log = logging.getLogger("lar_labels")

STREAM_LABELS_DIR = AL_RUNS_ROOT / "_stream_llm_labels"
STREAM_LABELS_PATH = STREAM_LABELS_DIR / "pp_stream_B.jsonl"

_LABEL_MAP = {"phish": 1, "benign": 0}


def _label_str_to_int(s: object) -> int | None:
    """'phish'->1, 'benign'->0, anything else (incl. 'uncertain')->None."""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        v = int(s)
        return v if v in (0, 1) else None
    if isinstance(s, str):
        return _LABEL_MAP.get(s.strip().lower())
    return None


def load_llm_label_map(
    variant: str = "B",
    *,
    cache_glob_root: Path = AL_RUNS_ROOT,
    prefer_materialised: bool = True,
) -> dict[int, int]:
    """Return {sample_id: label_int} for the dirty LLM oracle.

    If the materialised table exists and `prefer_materialised`, read it
    (single source of truth, may include the API backfill). Otherwise
    union every `*/oracle_cache/<variant>/<sid>.json.gz` on disk.

    Only parse-ok, non-uncertain labels are returned; uncertain/failed
    rows are omitted (the caller treats them as unlabelled).
    """
    if prefer_materialised and STREAM_LABELS_PATH.exists():
        out: dict[int, int] = {}
        for line in STREAM_LABELS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            li = _label_str_to_int(rec.get("label_llm"))
            if li is not None:
                out[int(rec["sample_id"])] = li
        log.info(f"loaded {len(out)} LLM labels from {STREAM_LABELS_PATH.name}")
        return out

    out_conf: dict[int, tuple[int, float]] = {}
    pat = str(cache_glob_root / "*" / "oracle_cache" / variant / "*.json.gz")
    n_files = 0
    for p in glob.glob(pat):
        n_files += 1
        try:
            sid = int(Path(p).name.split(".")[0])
        except ValueError:
            continue
        try:
            with gzip.open(p) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not d.get("ok", True):
            continue
        li = _label_str_to_int(d.get("label_llm"))
        if li is None:
            continue
        conf = float(d.get("confidence") or 0.0)
        # On collision, keep the higher-confidence verdict (deterministic
        # tie-break by not overwriting on equal confidence).
        prev = out_conf.get(sid)
        if prev is None or conf > prev[1]:
            out_conf[sid] = (li, conf)
    out = {sid: li for sid, (li, _) in out_conf.items()}
    log.info(f"unioned {len(out)} LLM labels from {n_files} cache files "
             f"under {pat}")
    return out


def stream_coverage(label_map: dict[int, int],
                    *, seed: int = 2025,
                    stream_order: str = "temporal") -> dict:
    """Diagnostic: how much of PP_stream the label map covers."""
    dfs = load_matrices()
    splits = build_splits(dfs, stream_order=stream_order, seed=seed)
    stream_ids = [int(x) for x in splits["pp_stream"]["sample_id"].tolist()]
    covered = [s for s in stream_ids if s in label_map]
    missing = [s for s in stream_ids if s not in label_map]
    return {
        "stream_len": len(stream_ids),
        "covered": len(covered),
        "missing": len(missing),
        "coverage_frac": len(covered) / max(len(stream_ids), 1),
        "missing_ids": missing,
    }


async def _backfill_missing(missing_ids: list[int], *, model: str | None) -> dict[int, int]:
    """Query the variant-B oracle for the still-missing stream rows.

    Reuses the exact Step-3 oracle path (CachedOracle + AsyncLLM) so the
    backfilled labels are produced identically to every other AL label.
    Returns {sample_id: label_int} for the rows that came back parse-ok.
    """
    import numpy as np
    from common import PoolSample
    from oracle import CachedOracle
    from llm_client import AsyncLLM

    dfs = load_matrices()
    splits = build_splits(dfs, stream_order="temporal", seed=2025)
    stream_df = splits["pp_stream"]
    by_id = {int(r["sample_id"]): r for _, r in stream_df.iterrows()}

    # Build minimal PoolSamples for the missing rows.
    from common import load_feature_cols
    feature_cols, fill_means = load_feature_cols()
    samples: list[PoolSample] = []
    for sid in missing_ids:
        row = by_id.get(sid)
        if row is None:
            continue
        raw = row.to_dict()
        fv = {f: float(raw.get(f, fill_means.get(f, 0.0)) or 0.0)
              for f in feature_cols}
        samples.append(PoolSample(
            sample_id=sid, row_idx=-1, feature_vec=fv,
            meta_prob=0.5, proposer_disagreement=0.0, raw_row=raw,
        ))

    backfill_dir = STREAM_LABELS_DIR / "_backfill_run"
    oracle = CachedOracle(
        run_dir=backfill_dir, mode="llm",
        gt_by_id=None, concurrency=4,
        prompt_variant="B", use_step2_cache=True,
    )
    out: dict[int, int] = {}
    async with AsyncLLM.from_env(model=model) as llm:
        log.info(f"backfilling {len(samples)} rows via {llm.backend}/{llm.model}")
        labels = await oracle.label_batch(samples, llm=llm)
    for lab in labels:
        if lab.label is not None:
            out[int(lab.sample_id)] = int(lab.label)
    log.info(f"backfill produced {len(out)}/{len(samples)} parse-ok labels")
    return out


def build_stream_label_table(*, backfill: bool, model: str | None) -> None:
    """Materialise the union (+optional backfill) to STREAM_LABELS_PATH."""
    STREAM_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    label_map = load_llm_label_map(prefer_materialised=False)
    cov = stream_coverage(label_map)
    log.info(f"pre-backfill coverage: {cov['covered']}/{cov['stream_len']} "
             f"({100*cov['coverage_frac']:.1f}%), missing={cov['missing']}")

    if backfill and cov["missing"]:
        import asyncio
        filled = asyncio.run(_backfill_missing(cov["missing_ids"], model=model))
        label_map.update(filled)
        cov = stream_coverage(label_map)
        log.info(f"post-backfill coverage: {cov['covered']}/{cov['stream_len']} "
                 f"({100*cov['coverage_frac']:.1f}%)")

    # Write one JSONL row per labelled stream sample (string label_llm to
    # mirror the raw cache; the loader maps it back to int).
    inv = {1: "phish", 0: "benign"}
    dfs = load_matrices()
    splits = build_splits(dfs, stream_order="temporal", seed=2025)
    stream_ids = [int(x) for x in splits["pp_stream"]["sample_id"].tolist()]
    n = 0
    with STREAM_LABELS_PATH.open("w") as fh:
        for sid in stream_ids:
            if sid in label_map:
                fh.write(json.dumps({
                    "sample_id": sid,
                    "label_llm": inv[label_map[sid]],
                }) + "\n")
                n += 1
    log.info(f"wrote {n} labels → {STREAM_LABELS_PATH}")
    if cov["missing"]:
        log.warning(
            f"{cov['missing']} stream rows remain UNLABELLED. The LAR "
            f"detector will skip them (resample within window). Re-run with "
            f"--backfill (and LLM creds) to reach 100% coverage."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backfill", action="store_true",
                   help="query the variant-B oracle for the missing rows "
                        "(needs AZURE_API_KEY/AZURE_BASE_URL or DEEPSEEK_API_KEY)")
    p.add_argument("--model", default=None,
                   help="LLM model for backfill (e.g. DeepSeek-V4-Flash)")
    p.add_argument("--coverage-only", action="store_true",
                   help="just report coverage of the on-disk caches, write nothing")
    args = p.parse_args()

    if args.coverage_only:
        cov = stream_coverage(load_llm_label_map(prefer_materialised=False))
        print(json.dumps({k: v for k, v in cov.items()
                          if k != "missing_ids"}, indent=2))
        print(f"(missing_ids: {len(cov['missing_ids'])} hidden)")
        return

    build_stream_label_table(backfill=args.backfill, model=args.model)


if __name__ == "__main__":
    main()
