"""make_subset.py — Build the canonical small-benchmark subsample.

Stratified subsample of PP_stream + PP_test for cheap multi-LLM /
multi-prompt comparative studies. The subset is persisted as a JSON
manifest (sample_ids only) so every downstream script loads identical
splits across runs and re-clones.

Stratification: preserve class balance per slice (phish/benign ratio
of the full PP) and preserve temporal order within stream.

KP_test and DP_probe are NOT subsampled — they remain full slices for
all small-benchmark evaluations, so OOD probes stay apples-to-apples
against the full-dataset Main Results.

Output: reboot/step5_smallbench/manifests/<name>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse Step-3 splits to start from the same PP_test/PP_stream as Main Results.
THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
from common import build_splits, load_matrices  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("make_subset")

MANIFESTS = THIS.parent / "manifests"


def stratified_sample(
    df: pd.DataFrame, n: int, seed: int, label_col: str = "label",
) -> pd.DataFrame:
    """Stratified sample preserving class proportions of `df`. Stable order."""
    rng = np.random.default_rng(seed)
    parts = []
    for cls, sub in df.groupby(label_col):
        share = len(sub) / len(df)
        n_cls = int(round(n * share))
        n_cls = max(1, min(n_cls, len(sub)))
        idx = rng.choice(len(sub), size=n_cls, replace=False)
        parts.append(sub.iloc[sorted(idx)])
    out = pd.concat(parts, axis=0)
    # Re-impose original order (so temporal sort downstream is well-defined)
    return df.loc[df.index.isin(out.index)].copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stream", type=int, default=500,
                   help="number of PP_stream rows to keep (stratified)")
    p.add_argument("--test", type=int, default=250,
                   help="number of PP_test rows to keep (stratified)")
    p.add_argument("--seed", type=int, default=2025,
                   help="seed for both the underlying Step-3 splits "
                        "AND the stratified subsample")
    p.add_argument("--stream-order", choices=("temporal", "shuffle"),
                   default="temporal")
    p.add_argument("--name", default="smallbench_v1",
                   help="manifest filename stem (manifests/<name>.json)")
    args = p.parse_args()

    log.info(f"Loading Z matrices via Step-3 build_splits "
             f"(stream_order={args.stream_order}, seed={args.seed})…")
    dfs = load_matrices()
    full = build_splits(dfs, stream_order=args.stream_order, seed=args.seed)

    full_stream = full["pp_stream"]
    full_test = full["pp_test"]
    log.info(f"Full PP_stream: {len(full_stream)} "
             f"(phish={(full_stream['label']==1).sum()}, "
             f"benign={(full_stream['label']==0).sum()})")
    log.info(f"Full PP_test:   {len(full_test)} "
             f"(phish={(full_test['label']==1).sum()}, "
             f"benign={(full_test['label']==0).sum()})")

    sub_stream = stratified_sample(full_stream, args.stream, args.seed)
    sub_test = stratified_sample(full_test, args.test, args.seed + 1)

    log.info(f"Subset stream:  {len(sub_stream)} "
             f"(phish={(sub_stream['label']==1).sum()}, "
             f"benign={(sub_stream['label']==0).sum()})")
    log.info(f"Subset test:    {len(sub_test)} "
             f"(phish={(sub_test['label']==1).sum()}, "
             f"benign={(sub_test['label']==0).sum()})")

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out = MANIFESTS / f"{args.name}.json"
    payload = {
        "name": args.name,
        "seed": args.seed,
        "stream_order": args.stream_order,
        "source_splits": {
            "kp_test_n": len(full["kp_test"]),
            "pp_test_n_full": len(full_test),
            "pp_stream_n_full": len(full_stream),
            "dp_probe_n": len(full["dp_probe"]),
        },
        "subset": {
            "pp_stream_n": len(sub_stream),
            "pp_test_n": len(sub_test),
            "pp_stream_phish": int((sub_stream["label"] == 1).sum()),
            "pp_stream_benign": int((sub_stream["label"] == 0).sum()),
            "pp_test_phish": int((sub_test["label"] == 1).sum()),
            "pp_test_benign": int((sub_test["label"] == 0).sum()),
            "pp_stream_sample_ids": [int(x) for x in
                                     sub_stream["sample_id"].tolist()],
            "pp_test_sample_ids": [int(x) for x in
                                   sub_test["sample_id"].tolist()],
        },
        "note": ("KP_test and DP_probe are NOT subsampled — they remain full "
                 "slices for every small-benchmark run, so OOD probes are "
                 "directly comparable to the full-dataset Main Results."),
    }
    out.write_text(json.dumps(payload, indent=2))
    log.info(f"Wrote {out.relative_to(REBOOT)}")


if __name__ == "__main__":
    main()
