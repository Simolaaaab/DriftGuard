"""load_subset.py — Load a small-benchmark manifest as Step-3 splits dict.

Drop-in replacement for `common.build_splits()`: returns the same dict
shape so the AL loop / evaluator can consume small-benchmark splits
without code changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
from common import build_splits, load_matrices  # noqa: E402

MANIFESTS = THIS.parent / "manifests"


def load_subset_splits(
    manifest_name: str = "smallbench_v1",
    dfs: dict | None = None,
) -> dict:
    """Load the manifest and return a splits dict shaped like Step-3's.

    Keys: kp_train, kp_test, pp_stream (subset), pp_test (subset),
          dp_probe.  Also injects '__manifest__' for traceability.
    """
    manifest_path = MANIFESTS / f"{manifest_name}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest {manifest_path} missing — run make_subset.py first"
        )
    manifest = json.loads(manifest_path.read_text())

    if dfs is None:
        dfs = load_matrices()
    full = build_splits(
        dfs,
        stream_order=manifest["stream_order"],
        seed=manifest["seed"],
    )

    stream_ids = set(manifest["subset"]["pp_stream_sample_ids"])
    test_ids = set(manifest["subset"]["pp_test_sample_ids"])

    pp_stream_sub = (
        full["pp_stream"][full["pp_stream"]["sample_id"]
                          .astype(int).isin(stream_ids)]
        .reset_index(drop=True)
    )
    pp_test_sub = (
        full["pp_test"][full["pp_test"]["sample_id"]
                        .astype(int).isin(test_ids)]
        .reset_index(drop=True)
    )

    if len(pp_stream_sub) != len(stream_ids):
        raise RuntimeError(
            f"manifest stream has {len(stream_ids)} ids but only "
            f"{len(pp_stream_sub)} matched in the full split — "
            f"manifest seed/stream_order mismatch?"
        )
    if len(pp_test_sub) != len(test_ids):
        raise RuntimeError(
            f"manifest test has {len(test_ids)} ids but only "
            f"{len(pp_test_sub)} matched in the full split"
        )

    return {
        "kp_train": full["kp_train"],     # full KP, never subsampled
        "kp_test": full["kp_test"],       # full KP test, never subsampled
        "pp_stream": pp_stream_sub,       # subset
        "pp_test": pp_test_sub,           # subset
        "dp_probe": full["dp_probe"],     # full DP, never subsampled
        "__manifest__": manifest,
    }
