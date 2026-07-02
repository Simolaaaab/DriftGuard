"""stream.py — Deterministic row-by-row PP iteration.

The stream is built from the 80% PP_stream slice. Order is
temporal-by-default (sorted by JSON `date`, then sample_id) — the
realistic configuration for the drift narrative. A `shuffle` mode is
exposed for ablation.

`PPStream` is intentionally a thin iterator: it does not own pool,
drift, or model state. The loop driver owns those.
"""

from __future__ import annotations

import json
from typing import Iterator

import pandas as pd

from common import (
    PP_JSON,
    StreamRow,
    build_feature_vector,
)


class PPStream:
    """Iterate the PP stream in the requested order, yielding StreamRow."""

    def __init__(
        self,
        df_pp_stream: pd.DataFrame,
        *,
        feature_cols: list[str],
        fill_means: dict[str, float],
    ) -> None:
        # We assume `df_pp_stream` is already ordered (see build_splits).
        self._df = df_pp_stream.reset_index(drop=True)
        self._feature_cols = feature_cols
        self._fill_means = fill_means
        self._url_by_id: dict[int, str] = {}
        if PP_JSON.exists():
            with PP_JSON.open() as f:
                for e in json.load(f):
                    if "id" in e:
                        self._url_by_id[int(e["id"])] = str(e.get("url") or "")

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[StreamRow]:
        for row_idx, (_, row) in enumerate(self._df.iterrows()):
            sid = int(row.get("sample_id", -1))
            domain = str(row.get("domain") or "").strip().lower()
            if not domain and "sample_name" in row.index:
                sn = str(row["sample_name"])
                if "_" in sn:
                    domain = sn.split("_", 1)[1].lower()
            url = self._url_by_id.get(sid, "")
            fv = build_feature_vector(row, self._feature_cols, self._fill_means)
            yield StreamRow(
                sample_id=sid,
                domain=domain,
                url=url,
                row_idx=row_idx,
                label_gt=int(row["label"]),
                feature_vec=fv,
                raw_row=row.to_dict(),
            )
