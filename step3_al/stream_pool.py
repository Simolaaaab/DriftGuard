"""stream_pool.py — Ungated parallel buffer for the pure-random-stream rung.

This is the *stream analogue* of `pool.UncertaintyPool`, built ADDITIVELY so
the canonical gated pool stays byte-for-byte unchanged (it lives in its own
module; nothing here touches it).

Why it exists
-------------
The AL ablation ladder needs three rungs to attribute the gain to its source:

    pure-random (from the raw stream)   ← THIS buffer
      -> random   (from the gated uncertainty pool)   ← pool.UncertaintyPool
        -> drift-anchored (smart selection on the gated pool)

The middle rung (`random` strategy) already samples uniformly from the gated
pool, whose admission rule is

    meta_prob ∈ [0.30, 0.70]   OR   |prob_sn − prob_pg| ≥ 0.35

`StreamPool` is identical to `UncertaintyPool` in EVERY respect — same FIFO
capacity, same replace-or-insert, same `remove`/`snapshot` semantics, same
accumulation across rounds — EXCEPT that it has no admission filter: every
seen stream row is a candidate, gated or not. Sampling K uniformly from this
buffer is therefore the clean isolation of the gating itself: the only thing
that differs from the `random` rung is whether the [0.30,0.70]/disagreement
filter is applied.

Note on the candidate universe: like the gated pool, this buffer accumulates
across rounds (consumed samples are removed; unconsumed ones persist, FIFO-
capped). That is deliberate — to isolate ONLY the admission filter, every
other aspect of the candidate-set lifecycle must match the gated pool exactly.
It does not graduate samples on retrain (there is no uncertainty criterion to
graduate against), which is the faithful ungated analogue of the gated pool's
rescore-eviction.
"""

from __future__ import annotations

from collections import OrderedDict

from common import POOL_CAPACITY, PoolSample, StreamRow


class StreamPool:
    """Bounded keyed-by-id buffer that admits EVERY row (no gating).

    Mirrors `pool.UncertaintyPool`'s public surface (`__len__`, `remove`,
    `snapshot`) so the loop can treat the two interchangeably.
    """

    def __init__(self, capacity: int = POOL_CAPACITY) -> None:
        self._cap = capacity
        self._items: OrderedDict[int, PoolSample] = OrderedDict()
        self._n_admitted = 0
        self._n_evicted = 0
        self._n_consumed = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def capacity(self) -> int:
        return self._cap

    def admit(
        self,
        row: StreamRow,
        *,
        meta_prob: float,
        prob_sn: float | None,
        prob_pg: float | None,
    ) -> bool:
        """Admit the row unconditionally. Always returns True.

        FIFO eviction at capacity and replace-or-insert are identical to
        `UncertaintyPool.admit_if_uncertain`, so the only behavioural
        difference between the two pools is the absence of the admission
        filter here.
        """
        diff = (abs(float(prob_sn) - float(prob_pg))
                if prob_sn is not None and prob_pg is not None else 0.0)
        sample = PoolSample(
            sample_id=row.sample_id,
            row_idx=row.row_idx,
            feature_vec=row.feature_vec,
            meta_prob=meta_prob,
            proposer_disagreement=diff,
            raw_row=row.raw_row,
        )
        if (sample.sample_id not in self._items
                and len(self._items) >= self._cap):
            self._items.popitem(last=False)
            self._n_evicted += 1
        self._items[sample.sample_id] = sample
        self._n_admitted += 1
        return True

    def remove(self, sample_ids: list[int]) -> int:
        n = 0
        for sid in sample_ids:
            if sid in self._items:
                del self._items[sid]
                n += 1
        self._n_consumed += n
        return n

    def snapshot(self) -> list[PoolSample]:
        """Return current items in insertion order."""
        return list(self._items.values())

    def stats(self) -> dict[str, int]:
        return {
            "n_admitted": self._n_admitted,
            "n_evicted": self._n_evicted,
            "n_consumed": self._n_consumed,
            "current_size": len(self._items),
        }
