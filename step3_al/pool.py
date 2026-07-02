"""pool.py — Bounded uncertainty pool with FIFO eviction + rescore.

Admission criterion (single rule, per Spec §3.2):
    meta_prob ∈ [0.30, 0.70]                            (uncertainty band)
    OR  |prob_sn − prob_pg| ≥ 0.35                      (proposer disagreement)

Why both: the band catches what the *meta-learner* hesitates on; the
disagreement catches what the *base learners* fight about even when
the meta-learner is confident (a blind spot of the meta-learner).
Either signal alone misses one of the two failure modes.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from common import (
    POOL_CAPACITY,
    PROPOSER_DISAGREEMENT_THRESHOLD,
    UNCERTAINTY_BAND,
    PoolSample,
    StreamRow,
)


@dataclass
class PoolStats:
    n_admitted: int = 0
    n_evicted: int = 0
    n_consumed: int = 0
    n_rescored_evicted: int = 0
    current_size: int = 0


class UncertaintyPool:
    """Bounded LIFO-keyed-by-id buffer. FIFO eviction at capacity.

    `admit_if_uncertain` returns True iff the row was admitted; the
    caller can use this for logging.
    """

    def __init__(self, capacity: int = POOL_CAPACITY) -> None:
        self._cap = capacity
        self._items: OrderedDict[int, PoolSample] = OrderedDict()
        self._stats = PoolStats()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def capacity(self) -> int:
        return self._cap

    def admit_if_uncertain(
        self,
        row: StreamRow,
        *,
        meta_prob: float,
        prob_sn: float | None,
        prob_pg: float | None,
    ) -> bool:
        """Apply the two-OR admission rule. Returns True if admitted."""
        lo, hi = UNCERTAINTY_BAND
        in_band = lo <= meta_prob <= hi
        diff = (abs(float(prob_sn) - float(prob_pg))
                if prob_sn is not None and prob_pg is not None else 0.0)
        proposer_disagree = diff >= PROPOSER_DISAGREEMENT_THRESHOLD
        if not (in_band or proposer_disagree):
            return False

        sample = PoolSample(
            sample_id=row.sample_id,
            row_idx=row.row_idx,
            feature_vec=row.feature_vec,
            meta_prob=meta_prob,
            proposer_disagreement=diff,
            raw_row=row.raw_row,
        )
        # If at capacity, evict oldest (FIFO).
        if (sample.sample_id not in self._items
                and len(self._items) >= self._cap):
            self._items.popitem(last=False)
            self._stats.n_evicted += 1
        # Replace-or-insert: keep most recent observation.
        self._items[sample.sample_id] = sample
        self._stats.n_admitted += 1
        self._stats.current_size = len(self._items)
        return True

    def remove(self, sample_ids: list[int]) -> int:
        n = 0
        for sid in sample_ids:
            if sid in self._items:
                del self._items[sid]
                n += 1
        self._stats.n_consumed += n
        self._stats.current_size = len(self._items)
        return n

    def snapshot(self) -> list[PoolSample]:
        """Return current items in insertion order."""
        return list(self._items.values())

    def rescore(
        self,
        *,
        rescorer,           # callable: PoolSample -> (meta_prob, prob_sn, prob_pg)
    ) -> tuple[int, int]:
        """Re-evaluate each pooled sample under the *new* model.

        Samples that no longer satisfy the admission criterion under
        the new model leave the pool ("graduated" — model now decides
        them confidently).

        Returns (n_updated, n_evicted).
        """
        lo, hi = UNCERTAINTY_BAND
        n_updated = 0
        n_evicted = 0
        ids = list(self._items.keys())
        for sid in ids:
            old = self._items[sid]
            new_mp, new_sn, new_pg = rescorer(old)
            new_diff = (abs(float(new_sn) - float(new_pg))
                        if new_sn is not None and new_pg is not None else 0.0)
            in_band = lo <= new_mp <= hi
            still_uncertain = in_band or (new_diff >= PROPOSER_DISAGREEMENT_THRESHOLD)
            if not still_uncertain:
                del self._items[sid]
                n_evicted += 1
                self._stats.n_rescored_evicted += 1
            else:
                # Update fields with new model's opinion.
                self._items[sid] = PoolSample(
                    sample_id=old.sample_id,
                    row_idx=old.row_idx,
                    feature_vec=old.feature_vec,
                    meta_prob=new_mp,
                    proposer_disagreement=new_diff,
                    raw_row=old.raw_row,
                )
                n_updated += 1
        self._stats.current_size = len(self._items)
        return n_updated, n_evicted

    def stats(self) -> PoolStats:
        return PoolStats(
            n_admitted=self._stats.n_admitted,
            n_evicted=self._stats.n_evicted,
            n_consumed=self._stats.n_consumed,
            n_rescored_evicted=self._stats.n_rescored_evicted,
            current_size=self._stats.current_size,
        )
