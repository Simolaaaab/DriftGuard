"""drift_detector.py — Composite drift trigger (delegate_ratio ∧ KL).

The detector observes only the meta-learner's `meta_prob` per row
and a `is_delegate` flag (whether the row landed in the uncertainty
band). It never sees ground-truth labels — the trigger must be
deployable in production where labels are absent.

Trigger rule (per Spec §3.3, AND logic):
    delegate_ratio  ≥ 0.10
  AND  KL(rolling_meta_prob || KP_reference) ≥ 0.50
"""

from __future__ import annotations

from collections import deque

import numpy as np

from common import (
    DRIFT_COOLDOWN_ROWS,
    DRIFT_DELEGATE_THRESHOLD,
    DRIFT_KL_THRESHOLD,
    DRIFT_WINDOW,
    DriftEvent,
    DriftSignal,
)


def build_reference_hist(
    meta_probs: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the KP_train reference histogram of meta_prob.

    Returns (hist, bin_edges). `hist` is a normalised probability mass
    function (sums to 1), `bin_edges` has `n_bins + 1` entries.
    Laplace-smoothed downstream in `_kl` to avoid log(0).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(meta_probs, bins=bins, density=False)
    total = counts.sum() or 1
    return counts.astype(float) / total, bins


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) with Laplace smoothing."""
    p_s = (p + eps) / (p.sum() + eps * len(p))
    q_s = (q + eps) / (q.sum() + eps * len(q))
    return float(np.sum(p_s * np.log(p_s / q_s)))


class CompositeDriftDetector:
    """Stateful trigger with cooldown and six modes:

      mode='and'              fire iff (delegate_ratio ≥ t_del) AND (KL ≥ t_kl)
      mode='or'               fire iff (delegate_ratio ≥ t_del) OR (KL ≥ t_kl).
                              More sensitive than AND: catches either
                              output-uncertainty drift or input-distribution
                              drift, accepting the higher fire rate as
                              cost-of-recall trade-off.
      mode='periodic'         fire every `periodic_every` rows, ignoring
                              delegate_ratio and KL entirely
      mode='and_or_periodic'  fire if AND-condition OR periodic spacing met
      mode='kl_only'          fire iff (KL ≥ t_kl), ignoring delegate_ratio.
                              Input-based drift signal, robust to model
                              confidence saturation. The right tool for a
                              well-calibrated leakage-clean LogReg whose
                              delegate_ratio collapses to ~0.
      mode='random_batching'  fire at pre-sampled row indices passed via
                              `random_fire_rows`. No gates. Ablation baseline:
                              isolates whether the *timing* of the trigger
                              matters at all, holding the selector fixed.
      mode='random_lar'       supervisor's LLM-Agreement-Rate trigger. Every
                              `lar_window` rows, probe `lar_probe_k` of them
                              (chosen uniformly at random among the rows whose
                              cached LLM label is known) and compute
                              LAR = mean(model_pred == llm_label). Fire when
                              LAR has dropped ≥ `lar_drop_delta` below the
                              post-retrain baseline. Spends LLM probe-calls to
                              decide (unlike kl_only/periodic which are
                              label-free) — that cost is reported separately.
      mode='kl_guided_lar'    identical to random_lar but the `lar_probe_k`
                              probed rows are the ones whose meta_prob lands in
                              the highest-KL-divergence bins of the window
                              histogram vs the KP reference, rather than a
                              uniform draw.

    KL and delegate_ratio are still *computed* in every mode for telemetry
    (rounds.jsonl). The mode only governs whether they gate firing.
    """

    _LAR_MODES = ("random_lar", "kl_guided_lar")

    def __init__(
        self,
        *,
        reference_hist: np.ndarray,
        bin_edges: np.ndarray,
        window: int = DRIFT_WINDOW,
        delegate_threshold: float = DRIFT_DELEGATE_THRESHOLD,
        kl_threshold: float = DRIFT_KL_THRESHOLD,
        cooldown_rows: int = DRIFT_COOLDOWN_ROWS,
        mode: str = "and",
        periodic_every: int = 0,
        random_fire_rows: tuple = (),
        # LAR-mode parameters (ignored by the non-LAR modes).
        llm_labels: dict[int, int] | None = None,
        lar_window: int = 0,
        lar_probe_k: int = 0,
        lar_select: str = "random",
        lar_rule: str = "drop",
        lar_drop_delta: float = 0.0,
        lar_abs_threshold: float = 0.7,
        lar_seed: int = 0,
    ) -> None:
        if window < 2:
            raise ValueError("window must be ≥ 2")
        if len(reference_hist) + 1 != len(bin_edges):
            raise ValueError("reference_hist length must equal len(bin_edges)-1")
        valid_modes = ("and", "or", "periodic", "and_or_periodic",
                       "kl_only", "random_batching") + self._LAR_MODES
        if mode not in valid_modes:
            raise ValueError(f"unknown mode: {mode!r}")
        if mode in ("periodic", "and_or_periodic") and periodic_every <= 0:
            raise ValueError(
                f"mode={mode!r} requires periodic_every > 0"
            )
        if mode == "random_batching" and not random_fire_rows:
            raise ValueError(
                "mode='random_batching' requires random_fire_rows to be "
                "non-empty (pre-sampled row indices from the caller)"
            )
        if mode in self._LAR_MODES:
            if not llm_labels:
                raise ValueError(
                    f"mode={mode!r} requires a non-empty llm_labels map "
                    "(build it with lar_labels.load_llm_label_map)"
                )
            if lar_window <= 0 or lar_probe_k <= 0:
                raise ValueError(
                    f"mode={mode!r} requires lar_window>0 and lar_probe_k>0"
                )
            if lar_select not in ("random", "kl_guided"):
                raise ValueError(f"unknown lar_select: {lar_select!r}")
            if lar_rule not in ("drop", "absolute"):
                raise ValueError(f"unknown lar_rule: {lar_rule!r}")
        self._ref = np.asarray(reference_hist, dtype=float)
        self._bins = np.asarray(bin_edges, dtype=float)
        self._win = int(window)
        self._t_del = float(delegate_threshold)
        self._t_kl = float(kl_threshold)
        self._cooldown = int(cooldown_rows)
        self._mode = mode
        self._periodic_every = int(periodic_every)
        self._random_fire_rows = frozenset(int(r) for r in random_fire_rows)
        self._buf: deque[DriftEvent] = deque(maxlen=self._win)
        self._cooldown_remaining = 0
        self._rows_since_last_trigger = 0
        self.last_trigger_row_idx: int | None = None
        # LAR state.
        self._llm_labels = dict(llm_labels or {})
        self._lar_window = int(lar_window)
        self._lar_probe_k = int(lar_probe_k)
        self._lar_select = lar_select
        self._lar_rule = lar_rule
        self._lar_drop_delta = float(lar_drop_delta)
        self._lar_abs_threshold = float(lar_abs_threshold)
        self._lar_rng = np.random.default_rng(int(lar_seed))
        self._lar_buf: deque[DriftEvent] = deque(maxlen=max(self._lar_window, 1))
        self._rows_since_lar_eval = 0
        self._lar_baseline: float | None = None
        self.last_lar: float | None = None
        # Telemetry
        self.n_ticks = 0
        self.n_triggers = 0
        self.n_suppressed_by_cooldown = 0
        self.n_periodic_fires = 0
        self.n_and_fires = 0
        self.n_or_fires = 0
        self.n_kl_only_fires = 0
        self.n_random_fires = 0
        self.n_lar_fires = 0
        self.n_lar_evals = 0
        self.n_probe_calls = 0           # total LLM probes consumed (cost)
        self.n_probe_unlabelled = 0      # probe slots that hit a missing label

    @property
    def window_size(self) -> int:
        return self._win

    def reset_window(self) -> None:
        """Called post-retrain. Fresh window so post-retrain drift is
        detected on its own merits, not contaminated by pre-retrain data.

        Note: rows_since_last_trigger is NOT reset here — that counter
        is reset only when an actual trigger fires, so the periodic
        spacing is consistent across rounds.

        For the LAR modes the post-retrain reset is semantically required:
        the model just changed, so the old LAR baseline no longer reflects
        model-vs-oracle agreement. We clear the LAR window and drop the
        baseline so the next full window re-establishes it (and never fires
        on that re-establishing window)."""
        self._buf.clear()
        self._cooldown_remaining = 0
        if self._mode in self._LAR_MODES:
            self._lar_buf.clear()
            self._rows_since_lar_eval = 0
            self._lar_baseline = None

    def revoke_last_trigger(self) -> None:
        """Loop-level veto: caller decided the trigger should not have
        fired (e.g. pool starvation). Roll back the counters so the
        detector tries again at the next opportunity. Telemetry counters
        (n_triggers etc.) are NOT decremented — the trigger occurred
        from the detector's perspective; only the spacing/cooldown reset."""
        self._rows_since_last_trigger = 0
        self._cooldown_remaining = 0
        if self._mode in self._LAR_MODES:
            self._rows_since_lar_eval = 0

    # ── LAR helpers ──────────────────────────────────────────────
    def _select_probe_indices(self, events: list[DriftEvent]) -> list[int]:
        """Pick up to `lar_probe_k` *labelled* rows from the window to probe.

        'random'    → uniform draw over the labelled rows.
        'kl_guided' → rows whose meta_prob falls in the window's
                      highest-KL-divergence bins vs the KP reference,
                      bin-priority order (within a bin: arrival order).
        Returns positional indices into `events`."""
        labelled = [i for i, e in enumerate(events)
                    if e.sample_id in self._llm_labels]
        if not labelled:
            return []
        k = min(self._lar_probe_k, len(labelled))

        if self._lar_select == "random":
            chosen = self._lar_rng.choice(labelled, size=k, replace=False)
            return [int(i) for i in chosen]

        # kl_guided: rank bins by their divergence contribution, then take
        # labelled rows from the most-divergent bins first.
        vals = np.array([e.meta_prob for e in events], dtype=float)
        counts, _ = np.histogram(vals, bins=self._bins, density=False)
        eps = 1e-9
        p = (counts + eps) / (counts.sum() + eps * len(counts))
        q = (self._ref + eps) / (self._ref.sum() + eps * len(self._ref))
        contrib = p * np.log(p / q)            # per-bin KL contribution
        bin_order = list(np.argsort(-contrib))  # most-divergent bin first
        # Map each labelled row to its bin (np.digitize: 1..n_bins).
        row_bin = np.digitize(vals, self._bins[1:-1])
        labelled_set = set(labelled)
        picked: list[int] = []
        for b in bin_order:
            members = [i for i in labelled
                       if row_bin[i] == b and i in labelled_set]
            members.sort()                      # arrival order, deterministic
            for i in members:
                picked.append(i)
                if len(picked) >= k:
                    return picked
        return picked

    def _measure_lar(self, events: list[DriftEvent]) -> tuple[float | None, int]:
        """Compute LAR over the probed rows of the window. Returns
        (lar, n_probes). `lar` is None when no labelled row is available."""
        idxs = self._select_probe_indices(events)
        self.n_probe_calls += len(idxs)
        if not idxs:
            return None, 0
        agree = 0
        for i in idxs:
            e = events[i]
            model_pred = 1 if e.meta_prob >= 0.5 else 0
            llm_lab = self._llm_labels[e.sample_id]
            agree += int(model_pred == llm_lab)
        return agree / len(idxs), len(idxs)

    def _lar_tick(self, event: DriftEvent,
                  delegate_ratio: float, kl: float) -> DriftSignal:
        """LAR-mode tick: accumulate a window, measure LAR every
        `lar_window` rows, fire when LAR drops ≥ `lar_drop_delta` below the
        post-retrain baseline. `delegate_ratio`/`kl` are passed only so the
        DriftSignal still carries them for telemetry (rounds.jsonl)."""
        self._lar_buf.append(event)
        self._rows_since_lar_eval += 1

        def _sig(triggered: bool, reason: str,
                 lar: float | None, probes: int) -> DriftSignal:
            return DriftSignal(
                triggered=triggered, delegate_ratio=delegate_ratio, kl=kl,
                reason=reason, window_size=len(self._lar_buf),
                lar=lar, lar_baseline=self._lar_baseline, lar_probes=probes,
            )

        # Not yet a full window since the last evaluation/reset → accumulate.
        if (len(self._lar_buf) < self._lar_window
                or self._rows_since_lar_eval < self._lar_window):
            return _sig(False, "lar_accumulating", self.last_lar, 0)

        # A measurement is due.
        events = list(self._lar_buf)
        lar, n_probes = self._measure_lar(events)
        self._rows_since_lar_eval = 0
        self.n_lar_evals += 1
        if n_probes < self._lar_probe_k:
            self.n_probe_unlabelled += (self._lar_probe_k - n_probes)
        self.last_lar = lar

        if lar is None:
            return _sig(False, "lar_no_labels_in_window", None, 0)

        # Decide whether this measurement is a fire CANDIDATE, per rule.
        if self._lar_rule == "absolute":
            # No baseline: fire whenever the agreement is low in absolute
            # terms. Fires on every drifted window, hence many more fires.
            if lar >= self._lar_abs_threshold:
                return _sig(
                    False,
                    f"lar_abs_ok|lar={lar:.3f}≥{self._lar_abs_threshold:.3f}",
                    lar, n_probes,
                )
            fire_reason = (f"lar_abs|lar={lar:.3f}<{self._lar_abs_threshold:.3f},"
                           f"select={self._lar_select}")
        else:
            # 'drop' rule: relative to the post-retrain baseline. First
            # measurement after bootstrap/retrain sets the baseline and
            # never fires on the window that sets it.
            if self._lar_baseline is None:
                self._lar_baseline = lar
                return _sig(False, f"lar_baseline_set|lar={lar:.3f}",
                            lar, n_probes)
            drop = self._lar_baseline - lar
            if drop < self._lar_drop_delta:
                return _sig(
                    False,
                    f"lar_no_drop|lar={lar:.3f},base={self._lar_baseline:.3f},"
                    f"drop={drop:.3f}<{self._lar_drop_delta:.3f}",
                    lar, n_probes,
                )
            fire_reason = (f"lar_drop|lar={lar:.3f},base={self._lar_baseline:.3f},"
                           f"drop={drop:.3f}≥{self._lar_drop_delta:.3f},"
                           f"select={self._lar_select}")

        # Fire candidate (cooldown can still suppress).
        if self._cooldown_remaining > 0:
            self.n_suppressed_by_cooldown += 1
            return _sig(False, "cooldown", lar, n_probes)

        self.n_triggers += 1
        self.n_lar_fires += 1
        self._cooldown_remaining = self._cooldown
        self._rows_since_last_trigger = 0
        self.last_trigger_row_idx = event.row_idx
        return _sig(True, fire_reason, lar, n_probes)

    def tick(self, event: DriftEvent) -> DriftSignal:
        self.n_ticks += 1
        self._buf.append(event)
        self._rows_since_last_trigger += 1
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Need a populated half-window before deciding (only matters for
        # AND-style modes — periodic can fire from row 1 if scheduled).
        in_warmup = len(self._buf) < max(2, self._win // 2)

        # Always compute the diagnostics for telemetry.
        delegate_ratio = 0.0
        kl = 0.0
        if len(self._buf) >= 2:
            delegate_ratio = sum(1 for e in self._buf if e.is_delegate) / len(self._buf)
            vals = np.array([e.meta_prob for e in self._buf], dtype=float)
            counts, _ = np.histogram(vals, bins=self._bins, density=False)
            kl = _kl(counts.astype(float), self._ref)

        # ── LAR modes: own cadence + baseline-drop logic, return early ──
        if self._mode in self._LAR_MODES:
            return self._lar_tick(event, delegate_ratio, kl)

        delegate_passes = delegate_ratio >= self._t_del
        kl_passes = kl >= self._t_kl
        and_fires = (not in_warmup) and delegate_passes and kl_passes
        or_fires = (not in_warmup) and (delegate_passes or kl_passes)

        # Periodic clause. Honour `periodic_every` only when the mode
        # asks for it. AND-only / kl_only / random_batching ignore this.
        periodic_fires = (
            self._mode in ("periodic", "and_or_periodic")
            and self._periodic_every > 0
            and self._rows_since_last_trigger >= self._periodic_every
        )

        # kl_only: KL signal alone, no delegate_ratio gate. Robust to
        # confidence saturation. Still respects warmup.
        kl_only_fires = (
            self._mode == "kl_only"
            and (not in_warmup)
            and kl_passes
        )

        # random_batching: fire iff this row was pre-sampled. No gates,
        # no warmup, no cooldown — exists purely to ablate timing.
        random_fires = (
            self._mode == "random_batching"
            and event.row_idx in self._random_fire_rows
        )

        if self._mode == "and":
            fire = and_fires
        elif self._mode == "or":
            fire = or_fires
        elif self._mode == "periodic":
            fire = periodic_fires
        elif self._mode == "and_or_periodic":
            fire = and_fires or periodic_fires
        elif self._mode == "kl_only":
            fire = kl_only_fires
        else:                    # random_batching
            fire = random_fires

        if not fire:
            if in_warmup and self._mode not in ("periodic", "random_batching"):
                reason = "warmup"
            elif self._mode == "random_batching":
                reason = "not_scheduled"
            elif self._mode == "or":
                # OR-mode never reports "below_kl"/"below_delegate" as a
                # rejection reason — if either passes, it fires. Only
                # "below_both" is meaningful here.
                reason = "below_both"
            elif delegate_passes and not kl_passes:
                reason = "below_kl"
            elif kl_passes and not delegate_passes:
                reason = "below_delegate"
            else:
                reason = "below_both"
            return DriftSignal(
                triggered=False, delegate_ratio=delegate_ratio, kl=kl,
                reason=reason, window_size=len(self._buf),
            )

        # Trigger CANDIDATE. Cooldown can still suppress it, except in
        # random_batching mode where fires are sparse by construction.
        if self._cooldown_remaining > 0 and self._mode != "random_batching":
            self.n_suppressed_by_cooldown += 1
            return DriftSignal(
                triggered=False, delegate_ratio=delegate_ratio, kl=kl,
                reason="cooldown", window_size=len(self._buf),
            )

        # Fire.
        self.n_triggers += 1
        if and_fires and periodic_fires:
            why = "and+periodic"
        elif self._mode == "or" and or_fires:
            self.n_or_fires += 1
            if delegate_passes and kl_passes:
                why = "or(both)"
            elif kl_passes:
                why = "or(kl)"
            else:
                why = "or(delegate)"
        elif and_fires:
            self.n_and_fires += 1
            why = "and"
        elif kl_only_fires:
            self.n_kl_only_fires += 1
            why = "kl_only"
        elif random_fires:
            self.n_random_fires += 1
            why = "random_batching"
        else:
            self.n_periodic_fires += 1
            why = "periodic"
        self._cooldown_remaining = self._cooldown
        self._rows_since_last_trigger = 0
        self.last_trigger_row_idx = event.row_idx
        return DriftSignal(
            triggered=True, delegate_ratio=delegate_ratio, kl=kl,
            reason=f"{why}|delegate={delegate_ratio:.3f},kl={kl:.3f}",
            window_size=len(self._buf),
        )
