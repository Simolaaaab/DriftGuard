"""oracle_health.py — Label-free runtime monitor for LLM oracle quality.

The problem (motivated by the Kimi-K2.6 ablation):
  Production AL has no ground truth. If the LLM oracle silently
  degrades — wrong model deployed, prompt-injection attack, rate-
  limited provider returning malformed JSON — the AL loop will
  inject noise into retraining without anyone noticing.

This module computes four runtime signals per AL round, all
label-free, and produces a `HealthSignal` enum + actionable verdict.
Calibrated against the DeepSeek-V4-Flash canonical run (healthy) and
the Kimi-K2.6 ablation (broken). Default thresholds catch both
failure modes that ablation exposed.

Signals:
  1. **parse_ok_rate**  — % of LLM responses that parse to valid JSON.
                          DeepSeek 99.8% / Kimi 31.9% → tight detector.
  2. **class_skew**     — abs(0.5 − P(phish_label)). If > 0.4, the
                          oracle is locked into one class.
                          DeepSeek 0.27 / Kimi 0.50 → catches Kimi.
  3. **yield_rate**     — accepted_labels / queried. < 0.7 indicates
                          oracle distress.
  4. **trust_collapse** — if the Step-4 verifier is on, average trust
                          score below floor signals semantic disagreement
                          between LLM verdict and OSINT/evidence.
                          Optional — only used when verifier is active.

Output: an enum {HEALTHY, WARN, CRITICAL} + breakdown for logging.
A CRITICAL verdict in two consecutive rounds should HALT the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from common import OracleLabel


class HealthSignal(str, Enum):
    HEALTHY = "healthy"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass(frozen=True)
class HealthThresholds:
    """All thresholds in their 'CRITICAL' direction.

    Calibrated against:
      - DeepSeek-V4-Flash (canonical, healthy): parse_ok≈1.00,
        class_skew 0.85-1.00 (naturally varies with sample distribution),
        yield≈1.00.
      - Kimi-K2.6 (broken): parse_ok 0.20-0.60, class_skew always 1.00,
        yield 0.20-0.60.

    Insight: parse_ok is the discriminator. DeepSeek's class_skew can
    naturally hit 0.96-1.00 in early rounds (drift_anchored picks early-
    stream benign-heavy boundary samples). class_skew CRITICAL is set at
    0.99 (only true mono-class) to avoid false alarms; halt logic
    requires ≥2 critical signals simultaneously, OR parse_ok alone
    persisting below threshold for 2 consecutive rounds.
    """
    # parse success — anything below this is a malformed-JSON
    # epidemic, indicates wrong model or prompt-injection.
    parse_ok_critical: float = 0.80
    parse_ok_warn: float = 0.95

    # Class skew — naturally varies with sample distribution. Only
    # treat 100% mono-class as critical (Kimi).
    class_skew_critical: float = 0.99
    class_skew_warn: float = 0.90

    # Yield — accepted / queried. Below this → oracle is dropping labels.
    yield_critical: float = 0.50
    yield_warn: float = 0.70

    # Trust collapse (only if verifier_mode != off)
    trust_critical: float = 0.30
    trust_warn: float = 0.45


@dataclass(frozen=True)
class HealthAudit:
    """One round's health snapshot."""
    round_num: int
    parse_ok_rate: float
    class_skew: float       # 1.0 = mono-class, 0.5 = perfectly balanced
    majority_class: int     # 0 or 1; the class the oracle leans toward
    yield_rate: float
    mean_trust: float | None
    signal: HealthSignal
    n_critical_signals: int  # how many independent signals fired CRITICAL
    reasons: tuple[str, ...]


def compute_health(
    *,
    round_num: int,
    queried_labels: Iterable[OracleLabel],
    accepted_count: int,
    mean_trust: float | None = None,
    thresholds: HealthThresholds = HealthThresholds(),
) -> HealthAudit:
    """Compute a HealthAudit from one round's oracle outputs.

    `queried_labels` is the full list of labels returned by the oracle
    for the round (including label=None for parse errors / uncertain).
    `accepted_count` is how many were accepted into the retrain.
    `mean_trust` is the average trust score (if verifier is active).
    """
    labels_list = list(queried_labels)
    n_total = len(labels_list)
    if n_total == 0:
        return HealthAudit(
            round_num=round_num, parse_ok_rate=1.0, class_skew=0.5,
            majority_class=-1, yield_rate=1.0, mean_trust=mean_trust,
            signal=HealthSignal.HEALTHY, reasons=("empty_round",),
        )

    # 1. Parse-OK rate: how many returned a structured (label!=None) verdict.
    n_parsed = sum(1 for lab in labels_list if lab.label is not None)
    parse_ok = n_parsed / n_total

    # 2. Class skew: of the parsed labels, what's the share of majority class?
    if n_parsed > 0:
        n_phish = sum(1 for lab in labels_list if lab.label == 1)
        share_phish = n_phish / n_parsed
        majority = 1 if share_phish >= 0.5 else 0
        skew = max(share_phish, 1.0 - share_phish)
    else:
        majority = -1
        skew = 1.0   # degenerate — call it max skew

    # 3. Yield: accepted of queried.
    yield_rate = accepted_count / n_total

    # Verdict aggregation — count critical signals independently.
    reasons: list[str] = []
    n_critical = 0
    n_warn = 0

    def _flag(metric: str, v: float, crit: float, warn: float,
              direction: str = "below") -> None:
        nonlocal n_critical, n_warn
        if direction == "below":
            is_crit = v < crit
            is_warn = (not is_crit) and (v < warn)
        else:                                    # 'above'
            is_crit = v > crit
            is_warn = (not is_crit) and (v > warn)
        if is_crit:
            n_critical += 1
            reasons.append(f"{metric}={v:.2f} CRIT(threshold={crit})")
        elif is_warn:
            n_warn += 1
            reasons.append(f"{metric}={v:.2f} warn(threshold={warn})")

    _flag("parse_ok", parse_ok,
          thresholds.parse_ok_critical, thresholds.parse_ok_warn, "below")
    _flag("class_skew", skew,
          thresholds.class_skew_critical, thresholds.class_skew_warn, "above")
    _flag("yield", yield_rate,
          thresholds.yield_critical, thresholds.yield_warn, "below")
    if mean_trust is not None:
        _flag("trust", mean_trust,
              thresholds.trust_critical, thresholds.trust_warn, "below")

    # Severity aggregation. A single CRITICAL on class_skew alone is
    # acceptable (natural distribution); two or more critical signals
    # together means the oracle is structurally broken.
    if n_critical >= 2:
        sev = HealthSignal.CRITICAL
    elif n_critical == 1 or n_warn >= 2:
        sev = HealthSignal.WARN
    else:
        sev = HealthSignal.HEALTHY

    if not reasons:
        reasons.append("ok")
    return HealthAudit(
        round_num=round_num, parse_ok_rate=parse_ok,
        class_skew=skew, majority_class=majority,
        yield_rate=yield_rate, mean_trust=mean_trust,
        signal=sev, n_critical_signals=n_critical, reasons=tuple(reasons),
    )


class HealthMonitor:
    """Stateful monitor across an AL run. Use:

        mon = HealthMonitor(consecutive_critical_to_halt=2)
        for each round:
            audit = mon.tick(round_num, queried_labels, accepted_count)
            log(audit)
            if mon.should_halt():
                log.error("Oracle health critical for 2 consecutive rounds — HALT")
                break
    """

    def __init__(
        self,
        *,
        thresholds: HealthThresholds = HealthThresholds(),
        consecutive_critical_to_halt: int = 2,
        consecutive_parse_warn_to_halt: int = 3,
    ) -> None:
        self.thresholds = thresholds
        self.consec_critical_to_halt = consecutive_critical_to_halt
        self.consec_parse_warn_to_halt = consecutive_parse_warn_to_halt
        self._consec_critical = 0
        self._consec_parse_low = 0
        self.history: list[HealthAudit] = []

    def tick(
        self, round_num: int,
        queried_labels: Iterable[OracleLabel],
        accepted_count: int,
        mean_trust: float | None = None,
    ) -> HealthAudit:
        audit = compute_health(
            round_num=round_num, queried_labels=queried_labels,
            accepted_count=accepted_count, mean_trust=mean_trust,
            thresholds=self.thresholds,
        )
        self.history.append(audit)
        if audit.signal == HealthSignal.CRITICAL:
            self._consec_critical += 1
        else:
            self._consec_critical = 0
        # Track parse_ok specifically — single-signal sustained collapse.
        if audit.parse_ok_rate < self.thresholds.parse_ok_warn:
            self._consec_parse_low += 1
        else:
            self._consec_parse_low = 0
        return audit

    def should_halt(self) -> bool:
        """Halt if either:
          - 2+ rounds with multi-signal CRITICAL (structural breakage), or
          - 3+ rounds with parse_ok in warning zone (sustained malformed
            JSON, indicates wrong-model / prompt-injection).
        """
        return (
            self._consec_critical >= self.consec_critical_to_halt
            or self._consec_parse_low >= self.consec_parse_warn_to_halt
        )
