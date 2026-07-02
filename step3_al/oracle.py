"""oracle.py — Cached LLM oracle (D_html_osint) for the AL loop.

Three-tier label resolution:
  1. Step-2 cache  (the 100 hard PP samples already labeled).
  2. Step-3 cache  (samples queried in earlier rounds of this run).
  3. Fresh LLM call → write to Step-3 cache.

A `gt_dryrun` mode is exposed for smoke testing: it returns the
ground-truth label without any API call. Convenient during the
build sequence so we never wait on credentials.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from common import (
    AL_RUNS_ROOT,
    ORACLE_CONCURRENCY,
    STEP2_CACHE,
    OracleLabel,
    PoolSample,
)

# Step 2 owns the prompt builder + LLM client.
from html_loader import HtmlCacheIndex, clean_html, load_pp_html  # noqa: E402
from llm_client import AsyncLLM, LLMError, parse_json_object       # noqa: E402
from prompts import (                                              # noqa: E402
    build_variant_A, build_variant_B, build_variant_C, build_variant_D,
)

# Step 3 finding: on AL-stream boundary samples Variant B (probs+OSINT)
# beats Variant D by +17pp accuracy / +22pp phish recall, because the
# proposer probs in the 0.3–0.7 band act as "look harder" signal rather
# than misleading anchor (which they were on the Step 2 FN/FP test set).
# B is therefore the new default oracle for AL queries. D, A, C remain
# selectable for ablation table.
_PROMPT_BUILDERS = {
    "A": build_variant_A,
    "B": build_variant_B,
    "C": build_variant_C,
    "D": build_variant_D,
}

log = logging.getLogger("oracle")


# ── Deterministic noise injection (gt_dryrun mode only) ─────────


_NOISE_MODES = (
    "none",
    "random_flip",
    "adversarial_p2b",   # phish → benign at `rate` (worst-case for R_phish)
    "adversarial_b2p",   # benign → phish at `rate` (mirror, R_benign)
)


def _stable_uniform(sample_id: int, noise_seed: int) -> float:
    """Stable uniform draw in [0,1) keyed by (sample_id, noise_seed).

    Deterministic across runs and across stream orderings — the same
    sample always receives the same draw given the same noise_seed.
    Uses SHA-256 of the joined key to avoid Python hash randomisation.
    """
    h = hashlib.sha256(f"{int(noise_seed)}|{int(sample_id)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def apply_noise(
    gt_label: int, sample_id: int, *,
    mode: str, rate: float, noise_seed: int,
) -> tuple[int, bool]:
    """Apply noise transformation to a GT label.

    Returns (new_label, was_flipped). `was_flipped=False` always for
    mode='none' or rate<=0.

    The transformation is deterministic per (sample_id, noise_seed).
    """
    if mode == "none" or rate <= 0.0:
        return int(gt_label), False
    if mode not in _NOISE_MODES:
        raise ValueError(f"noise_mode must be in {_NOISE_MODES}; got {mode!r}")
    u = _stable_uniform(sample_id, noise_seed)
    flip = u < rate
    if not flip:
        return int(gt_label), False
    if mode == "random_flip":
        return 1 - int(gt_label), True
    if mode == "adversarial_p2b":
        if int(gt_label) == 1:
            return 0, True
        return int(gt_label), False           # benign untouched
    if mode == "adversarial_b2p":
        if int(gt_label) == 0:
            return 1, True
        return int(gt_label), False           # phish untouched
    return int(gt_label), False                # unreachable


def _llm_label_to_int(lbl: Any) -> int | None:
    if lbl is None:
        return None
    s = str(lbl).strip().lower()
    if s in ("phish", "phishing", "1", "true"):
        return 1
    if s in ("benign", "legitimate", "0", "false"):
        return 0
    return None     # 'uncertain' or unparseable → skip


def _coerce_confidence(v: Any) -> float:
    """Best-effort coerce LLM 'confidence' field to float in [0, 1].

    LLMs occasionally emit categorical strings ('high', 'medium', 'low'),
    percentage strings ('85%'), or out-of-range numbers. We map them all to
    [0, 1] rather than crashing the AL loop on one stray sample.
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        # heuristic: 1 < f ≤ 100 likely percent ('85'); >100 nonsense → 1.0;
        # 1 < f < 2 likely a mis-scaled proportion ('1.0' off-by-one) → 1.0.
        if 1.0 < f <= 2.0:
            f = 1.0
        elif 2.0 < f <= 100.0:
            f = f / 100.0
        elif f > 100.0:
            f = 1.0
        return max(0.0, min(1.0, f))
    s = str(v).strip().lower().rstrip("%")
    # Categorical fallback (matches the failure mode observed in
    # margin_seed2029 R8: parsed.get('confidence') == 'high').
    categorical = {
        "very_low": 0.10, "very low": 0.10, "lowest": 0.10,
        "low": 0.30, "weak": 0.30,
        "medium": 0.55, "moderate": 0.55, "mid": 0.55,
        "high": 0.80, "strong": 0.80,
        "very_high": 0.95, "very high": 0.95, "highest": 0.95,
        "certain": 1.0, "confident": 0.85,
    }
    if s in categorical:
        return categorical[s]
    try:
        f = float(s)
    except (TypeError, ValueError):
        return 0.0
    if f > 1.0:
        f = f / 100.0 if f <= 100.0 else 1.0
    return max(0.0, min(1.0, f))


def _cache_load_step2(sample_id: int) -> dict | None:
    p = STEP2_CACHE / f"{sample_id}.json.gz"
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class CachedOracle:
    """Async D_html_osint oracle with two-tier cache + dry-run fallback.

    Construct one per AL run. Inject `gt_by_id` only for dryrun mode."""

    def __init__(
        self,
        *,
        run_dir: Path,
        html_index: HtmlCacheIndex | None = None,
        mode: str = "llm",            # "llm" or "gt_dryrun"
        gt_by_id: dict[int, int] | None = None,
        concurrency: int = ORACLE_CONCURRENCY,
        max_html_chars: int = 8000,
        prompt_variant: str = "B",     # B = AL-canonical (probs+OSINT)
        use_step2_cache: bool = True,  # set False to force variant purity
        noise_mode: str = "none",
        noise_rate: float = 0.0,
        noise_seed: int = 0,
    ) -> None:
        if mode not in ("llm", "gt_dryrun"):
            raise ValueError(f"Unknown oracle mode: {mode!r}")
        if prompt_variant not in _PROMPT_BUILDERS:
            raise ValueError(
                f"prompt_variant must be one of {list(_PROMPT_BUILDERS)}; "
                f"got {prompt_variant!r}"
            )
        if noise_mode not in _NOISE_MODES:
            raise ValueError(
                f"noise_mode must be in {_NOISE_MODES}; got {noise_mode!r}"
            )
        if noise_mode != "none" and mode != "gt_dryrun":
            raise ValueError(
                "noise injection is only supported in mode='gt_dryrun' "
                "(we need the ground-truth label to flip)"
            )
        if not (0.0 <= noise_rate <= 1.0):
            raise ValueError(f"noise_rate must be in [0,1]; got {noise_rate}")
        self.mode = mode
        self.gt_by_id = gt_by_id or {}
        self.prompt_variant = prompt_variant
        self._build_prompt = _PROMPT_BUILDERS[prompt_variant]
        # Disable Step-2 cache reuse under noise — those cached labels were
        # produced by the *real* LLM, not by the noise-injected oracle, so
        # using them would silently contaminate the ablation.
        self.use_step2_cache = use_step2_cache and (noise_mode == "none")
        self.noise_mode = noise_mode
        self.noise_rate = float(noise_rate)
        self.noise_seed = int(noise_seed)
        # Cache layout: oracle_cache/<variant>/<sid>.json.gz so a run that
        # changes the variant doesn't collide with a previous run's labels.
        # When noise is on we ALSO scope by noise config so the on-disk cache
        # is unambiguously a noisy run's output and can never be picked up
        # by a clean run via misconfiguration.
        if noise_mode == "none":
            self.cache_dir = run_dir / "oracle_cache" / prompt_variant
        else:
            tag = f"noise_{noise_mode}_r{noise_rate:.2f}_s{noise_seed}"
            self.cache_dir = run_dir / "oracle_cache" / prompt_variant / tag
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.html_index = html_index or HtmlCacheIndex()
        self.concurrency = concurrency
        self.max_html_chars = max_html_chars
        self._sem = asyncio.Semaphore(concurrency)
        # Telemetry.
        self.n_step2_hits = 0
        self.n_step3_hits = 0
        self.n_llm_calls = 0
        self.n_uncertain = 0
        self.n_errors = 0
        self.n_noise_flipped = 0

    # ── Cache helpers ─────────────────────────────────────────────

    def _step3_path(self, sample_id: int) -> Path:
        return self.cache_dir / f"{sample_id}.json.gz"

    def _step3_load(self, sample_id: int) -> dict | None:
        p = self._step3_path(sample_id)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _step3_save(self, sample_id: int, payload: dict) -> None:
        p = self._step3_path(sample_id)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

    # ── Public interface ─────────────────────────────────────────

    async def label_batch(
        self, samples: list[PoolSample], llm: AsyncLLM | None = None,
    ) -> list[OracleLabel]:
        """Resolve labels for a batch. Cache-first, LLM on miss.

        The LLM client may be `None` if every sample is already in
        the Step-2 / Step-3 cache or if mode='gt_dryrun'. We only
        complain if we actually need to issue an HTTP call.
        """
        if not samples:
            return []
        tasks = [self._label_one(s, llm) for s in samples]
        return await asyncio.gather(*tasks)

    async def _label_one(
        self, sample: PoolSample, llm: AsyncLLM | None,
    ) -> OracleLabel:
        sid = sample.sample_id

        # Tier 1: Step-2 cache (only if variant matches D or explicitly enabled).
        # Step-2 cache was built with Variant D. By default we still reuse
        # it: D's labels on those 100 hard cherry-picked samples are
        # 91% accurate, no point re-paying. Set use_step2_cache=False to
        # force variant purity for the ablation.
        hit = _cache_load_step2(sid) if self.use_step2_cache else None
        if hit is not None and hit.get("ok"):
            self.n_step2_hits += 1
            lbl = _llm_label_to_int(hit.get("label_llm"))
            return OracleLabel(
                sample_id=sid, label=lbl,
                confidence=_coerce_confidence(hit.get("confidence")),
                reasoning=str(hit.get("reasoning") or ""),
                source="step2_cache",
                raw=hit,
            )

        # Tier 2: Step-3 cache.
        hit = self._step3_load(sid)
        if hit is not None and hit.get("ok"):
            self.n_step3_hits += 1
            lbl = _llm_label_to_int(hit.get("label_llm"))
            return OracleLabel(
                sample_id=sid, label=lbl,
                confidence=_coerce_confidence(hit.get("confidence")),
                reasoning=str(hit.get("reasoning") or ""),
                source="step3_cache",
                raw=hit,
            )

        # Tier 3a: dry-run mode.
        if self.mode == "gt_dryrun":
            gt = self.gt_by_id.get(sid)
            if gt is None:
                self.n_errors += 1
                return OracleLabel(
                    sample_id=sid, label=None, confidence=0.0,
                    reasoning="gt_dryrun: no GT available",
                    source="gt_dryrun", raw=None,
                )
            # Apply noise transformation. For mode='none' this is a no-op
            # and `served == gt`. Decision is deterministic per
            # (sample_id, noise_seed) and independent of selection order.
            served, was_flipped = apply_noise(
                gt, sid,
                mode=self.noise_mode, rate=self.noise_rate,
                noise_seed=self.noise_seed,
            )
            if was_flipped:
                self.n_noise_flipped += 1
            src_tag = "gt_dryrun" if self.noise_mode == "none" \
                else f"gt_dryrun_noisy[{self.noise_mode}@{self.noise_rate}]"
            payload = {
                "ok": True,
                "label_llm": "phish" if served == 1 else "benign",
                "confidence": 1.0,
                "reasoning": (
                    "gt_dryrun perfect oracle" if not was_flipped
                    else f"gt_dryrun NOISE-FLIPPED ({self.noise_mode}): "
                         f"true={gt} served={served}"
                ),
                "indicators_found": [],
                "indicators_not_found": [],
                "phishing_type": "legitimate" if served == 0 else "other",
                "source": src_tag,
                # Audit trail. Lets bootstrap_noise_ablation cross-check
                # cached noise composition without re-running.
                "noise_audit": {
                    "gt": int(gt),
                    "served": int(served),
                    "flipped": bool(was_flipped),
                    "mode": self.noise_mode,
                    "rate": self.noise_rate,
                    "noise_seed": self.noise_seed,
                },
            }
            self._step3_save(sid, payload)
            return OracleLabel(
                sample_id=sid, label=served, confidence=1.0,
                reasoning=payload["reasoning"],
                source=src_tag, raw=payload,
            )

        # Tier 3b: real LLM call.
        if llm is None:
            self.n_errors += 1
            return OracleLabel(
                sample_id=sid, label=None, confidence=0.0,
                reasoning="cache miss and no LLM client supplied",
                source="cache_miss_no_llm", raw=None,
            )
        return await self._llm_call(sample, llm)

    async def _llm_call(
        self, sample: PoolSample, llm: AsyncLLM,
    ) -> OracleLabel:
        sid = sample.sample_id
        # Pull HTML.
        sn = str(sample.raw_row.get("sample_name") or "")
        url = str(sample.raw_row.get("url") or "")
        raw_html = load_pp_html(sid, sample_name=sn, url=url,
                                index=self.html_index)
        cleaned = (
            clean_html(raw_html, max_chars=self.max_html_chars)
            if raw_html else "(HTML not available)"
        )
        # Build the D_html_osint prompt: no probs, OSINT dossier + HTML.
        # build_variant_D expects a Mapping-like row.
        prompt_row = dict(sample.raw_row)
        prompt_row["url"] = url
        prompt_row["domain"] = sample.raw_row.get("domain") or ""
        system_prompt, user_prompt = self._build_prompt(prompt_row, cleaned)

        async with self._sem:
            try:
                resp = await llm.chat(
                    system=system_prompt, user=user_prompt,
                    temperature=0.1, max_tokens=2000,
                )
            except LLMError as e:
                self.n_errors += 1
                # Surface the first few error bodies per batch so 400-style
                # gateway failures don't silently hide behind label=None.
                if self.n_errors <= 5 or self.n_errors % 50 == 0:
                    log.warning(
                        f"oracle LLM error #{self.n_errors} for sid={sid}: "
                        f"{str(e)[:400]}"
                    )
                # Don't cache transient LLM failures — caller can retry.
                return OracleLabel(
                    sample_id=sid, label=None, confidence=0.0,
                    reasoning=f"llm_error: {e}",
                    source="llm_error", raw=None,
                )
        self.n_llm_calls += 1
        parsed = parse_json_object(resp.text) or {}
        lbl = _llm_label_to_int(parsed.get("label"))
        if lbl is None:
            self.n_uncertain += 1
        payload = {
            "ok": parsed is not None and "label" in parsed,
            "label_llm": parsed.get("label"),
            "confidence": parsed.get("confidence"),
            "reasoning": parsed.get("reasoning"),
            "phishing_type": parsed.get("phishing_type"),
            "indicators_found": parsed.get("indicators_found"),
            "indicators_not_found": parsed.get("indicators_not_found"),
            "top_factors": parsed.get("top_factors"),
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_ms": resp.latency_ms,
            "raw_text": resp.text,
            "source": "llm",
        }
        self._step3_save(sid, payload)
        return OracleLabel(
            sample_id=sid, label=lbl,
            confidence=_coerce_confidence(parsed.get("confidence")),
            reasoning=str(parsed.get("reasoning") or ""),
            source="llm", raw=payload,
        )

    def telemetry(self) -> dict:
        return {
            "n_step2_hits": self.n_step2_hits,
            "n_step3_hits": self.n_step3_hits,
            "n_llm_calls": self.n_llm_calls,
            "n_uncertain": self.n_uncertain,
            "n_errors": self.n_errors,
            "n_noise_flipped": self.n_noise_flipped,
            "noise_mode": self.noise_mode,
            "noise_rate": self.noise_rate,
            "noise_seed": self.noise_seed,
        }
