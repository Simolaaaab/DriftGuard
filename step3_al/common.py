"""common.py — Shared types, paths, and config for the Step-3 AL loop.

Everything that crosses module boundaries lives here. The intent is
that downstream modules (stream/pool/drift/...) import dataclasses
and constants from this file and never from each other directly,
which keeps the dependency graph a fan-out tree rather than a mesh.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────

REBOOT = Path(__file__).resolve().parents[1]

Z_KP = REBOOT / "runs" / "osint" / "z_matrix_kp_osint.csv"
Z_DP = REBOOT / "runs" / "osint" / "z_matrix_dp_osint.csv"
Z_PP = REBOOT / "runs" / "osint" / "z_matrix_pp_osint.csv"

STEP1_MANIFEST = REBOOT / "runs" / "ensemble" / "osint" / "feature_manifest.json"
STEP2_CACHE = REBOOT / "runs" / "oracle_ablation" / "cache" / "D_html_osint"
STEP2_TEST_SET = REBOOT / "runs" / "oracle_ablation" / "test_set.csv"

PP_JSON = REBOOT / "datasets" / "phreshphish_data.json"

AL_RUNS_ROOT = REBOOT / "runs" / "al"

# ── Step-3 hyperparameters (frozen per spec) ─────────────────────

SEED = 2025
K_PER_ROUND = 50
MAX_ROUNDS = 30
TOTAL_LLM_CAP = 2000              # safety brake
DRIFT_WINDOW = 200
DRIFT_DELEGATE_THRESHOLD = 0.10
DRIFT_KL_THRESHOLD = 0.50
DRIFT_COOLDOWN_ROWS = 50
POOL_CAPACITY = 2000
UNCERTAINTY_BAND = (0.30, 0.70)
PROPOSER_DISAGREEMENT_THRESHOLD = 0.35
ORACLE_CONCURRENCY = 4            # per the user's call: avoid rate-limit on heavy HTML

# Train/test split ratio for KP (matches Step 1 with the same seed).
KP_TEST_RATIO = 0.20
PP_TEST_RATIO = 0.20


# ── Step-3 module imports from earlier steps ─────────────────────
# Reuse rather than duplicate. Step 1 owns the feature gate, the
# imputation logic, and the AUC gates; Step 2 owns the prompt builder
# and the LLM client.

sys.path.insert(0, str(REBOOT / "step1_ensemble"))
sys.path.insert(0, str(REBOOT / "step2_oracle"))

# (Imports performed lazily inside helpers to avoid import-time side
# effects in modules that only need the types from here.)


# ── Dataclasses (the wire-types between modules) ─────────────────


@dataclass(frozen=True)
class StreamRow:
    """One sample as it arrives in the simulated stream."""
    sample_id: int
    domain: str
    url: str
    row_idx: int                  # arrival index within the stream
    label_gt: int                 # ground truth — for evaluation/audit ONLY
    feature_vec: dict[str, float] # ML-ready post-imputation feature vector
    raw_row: dict[str, Any]       # the original Z-matrix row (for OSINT dossier)


@dataclass(frozen=True)
class PoolSample:
    """An uncertain sample buffered for potential oracle querying."""
    sample_id: int
    row_idx: int
    feature_vec: dict[str, float]
    meta_prob: float
    proposer_disagreement: float
    raw_row: dict[str, Any]


@dataclass(frozen=True)
class DriftEvent:
    """One tick fed into the drift detector."""
    row_idx: int
    meta_prob: float
    is_delegate: bool
    sample_id: int = -1            # needed by the LAR modes to look up the
                                   # cached LLM label for this row; -1 means
                                   # "unknown" (the legacy non-LAR modes don't
                                   # use it).


@dataclass(frozen=True)
class DriftSignal:
    """Detector output for one tick."""
    triggered: bool
    delegate_ratio: float
    kl: float
    reason: str
    window_size: int
    # LAR telemetry — populated only by the random_lar / kl_guided_lar
    # modes, None/0 otherwise. `lar` is the LLM-Agreement Rate measured on
    # the window just evaluated; `lar_baseline` is the post-retrain baseline
    # the drop is measured against; `lar_probes` is how many LLM labels were
    # consulted to compute it (the per-window probe cost).
    lar: float | None = None
    lar_baseline: float | None = None
    lar_probes: int = 0


@dataclass(frozen=True)
class OracleLabel:
    """Result of one LLM (or cached) oracle query."""
    sample_id: int
    label: int | None              # None = LLM uncertain → not retrainable
    confidence: float
    reasoning: str
    source: str                    # "step2_cache" / "step3_cache" / "llm" / "gt_dryrun"
    raw: dict[str, Any] | None


@dataclass(frozen=True)
class RoundEvalRecord:
    """Per-round evaluation snapshot."""
    round_num: int
    n_labels_accepted_cumulative: int
    kp_test: dict[str, float]
    pp_test: dict[str, float]
    dp_probe: dict[str, float]
    pool_size: int
    last_trigger_row_idx: int | None


@dataclass
class ALRunConfig:
    """Aggregated run configuration. Saved verbatim to config.json."""
    run_id: str
    strategy: str
    oracle_mode: str               # "llm" / "gt_dryrun"
    stream_order: str              # "temporal" / "shuffle"
    k_per_round: int = K_PER_ROUND
    max_rounds: int = MAX_ROUNDS
    total_llm_cap: int = TOTAL_LLM_CAP
    drift_window: int = DRIFT_WINDOW
    drift_delegate_threshold: float = DRIFT_DELEGATE_THRESHOLD
    drift_kl_threshold: float = DRIFT_KL_THRESHOLD
    drift_cooldown_rows: int = DRIFT_COOLDOWN_ROWS
    drift_mode: str = "and"        # "and" | "or" | "periodic"
                                   # | "and_or_periodic" | "kl_only"
                                   # | "random_batching"
    periodic_every: int = 0        # rows between periodic triggers
    # For "random_batching" mode: pre-sampled row indices where the
    # detector fires unconditionally. Computed by run_al.py from the
    # stream length and a seed so the trigger is reproducible.
    random_fire_rows: tuple = ()
    pool_capacity: int = POOL_CAPACITY
    uncertainty_band: tuple = UNCERTAINTY_BAND
    proposer_disagreement_threshold: float = PROPOSER_DISAGREEMENT_THRESHOLD
    min_pool_for_trigger: int = K_PER_ROUND   # don't fire if pool < this
    # Oracle prompt variant. B = AL-canonical (probs+OSINT), based on the
    # B1 diagnostic (B beats D by +17pp on boundary samples).
    prompt_variant: str = "B"
    use_step2_cache: bool = True
    # Step-4 verifier mode. 'full' = class-asymmetric soft trust scoring
    # (T1 confidence + T2 evidence + T3 OSINT coherence) as sample_weight
    # in retrain. 'off' reproduces Step 3 baseline. See verifier.py.
    verifier_mode: str = "full"
    # Step-4.5 health monitor (label-free oracle quality guard-rail).
    # On by default. When the LLM is structurally broken (Kimi-K2.6
    # case: 32% parse_ok + 100% mono-class), the monitor halts at
    # round 2, preventing toxic labels from entering retrain.
    # Calibrated to never trigger on the healthy DeepSeek run (75 rounds
    # validated across K=3 seeds). Set --disable-health-monitor to skip.
    disable_health_monitor: bool = False
    # Noise-ablation knobs (only meaningful when oracle_mode='gt_dryrun').
    # Used to isolate the *causal* origin of the DP robustness effect:
    # is the OOD lift from *structured* LLM noise, or would *any* label
    # noise produce the same regularisation? See bootstrap_noise_ablation.py.
    #   none          → no flip (clean GT, control arm)
    #   random_flip   → flip GT with prob=noise_rate (symmetric noise)
    #   adversarial_p2b → flip ONLY phish→benign at rate=noise_rate
    #   adversarial_b2p → flip ONLY benign→phish at rate=noise_rate
    # Noise is deterministic per (sample_id, noise_seed): same sample
    # always gets the same flip decision, so different selection
    # trajectories that happen to pick the same sample see identical
    # noise.
    noise_mode: str = "none"
    noise_rate: float = 0.0
    noise_seed: int = 0
    # drift_anchored score weight on the per-feature drift term:
    #   s(x) = (1/margin) * (1 + drift_alpha * Σ_i δ_i·|x_i-μ_kp,i|)
    # Default 0.5 = the canonical strategy-table value. drift_alpha=0.0
    # reduces drift_anchored to pure margin sampling.
    drift_alpha: float = 0.5
    # ── LAR triggers (random_lar / kl_guided_lar) ────────────────────
    # Supervisor's "LLM-Agreement Rate" triggers. Every `lar_window` rows
    # we probe `lar_probe_k` of them with the (cached) dirty LLM oracle and
    # measure LAR = fraction where model_pred == llm_label. The trigger
    # fires when LAR drops by ≥ `lar_drop_delta` below the post-retrain
    # baseline. `lar_select` chooses *which* window rows to probe:
    #   'random'      → uniform over labelled rows in the window
    #   'kl_guided'   → rows whose meta_prob falls in the highest
    #                   KL-divergence bins vs the KP reference histogram
    # `lar_seed` seeds the random probe draw for reproducibility. All
    # zero/empty ⇒ the LAR machinery is inert (non-LAR modes).
    lar_window: int = 0
    lar_probe_k: int = 0
    lar_select: str = "random"     # "random" | "kl_guided"
    # Trigger rule for the LAR modes:
    #   'drop'     → fire when LAR drops ≥ lar_drop_delta below the
    #                post-retrain baseline (reactive to *change*).
    #   'absolute' → fire when LAR < lar_abs_threshold, no baseline
    #                (the supervisor's literal "fire when LAR is low"
    #                reading; fires on every drifted window, so more fires).
    lar_rule: str = "drop"         # "drop" | "absolute"
    lar_drop_delta: float = 0.0
    lar_abs_threshold: float = 0.7
    lar_seed: int = 0
    seed: int = SEED


# ── Helpers shared across modules ────────────────────────────────


def load_feature_cols() -> tuple[list[str], dict[str, float]]:
    """Return (feature_cols, seed_train_means) from the Step-1 manifest."""
    if not STEP1_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {STEP1_MANIFEST}. Run Step-1 ensemble_reboot.py first."
        )
    m = json.loads(STEP1_MANIFEST.read_text())
    return list(m["final_feature_set"]), dict(m["seed_train_means"])


def load_matrices() -> dict[str, pd.DataFrame]:
    """Load the three OSINT-enriched Z matrices."""
    for p in (Z_KP, Z_DP, Z_PP):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Step-1.5 OSINT enrichment is a prerequisite."
            )
    return {
        "kp": pd.read_csv(Z_KP),
        "dp": pd.read_csv(Z_DP),
        "pp": pd.read_csv(Z_PP),
    }


def build_splits(
    dfs: dict[str, pd.DataFrame],
    *,
    stream_order: str = "temporal",
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Produce the 5 immutable slices: KP_train, KP_test, PP_stream,
    PP_test, DP_probe.

    KP split: stratified 80/20 with seed (matches Step 1 exactly).
    PP split: stratified 80/20 with seed; `stream_order` controls the
              ORDER of the resulting PP_stream:
                'temporal' → sort by JSON `date` then by sample_id
                'shuffle'  → uniform shuffle with seed
    """
    df_kp = dfs["kp"]
    df_dp = dfs["dp"]
    df_pp = dfs["pp"]

    # KP 80/20 stratified.
    kp_train, kp_test = train_test_split(
        df_kp,
        test_size=KP_TEST_RATIO,
        random_state=seed,
        stratify=df_kp["label"].values,
    )

    # PP 80/20 stratified.
    pp_stream_raw, pp_test = train_test_split(
        df_pp,
        test_size=PP_TEST_RATIO,
        random_state=seed,
        stratify=df_pp["label"].values,
    )

    # Order the PP stream.
    if stream_order == "temporal":
        # Pull dates from the JSON.
        with PP_JSON.open() as f:
            entries = json.load(f)
        date_by_id = {int(e["id"]): str(e.get("date") or "")
                      for e in entries}
        pp_stream_raw = pp_stream_raw.copy()
        pp_stream_raw["__date"] = (
            pp_stream_raw["sample_id"].astype(int).map(date_by_id)
        )
        # Empty/missing dates sort last (deterministic).
        pp_stream = pp_stream_raw.sort_values(
            by=["__date", "sample_id"], ascending=[True, True],
            kind="mergesort",   # stable
        ).drop(columns="__date").reset_index(drop=True)
    elif stream_order == "shuffle":
        pp_stream = pp_stream_raw.sample(
            frac=1.0, random_state=seed,
        ).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown stream_order: {stream_order!r}")

    return {
        "kp_train": kp_train.reset_index(drop=True),
        "kp_test": kp_test.reset_index(drop=True),
        "pp_stream": pp_stream,
        "pp_test": pp_test.reset_index(drop=True),
        "dp_probe": df_dp.reset_index(drop=True),
    }


def build_feature_vector(
    row: pd.Series,
    feature_cols: list[str],
    fill_means: dict[str, float],
) -> dict[str, float]:
    """Build a single ML-ready feature dict, mirroring Step 1's
    `prepare_X` behaviour: -1 OSINT sentinels become NaN, then NaN
    becomes the seed-train mean. Never zero-impute.
    """
    from ensemble_reboot import OSINT_NEG1_SENTINEL    # local import

    out: dict[str, float] = {}
    for f in feature_cols:
        mean = fill_means.get(f, 0.0)
        if f in row.index:
            v = row[f]
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = mean
            if f in OSINT_NEG1_SENTINEL and v == -1.0:
                v = mean
            if v != v:        # NaN
                v = mean
            out[f] = v
        else:
            out[f] = mean
    return out
