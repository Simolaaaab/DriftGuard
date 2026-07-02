"""loop.py — The AL orchestrator.

Pure function: `run_al(config, ...)` consumes immutable splits and
returns per-round trajectories. The CLI driver in `run_al.py` owns
all IO.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import (
    ALRunConfig, DriftEvent, OracleLabel, PoolSample, StreamRow,
    UNCERTAINTY_BAND, build_feature_vector,
)
from drift_detector import CompositeDriftDetector, build_reference_hist
from evaluator import evaluate
from oracle import CachedOracle
from pool import UncertaintyPool
from stream_pool import StreamPool
from retrain import retrain
from selection import select
from stream import PPStream
from verifier import TrustAudit, compute_trust
from oracle_health import HealthMonitor, HealthSignal

# Reuse the html loader from Step 2.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "step2_oracle"))
from html_loader import HtmlCacheIndex, load_pp_html  # noqa: E402

log = logging.getLogger("al_loop")


def _to_X(df, cols: list[str], means: dict[str, float]) -> np.ndarray:
    """Build a raw (unscaled) feature matrix for a Z-matrix slice.
    Mirrors `common.build_feature_vector` and `prepare_X` from Step 1.
    """
    from ensemble_reboot import OSINT_NEG1_SENTINEL
    X = np.empty((len(df), len(cols)))
    for j, f in enumerate(cols):
        m = means.get(f, 0.0)
        if f in df.columns:
            col = pd.to_numeric(df[f], errors="coerce")
            if f in OSINT_NEG1_SENTINEL:
                col = col.where(col != -1, other=np.nan)
            X[:, j] = col.fillna(m).to_numpy()
        else:
            X[:, j] = m
    return X


def _per_feature_drift_kl(
    pool_features: np.ndarray,
    kp_X: np.ndarray,
    feature_cols: list[str],
    n_bins: int = 10,
    eps: float = 1e-9,
) -> dict[str, float]:
    """Compute per-feature bin-wise KL(pool || kp) on standardised data.

    Used by drift_anchored selection. Returns 0.0 for degenerate
    features."""
    out: dict[str, float] = {}
    for j, f in enumerate(feature_cols):
        a = pool_features[:, j]
        b = kp_X[:, j]
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        if hi - lo < 1e-9:
            out[f] = 0.0
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        pa, _ = np.histogram(a, bins=bins, density=False)
        pb, _ = np.histogram(b, bins=bins, density=False)
        pa = (pa + eps) / (pa.sum() + eps * len(pa))
        pb = (pb + eps) / (pb.sum() + eps * len(pb))
        out[f] = float(np.sum(pa * np.log(pa / pb)))
    return out


def _save_round_record(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


async def run_al(
    *,
    config: ALRunConfig,
    splits: dict,
    feature_cols: list[str],
    fill_means: dict[str, float],
    run_dir: Path,
    llm_client=None,
    shap_top_features: tuple[str, ...] = (),
    snapshot_every: int = 1,
    llm_labels: dict[int, int] | None = None,
) -> dict:
    """The AL loop. Returns a summary dict; per-round records are
    streamed to `run_dir/rounds.jsonl`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rounds_log = run_dir / "rounds.jsonl"
    eval_log = run_dir / "eval_per_round.csv"
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    if rounds_log.exists():
        rounds_log.unlink()
    if eval_log.exists():
        eval_log.unlink()

    # 1. Bootstrap: round-0 model on KP_train.
    X_kp_train = _to_X(splits["kp_train"], feature_cols, fill_means)
    y_kp_train = splits["kp_train"]["label"].values
    X_kp_test = _to_X(splits["kp_test"], feature_cols, fill_means)
    y_kp_test = splits["kp_test"]["label"].values
    X_pp_test = _to_X(splits["pp_test"], feature_cols, fill_means)
    y_pp_test = splits["pp_test"]["label"].values
    X_dp = _to_X(splits["dp_probe"], feature_cols, fill_means)
    y_dp = splits["dp_probe"]["label"].values

    frozen_scaler = StandardScaler().fit(X_kp_train)
    X_kp_train_s = frozen_scaler.transform(X_kp_train)
    joblib.dump(frozen_scaler, snapshot_dir / "scaler.joblib")

    model, _ = retrain(
        seed_X=X_kp_train_s, seed_y=y_kp_train,
        accepted_feature_vecs=[], accepted_labels=[],
        feature_cols=feature_cols, fill_means=fill_means,
        frozen_scaler=frozen_scaler, seed=config.seed,
    )
    joblib.dump(model, snapshot_dir / "model_round_0.joblib")

    # 2. Build drift reference from KP_train meta_prob.
    p_kp = model.predict_proba(X_kp_train_s)[:, 1]
    ref_hist, bin_edges = build_reference_hist(p_kp, n_bins=10)
    np.save(run_dir / "reference_hist.npy", ref_hist)

    drift_det = CompositeDriftDetector(
        reference_hist=ref_hist, bin_edges=bin_edges,
        window=config.drift_window,
        delegate_threshold=config.drift_delegate_threshold,
        kl_threshold=config.drift_kl_threshold,
        cooldown_rows=config.drift_cooldown_rows,
        mode=config.drift_mode,
        periodic_every=config.periodic_every,
        random_fire_rows=config.random_fire_rows,
        # LAR-trigger wiring (inert for the non-LAR modes).
        llm_labels=llm_labels,
        lar_window=getattr(config, "lar_window", 0),
        lar_probe_k=getattr(config, "lar_probe_k", 0),
        lar_select=getattr(config, "lar_select", "random"),
        lar_rule=getattr(config, "lar_rule", "drop"),
        lar_drop_delta=getattr(config, "lar_drop_delta", 0.0),
        lar_abs_threshold=getattr(config, "lar_abs_threshold", 0.7),
        lar_seed=(getattr(config, "lar_seed", 0) or config.seed),
    )

    pool = UncertaintyPool(capacity=config.pool_capacity)
    # Pure-random-stream rung (ablation): an UNGATED parallel buffer that
    # admits every seen row, identical in lifecycle to the gated pool but
    # without the [0.30,0.70]/disagreement admission filter. Only used when
    # the strategy is "pure_random_stream"; for every other strategy the
    # canonical gated pool is the sole buffer and the code path below is
    # byte-for-byte the original behaviour.
    stream_pool = StreamPool(capacity=config.pool_capacity)
    use_stream_pool = (config.strategy == "pure_random_stream")
    active_pool = stream_pool if use_stream_pool else pool
    stream = PPStream(splits["pp_stream"], feature_cols=feature_cols,
                      fill_means=fill_means)

    # 3. Initial eval (round 0).
    eval_rows: list[dict] = []
    def _eval(rnd: int, n_accepted: int) -> dict:
        rec = {
            "round": rnd, "n_accepted_cumulative": n_accepted,
            "kp_test": evaluate(model, frozen_scaler, X_kp_test, y_kp_test),
            "pp_test": evaluate(model, frozen_scaler, X_pp_test, y_pp_test),
            "dp_probe": evaluate(model, frozen_scaler, X_dp, y_dp),
        }
        flat = {"round": rnd, "n_accepted": n_accepted}
        for slc in ("kp_test", "pp_test", "dp_probe"):
            for k, v in rec[slc].items():
                flat[f"{slc}_{k}"] = v
        eval_rows.append(flat)
        return rec

    initial = _eval(0, 0)
    log.info(f"[R0] KP F1w={initial['kp_test']['f1_weighted']:.4f}  "
             f"PP F1w={initial['pp_test']['f1_weighted']:.4f}  "
             f"DP F1w={initial['dp_probe']['f1_weighted']:.4f}")

    # 4. Accepted-label store (grows across rounds, never shrinks).
    accepted_fv: list[dict[str, float]] = []
    accepted_y: list[int] = []
    accepted_trust: list[float] = []     # Step-4 verifier weights
    cumulative_llm_calls = 0
    round_num = 0
    last_trigger_row = None
    verifier_mode = getattr(config, "verifier_mode", "off")
    html_index = HtmlCacheIndex() if verifier_mode != "off" else None
    log.info(f"Verifier mode: {verifier_mode}")

    # Step-4.5 health monitor. Pure observer of oracle outputs — never
    # modifies labels or weights. Only raises a structured halt signal
    # when the oracle is multi-signal critical for ≥2 consecutive rounds.
    health_disabled = getattr(config, "disable_health_monitor", False)
    health_mon: HealthMonitor | None = (
        None if health_disabled else HealthMonitor()
    )
    log.info(f"Health monitor: {'DISABLED' if health_disabled else 'ENABLED'}")

    oracle = CachedOracle(
        run_dir=run_dir,
        mode=config.oracle_mode,
        gt_by_id=(
            {int(r["sample_id"]): int(r["label"])
             for _, r in splits["pp_stream"].iterrows()}
            if config.oracle_mode == "gt_dryrun" else None
        ),
        concurrency=4,
        prompt_variant=getattr(config, "prompt_variant", "B"),
        use_step2_cache=getattr(config, "use_step2_cache", True),
        noise_mode=getattr(config, "noise_mode", "none"),
        noise_rate=getattr(config, "noise_rate", 0.0),
        noise_seed=getattr(config, "noise_seed", 0),
    )

    lo_band, hi_band = UNCERTAINTY_BAND
    t0 = time.monotonic()
    log.info(f"Stream length: {len(stream)}, max rounds: {config.max_rounds}")

    # 5. The main loop.
    for row in stream:
        # 5a. Score with the current model.
        fv = row.feature_vec
        X_row = frozen_scaler.transform(
            np.array([[fv[f] for f in feature_cols]])
        )
        meta_prob = float(model.predict_proba(X_row)[0, 1])
        prob_sn = float(row.raw_row.get("prob_sn", 0) or 0)
        prob_pg = float(row.raw_row.get("prob_pg", 0) or 0)
        is_delegate = lo_band <= meta_prob <= hi_band

        # 5b. Admission.
        if use_stream_pool:
            # Ungated: every seen row is a candidate (gating removed).
            stream_pool.admit(
                row, meta_prob=meta_prob, prob_sn=prob_sn, prob_pg=prob_pg,
            )
        else:
            pool.admit_if_uncertain(
                row, meta_prob=meta_prob, prob_sn=prob_sn, prob_pg=prob_pg,
            )

        # 5c. Drift tick. sample_id is needed by the LAR modes to look up
        # the cached dirty-oracle label for this row.
        sig = drift_det.tick(DriftEvent(
            row.row_idx, meta_prob, is_delegate, sample_id=row.sample_id,
        ))

        if not sig.triggered:
            continue

        # Pool-size guard: don't honour the trigger if pool can't supply
        # K samples. The detector counters are rolled back so the next
        # firing is on schedule from this row, not from the last accepted
        # trigger.
        snapshot = active_pool.snapshot()
        if len(snapshot) < config.min_pool_for_trigger:
            log.warning(
                f"Trigger at row={row.row_idx} vetoed: pool={len(snapshot)} "
                f"< min_pool_for_trigger={config.min_pool_for_trigger}"
            )
            drift_det.revoke_last_trigger()
            continue

        round_num += 1
        last_trigger_row = row.row_idx
        log.info(
            f"[R{round_num}] trigger at row={row.row_idx}  "
            f"pool={len(snapshot)}  reason={sig.reason}"
        )

        # 5d. Per-feature drift KL (only built lazily for drift_anchored).
        per_feat_drift = {}
        kp_means = {}
        if config.strategy == "drift_anchored":
            pool_mat = np.array([[s.feature_vec.get(f, fill_means.get(f, 0.0))
                                  for f in feature_cols] for s in snapshot])
            per_feat_drift = _per_feature_drift_kl(
                pool_mat, X_kp_train, feature_cols,
            )
            kp_means = {f: float(X_kp_train[:, j].mean())
                        for j, f in enumerate(feature_cols)}

        # 5e. Build accepted matrix for selection's QBC/Hybrid call.
        if accepted_fv:
            X_accepted_raw = np.empty((len(accepted_fv), len(feature_cols)))
            for i, av in enumerate(accepted_fv):
                for j, f in enumerate(feature_cols):
                    X_accepted_raw[i, j] = float(av.get(f, fill_means.get(f, 0.0)))
            X_accepted_s = frozen_scaler.transform(X_accepted_raw)
            y_accepted_arr = np.asarray(accepted_y)
        else:
            X_accepted_s = None
            y_accepted_arr = None

        # 5f. Selection.
        selected = select(
            config.strategy, snapshot, k=config.k_per_round,
            seed=config.seed + round_num,
            feature_cols=feature_cols, fill_means=fill_means,
            frozen_scaler=frozen_scaler,
            seed_X=X_kp_train_s, seed_y=y_kp_train,
            accepted_X=X_accepted_s, accepted_y=y_accepted_arr,
            logreg=model,
            shap_top_features=shap_top_features,
            per_feature_drift_kl=per_feat_drift,
            kp_means=kp_means,
            drift_alpha=getattr(config, "drift_alpha", 0.5),
        )

        # 5g. Oracle.
        oracle_labels = await oracle.label_batch(selected, llm=llm_client)
        # Step-4 verifier: per-label trust score becomes sample_weight.
        # gt_dryrun source is treated as trust=1.0 (perfect by construction).
        round_trust_audit: list[TrustAudit] = []
        accepted_this_round: list[tuple] = []     # (sample, oracle_label, trust)
        for s, lab in zip(selected, oracle_labels):
            if lab.label is None:
                continue
            if verifier_mode == "off" or lab.source == "gt_dryrun":
                audit = TrustAudit(
                    trust=1.0, t1_confidence=float(lab.confidence or 0.0),
                    t2_evidence=1.0, t3_osint=1.0,
                    n_indicators_verified=0, n_indicators_unverifiable=0,
                    n_indicators_contradicted=0, n_indicators_total=0,
                )
            else:
                # Load HTML for evidence verification.
                sn = str(s.raw_row.get("sample_name") or "")
                url = str(s.raw_row.get("url") or "")
                raw_html = load_pp_html(
                    s.sample_id, sample_name=sn, url=url, index=html_index,
                )
                inds_found = (lab.raw or {}).get("indicators_found") or []
                inds_not_found = (lab.raw or {}).get("indicators_not_found") or []
                audit = compute_trust(
                    label=int(lab.label),
                    confidence=float(lab.confidence or 0.0),
                    indicators_found=inds_found,
                    indicators_not_found=inds_not_found,
                    raw_html=raw_html,
                    url=url,
                    domain=str(s.raw_row.get("domain") or ""),
                    osint_row=s.raw_row,
                    mode=verifier_mode,
                )
            round_trust_audit.append(audit)
            # 'hard_filter' mode: trust==0 means drop the label entirely.
            if audit.trust > 0.0 or verifier_mode != "hard_filter":
                accepted_this_round.append((s, lab, audit.trust))

        # Oracle audit: GT vs LLM (only on accepted labels).
        gt_match = sum(
            1 for s, lab in zip(selected, oracle_labels)
            if lab.label is not None
            and lab.label == int(s.raw_row.get("label", -1))
        )

        # Step-4.5 health monitor — label-free oracle quality check.
        # Compute audit on the raw oracle output BEFORE labels enter
        # the retrain store. The monitor is a pure observer.
        health_audit = None
        if health_mon is not None:
            mean_trust_for_health = (
                sum(a.trust for a in round_trust_audit) / len(round_trust_audit)
                if round_trust_audit else None
            )
            health_audit = health_mon.tick(
                round_num, oracle_labels,
                len(accepted_this_round), mean_trust_for_health,
            )
            sev_marker = {"healthy": "🟢", "warn": "🟡",
                          "critical": "🔴"}.get(health_audit.signal.value, "?")
            log.info(
                f"[R{round_num}] oracle health: {sev_marker} "
                f"{health_audit.signal.value} "
                f"(n_critical_signals={health_audit.n_critical_signals}) "
                f"— {' | '.join(health_audit.reasons)}"
            )

        # Halt before retrain if the monitor flagged structural breakage.
        if health_mon is not None and health_mon.should_halt():
            halt_record = {
                "round": round_num,
                "trigger_row": row.row_idx,
                "halted": True,
                "halt_reason": "oracle_health_critical_sustained",
                "health_signal": health_audit.signal.value,
                "health_parse_ok_rate": health_audit.parse_ok_rate,
                "health_class_skew": health_audit.class_skew,
                "health_majority_class": health_audit.majority_class,
                "health_yield_rate": health_audit.yield_rate,
                "health_n_critical_signals": health_audit.n_critical_signals,
                "health_reasons": " | ".join(health_audit.reasons),
                "n_queried": len(oracle_labels),
                "n_would_have_accepted": len(accepted_this_round),
                "cumulative_labels_at_halt": len(accepted_y),
                "cumulative_llm_calls_at_halt": cumulative_llm_calls + len(oracle_labels),
            }
            _save_round_record(rounds_log, halt_record)
            log.error(
                f"HALT @ R{round_num}: oracle structurally broken. "
                f"Last reasons: {health_audit.reasons}. "
                f"NOT injecting {len(accepted_this_round)} labels into retrain. "
                f"Run --disable-health-monitor to bypass."
            )
            break

        # Add to the accepted store.
        for s, lab, trust in accepted_this_round:
            accepted_fv.append(s.feature_vec)
            accepted_y.append(int(lab.label))
            accepted_trust.append(float(trust))
        # Remove ALL queried (whether accepted or uncertain) from pool.
        active_pool.remove([s.sample_id for s in selected])
        cumulative_llm_calls += len(oracle_labels)

        # 5h. Retrain with verifier-weighted samples (sample_weight, not class_weight).
        model, rr = retrain(
            seed_X=X_kp_train_s, seed_y=y_kp_train,
            accepted_feature_vecs=accepted_fv, accepted_labels=accepted_y,
            accepted_trust_weights=(accepted_trust
                                    if verifier_mode != "off" else None),
            feature_cols=feature_cols, fill_means=fill_means,
            frozen_scaler=frozen_scaler, seed=config.seed,
        )
        # Save snapshot every N rounds + at the end.
        if (round_num % snapshot_every == 0 or round_num == config.max_rounds):
            joblib.dump(model, snapshot_dir / f"model_round_{round_num}.joblib")

        # 5i. Living pool: rescore under the new model.
        def _rescorer(s):
            X_one = frozen_scaler.transform(
                np.array([[s.feature_vec.get(f, fill_means.get(f, 0.0))
                           for f in feature_cols]])
            )
            new_mp = float(model.predict_proba(X_one)[0, 1])
            return new_mp, s.raw_row.get("prob_sn"), s.raw_row.get("prob_pg")
        if use_stream_pool:
            # The ungated buffer has no admission criterion, so nothing
            # "graduates" on retrain — the faithful analogue of the gated
            # pool's rescore-eviction is a no-op (samples persist, FIFO-
            # capped, until consumed). Keeps the only difference the filter.
            n_evicted = 0
        else:
            n_kept, n_evicted = pool.rescore(rescorer=_rescorer)
        drift_det.reset_window()

        # 5j. Eval & log.
        rec = _eval(round_num, len(accepted_y))
        # Aggregate trust telemetry for the round.
        if round_trust_audit:
            trust_vals = [a.trust for a in round_trust_audit]
            avg_trust = float(sum(trust_vals) / len(trust_vals))
            min_trust = float(min(trust_vals))
            max_trust = float(max(trust_vals))
        else:
            avg_trust = min_trust = max_trust = 1.0

        round_record = {
            "round": round_num,
            "trigger_row": row.row_idx,
            "delegate_ratio": sig.delegate_ratio,
            "kl": sig.kl,
            # LAR telemetry (None/0 for the non-LAR triggers).
            "lar": sig.lar,
            "lar_baseline": sig.lar_baseline,
            "lar_probes_this_fire": sig.lar_probes,
            "cumulative_probe_calls": drift_det.n_probe_calls,
            "pool_size_pre": len(snapshot),
            "pool_size_post": len(active_pool),
            "pool_rescored_evicted": n_evicted,
            "n_queried": len(oracle_labels),
            "n_accepted": len(accepted_this_round),
            "n_oracle_uncertain": sum(
                1 for lab in oracle_labels if lab.label is None
            ),
            "oracle_gt_agreement": gt_match,
            "trust_mean": avg_trust,
            "trust_min": min_trust,
            "trust_max": max_trust,
            "health_signal": (health_audit.signal.value
                              if health_audit else "disabled"),
            "health_parse_ok_rate": (health_audit.parse_ok_rate
                                     if health_audit else 1.0),
            "health_class_skew": (health_audit.class_skew
                                  if health_audit else 0.5),
            "health_yield_rate": (health_audit.yield_rate
                                  if health_audit else 1.0),
            "health_n_critical_signals": (health_audit.n_critical_signals
                                          if health_audit else 0),
            "cumulative_labels": len(accepted_y),
            "cumulative_llm_calls": cumulative_llm_calls,
            "queried_sample_ids": [s.sample_id for s in selected],
            "kp_test_f1w": rec["kp_test"]["f1_weighted"],
            "pp_test_f1w": rec["pp_test"]["f1_weighted"],
            "dp_probe_f1w": rec["dp_probe"]["f1_weighted"],
            "kp_test_recall_phish": rec["kp_test"]["recall_phish"],
            "pp_test_recall_phish": rec["pp_test"]["recall_phish"],
            "pp_test_ece": rec["pp_test"]["ece"],
        }
        _save_round_record(rounds_log, round_record)
        log.info(
            f"[R{round_num}] PP F1w={rec['pp_test']['f1_weighted']:.4f}  "
            f"KP F1w={rec['kp_test']['f1_weighted']:.4f}  "
            f"DP F1w={rec['dp_probe']['f1_weighted']:.4f}  "
            f"+{len(accepted_this_round)} labels  "
            f"(cum {len(accepted_y)})"
        )

        # 5k. Stop conditions.
        if round_num >= config.max_rounds:
            log.info("Stop: max_rounds reached")
            break
        if cumulative_llm_calls >= config.total_llm_cap:
            log.info("Stop: total LLM cap reached")
            break

    # 6. Save final eval CSV + summary.
    pd.DataFrame(eval_rows).to_csv(eval_log, index=False)
    elapsed = time.monotonic() - t0
    summary = {
        "config": asdict(config),
        "n_rounds_completed": round_num,
        "cumulative_labels": len(accepted_y),
        "cumulative_llm_calls": cumulative_llm_calls,
        # LAR probe cost: LLM calls spent only to *decide when* to fire
        # (0 for periodic/kl_only/or). `total_llm_calls` is the full oracle
        # bill = label calls (K×fires) + probe calls.
        "probe_calls": int(getattr(drift_det, "n_probe_calls", 0)),
        "probe_unlabelled_slots": int(getattr(drift_det, "n_probe_unlabelled", 0)),
        "n_lar_evals": int(getattr(drift_det, "n_lar_evals", 0)),
        "total_llm_calls": int(cumulative_llm_calls
                               + getattr(drift_det, "n_probe_calls", 0)),
        "elapsed_s": elapsed,
        "last_trigger_row": last_trigger_row,
        "oracle_telemetry": oracle.telemetry(),
        "initial_eval": initial,
        "final_eval": eval_rows[-1] if eval_rows else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    log.info(f"Run complete: {round_num} rounds, "
             f"{len(accepted_y)} accepted, "
             f"{cumulative_llm_calls} LLM calls in {elapsed/60:.1f} min")
    return summary
