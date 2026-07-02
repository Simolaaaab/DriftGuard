"""run_al.py — CLI entrypoint for one AL run.

Usage:
    # smoke (gt_dryrun, no LLM key needed)
    python3 reboot/step3_al/run_al.py \\
        --strategy margin --oracle gt_dryrun --max-rounds 5

    # real run with the canonical hybrid strategy
    export AZURE_API_KEY=... AZURE_BASE_URL=... AZURE_MODEL=DeepSeek-V4-Flash
    python3 reboot/step3_al/run_al.py \\
        --strategy hybrid --oracle llm --max-rounds 30

    # threshold preset (data on PP shows the spec defaults never fire)
    python3 reboot/step3_al/run_al.py \\
        --strategy hybrid --oracle llm \\
        --threshold-preset permissive
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from common import (
    AL_RUNS_ROOT, ALRunConfig,
    DRIFT_DELEGATE_THRESHOLD, DRIFT_KL_THRESHOLD,
    K_PER_ROUND, MAX_ROUNDS, TOTAL_LLM_CAP, SEED,
    load_feature_cols, load_matrices, build_splits,
)
from loop import run_al
from llm_client import AsyncLLM

# Lazy/optional: only needed when --provider is used.
_THIS = Path(__file__).resolve()
_REBOOT = _THIS.parents[1]
sys.path.insert(0, str(_REBOOT / "step5_smallbench"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("run_al")


# Defaults derived from PP-stream calibration diagnostics:
#  PP delegate_ratio max=0.100, p95=0.085
#  PP kl max=0.561,  p95=0.476
# The "strict" preset is what the spec approved (AND 0.10 / 0.50).
# Empirically it produces 0 triggers, because the leakage-clean
# LogReg is *too confident* on PP — delegate band catches nothing.
# The "permissive" preset is a calibrated alternative that yields a
# usable trigger rate while preserving AND semantics.
THRESHOLD_PRESETS = {
    # "and"-mode (KL ∧ delegate_ratio gate). On PP these never fire on
    # 'strict'; 'permissive' fires ~3-5×; useful for ablation only.
    "strict":          dict(delegate=0.10, kl=0.50,
                            mode="and",  periodic_every=0),
    "permissive":      dict(delegate=0.05, kl=0.30,
                            mode="and",  periodic_every=0),
    "very_permissive": dict(delegate=0.03, kl=0.20,
                            mode="and",  periodic_every=0),

    # "periodic"-mode (Scheduled Batching). Ignores KL/delegate, fires
    # every N rows. THIS IS THE CANONICAL preset for the AL paper —
    # guarantees a bounded round count on a polarised model.
    # 150 rows ⇒ ~25 rounds on the 3821-row PP_stream.
    "periodic_audit":  dict(delegate=2.0,  kl=1e9,
                            mode="periodic", periodic_every=150),

    # "hybrid_trigger": permissive AND OR periodic fallback. Captures
    # genuine drift events when they happen, falls back to scheduled
    # batching otherwise. Reported alongside periodic_audit as an
    # ablation showing the AND signal is rarely additive.
    "hybrid_trigger":  dict(delegate=0.05, kl=0.30,
                            mode="and_or_periodic", periodic_every=200),

    # "or_audit": pure OR gate on (delegate, KL). Fires if EITHER
    # signal crosses its threshold. More sensitive than AND — catches
    # both output-uncertainty drift (delegate) and input-distribution
    # drift (KL). Uses the same per-signal thresholds as the AND spec
    # (delegate≥0.10, KL≥0.50) so the only knob varied vs `strict` is
    # the gate logic. Expected to fire 2-4× more often than AND on PP.
    "or_audit":        dict(delegate=0.10, kl=0.50,
                            mode="or", periodic_every=0),

    # "kl_only": input-based drift gate. Uses the KL between the
    # rolling meta_prob histogram and the KP_train reference, ignoring
    # delegate_ratio entirely. Defended against the reviewer critique
    # that on a confident leakage-clean LogReg the delegate band is
    # empty: KL is the right signal because drift on a calibrated
    # model manifests as input shift, not output uncertainty.
    # Threshold 0.30 is permissive enough to fire several times across
    # the PP_stream (empirical max KL on PP ≈ 0.56).
    "kl_only":         dict(delegate=2.0,  kl=0.30,
                            mode="kl_only", periodic_every=0),

    # "random_batching": ablation baseline. Fires at random row
    # indices (pre-sampled deterministically from --seed), matching
    # the firing count of periodic_audit. Isolates the contribution
    # of *timing*: if random_batching + drift_anchored ≈ periodic +
    # drift_anchored, the selector is what matters, not the trigger.
    # `periodic_every` is reused here as the cadence proxy that sets
    # n_fires = stream_len // periodic_every.
    "random_batching": dict(delegate=2.0,  kl=1e9,
                            mode="random_batching", periodic_every=150),

    # ── LAR triggers (supervisor's LLM-Agreement-Rate idea) ──────────
    # Every `lar_window` rows, probe `lar_probe_k` of them with the cached
    # dirty oracle and measure LAR = mean(model_pred == llm_label); fire
    # when LAR drops ≥ `lar_drop_delta` below the post-retrain baseline.
    #   *_w100 → window 100, 10 probes (10%);  *_w200 → window 200 (5%).
    #   random_lar    → uniform probe draw within the window
    #   kl_guided_lar → probe the rows in the highest-KL-divergence bins
    # `lar_drop_delta` is a PLACEHOLDER (0.10) pending calibration: run
    # calibrate_lar.py to pick a Δ that yields a #fires comparable to
    # kl_only/periodic, then override with --lar-drop-delta. delegate/kl
    # are set inert so only the LAR machinery gates firing.
    "random_lar_w100":    dict(delegate=2.0, kl=1e9, mode="random_lar",
                               periodic_every=0, lar_window=100,
                               lar_probe_k=10, lar_select="random",
                               lar_drop_delta=0.10),
    "random_lar_w200":    dict(delegate=2.0, kl=1e9, mode="random_lar",
                               periodic_every=0, lar_window=200,
                               lar_probe_k=10, lar_select="random",
                               lar_drop_delta=0.10),
    "kl_guided_lar_w100": dict(delegate=2.0, kl=1e9, mode="kl_guided_lar",
                               periodic_every=0, lar_window=100,
                               lar_probe_k=10, lar_select="kl_guided",
                               lar_drop_delta=0.10),
    "kl_guided_lar_w200": dict(delegate=2.0, kl=1e9, mode="kl_guided_lar",
                               periodic_every=0, lar_window=200,
                               lar_probe_k=10, lar_select="kl_guided",
                               lar_drop_delta=0.10),

    # ── Absolute-threshold LAR (supervisor's "fire when LAR is low") ──
    # Same probing, but fire when LAR < lar_abs_threshold (no baseline).
    # Fires on every drifted window → many more fires than the drop rule.
    # τ=0.7 ⇒ fires whenever agreement ≤ 0.6 (LAR has 0.1 resolution).
    "random_abs_lar_w100":    dict(delegate=2.0, kl=1e9, mode="random_lar",
                                   periodic_every=0, lar_window=100,
                                   lar_probe_k=10, lar_select="random",
                                   lar_rule="absolute", lar_abs_threshold=0.7),
    "random_abs_lar_w200":    dict(delegate=2.0, kl=1e9, mode="random_lar",
                                   periodic_every=0, lar_window=200,
                                   lar_probe_k=10, lar_select="random",
                                   lar_rule="absolute", lar_abs_threshold=0.7),
    "kl_guided_abs_lar_w100": dict(delegate=2.0, kl=1e9, mode="kl_guided_lar",
                                   periodic_every=0, lar_window=100,
                                   lar_probe_k=10, lar_select="kl_guided",
                                   lar_rule="absolute", lar_abs_threshold=0.7),
    "kl_guided_abs_lar_w200": dict(delegate=2.0, kl=1e9, mode="kl_guided_lar",
                                   periodic_every=0, lar_window=200,
                                   lar_probe_k=10, lar_select="kl_guided",
                                   lar_rule="absolute", lar_abs_threshold=0.7),
}

LAR_MODES = ("random_lar", "kl_guided_lar")


def _validate_strategy(s: str) -> str:
    allowed = ("margin", "qbc", "core_set", "badge",
               "drift_anchored", "hybrid", "random", "pure_random_stream")
    if s not in allowed:
        raise argparse.ArgumentTypeError(
            f"strategy must be one of {allowed}, got {s!r}"
        )
    return s


async def main_async(args) -> None:
    feature_cols, fill_means = load_feature_cols()
    dfs = load_matrices()
    run_seed = args.seed if args.seed is not None else SEED
    splits = build_splits(dfs, stream_order=args.stream_order, seed=run_seed)
    log.info(f"Splits: kp_train={len(splits['kp_train'])}  "
             f"kp_test={len(splits['kp_test'])}  "
             f"pp_stream={len(splits['pp_stream'])}  "
             f"pp_test={len(splits['pp_test'])}  "
             f"dp_probe={len(splits['dp_probe'])}")

    # Resolve thresholds + mode + periodic.
    # LAR-mode params default to inert; a preset or CLI override turns them on.
    lar_window = 0
    lar_probe_k = 0
    lar_select = "random"
    lar_rule = "drop"
    lar_drop_delta = 0.0
    lar_abs_threshold = 0.7
    if args.threshold_preset:
        preset = THRESHOLD_PRESETS[args.threshold_preset]
        del_thr = preset["delegate"]
        kl_thr = preset["kl"]
        drift_mode = preset["mode"]
        periodic_every = preset["periodic_every"]
        lar_window = preset.get("lar_window", 0)
        lar_probe_k = preset.get("lar_probe_k", 0)
        lar_select = preset.get("lar_select", "random")
        lar_rule = preset.get("lar_rule", "drop")
        lar_drop_delta = preset.get("lar_drop_delta", 0.0)
        lar_abs_threshold = preset.get("lar_abs_threshold", 0.7)
    else:
        del_thr = (args.drift_delegate_threshold
                   if args.drift_delegate_threshold is not None
                   else DRIFT_DELEGATE_THRESHOLD)
        kl_thr = (args.drift_kl_threshold
                  if args.drift_kl_threshold is not None
                  else DRIFT_KL_THRESHOLD)
        drift_mode = args.drift_mode or "and"
        periodic_every = args.periodic_every or 0
    # CLI overrides (useful for the Δ calibration sweep).
    if args.lar_window is not None:
        lar_window = args.lar_window
    if args.lar_probe_k is not None:
        lar_probe_k = args.lar_probe_k
    if args.lar_select is not None:
        lar_select = args.lar_select
    if args.lar_rule is not None:
        lar_rule = args.lar_rule
    if args.lar_drop_delta is not None:
        lar_drop_delta = args.lar_drop_delta
    if args.lar_abs_threshold is not None:
        lar_abs_threshold = args.lar_abs_threshold
    log.info(f"Drift detector: mode={drift_mode}  delegate≥{del_thr}  "
             f"KL≥{kl_thr}  periodic_every={periodic_every}")

    # LAR triggers need the per-stream dirty-oracle label map up front.
    llm_labels = None
    if drift_mode in LAR_MODES:
        from lar_labels import load_llm_label_map, stream_coverage
        llm_labels = load_llm_label_map()
        cov = stream_coverage(llm_labels, seed=run_seed,
                              stream_order=args.stream_order)
        log.info(
            f"LAR mode={drift_mode}  window={lar_window}  probe_k={lar_probe_k}"
            f"  select={lar_select}  drop_delta={lar_drop_delta}  | "
            f"LLM-label coverage {cov['covered']}/{cov['stream_len']} "
            f"({100*cov['coverage_frac']:.1f}%)"
        )
        if cov["coverage_frac"] < 0.999:
            log.warning(
                f"{cov['missing']} stream rows have NO cached LLM label; the "
                f"LAR detector will skip them within each window. For a clean "
                f"experiment run: python3 reboot/step3_al/lar_labels.py "
                f"--backfill"
            )

    # Pre-sample fire rows for random_batching. Uses periodic_every as
    # the cadence proxy → same expected fire count as periodic_audit
    # for an apples-to-apples timing-ablation comparison.
    random_fire_rows: tuple = ()
    if drift_mode == "random_batching":
        import numpy as _np
        stream_len = len(splits["pp_stream"])
        if periodic_every <= 0:
            periodic_every = 150
        n_fires = max(1, stream_len // periodic_every)
        rng = _np.random.default_rng(run_seed)
        # Avoid the first DRIFT_WINDOW rows (warmup-equivalent) and the
        # very last rows (no pool slack left).
        lo, hi = 50, max(51, stream_len - 50)
        sampled = sorted(int(x) for x in rng.choice(
            _np.arange(lo, hi), size=n_fires, replace=False
        ))
        random_fire_rows = tuple(sampled)
        log.info(f"random_batching: pre-sampled {n_fires} fire rows "
                 f"(seed={run_seed}, range=[{lo},{hi})): "
                 f"first 5 = {sampled[:5]}")

    # SHAP-top-N features for core-set and hybrid: pull from the
    # Step-1 importance CSV if available, else use a sensible default.
    shap_top = tuple(args.shap_top.split(","))
    log.info(f"SHAP-top features for diversity: {shap_top}")

    run_id = args.run_id or f"{args.strategy}_{args.oracle}_{args.stream_order}"
    run_dir = AL_RUNS_ROOT / run_id

    cfg = ALRunConfig(
        run_id=run_id,
        strategy=args.strategy,
        oracle_mode=args.oracle,
        stream_order=args.stream_order,
        k_per_round=args.k,
        max_rounds=args.max_rounds,
        total_llm_cap=args.total_llm_cap,
        drift_delegate_threshold=del_thr,
        drift_kl_threshold=kl_thr,
        drift_mode=drift_mode,
        periodic_every=periodic_every,
        random_fire_rows=random_fire_rows,
        min_pool_for_trigger=args.min_pool_for_trigger or args.k,
        lar_window=lar_window,
        lar_probe_k=lar_probe_k,
        lar_select=lar_select,
        lar_rule=lar_rule,
        lar_drop_delta=lar_drop_delta,
        lar_abs_threshold=lar_abs_threshold,
        lar_seed=(args.lar_seed if args.lar_seed is not None else run_seed),
        prompt_variant=args.oracle_prompt,
        use_step2_cache=not args.no_step2_cache,
        verifier_mode=args.verifier_mode,
        disable_health_monitor=args.disable_health_monitor,
        noise_mode=args.noise_mode,
        noise_rate=args.noise_rate,
        noise_seed=(args.noise_seed if args.noise_seed != 0 else run_seed),
        drift_alpha=args.drift_alpha,
        seed=run_seed,
    )
    if args.noise_mode != "none" and args.oracle != "gt_dryrun":
        raise SystemExit(
            f"--noise-mode={args.noise_mode!r} requires --oracle gt_dryrun "
            f"(got --oracle {args.oracle!r}); we need GT to flip."
        )
    if args.noise_mode != "none":
        log.info(
            f"Noise injection: mode={args.noise_mode} "
            f"rate={args.noise_rate} noise_seed={cfg.noise_seed}"
        )

    # Wire the LLM client only if needed.
    if args.oracle == "llm":
        # Allow graceful fallback to cache-only when credentials are
        # absent. The loop will error out on a true cache miss
        # (returns "cache_miss_no_llm" label which is skipped).
        try:
            if args.provider:
                # Multi-provider bench path: dispatch through the
                # llm_providers registry (Azure-routed for all 7 models).
                from llm_providers import get_llm_client
                llm_ctx = get_llm_client(args.provider)
                log.info(f"Provider dispatch: {args.provider}")
            else:
                llm_ctx = AsyncLLM.from_env(model=args.model)
        except Exception as e:
            log.warning(f"No LLM credentials available ({e}). "
                        f"Running in cache-only mode — any cache miss "
                        f"will produce label=None.")
            llm_ctx = None
        if llm_ctx is not None:
            async with llm_ctx as llm:
                log.info(f"LLM backend={llm.backend} model={llm.model}")
                summary = await run_al(
                    config=cfg, splits=splits,
                    feature_cols=feature_cols, fill_means=fill_means,
                    run_dir=run_dir, llm_client=llm,
                    shap_top_features=shap_top,
                    snapshot_every=args.snapshot_every,
                    llm_labels=llm_labels,
                )
        else:
            summary = await run_al(
                config=cfg, splits=splits,
                feature_cols=feature_cols, fill_means=fill_means,
                run_dir=run_dir, llm_client=None,
                shap_top_features=shap_top,
                snapshot_every=args.snapshot_every,
                llm_labels=llm_labels,
            )
    else:
        summary = await run_al(
            config=cfg, splits=splits,
            feature_cols=feature_cols, fill_means=fill_means,
            run_dir=run_dir, llm_client=None,
            shap_top_features=shap_top,
            snapshot_every=args.snapshot_every,
            llm_labels=llm_labels,
        )

    print()
    print("=" * 60)
    print(f"Run {run_id} summary:")
    print(f"  rounds: {summary['n_rounds_completed']}")
    print(f"  cumulative labels: {summary['cumulative_labels']}")
    print(f"  cumulative LLM calls: {summary['cumulative_llm_calls']}")
    if summary["final_eval"]:
        fe = summary["final_eval"]
        print(f"  final PP F1w: {fe['pp_test_f1_weighted']:.4f}  "
              f"KP F1w: {fe['kp_test_f1_weighted']:.4f}  "
              f"DP F1w: {fe['dp_probe_f1_weighted']:.4f}")
    print(f"  artifacts: {run_dir.relative_to(AL_RUNS_ROOT.parents[2])}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True, type=_validate_strategy,
                   help="margin|qbc|core_set|badge|drift_anchored|hybrid|"
                        "random|pure_random_stream")
    p.add_argument("--oracle", choices=("llm", "gt_dryrun"), default="llm")
    p.add_argument("--stream-order", choices=("temporal", "shuffle"),
                   default="temporal",
                   help="temporal = chronological by JSON 'date' (canonical)")
    p.add_argument("--k", type=int, default=K_PER_ROUND)
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    p.add_argument("--total-llm-cap", type=int, default=TOTAL_LLM_CAP)
    p.add_argument("--threshold-preset",
                   choices=list(THRESHOLD_PRESETS), default=None,
                   help="strict (spec) | permissive (PP-calibrated)")
    p.add_argument("--drift-delegate-threshold", type=float, default=None)
    p.add_argument("--drift-kl-threshold", type=float, default=None)
    p.add_argument("--drift-mode",
                   choices=("and", "or", "periodic", "and_or_periodic",
                            "kl_only", "random_batching",
                            "random_lar", "kl_guided_lar"),
                   default=None,
                   help="override preset's mode")
    p.add_argument("--periodic-every", type=int, default=None,
                   help="rows between periodic triggers (default 150 in preset)")
    # ── LAR trigger overrides (random_lar / kl_guided_lar) ───────────
    p.add_argument("--lar-window", type=int, default=None,
                   help="rows per LAR measurement window (e.g. 100 or 200)")
    p.add_argument("--lar-probe-k", type=int, default=None,
                   help="LLM probes per window for the LAR estimate (e.g. 10)")
    p.add_argument("--lar-select", choices=("random", "kl_guided"),
                   default=None,
                   help="how to pick the probed rows within the window")
    p.add_argument("--lar-rule", choices=("drop", "absolute"), default=None,
                   help="'drop' = fire on LAR drop vs baseline; 'absolute' = "
                        "fire when LAR < --lar-abs-threshold (no baseline).")
    p.add_argument("--lar-drop-delta", type=float, default=None,
                   help="fire when LAR drops ≥ this below the post-retrain "
                        "baseline. Calibrate with calibrate_lar.py.")
    p.add_argument("--lar-abs-threshold", type=float, default=None,
                   help="for --lar-rule absolute: fire when LAR < this "
                        "(default 0.7).")
    p.add_argument("--lar-seed", type=int, default=None,
                   help="seed for the random LAR probe draw (default: run seed)")
    p.add_argument("--min-pool-for-trigger", type=int, default=None,
                   help="veto trigger if pool < this. Default: --k value.")
    p.add_argument("--oracle-prompt", choices=("A", "B", "C", "D"),
                   default="B",
                   help="LLM prompt variant for AL queries. Default 'B' "
                        "(probs+OSINT) — diag_b1 showed +17pp vs 'D' on "
                        "AL boundary samples. Use 'D' to reproduce the "
                        "old behaviour.")
    p.add_argument("--no-step2-cache", action="store_true",
                   help="ignore the Step-2 D cache for variant purity")
    p.add_argument(
        "--verifier-mode",
        choices=("off", "t1_only", "t2_only", "t3_only", "full",
                 "hard_filter"),
        default="full",
        help="Step-4 verifier mode. 'full' = class-asymmetric soft trust "
             "(T1+T2+T3) as sample_weight in retrain. 'off' reproduces "
             "Step-3 baseline. Single-tier and 'hard_filter' are ablations.",
    )
    p.add_argument(
        "--disable-health-monitor",
        action="store_true",
        help="Disable the Step-4.5 oracle health monitor. By default the "
             "monitor observes oracle outputs and halts the run if the "
             "LLM is structurally broken (calibrated against Kimi-K2.6 "
             "ablation — 31.9%% parse_ok / 100%% mono-class). Use this "
             "flag to force a run regardless of health signals.",
    )
    p.add_argument("--drift-alpha", type=float, default=0.5,
                   help="drift_anchored score weight on the per-feature drift "
                        "term. 0.5 = canonical. 0.0 = pure margin sampling. "
                        "Only affects --strategy drift_anchored.")
    p.add_argument("--shap-top",
                   default="min_prob,prob_pg,rdap_age_risk,"
                           "rdap_has_registrar,has_https",
                   help="comma-separated feature names for core-set diversity")
    p.add_argument("--model", default=None,
                   help="override LLM model (e.g. DeepSeek-V4-Flash). "
                        "Ignored when --provider is used.")
    p.add_argument("--provider", default=None,
                   help="symbolic provider name from step5_smallbench/"
                        "llm_providers.PROVIDERS "
                        "(deepseek_v4|gpt_5_4|gpt_5_4_mini|gpt_5_1|"
                        "Llama|mistral_large|grok_4|kimi_k2_6). "
                        "When set, overrides --model and reads creds via "
                        "the registry instead of AsyncLLM.from_env.")
    p.add_argument("--run-id", default=None,
                   help="output dir name; default = strategy_oracle_streamorder")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for splits, selection, and model. "
                        "Default 2025. Use {2025,2026,2027} for K=3 bands.")
    p.add_argument("--snapshot-every", type=int, default=1,
                   help="save a model snapshot every N rounds. Default 1 "
                        "(every round) — required for the per-stream-row "
                        "cost-of-delay eval. Bump to 5 to save disk when "
                        "running cheap sanity sweeps.")
    # ── Noise ablation (only meaningful with --oracle gt_dryrun) ────
    p.add_argument(
        "--noise-mode",
        choices=("none", "random_flip",
                 "adversarial_p2b", "adversarial_b2p"),
        default="none",
        help="Inject deterministic noise into gt_dryrun labels. "
             "random_flip: flip GT with prob=rate (symmetric). "
             "adversarial_p2b: flip ONLY phish→benign at rate. "
             "adversarial_b2p: flip ONLY benign→phish at rate. "
             "Requires --oracle gt_dryrun. Used by Step-4 noise ablation.",
    )
    p.add_argument(
        "--noise-rate", type=float, default=0.0,
        help="Flip probability for --noise-mode (0.0..1.0).",
    )
    p.add_argument(
        "--noise-seed", type=int, default=0,
        help="Seed for the deterministic flip decision. Same "
             "(sample_id, noise_seed) → same flip. Vary across K seeds "
             "for variance bands.",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
