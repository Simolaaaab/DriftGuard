"""run_smallbench_loop.py — Step-3 AL loop, restricted to the small benchmark.

Thin wrapper around `step3_al.run_al.main_async` that swaps the splits
for the small-benchmark subset BEFORE the loop sees them. Everything
else (selection strategies, drift detector, oracle, verifier, retrain,
evaluator) is reused unchanged so results are directly comparable to
Step-3 numbers.

Provider selection is done via the small-benchmark `llm_providers.py`
registry (--provider deepseek_v4|gpt_5_4|mistral_large|grok_4|…).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step5_smallbench"))

from common import (  # noqa: E402
    AL_RUNS_ROOT, ALRunConfig,
    K_PER_ROUND, MAX_ROUNDS, TOTAL_LLM_CAP, SEED,
    load_feature_cols, load_matrices,
)
from loop import run_al                  # noqa: E402

from load_subset import load_subset_splits  # noqa: E402
from llm_providers import get_llm_client    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_smallbench")


THRESHOLD_PRESET = dict(
    delegate=2.0, kl=1e9, mode="periodic", periodic_every=30,
)
# periodic_every=30 → ~16 rounds on a 500-row stream → ~800 LLM calls/run.
# Cap with --max-rounds for tighter budgets.


async def main_async(args) -> None:
    feature_cols, fill_means = load_feature_cols()
    dfs = load_matrices()
    splits = load_subset_splits(args.manifest, dfs=dfs)
    log.info(f"Manifest: {args.manifest}")
    log.info(
        f"Splits: kp_train={len(splits['kp_train'])} "
        f"kp_test={len(splits['kp_test'])} "
        f"pp_stream(SUBSET)={len(splits['pp_stream'])} "
        f"pp_test(SUBSET)={len(splits['pp_test'])} "
        f"dp_probe={len(splits['dp_probe'])}"
    )

    run_id = args.run_id or f"smallbench_{args.strategy}_{args.provider}"
    run_dir = AL_RUNS_ROOT / run_id

    cfg = ALRunConfig(
        run_id=run_id,
        strategy=args.strategy,
        oracle_mode="llm",
        stream_order=splits["__manifest__"]["stream_order"],
        k_per_round=args.k,
        max_rounds=args.max_rounds,
        total_llm_cap=args.total_llm_cap,
        drift_delegate_threshold=THRESHOLD_PRESET["delegate"],
        drift_kl_threshold=THRESHOLD_PRESET["kl"],
        drift_mode=THRESHOLD_PRESET["mode"],
        periodic_every=THRESHOLD_PRESET["periodic_every"],
        min_pool_for_trigger=args.min_pool_for_trigger or args.k,
        prompt_variant=args.oracle_prompt,
        use_step2_cache=not args.no_step2_cache,
        verifier_mode=args.verifier_mode,
        disable_health_monitor=args.disable_health_monitor,
        seed=args.seed,
    )
    log.info(f"Provider: {args.provider}  run_id={run_id}")

    llm_ctx = get_llm_client(args.provider)
    async with llm_ctx as llm:
        log.info(f"LLM backend={llm.backend} model={llm.model}")
        summary = await run_al(
            config=cfg, splits=splits,
            feature_cols=feature_cols, fill_means=fill_means,
            run_dir=run_dir, llm_client=llm,
            shap_top_features=tuple(args.shap_top.split(",")),
        )

    print()
    print("=" * 60)
    print(f"Smallbench run {run_id} summary:")
    print(f"  provider: {args.provider}  model={llm.model}")
    print(f"  rounds: {summary['n_rounds_completed']}")
    print(f"  cumulative labels: {summary['cumulative_labels']}")
    print(f"  cumulative LLM calls: {summary['cumulative_llm_calls']}")
    if summary["final_eval"]:
        fe = summary["final_eval"]
        print(f"  final PP F1w: {fe['pp_test_f1_weighted']:.4f}  "
              f"KP F1w: {fe['kp_test_f1_weighted']:.4f}  "
              f"DP F1w: {fe['dp_probe_f1_weighted']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="smallbench_v1")
    p.add_argument("--provider", required=True,
                   help="symbolic provider name from llm_providers.PROVIDERS")
    p.add_argument("--strategy", default="drift_anchored",
                   choices=("margin", "qbc", "core_set", "badge",
                            "drift_anchored", "hybrid", "random"))
    p.add_argument("--oracle-prompt", choices=("A", "B", "C", "D"),
                   default="B")
    p.add_argument("--no-step2-cache", action="store_true")
    p.add_argument("--verifier-mode",
                   choices=("off", "t1_only", "t2_only", "t3_only", "full",
                            "hard_filter"),
                   default="off")
    p.add_argument("--disable-health-monitor", action="store_true",
                   default=True,
                   help="default ON for small-benchmark — short runs would "
                        "false-trigger the monitor")
    p.add_argument("--k", type=int, default=K_PER_ROUND)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--total-llm-cap", type=int, default=TOTAL_LLM_CAP)
    p.add_argument("--min-pool-for-trigger", type=int, default=10)
    p.add_argument("--shap-top",
                   default="min_prob,prob_pg,rdap_age_risk,"
                           "rdap_has_registrar,has_https")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
