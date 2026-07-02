"""run_pure_random_al.py — Pure-random AL baseline (NO drift, NO pool).

What this is and is NOT:
  - This is **not** Step-3's `--strategy random`. That one samples
    uniformly *from the uncertainty pool*, which is itself filtered
    by the uncertainty band (0.30, 0.70) ∪ proposer_disagreement ≥ 0.35.
    Therefore it's "random *conditional on being uncertain*".
  - This IS the baseline the supervisor asked for: pre-sample `budget`
    row indices uniformly from [0, len(stream)) once at the start
    (deterministic by --seed), then iterate the stream and send those
    rows — and only those rows — to the oracle. No drift detection,
    no pool filtering, no selection logic at all. The model is retrained
    every `--k` accumulated labels, matching Step-3's round cadence.

This isolates the **label-budget contribution** from the **selection
contribution**. The gap drift_anchored+B − pure_random is exactly the
"selection strategy adds value beyond labeling more samples" claim
that Reviewer 2 will ask about.

Runs on either the small benchmark (default) or the full dataset
(--full-dataset) so the same baseline can populate both the Main
Results table and the small-bench comparative table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step5_smallbench"))
sys.path.insert(0, str(REBOOT / "step2_oracle"))

from common import (  # noqa: E402
    AL_RUNS_ROOT, ALRunConfig, OracleLabel,
    K_PER_ROUND, MAX_ROUNDS, TOTAL_LLM_CAP, SEED,
    build_feature_vector, build_splits, load_feature_cols, load_matrices,
)
from evaluator import evaluate            # noqa: E402
from oracle import CachedOracle           # noqa: E402
from retrain import retrain               # noqa: E402
from stream import PPStream               # noqa: E402

from html_loader import HtmlCacheIndex    # noqa: E402

from load_subset import load_subset_splits  # noqa: E402
from llm_providers import get_llm_client    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pure_random_al")


def _to_X(df: pd.DataFrame, cols: list[str],
          means: dict[str, float]) -> np.ndarray:
    """Same imputation policy as Step-3's loop._to_X."""
    sys.path.insert(0, str(REBOOT / "step1_ensemble"))
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


async def run_pure_random(
    *, splits: dict, feature_cols: list[str], fill_means: dict[str, float],
    run_dir: Path, llm_client, provider_name: str,
    budget: int, k_per_round: int, seed: int,
    prompt_variant: str, use_step2_cache: bool, max_rounds: int,
    snapshot_every: int = 1,
) -> dict:
    """The pure-random AL loop. Mirrors run_al() but with NO drift trigger
    and NO pool — just oracle-the-sampled-positions, retrain every K."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rounds_log = run_dir / "rounds.jsonl"
    eval_log = run_dir / "eval_per_round.csv"
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    if rounds_log.exists():
        rounds_log.unlink()
    if eval_log.exists():
        eval_log.unlink()

    # 1. Eval slices & bootstrap model.
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
        frozen_scaler=frozen_scaler, seed=seed,
    )
    joblib.dump(model, snapshot_dir / "model_round_0.joblib")

    # 2. Pre-sample budget positions from the stream — uniform, no replacement.
    stream_len = len(splits["pp_stream"])
    if budget > stream_len:
        log.warning(f"budget={budget} exceeds stream_len={stream_len}; "
                    f"using full stream")
        budget = stream_len
    rng = np.random.default_rng(seed)
    sampled_positions = sorted(int(x) for x in rng.choice(
        stream_len, size=budget, replace=False
    ))
    sampled_set = set(sampled_positions)
    log.info(f"Pre-sampled {budget} stream positions "
             f"(uniform over [0, {stream_len}))")
    log.info(f"  first 5 positions: {sampled_positions[:5]}")

    # 3. Set up oracle (re-uses Step-3 CachedOracle).
    if llm_client is None:
        # gt_dryrun fallback: GT labels, no LLM.
        oracle = CachedOracle(
            run_dir=run_dir, mode="gt_dryrun",
            gt_by_id={int(r["sample_id"]): int(r["label"])
                      for _, r in splits["pp_stream"].iterrows()},
            concurrency=4,
            prompt_variant=prompt_variant,
            use_step2_cache=False,
        )
    else:
        oracle = CachedOracle(
            run_dir=run_dir, mode="llm",
            gt_by_id=None,
            concurrency=4,
            prompt_variant=prompt_variant,
            use_step2_cache=use_step2_cache,
        )

    # 4. Eval helper.
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

    # 5. Walk the stream; batch-query every k_per_round accumulated samples.
    stream = PPStream(splits["pp_stream"],
                      feature_cols=feature_cols, fill_means=fill_means)

    accepted_fv: list[dict[str, float]] = []
    accepted_y: list[int] = []
    cumulative_llm_calls = 0
    round_num = 0
    pending: list = []        # PoolSample-like objects waiting to be batched

    # PoolSample import for type-compatibility with oracle.label_batch.
    from common import PoolSample as _PoolSample

    t0 = time.monotonic()
    for row_idx, row in enumerate(stream):
        if row_idx not in sampled_set:
            continue
        ps = _PoolSample(
            sample_id=row.sample_id, row_idx=row.row_idx,
            feature_vec=row.feature_vec,
            meta_prob=0.5, proposer_disagreement=0.0,
            raw_row=row.raw_row,
        )
        pending.append(ps)

        if len(pending) < k_per_round:
            continue

        round_num += 1
        log.info(f"[R{round_num}] dispatching batch of {len(pending)} "
                 f"oracle queries (stream pos≈{row_idx})")
        oracle_labels = await oracle.label_batch(pending, llm=llm_client)
        accepted_this_round = [(s, lab)
                               for s, lab in zip(pending, oracle_labels)
                               if lab.label is not None]
        for s, lab in accepted_this_round:
            accepted_fv.append(s.feature_vec)
            accepted_y.append(int(lab.label))
        cumulative_llm_calls += len(oracle_labels)

        model, _ = retrain(
            seed_X=X_kp_train_s, seed_y=y_kp_train,
            accepted_feature_vecs=accepted_fv, accepted_labels=accepted_y,
            feature_cols=feature_cols, fill_means=fill_means,
            frozen_scaler=frozen_scaler, seed=seed,
        )
        if (round_num % snapshot_every == 0 or round_num == max_rounds):
            joblib.dump(model, snapshot_dir / f"model_round_{round_num}.joblib")

        rec = _eval(round_num, len(accepted_y))
        round_record = {
            "round": round_num,
            "trigger_row": int(row_idx),
            "n_queried": len(oracle_labels),
            "n_accepted": len(accepted_this_round),
            "cumulative_labels": len(accepted_y),
            "cumulative_llm_calls": cumulative_llm_calls,
            "queried_sample_ids": [s.sample_id for s in pending],
            "pp_test_f1w": rec["pp_test"]["f1_weighted"],
            "kp_test_f1w": rec["kp_test"]["f1_weighted"],
            "dp_probe_f1w": rec["dp_probe"]["f1_weighted"],
        }
        with rounds_log.open("a") as f:
            f.write(json.dumps(round_record) + "\n")
        log.info(f"[R{round_num}] PP F1w={rec['pp_test']['f1_weighted']:.4f}  "
                 f"KP F1w={rec['kp_test']['f1_weighted']:.4f}  "
                 f"DP F1w={rec['dp_probe']['f1_weighted']:.4f}  "
                 f"(+{len(accepted_this_round)} labels, "
                 f"cum {len(accepted_y)})")
        pending = []

        if round_num >= max_rounds:
            log.info("Stop: max_rounds reached")
            break
        if cumulative_llm_calls >= budget:
            log.info("Stop: budget exhausted")
            break

    # Drain pending if any remain (don't waste paid budget).
    if pending and cumulative_llm_calls < budget and round_num < max_rounds:
        round_num += 1
        log.info(f"[R{round_num}] (drain) dispatching final "
                 f"{len(pending)} samples")
        oracle_labels = await oracle.label_batch(pending, llm=llm_client)
        accepted_this_round = [(s, lab)
                               for s, lab in zip(pending, oracle_labels)
                               if lab.label is not None]
        for s, lab in accepted_this_round:
            accepted_fv.append(s.feature_vec)
            accepted_y.append(int(lab.label))
        cumulative_llm_calls += len(oracle_labels)
        if accepted_this_round:
            model, _ = retrain(
                seed_X=X_kp_train_s, seed_y=y_kp_train,
                accepted_feature_vecs=accepted_fv, accepted_labels=accepted_y,
                feature_cols=feature_cols, fill_means=fill_means,
                frozen_scaler=frozen_scaler, seed=seed,
            )
            joblib.dump(model, snapshot_dir / f"model_round_{round_num}.joblib")
        rec = _eval(round_num, len(accepted_y))
        log.info(f"[R{round_num}] PP F1w={rec['pp_test']['f1_weighted']:.4f}")

    pd.DataFrame(eval_rows).to_csv(eval_log, index=False)
    elapsed = time.monotonic() - t0
    summary = {
        "provider": provider_name,
        "model": (llm_client.model if llm_client is not None else "gt_dryrun"),
        "strategy": "pure_random",
        "budget": budget,
        "k_per_round": k_per_round,
        "seed": seed,
        "n_rounds_completed": round_num,
        "cumulative_labels": len(accepted_y),
        "cumulative_llm_calls": cumulative_llm_calls,
        "elapsed_s": elapsed,
        "oracle_telemetry": oracle.telemetry(),
        "initial_eval": initial,
        "final_eval": eval_rows[-1] if eval_rows else None,
        "manifest_name": splits.get("__manifest__", {}).get("name"),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    log.info(f"Run complete: {round_num} rounds, {len(accepted_y)} accepted, "
             f"{cumulative_llm_calls} LLM calls in {elapsed/60:.1f} min")
    return summary


async def main_async(args) -> None:
    feature_cols, fill_means = load_feature_cols()
    dfs = load_matrices()

    if args.full_dataset:
        log.info("Mode: FULL DATASET (no subset)")
        splits = build_splits(dfs, stream_order=args.stream_order,
                              seed=args.seed)
    else:
        log.info(f"Mode: SMALL BENCHMARK (manifest={args.manifest})")
        splits = load_subset_splits(args.manifest, dfs=dfs)

    run_id = args.run_id or (
        f"pure_random_{args.provider}"
        f"_{'full' if args.full_dataset else 'smallbench'}"
        f"_seed{args.seed}"
    )
    run_dir = AL_RUNS_ROOT / run_id

    if args.provider == "gt_dryrun":
        log.info("Oracle: gt_dryrun (no LLM call) — perfect-oracle baseline")
        # Sentinel for in-script identification; not a real LLM client.
        class _GTStub:
            model = "gt_dryrun"
            backend = "gt_dryrun"
        summary = await run_pure_random(
            splits=splits, feature_cols=feature_cols, fill_means=fill_means,
            run_dir=run_dir, llm_client=None, provider_name="gt_dryrun",
            budget=args.budget, k_per_round=args.k, seed=args.seed,
            prompt_variant=args.oracle_prompt,
            use_step2_cache=False,
            max_rounds=args.max_rounds,
        )
        llm = _GTStub()
    else:
        llm_ctx = get_llm_client(args.provider)
        async with llm_ctx as llm:
            log.info(f"LLM backend={llm.backend} model={llm.model}")
            summary = await run_pure_random(
                splits=splits, feature_cols=feature_cols,
                fill_means=fill_means,
                run_dir=run_dir, llm_client=llm,
                provider_name=args.provider,
                budget=args.budget, k_per_round=args.k, seed=args.seed,
                prompt_variant=args.oracle_prompt,
                use_step2_cache=not args.no_step2_cache,
                max_rounds=args.max_rounds,
            )

    print()
    print("=" * 60)
    print(f"PURE-RANDOM AL  run_id={run_id}")
    print(f"  provider:    {args.provider}  model={llm.model}")
    print(f"  rounds:      {summary['n_rounds_completed']}")
    print(f"  labels:      {summary['cumulative_labels']}")
    print(f"  LLM calls:   {summary['cumulative_llm_calls']}")
    if summary["final_eval"]:
        fe = summary["final_eval"]
        print(f"  final PP F1w: {fe['pp_test_f1_weighted']:.4f}  "
              f"KP F1w: {fe['kp_test_f1_weighted']:.4f}  "
              f"DP F1w: {fe['dp_probe_f1_weighted']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="smallbench_v1")
    p.add_argument("--full-dataset", action="store_true",
                   help="run on full PP_stream/PP_test instead of subset")
    p.add_argument("--stream-order", choices=("temporal", "shuffle"),
                   default="temporal")
    p.add_argument("--provider", required=True)
    p.add_argument("--budget", type=int, default=250,
                   help="total max LLM calls. Default 250 (small bench).")
    p.add_argument("--k", type=int, default=K_PER_ROUND,
                   help="labels per retrain round. Default 50.")
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    p.add_argument("--oracle-prompt", choices=("A", "B", "C", "D"),
                   default="B")
    p.add_argument("--no-step2-cache", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
