"""eval_stream_cumulative.py — Per-stream-row retrospective evaluation.

For one AL run, walks PP_stream in temporal order and at each row
predicts under the *live* model snapshot — i.e. the snapshot that was
in production at that row index. Outputs per-row predictions plus
cumulative counters (FN, FP, errors) and rolling F1w / recall_phish.

Produces:
  run_dir/stream_per_row.csv     — one row per stream sample
  run_dir/stream_summary.json    — headline numbers for the trigger
  run_dir/stream_fires.csv       — retrain events with their row_idx

Live-model rule:
  for stream row i, live_round = max{r : trigger_row[r] ≤ i}
  (if no fire has happened yet, live_round = 0, i.e. the bootstrap model)

This is the post-hoc engine for the "saw-tooth Figure 1" and the
"cost of delay" comparison across trigger modes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step1_ensemble"))

from common import (  # noqa: E402
    AL_RUNS_ROOT, build_splits, load_feature_cols, load_matrices,
)
from stream import PPStream  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_stream_cumulative")

ROLLING_WINDOW = 200


def _load_fires(run_dir: Path) -> list[tuple[int, int]]:
    """Return list of (round_idx, trigger_row) sorted by row."""
    p = run_dir / "rounds.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    out: list[tuple[int, int]] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("halted"):
            continue
        out.append((int(r["round"]), int(r["trigger_row"])))
    out.sort(key=lambda x: x[1])
    return out


def _load_snapshots(run_dir: Path, max_round: int) -> dict[int, object]:
    """Load all model_round_N.joblib for N ∈ [0, max_round]."""
    snap_dir = run_dir / "snapshots"
    models: dict[int, object] = {}
    for n in range(max_round + 1):
        p = snap_dir / f"model_round_{n}.joblib"
        if not p.exists():
            continue
        models[n] = joblib.load(p)
    if 0 not in models:
        raise FileNotFoundError(
            f"missing model_round_0.joblib in {snap_dir} — "
            f"cannot bootstrap stream eval"
        )
    log.info(f"loaded {len(models)} snapshots: rounds {sorted(models)}")
    return models


def _build_X(stream_df: pd.DataFrame, feature_cols: list[str],
             fill_means: dict[str, float]) -> np.ndarray:
    from ensemble_reboot import OSINT_NEG1_SENTINEL
    X = np.empty((len(stream_df), len(feature_cols)))
    for j, f in enumerate(feature_cols):
        m = fill_means.get(f, 0.0)
        if f in stream_df.columns:
            col = pd.to_numeric(stream_df[f], errors="coerce")
            if f in OSINT_NEG1_SENTINEL:
                col = col.where(col != -1, other=np.nan)
            X[:, j] = col.fillna(m).to_numpy()
        else:
            X[:, j] = m
    return X


def eval_run_stream(run_id: str) -> dict:
    run_dir = AL_RUNS_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} not found")

    # Pull config to reproduce the same splits the run used.
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path}: run did not finish")
    summary = json.loads(summary_path.read_text())
    cfg = summary.get("config") or {}
    seed = int(cfg.get("seed", 2025))
    stream_order = str(cfg.get("stream_order", "temporal"))

    feature_cols, fill_means = load_feature_cols()
    dfs = load_matrices()
    splits = build_splits(dfs, stream_order=stream_order, seed=seed)
    stream_df = splits["pp_stream"].reset_index(drop=True)

    # Load the frozen scaler from the run.
    scaler = joblib.load(run_dir / "snapshots" / "scaler.joblib")

    # Fires + snapshots.
    fires = _load_fires(run_dir)
    max_round = max((r for r, _ in fires), default=0)
    snapshots = _load_snapshots(run_dir, max_round=max_round)

    # Build the full stream feature matrix once.
    X_raw = _build_X(stream_df, feature_cols, fill_means)
    X_s = scaler.transform(X_raw)
    y_gt = stream_df["label"].astype(int).to_numpy()

    # Precompute "live round at row i" via binary search.
    fire_rows = np.array([r for _, r in fires], dtype=int)
    fire_rounds = np.array([rd for rd, _ in fires], dtype=int)

    # Predict in one shot per snapshot, then mux by live_round.
    # For each row i, live_round = last fire with trigger_row <= i, or 0.
    live_round = np.zeros(len(stream_df), dtype=int)
    if len(fire_rows):
        # np.searchsorted: index of insertion → number of fires ≤ i
        idx = np.searchsorted(fire_rows, np.arange(len(stream_df)),
                              side="right")
        for i in range(len(stream_df)):
            if idx[i] > 0:
                live_round[i] = fire_rounds[idx[i] - 1]

    # Resolve live_round → effective snapshot. When snapshot_every>1 the
    # run only saved a subset of rounds; for any live_round R without a
    # snapshot, fall back to the most recent snapshot ≤ R (the model
    # that was actually running on disk at that point).
    available = sorted(snapshots)
    avail_arr = np.array(available, dtype=int)
    effective_round = np.empty(len(stream_df), dtype=int)
    for i in range(len(stream_df)):
        # largest avail ≤ live_round[i]
        j = np.searchsorted(avail_arr, live_round[i], side="right") - 1
        effective_round[i] = int(avail_arr[max(j, 0)])

    # Predict batch per snapshot for speed.
    preds = np.full(len(stream_df), -1, dtype=int)        # sentinel
    probs = np.full(len(stream_df), np.nan, dtype=float)
    for rnd, model in snapshots.items():
        mask = (effective_round == rnd)
        if not mask.any():
            continue
        p = model.predict_proba(X_s[mask])[:, 1]
        preds[mask] = (p >= 0.5).astype(int)
        probs[mask] = p
    if (preds == -1).any():
        n_missing = int((preds == -1).sum())
        raise RuntimeError(
            f"{n_missing} stream rows were not assigned a prediction "
            f"— likely a snapshot/live_round mismatch bug."
        )

    # Per-row correctness + cumulative counters.
    fn_mask = (preds == 0) & (y_gt == 1)        # phishing missed
    fp_mask = (preds == 1) & (y_gt == 0)        # false alarm
    err_mask = (preds != y_gt)
    cum_fn = np.cumsum(fn_mask.astype(int))
    cum_fp = np.cumsum(fp_mask.astype(int))
    cum_err = np.cumsum(err_mask.astype(int))

    # Rolling window F1w and recall_phish (window = ROLLING_WINDOW).
    roll_f1w = np.full(len(stream_df), np.nan)
    roll_recall_phish = np.full(len(stream_df), np.nan)
    W = ROLLING_WINDOW
    for i in range(W - 1, len(stream_df)):
        sl = slice(i - W + 1, i + 1)
        y_w = y_gt[sl]
        p_w = preds[sl]
        # avoid degenerate cases (no phish in window for recall metric)
        roll_f1w[i] = f1_score(y_w, p_w, average="weighted",
                                zero_division=0)
        if (y_w == 1).any():
            roll_recall_phish[i] = recall_score(y_w, p_w,
                                                pos_label=1,
                                                zero_division=0)

    per_row = pd.DataFrame({
        "row_idx": np.arange(len(stream_df)),
        "sample_id": stream_df["sample_id"].astype(int).to_numpy(),
        "gt": y_gt,
        "pred": preds,
        "prob_phish": probs,
        "live_round": live_round,
        "cum_FN": cum_fn,
        "cum_FP": cum_fp,
        "cum_errors": cum_err,
        "rolling_f1w_w200": roll_f1w,
        "rolling_recall_phish_w200": roll_recall_phish,
    })
    per_row.to_csv(run_dir / "stream_per_row.csv", index=False)

    # Fires CSV (compact, for plot annotations).
    fires_df = pd.DataFrame({
        "round": [r for r, _ in fires],
        "trigger_row": [t for _, t in fires],
    })
    fires_df.to_csv(run_dir / "stream_fires.csv", index=False)

    # Headline summary numbers.
    total_phish = int((y_gt == 1).sum())
    total_benign = int((y_gt == 0).sum())
    stream_summary = {
        "run_id": run_id,
        "drift_mode": cfg.get("drift_mode"),
        "strategy": cfg.get("strategy"),
        "stream_len": int(len(stream_df)),
        "n_fires": int(len(fires)),
        "first_fire_row": int(fire_rows[0]) if len(fire_rows) else None,
        "last_fire_row": int(fire_rows[-1]) if len(fire_rows) else None,
        "total_phish_in_stream": total_phish,
        "total_benign_in_stream": total_benign,
        "cum_FN_total": int(cum_fn[-1]),
        "cum_FP_total": int(cum_fp[-1]),
        "cum_errors_total": int(cum_err[-1]),
        # Stream-wide rates (NOT rolling).
        "stream_recall_phish": float(
            1.0 - cum_fn[-1] / total_phish if total_phish else 0.0
        ),
        "stream_f1w": float(f1_score(y_gt, preds, average="weighted",
                                      zero_division=0)),
        "stream_recall_phish_sklearn": float(
            recall_score(y_gt, preds, pos_label=1, zero_division=0)
        ),
        # Area-under-(1−F1) ≈ time-integrated cost. Use rolling F1w, skip
        # the warmup NaN region (first 199 rows).
        "area_under_1_minus_f1w_w200": float(
            np.nansum(1.0 - roll_f1w[~np.isnan(roll_f1w)])
        ),
        "final_pp_test_f1w": (
            summary.get("final_eval", {}).get("pp_test_f1_weighted")
        ),
        # Oracle cost decomposition. For LAR triggers the probe calls are the
        # LLM queries spent only to *decide when* to fire (0 for periodic /
        # kl_only / or); label_calls is the K×fires spent on the actual
        # retrain labels; total_llm_calls is the full bill.
        "label_calls": int(summary.get("cumulative_llm_calls", 0)),
        "probe_calls": int(summary.get("probe_calls", 0)),
        "total_llm_calls": int(
            summary.get("total_llm_calls",
                        summary.get("cumulative_llm_calls", 0)
                        + summary.get("probe_calls", 0))
        ),
    }
    (run_dir / "stream_summary.json").write_text(
        json.dumps(stream_summary, indent=2)
    )
    log.info(f"[{run_id}] fires={stream_summary['n_fires']}  "
             f"cum_FN={stream_summary['cum_FN_total']}/{total_phish}  "
             f"stream_recall_phish={stream_summary['stream_recall_phish']:.4f}  "
             f"stream_F1w={stream_summary['stream_f1w']:.4f}  "
             f"AUC(1−F1w)={stream_summary['area_under_1_minus_f1w_w200']:.1f}")
    return stream_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=False, default=None,
                   help="one run id to evaluate. If absent, --runs-glob "
                        "must be passed.")
    p.add_argument("--runs-glob", default=None,
                   help="bash-glob over AL_RUNS_ROOT, e.g. "
                        "'margin_*_B_seed2025_full_snap'")
    args = p.parse_args()
    if not args.run_id and not args.runs_glob:
        raise SystemExit("pass --run-id or --runs-glob")

    targets: list[str] = []
    if args.run_id:
        targets.append(args.run_id)
    if args.runs_glob:
        for d in sorted(AL_RUNS_ROOT.glob(args.runs_glob)):
            if d.is_dir():
                targets.append(d.name)

    results = []
    for rid in targets:
        try:
            res = eval_run_stream(rid)
            results.append(res)
        except Exception as e:
            log.error(f"[{rid}] FAILED: {e}")

    if len(results) > 1:
        # Print a compact comparison table.
        print()
        print(f"{'run_id':<55s} {'fires':>6s} {'1st':>5s} {'cum_FN':>7s} "
              f"{'recall_p':>9s} {'F1w':>7s} {'AUC(1-F1)':>10s} "
              f"{'probe':>6s} {'totLLM':>7s}")
        for r in results:
            print(f"{r['run_id']:<55s} "
                  f"{r['n_fires']:>6d} "
                  f"{(r['first_fire_row'] if r['first_fire_row'] is not None else -1):>5d} "
                  f"{r['cum_FN_total']:>7d} "
                  f"{r['stream_recall_phish']:>9.4f} "
                  f"{r['stream_f1w']:>7.4f} "
                  f"{r['area_under_1_minus_f1w_w200']:>10.1f} "
                  f"{r['probe_calls']:>6d} "
                  f"{r['total_llm_calls']:>7d}")


if __name__ == "__main__":
    main()
