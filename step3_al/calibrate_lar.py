"""calibrate_lar.py — Offline Δ calibration for the LAR triggers.

The LAR trigger fires when the LLM-Agreement Rate drops ≥ Δ below its
post-retrain baseline. Δ cannot be picked blind: too small ⇒ it fires every
window (budget blow-up), too large ⇒ it never fires. This script computes the
LAR trajectory of the *round-0* model (no retraining) over the whole PP_stream
and reports, for a grid of Δ, how many windows would have triggered against
the round-0 baseline.

Caveat (stated honestly): the real loop retrains and resets the baseline after
every fire, so this no-retrain trajectory only brackets Δ for the *first* fire
and gives a rough fire-count. Use it to pick 1–2 candidate Δ, then run the real
loop and read #fires from eval_stream_cumulative.py; adjust if far from the
kl_only/periodic band (≈25–30 fires).

Usage:
    python3 reboot/step3_al/calibrate_lar.py --window 200 --probe-k 10 \\
        --select random          # or kl_guided
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

THIS = Path(__file__).resolve()
REBOOT = THIS.parents[1]
sys.path.insert(0, str(REBOOT / "step3_al"))
sys.path.insert(0, str(REBOOT / "step1_ensemble"))

from common import (  # noqa: E402
    DriftEvent, UNCERTAINTY_BAND,
    build_splits, load_feature_cols, load_matrices,
)
from drift_detector import CompositeDriftDetector, build_reference_hist  # noqa: E402
from lar_labels import load_llm_label_map, stream_coverage  # noqa: E402
from retrain import retrain  # noqa: E402
from stream import PPStream  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("calibrate_lar")


def _to_X(df, cols, means):
    from ensemble_reboot import OSINT_NEG1_SENTINEL
    import pandas as pd
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--probe-k", type=int, default=10)
    p.add_argument("--select", choices=("random", "kl_guided"), default="random")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--stream-order", default="temporal")
    args = p.parse_args()

    feature_cols, fill_means = load_feature_cols()
    dfs = load_matrices()
    splits = build_splits(dfs, stream_order=args.stream_order, seed=args.seed)

    # Round-0 model on KP_train, frozen scaler.
    X_kp = _to_X(splits["kp_train"], feature_cols, fill_means)
    y_kp = splits["kp_train"]["label"].values
    scaler = StandardScaler().fit(X_kp)
    X_kp_s = scaler.transform(X_kp)
    model, _ = retrain(seed_X=X_kp_s, seed_y=y_kp,
                       accepted_feature_vecs=[], accepted_labels=[],
                       feature_cols=feature_cols, fill_means=fill_means,
                       frozen_scaler=scaler, seed=args.seed)
    ref_hist, bin_edges = build_reference_hist(model.predict_proba(X_kp_s)[:, 1])

    llm_labels = load_llm_label_map()
    cov = stream_coverage(llm_labels, seed=args.seed, stream_order=args.stream_order)
    log.info(f"LLM-label coverage: {cov['covered']}/{cov['stream_len']} "
             f"({100*cov['coverage_frac']:.1f}%)")

    # Detector in LAR mode with an unreachable Δ so it never fires/resets →
    # we read the raw LAR series from the telemetry.
    det = CompositeDriftDetector(
        reference_hist=ref_hist, bin_edges=bin_edges,
        mode=args.select == "kl_guided" and "kl_guided_lar" or "random_lar",
        llm_labels=llm_labels, lar_window=args.window, lar_probe_k=args.probe_k,
        lar_select=args.select, lar_drop_delta=2.0, lar_seed=args.seed,
    )

    stream = PPStream(splits["pp_stream"], feature_cols=feature_cols,
                      fill_means=fill_means)
    lo, hi = UNCERTAINTY_BAND
    lar_series: list[tuple[int, float]] = []     # (row_idx, lar)
    for row in stream:
        X_row = scaler.transform(np.array([[row.feature_vec[f] for f in feature_cols]]))
        mp = float(model.predict_proba(X_row)[0, 1])
        sig = det.tick(DriftEvent(row.row_idx, mp, lo <= mp <= hi,
                                  sample_id=row.sample_id))
        if sig.reason.startswith("lar_") and sig.lar is not None \
                and "accumulating" not in sig.reason:
            lar_series.append((row.row_idx, sig.lar))

    if not lar_series:
        log.error("No LAR measurements produced — window too large for stream?")
        return

    rows, lars = zip(*lar_series)
    lars = np.array(lars)
    baseline = lars[0]
    print(f"\n=== LAR trajectory (round-0 model, no retrain) ===")
    print(f"select={args.select}  window={args.window}  probe_k={args.probe_k}")
    print(f"measurements={len(lars)}  baseline(LAR@first window)={baseline:.3f}")
    print(f"LAR  min={lars.min():.3f}  max={lars.max():.3f}  "
          f"mean={lars.mean():.3f}  last={lars[-1]:.3f}")
    print(f"max drop below baseline = {baseline - lars.min():.3f}\n")
    print(f"{'Δ':>6s} {'#windows drop≥Δ vs fixed baseline':>34s}")
    for d in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30):
        n = int(np.sum((baseline - lars) >= d))
        print(f"{d:>6.2f} {n:>34d}")
    print("\n(Use this to bracket Δ. Real #fires will differ because the loop "
          "resets the baseline after each retrain — verify with the real run.)")
    print("Per-window LAR:")
    print("  " + "  ".join(f"r{r}:{v:.2f}" for r, v in lar_series))


if __name__ == "__main__":
    main()
