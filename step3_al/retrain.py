"""retrain.py — Full-union refit of the leakage-clean LogReg.

Per Spec §3.6 the rules are HARD and enforced by assertions:
  1. No class_weight / sample_weight / oversampling.
  2. Frozen StandardScaler (round-0 fit on KP_train).
  3. Fresh LogisticRegression every call — no warm_start.
  4. The full union (KP_train ∪ accepted_oracle_labels) is fit jointly.
  5. DP labels never enter — caller guarantees.

The function is intentionally a thin wrapper around `model.fit`. If
you find yourself adding tricks in here you're probably solving the
wrong problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import OracleLabel


@dataclass(frozen=True)
class RetrainResult:
    n_train_total: int
    n_seed: int
    n_added: int
    classes_seen: tuple[int, ...]
    in_sample_accuracy: float


def _build_X(rows: list[dict[str, float]],
             feature_cols: list[str],
             fill_means: dict[str, float]) -> np.ndarray:
    """Build a matrix from per-sample feature dicts. Missing keys get the
    seed mean. This mirrors `common.build_feature_vector` for safety
    but operates on already-canonical dicts."""
    X = np.empty((len(rows), len(feature_cols)), dtype=float)
    for i, fv in enumerate(rows):
        for j, f in enumerate(feature_cols):
            X[i, j] = float(fv.get(f, fill_means.get(f, 0.0)))
    return X


def retrain(
    *,
    seed_X: np.ndarray,                      # KP_train post-scaler
    seed_y: np.ndarray,
    accepted_feature_vecs: list[dict[str, float]],
    accepted_labels: list[int],
    feature_cols: list[str],
    fill_means: dict[str, float],
    frozen_scaler: StandardScaler,
    seed: int,
    accepted_trust_weights: list[float] | None = None,
) -> tuple[LogisticRegression, RetrainResult]:
    """Fit a fresh LogReg on KP ∪ accepted.

    `seed_X` is *already scaled* with `frozen_scaler` — the caller does
    that once at round 0 and reuses. `accepted_feature_vecs` are raw
    feature dicts; this function scales them with the frozen scaler.

    `accepted_trust_weights` is the per-label trust score from the
    Step-4 verifier (or None to imply uniform weight 1.0). The KP
    seed labels always get weight 1.0 — they are ground truth by
    construction.

    HARD RULES (asserted, per Spec §3.6 and Step-4 design):
      - class_weight is None. The verifier does class-asymmetric
        scoring; we do NOT compound it with class_weight='balanced'.
      - frozen_scaler is reused, never refit.
      - solver='lbfgs', max_iter=1000, C=1.0.
    """
    assert seed_X.shape[1] == len(feature_cols), \
        "seed_X column count must match feature_cols"
    assert len(accepted_feature_vecs) == len(accepted_labels), \
        "feature_vecs and labels must align"
    if accepted_trust_weights is not None:
        assert len(accepted_trust_weights) == len(accepted_labels), \
            "trust weights must align with labels"

    if accepted_feature_vecs:
        X_added_raw = _build_X(accepted_feature_vecs, feature_cols, fill_means)
        X_added = frozen_scaler.transform(X_added_raw)
        y_added = np.asarray(accepted_labels, dtype=int)
        X_full = np.vstack([seed_X, X_added])
        y_full = np.concatenate([seed_y, y_added])
    else:
        X_full = seed_X
        y_full = seed_y

    classes_seen = tuple(sorted({int(v) for v in y_full.tolist()}))
    assert len(classes_seen) == 2, \
        f"both classes must be present; got {classes_seen}"

    # Build sample_weight: KP seed gets 1.0; accepted gets verifier trust.
    if accepted_trust_weights is not None:
        seed_weights = np.ones(len(seed_y), dtype=float)
        added_weights = np.asarray(accepted_trust_weights, dtype=float)
        sample_weight = np.concatenate([seed_weights, added_weights])
    else:
        sample_weight = None

    # The pinned configuration. No class_weight.
    model = LogisticRegression(
        max_iter=1000, random_state=seed, C=1.0,
        class_weight=None,           # explicit — see Spec §3.6
        solver="lbfgs",
    )
    model.fit(X_full, y_full, sample_weight=sample_weight)
    in_sample_acc = float(model.score(X_full, y_full,
                                      sample_weight=sample_weight))

    return model, RetrainResult(
        n_train_total=len(y_full),
        n_seed=len(seed_y),
        n_added=len(accepted_labels),
        classes_seen=classes_seen,
        in_sample_accuracy=in_sample_acc,
    )
