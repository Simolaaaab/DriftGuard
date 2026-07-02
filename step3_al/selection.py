"""selection.py — Seven AL selection strategies behind one dispatch.

Strategies (per Spec §3.4 + reviewer baseline):

  margin            top-K |meta_prob - 0.5| smallest
  qbc               top-K committee disagreement (LogReg/GB/RF)
  core_set          k-Center greedy on SHAP-top-5 vectors
  badge             k-Means++ on LogReg gradient embeddings
  drift_anchored    margin reweighted by per-feature drift KL  (ours)
  hybrid            60% margin + 25% core-set + 15% qbc        (canonical)
  random            uniform random over the (gated) pool (reviewer baseline)
  pure_random_stream  uniform random over the UNGATED stream buffer
                      (ablation: isolates the value of the uncertainty gating)

Every strategy is a pure function:
    select(strategy, pool, k, model, ...) -> list[PoolSample]
Order is deterministic for the same seed + same pool.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from common import PoolSample


# ── Strategy A: margin sampling ──────────────────────────────────


def _margin(pool: list[PoolSample], *, k: int) -> list[PoolSample]:
    ordered = sorted(pool, key=lambda s: (abs(s.meta_prob - 0.5), s.row_idx))
    return ordered[:k]


# ── Strategy G: random (reviewer baseline) ───────────────────────


def _random(pool: list[PoolSample], *, k: int, seed: int) -> list[PoolSample]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[int(i)] for i in idx]


# ── Strategy G': pure-random-from-stream (ablation rung) ─────────
#
# Identical draw logic to `_random`, but the caller hands it the UNGATED
# StreamPool snapshot (every seen row) instead of the gated UncertaintyPool
# snapshot. Keeping it a separate, explicitly named function makes the rung
# self-documenting in the dispatch table; the difference between this and the
# `random` rung is purely the candidate universe assembled in loop.py, which
# is exactly the uncertainty-gating effect we want to isolate.
def _pure_random_stream(pool: list[PoolSample], *, k: int, seed: int) -> list[PoolSample]:
    return _random(pool, k=k, seed=seed)


# ── Strategy B: query-by-committee ───────────────────────────────


def _qbc(
    pool: list[PoolSample], *, k: int,
    feature_cols: list[str],
    fill_means: dict[str, float],
    frozen_scaler,
    seed_X: np.ndarray,
    seed_y: np.ndarray,
    accepted_X: np.ndarray | None,
    accepted_y: np.ndarray | None,
    seed: int,
) -> list[PoolSample]:
    """Fit a small committee (LogReg, GB, RF) on KP ∪ accepted and
    pick top-K by `Var(prob across committee)`. Quick to fit for our
    20K-row scale."""
    if accepted_X is not None and len(accepted_X):
        X_train = np.vstack([seed_X, accepted_X])
        y_train = np.concatenate([seed_y, accepted_y])
    else:
        X_train = seed_X
        y_train = seed_y

    members = [
        LogisticRegression(max_iter=1000, random_state=seed, C=1.0).fit(X_train, y_train),
        GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                   learning_rate=0.1, random_state=seed).fit(X_train, y_train),
        RandomForestClassifier(n_estimators=100, max_depth=10,
                               random_state=seed, n_jobs=-1).fit(X_train, y_train),
    ]
    # Pool features → matrix.
    X_pool = np.empty((len(pool), len(feature_cols)))
    for i, s in enumerate(pool):
        for j, f in enumerate(feature_cols):
            X_pool[i, j] = float(s.feature_vec.get(f, fill_means.get(f, 0.0)))
    X_pool_s = frozen_scaler.transform(X_pool)
    probs = np.column_stack([m.predict_proba(X_pool_s)[:, 1] for m in members])
    var = probs.var(axis=1)
    order = np.argsort(-var)
    return [pool[int(i)] for i in order[:k]]


# ── Strategy C: core-set / k-Center on SHAP-top-5 ───────────────


def _shap_top_vectors(
    pool: list[PoolSample], shap_top_features: tuple[str, ...],
) -> np.ndarray:
    mat = np.zeros((len(pool), len(shap_top_features)), dtype=float)
    for i, s in enumerate(pool):
        for j, f in enumerate(shap_top_features):
            mat[i, j] = float(s.feature_vec.get(f, 0.0))
    return mat


def _core_set(
    pool: list[PoolSample], *, k: int,
    shap_top_features: tuple[str, ...],
) -> list[PoolSample]:
    """Greedy farthest-point traversal on the SHAP-top feature space.
    Seed point: smallest margin (most uncertain) in pool."""
    if k >= len(pool):
        return list(pool)
    mat = _shap_top_vectors(pool, shap_top_features)
    margins = np.array([abs(s.meta_prob - 0.5) for s in pool])
    seed_idx = int(np.argmin(margins))
    picked = [seed_idx]
    min_dist = np.linalg.norm(mat - mat[seed_idx], axis=1)
    min_dist[seed_idx] = -1.0
    while len(picked) < k:
        next_i = int(np.argmax(min_dist))
        picked.append(next_i)
        new_d = np.linalg.norm(mat - mat[next_i], axis=1)
        min_dist = np.minimum(min_dist, new_d)
        min_dist[next_i] = -1.0
    return [pool[i] for i in picked]


# ── Strategy D: BADGE — k-Means++ on gradient embeddings ─────────


def _badge(
    pool: list[PoolSample], *, k: int,
    feature_cols: list[str],
    fill_means: dict[str, float],
    frozen_scaler,
    logreg: LogisticRegression,
    seed: int,
) -> list[PoolSample]:
    """For each pool sample, compute the LogReg gradient embedding
    g(x) = x_scaled * (p(x) - argmax p(x)). Cluster with k-Means++
    init only (no Lloyd iterations needed for sample selection)."""
    if k >= len(pool):
        return list(pool)
    X = np.empty((len(pool), len(feature_cols)))
    for i, s in enumerate(pool):
        for j, f in enumerate(feature_cols):
            X[i, j] = float(s.feature_vec.get(f, fill_means.get(f, 0.0)))
    X_s = frozen_scaler.transform(X)
    p = logreg.predict_proba(X_s)[:, 1]
    pred = (p > 0.5).astype(int)
    grad = X_s * (p - pred).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    n = grad.shape[0]
    # k-Means++ init: first center is a random point, subsequent
    # centers are picked with prob proportional to D(x)^2.
    first = int(rng.integers(n))
    centers = [first]
    dists = np.linalg.norm(grad - grad[first], axis=1) ** 2
    while len(centers) < k:
        total = dists.sum()
        if total <= 0:
            # Degenerate: all remaining gradients equal → fall back to random.
            remaining = [i for i in range(n) if i not in centers]
            rng.shuffle(remaining)
            centers.extend(remaining[: k - len(centers)])
            break
        probs = dists / total
        nxt = int(rng.choice(n, p=probs))
        if nxt in centers:
            # extremely unlikely with continuous distances; resample.
            nxt = int(np.argmax(dists))
            if nxt in centers:
                break
        centers.append(nxt)
        new_d = np.linalg.norm(grad - grad[nxt], axis=1) ** 2
        dists = np.minimum(dists, new_d)
    return [pool[i] for i in centers[:k]]


# ── Strategy E: drift-anchored (novel) ──────────────────────────


def _drift_anchored(
    pool: list[PoolSample], *, k: int,
    feature_cols: list[str],
    fill_means: dict[str, float],
    per_feature_drift_kl: dict[str, float],
    kp_means: dict[str, float],
    alpha: float = 0.5,
) -> list[PoolSample]:
    """Margin reweighted by per-feature drift.

    score(x) = (1 / (margin(x) + eps)) * (1 + α · Σ_i δ_i · |x_i - μ_kp,i|)

    where δ_i is the bin-wise KL drift on feature i and μ_kp,i is the
    KP_train mean. High score = uncertain AND lies along the drifted
    dimensions."""
    if k >= len(pool):
        return list(pool)
    drift_features = [f for f in feature_cols if f in per_feature_drift_kl]
    if not drift_features:
        return _margin(pool, k=k)
    scores = np.zeros(len(pool))
    eps = 1e-6
    for i, s in enumerate(pool):
        margin = abs(s.meta_prob - 0.5) + eps
        drift_score = 0.0
        for f in drift_features:
            x = float(s.feature_vec.get(f, fill_means.get(f, 0.0)))
            mu = float(kp_means.get(f, 0.0))
            delta = float(per_feature_drift_kl.get(f, 0.0))
            drift_score += delta * abs(x - mu)
        scores[i] = (1.0 / margin) * (1.0 + alpha * drift_score)
    order = np.argsort(-scores)
    return [pool[int(i)] for i in order[:k]]


# ── Strategy F: hybrid 60/25/15 (canonical) ──────────────────────


def _hybrid(pool: list[PoolSample], *, k: int, **kwargs) -> list[PoolSample]:
    """60% margin + 25% core-set + 15% qbc, dedup-by-id with margin refill."""
    k_margin = round(k * 0.60)
    k_core = round(k * 0.25)
    k_qbc = k - k_margin - k_core

    seen: set[int] = set()
    merged: list[PoolSample] = []

    def _take(items: list[PoolSample]) -> None:
        for s in items:
            if s.sample_id in seen:
                continue
            seen.add(s.sample_id)
            merged.append(s)
            if len(merged) >= k:
                return

    _take(_margin(pool, k=k_margin))
    _take(_core_set(pool, k=k_core,
                    shap_top_features=kwargs["shap_top_features"]))
    if len(merged) < k:
        _take(_qbc(
            pool, k=k_qbc,
            feature_cols=kwargs["feature_cols"],
            fill_means=kwargs["fill_means"],
            frozen_scaler=kwargs["frozen_scaler"],
            seed_X=kwargs["seed_X"],
            seed_y=kwargs["seed_y"],
            accepted_X=kwargs.get("accepted_X"),
            accepted_y=kwargs.get("accepted_y"),
            seed=kwargs["seed"],
        ))
    # Refill from margin if dedup left us short.
    if len(merged) < k:
        _take(_margin(pool, k=k))
    return merged[:k]


# ── Dispatcher ───────────────────────────────────────────────────


def select(
    strategy: str,
    pool: list[PoolSample],
    *,
    k: int,
    seed: int,
    feature_cols: list[str],
    fill_means: dict[str, float],
    frozen_scaler,
    seed_X: np.ndarray,
    seed_y: np.ndarray,
    accepted_X: np.ndarray | None = None,
    accepted_y: np.ndarray | None = None,
    logreg: LogisticRegression | None = None,
    shap_top_features: tuple[str, ...] = (),
    per_feature_drift_kl: dict[str, float] | None = None,
    kp_means: dict[str, float] | None = None,
    drift_alpha: float = 0.5,
) -> list[PoolSample]:
    if not pool:
        return []
    k = min(k, len(pool))
    if strategy == "margin":
        return _margin(pool, k=k)
    if strategy == "random":
        return _random(pool, k=k, seed=seed)
    if strategy == "pure_random_stream":
        return _pure_random_stream(pool, k=k, seed=seed)
    if strategy == "qbc":
        return _qbc(pool, k=k,
                    feature_cols=feature_cols, fill_means=fill_means,
                    frozen_scaler=frozen_scaler,
                    seed_X=seed_X, seed_y=seed_y,
                    accepted_X=accepted_X, accepted_y=accepted_y,
                    seed=seed)
    if strategy == "core_set":
        return _core_set(pool, k=k, shap_top_features=shap_top_features)
    if strategy == "badge":
        if logreg is None:
            raise ValueError("badge requires a logreg model")
        return _badge(pool, k=k,
                      feature_cols=feature_cols, fill_means=fill_means,
                      frozen_scaler=frozen_scaler, logreg=logreg,
                      seed=seed)
    if strategy == "drift_anchored":
        return _drift_anchored(
            pool, k=k, feature_cols=feature_cols, fill_means=fill_means,
            per_feature_drift_kl=per_feature_drift_kl or {},
            kp_means=kp_means or {},
            alpha=drift_alpha,
        )
    if strategy == "hybrid":
        return _hybrid(
            pool, k=k,
            feature_cols=feature_cols, fill_means=fill_means,
            frozen_scaler=frozen_scaler,
            seed_X=seed_X, seed_y=seed_y,
            accepted_X=accepted_X, accepted_y=accepted_y,
            shap_top_features=shap_top_features,
            seed=seed,
        )
    raise ValueError(f"Unknown strategy: {strategy!r}")
