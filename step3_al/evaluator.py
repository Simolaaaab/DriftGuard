"""evaluator.py — Per-round metrics on KP_test / PP_test / DP_probe.

Reports all four directionally-meaningful numbers (acc, f1w, p/r
per class, AUC, ECE) so the paper's tables can be built without
re-running anything.

ECE = Expected Calibration Error (Naeini 2015, 10-bin).
Reported because incremental retraining can drift calibration; we
verify it stays bounded along the AL trajectory.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard 10-bin ECE for the positive class."""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(in_bin):
            continue
        acc = float(np.mean(y_true[in_bin] == 1))
        conf = float(np.mean(y_prob[in_bin]))
        ece += abs(acc - conf) * (in_bin.sum() / len(y_true))
    return ece


def evaluate(model, scaler, X_raw: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate `model` (post-scaler) on `(X_raw, y)`. Returns a dict
    with all metrics. `X_raw` is unscaled — scaler.transform applied
    inside."""
    if len(y) == 0:
        return {k: float("nan") for k in
                ("accuracy", "f1_weighted", "f1_phish",
                 "precision_phish", "recall_phish",
                 "precision_benign", "recall_benign",
                 "auc", "ece")} | {"n": 0}
    X = scaler.transform(X_raw)
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    n_classes_truth = len(set(y.tolist()))
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "f1_phish": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision_phish": float(
            precision_score(y, pred, pos_label=1, zero_division=0)
        ),
        "recall_phish": float(
            recall_score(y, pred, pos_label=1, zero_division=0)
        ),
        "precision_benign": float(
            precision_score(y, pred, pos_label=0, zero_division=0)
        ),
        "recall_benign": float(
            recall_score(y, pred, pos_label=0, zero_division=0)
        ),
        "auc": (float(roc_auc_score(y, prob))
                if n_classes_truth > 1 else float("nan")),
        "ece": _ece(y, prob),
    }
